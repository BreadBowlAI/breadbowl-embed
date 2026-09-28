"""Retrieval scoring primitives shared by training and eval."""

from __future__ import annotations

import math
from functools import wraps
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .architecture import QKVOutput
def normalize_embeddings(x):
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    return F.normalize(x, p=2, dim=-1)

QKMode = Literal["max", "logsumexp", "logmeanexp"]
QK_MODES: tuple[QKMode, ...] = ("max", "logsumexp", "logmeanexp")


def validate_qk_mode(mode: str, name: str = "qk_mode") -> QKMode:
    if mode not in QK_MODES:
        raise ValueError(f"Unsupported {name}: {mode!r}; expected one of {QK_MODES}.")
    return cast(QKMode, mode)


# Bound each materialized value readout to ~16 MiB in float32. Large training
# calls recompute blocks during backward instead of retaining every readout.
_VALUE_BLOCK_ELEMENTS = 4 * 1024 * 1024


def validate_temperature(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return value


def _attention_temperature(tau_v: float | None, dimension: int) -> float:
    # Missing temperature preserves the routing of pre-v1.1 checkpoints.
    return (
        math.sqrt(dimension) if tau_v is None else validate_temperature(tau_v, "tau_v")
    )


def _stable_scoring(fn):
    """Disable autocast locally and promote half inputs, preserving float64."""

    @wraps(fn)
    def wrapped(*args, **kwargs):
        def promote(x):
            if isinstance(x, torch.Tensor) and x.dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                return x.float()
            return x

        tensors = [x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)]
        with torch.autocast(device_type=tensors[0].device.type, enabled=False):
            return fn(
                *(promote(x) for x in args),
                **{k: promote(v) for k, v in kwargs.items()},
            )

    return wrapped


def _value_scores(q, k, qv, v, temperature: float, *, grouped: bool):
    """Direct readout cosine, with bounded blocks and differentiable recomputation."""
    candidates = k.size(1 if grouped else 0)
    if q.size(0) == 0 or candidates == 0:
        return q.new_empty((q.size(0), candidates))
    pair_budget = max(1, _VALUE_BLOCK_ELEMENTS // (q.size(1) * v.size(-1)))
    query_chunk = min(q.size(0), max(1, math.isqrt(pair_budget)))
    candidate_chunk = max(1, pair_budget // query_chunk)
    recompute = (
        torch.is_grad_enabled()
        and any(x.requires_grad for x in (q, k, qv, v))
        and q.size(0) * candidates > pair_budget
    )

    @_stable_scoring
    def block(qb, kb, qvb, vb):
        # Normalize and promote inside the checkpointed block so grouped
        # candidates do not require a full additional float32 value tensor.
        qb = normalize_embeddings(qb)
        kb = normalize_embeddings(kb)
        qvb = normalize_embeddings(qvb)
        route_eq = "bmd,bknd->bkmn" if grouped else "bmd,knd->bkmn"
        read_eq = "bkmn,bknv->bkmv" if grouped else "bkmn,knv->bkmv"
        attention = torch.softmax(torch.einsum(route_eq, qb, kb) / temperature, dim=-1)
        readout = torch.einsum(read_eq, attention, vb)
        return (qvb[:, None] * normalize_embeddings(readout)).sum(-1).mean(-1)

    rows = []
    for start in range(0, q.size(0), query_chunk):
        end = start + query_chunk
        columns = []
        for offset in range(0, candidates, candidate_chunk):
            stop = offset + candidate_chunk
            kb = k[start:end, offset:stop] if grouped else k[offset:stop]
            vb = v[start:end, offset:stop] if grouped else v[offset:stop]
            inputs = (q[start:end], kb, qv[start:end], vb)
            columns.append(
                checkpoint(block, *inputs, use_reentrant=False)
                if recompute
                else block(*inputs)
            )
        rows.append(torch.cat(columns, dim=1))
    return torch.cat(rows, dim=0)


def _aggregate_routing(sim: torch.Tensor, tau_k: float, mode: QKMode) -> torch.Tensor:
    """Reduce document then query slots, keeping historical modes unchanged."""
    if sim.size(-2) == 0 or sim.size(-1) == 0:
        raise ValueError("Routing scoring requires at least one query and document slot.")
    if mode == "max":
        return sim.max(dim=-1).values.mean(dim=-1)
    log_mass = torch.logsumexp(sim / tau_k, dim=-1)
    if mode == "logmeanexp":
        # Mellowmax: tau * log(mean(exp(cosine / tau))), averaged over query
        # slots. The score stays in cosine units; loss temperature is separate.
        return (tau_k * (log_mass - math.log(sim.size(-1)))).mean(dim=-1)
    return log_mass.sum(dim=-1)


@_stable_scoring
def score_qk(
    query_routing: torch.Tensor,
    doc_routing: torch.Tensor,
    tau_k: float = 0.02,
    mode: QKMode = "logsumexp",
) -> torch.Tensor:
    """Routing score per pair. The default preserves legacy checkpoint APIs.

    New training recipes select ``logmeanexp`` explicitly. Scoring returns
    relevance scores; contrastive temperature is applied only in the loss.
    """
    validate_temperature(tau_k, "tau_k")
    validate_qk_mode(mode)
    q = normalize_embeddings(query_routing)
    k = normalize_embeddings(doc_routing)
    sim = torch.einsum("bmd,cnd->bcmn", q, k)
    return _aggregate_routing(sim, tau_k, mode)


def score_qkv(
    query_routing: torch.Tensor,
    doc_routing: torch.Tensor,
    query_value: torch.Tensor,
    doc_value: torch.Tensor,
    tau_v: float | None = None,
) -> torch.Tensor:
    """Value-readout cosine scores; tau_v=None preserves legacy routing."""
    temperature = _attention_temperature(tau_v, query_routing.size(-1))
    return _value_scores(
        query_routing,
        doc_routing,
        query_value,
        doc_value,
        temperature,
        grouped=False,
    )


@_stable_scoring
def score_qk_grouped(
    query_routing: torch.Tensor,
    doc_routing: torch.Tensor,
    tau_k: float = 0.02,
    mode: QKMode = "logsumexp",
) -> torch.Tensor:
    """Coarse Q·K score over grouped candidates. Shape: (B, K)."""
    validate_temperature(tau_k, "tau_k")
    validate_qk_mode(mode)
    q = normalize_embeddings(query_routing)
    k = normalize_embeddings(doc_routing)
    sim = torch.einsum("bmd,bknd->bkmn", q, k)
    return _aggregate_routing(sim, tau_k, mode)


def score_qkv_grouped(
    query_routing: torch.Tensor,
    doc_routing: torch.Tensor,
    query_value: torch.Tensor,
    doc_value: torch.Tensor,
    tau_v: float | None = None,
) -> torch.Tensor:
    """Value-readout cosine scores; tau_v=None preserves legacy routing."""
    temperature = _attention_temperature(tau_v, query_routing.size(-1))
    return _value_scores(
        query_routing,
        doc_routing,
        query_value,
        doc_value,
        temperature,
        grouped=True,
    )


def score_final(
    query_out: QKVOutput,
    doc_out: QKVOutput,
    alpha: float = 0.8,
    beta: float = 0.2,
    tau_k: float = 0.02,
    qk_mode: QKMode = "logsumexp",
    tau_v: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined score and per-term scores: (final, coarse, exact)."""
    coarse = score_qk(query_out.routing, doc_out.routing, tau_k=tau_k, mode=qk_mode)
    exact = score_qkv(
        query_out.routing,
        doc_out.routing,
        query_out.value,
        doc_out.value,
        tau_v=tau_v,
    )
    final = alpha * coarse + beta * exact
    return final, coarse, exact


def _slice_output(output: QKVOutput, start: int, end: int) -> QKVOutput:
    return QKVOutput(
        routing=output.routing[start:end],
        value=output.value[start:end],
        attention=output.attention[start:end],
        slot_states=None
        if output.slot_states is None
        else output.slot_states[start:end],
        slot_queries=output.slot_queries,
    )


def score_final_chunked(
    query_out: QKVOutput,
    doc_out: QKVOutput,
    alpha: float = 0.8,
    beta: float = 0.2,
    tau_k: float = 0.02,
    qk_mode: QKMode = "logsumexp",
    tau_v: float | None = None,
    *,
    doc_chunk_size: int | None = None,
    compute_qkv: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined score with optional document-axis chunking.

    The returned tensors have the same shape as :func:`score_final`. Chunking
    only reduces intermediate all-pairs memory; it intentionally does not change
    the loss surface.
    """
    doc_count = int(doc_out.routing.size(0))
    if doc_chunk_size is None or doc_chunk_size <= 0 or doc_chunk_size >= doc_count:
        coarse = score_qk(query_out.routing, doc_out.routing, tau_k=tau_k, mode=qk_mode)
        if compute_qkv:
            exact = score_qkv(
                query_out.routing,
                doc_out.routing,
                query_out.value,
                doc_out.value,
                tau_v=tau_v,
            )
            return alpha * coarse + beta * exact, coarse, exact
        exact = coarse.new_zeros(coarse.shape)
        return alpha * coarse, coarse, exact

    coarse_parts: list[torch.Tensor] = []
    exact_parts: list[torch.Tensor] = []
    final_parts: list[torch.Tensor] = []
    for start in range(0, doc_count, doc_chunk_size):
        chunk = _slice_output(doc_out, start, min(doc_count, start + doc_chunk_size))
        coarse = score_qk(query_out.routing, chunk.routing, tau_k=tau_k, mode=qk_mode)
        if compute_qkv:
            exact = score_qkv(
                query_out.routing,
                chunk.routing,
                query_out.value,
                chunk.value,
                tau_v=tau_v,
            )
            final = alpha * coarse + beta * exact
        else:
            exact = coarse.new_zeros(coarse.shape)
            final = alpha * coarse
        coarse_parts.append(coarse)
        exact_parts.append(exact)
        final_parts.append(final)
    return (
        torch.cat(final_parts, dim=1),
        torch.cat(coarse_parts, dim=1),
        torch.cat(exact_parts, dim=1),
    )


def score_final_grouped(
    query_out: QKVOutput,
    doc_out: QKVOutput,
    alpha: float = 0.8,
    beta: float = 0.2,
    tau_k: float = 0.02,
    qk_mode: QKMode = "logsumexp",
    tau_v: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined score and per-term scores over grouped candidates: (final, coarse, exact)."""
    coarse = score_qk_grouped(
        query_out.routing, doc_out.routing, tau_k=tau_k, mode=qk_mode
    )
    exact = score_qkv_grouped(
        query_out.routing,
        doc_out.routing,
        query_out.value,
        doc_out.value,
        tau_v=tau_v,
    )
    final = alpha * coarse + beta * exact
    return final, coarse, exact


def infonce_loss(score_matrix: torch.Tensor) -> torch.Tensor:
    """In-batch negative cross-entropy; labels = diagonal (query i matches doc i)."""
    batch_size = score_matrix.size(0)
    labels = torch.arange(batch_size, device=score_matrix.device)
    return F.cross_entropy(score_matrix, labels)


@_stable_scoring
def pair_route_attention(
    query_routing: torch.Tensor,
    doc_routing: torch.Tensor,
    tau_v: float | None = None,
) -> torch.Tensor:
    """Positive-pair slot routing attention. Shape: (B, query_slots, doc_slots)."""
    q = normalize_embeddings(query_routing)
    k = normalize_embeddings(doc_routing)
    logits = torch.einsum("bsd,btd->bst", q, k) / _attention_temperature(
        tau_v, q.size(-1)
    )
    return torch.softmax(logits, dim=-1)


@_stable_scoring
def pair_value_readout(
    attention: torch.Tensor, doc_value: torch.Tensor
) -> torch.Tensor:
    """Aggregate document value slots for aligned positive pairs."""
    return torch.einsum("bst,btv->bsv", attention, doc_value)

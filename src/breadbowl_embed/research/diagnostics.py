"""Slot diagnostics over a collection, and diagnostics of the per-slot search pipeline.

`slot_statistics` answers: do slots differ from each other (or collapse onto the same
direction)? Does each slot vary across texts? How many directions do the slots use? Which
slots win in scoring? `shortlist_diagnostics` answers: how close does a slot-shortlist profile
get to exact routing search, which query slots find the exact top documents, and how many
queries reach fewer candidates than the rerank width (underfill)?
"""
from __future__ import annotations

from dataclasses import dataclass
import statistics as stats

import numpy as np
import torch
import torch.nn.functional as F

from ..embeddings import Embeddings
from ..retrieval import EXACT, SLOT_SHORTLIST, Index, resolve_profile
from ..scoring import score
from ._base import ResearchObject
from ._metrics import effective_rank
from .decomposition import _pair
from .rules import base_config, resolve_rule


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    return matrix[mask]


@dataclass(frozen=True, eq=False)
class SlotStatistics(ResearchObject):
    """Per-slot and between-slot statistics of a collection of encoded texts.

    per_slot: one row per slot with
      routing_resultant   |mean over texts of the unit routing vector| (1: the slot points the same
                          way for every text; near 0: it varies freely);
      routing_variance    1 - routing_resultant^2, the total variance of the unit routing vectors;
      routing_effective_rank  effective rank of the slot's centred routing vectors across texts;
      value_norm_mean/std mean and spread of the raw value norm;
      value_relative_variance  total variance of the raw value vectors / their mean squared norm;
      win_rate, score_share (with queries): how often the slot is a query slot's best match and its
                          share of the final score over the scored pairs.
    within_text_routing_cosine [S, S]: mean over texts of the cosine between two slots of the same
      text; off-diagonal values near 1 mean those slots collapsed onto one direction.
    centroid_cosine [S, S]: cosine between the slots' mean routing directions.
    per_text_effective_rank: effective rank of each text's [S, d_k] routing matrix (S = no collapse).
    """
    json_type = "slot_statistics"

    kind: str
    count: int
    slots: int
    per_slot: tuple
    within_text_routing_cosine: torch.Tensor
    within_text_value_cosine: torch.Tensor
    centroid_cosine: torch.Tensor
    per_text_effective_rank: dict
    collection_effective_rank: float
    collapse: dict
    usage: dict | None

    def _payload(self) -> dict:
        return {"kind": self.kind, "count": self.count, "slots": self.slots, "collapse": self.collapse,
                "per_text_effective_rank": self.per_text_effective_rank,
                "collection_effective_rank": self.collection_effective_rank, "per_slot": list(self.per_slot),
                "within_text_routing_cosine": self.within_text_routing_cosine,
                "within_text_value_cosine": self.within_text_value_cosine, "centroid_cosine": self.centroid_cosine,
                "usage": self.usage}

    def _html(self) -> str:
        from ._html import slot_statistics_html
        return slot_statistics_html(self)

    def plot(self, **kwargs):
        from .plot import slot_statistics
        return slot_statistics(self, **kwargs)


def _collection(source, sample: int | None) -> Embeddings:
    if isinstance(source, Index):
        positions = np.arange(len(source))
        if sample is not None and sample < len(source):
            positions = np.unique(np.linspace(0, len(source) - 1, sample).round().astype(np.int64))
        routing = F.normalize(source.routing_vectors(positions), dim=-1)
        return Embeddings(routing, source.value_vectors(positions), source.representation_id, "document")
    if not isinstance(source, Embeddings):
        raise TypeError("Pass Embeddings or an Index.")
    if sample is not None and sample < len(source):
        source = source.select(np.unique(np.linspace(0, len(source) - 1, sample).round().astype(np.int64)).tolist())
    return source.without_tokens()


def _usage(queries: Embeddings, documents: Embeddings, top_k: int, rule) -> dict:
    """Which slots carry the score over each query's top-k documents."""
    config = rule.config()
    scores = score(queries, documents, config=config).combined
    k = min(top_k, len(documents))
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    q = F.normalize(queries.routing.float(), dim=-1)
    d = F.normalize(documents.routing.float(), dim=-1)
    sq, sd = queries.num_slots, documents.num_slots
    wins = torch.zeros(sq, sd, dtype=torch.float64)
    weight = torch.zeros(sq, sd, dtype=torch.float64)
    contribution = torch.zeros(sq, sd, dtype=torch.float64)
    entropy = 0.0
    pairs = 0
    for i in range(len(queries)):
        for j in order[i].tolist():
            parts = _pair(q[i], d[j], queries.value[i], documents.value[j], rule)
            sim = parts["similarity"]
            wins[torch.arange(sq), sim.argmax(dim=1)] += 1
            weight += parts["routing_weight"]
            contribution += ((1 - rule.value_mix) * parts["routing_pair_contribution"]
                             + rule.value_mix * parts["value_pair_contribution"])
            entropy += (1 - rule.value_mix) * float(parts["routing_entropy_term"].sum())
            pairs += 1
    if not pairs:
        return {}
    total = float(contribution.sum()) + entropy
    return {"pairs": pairs, "top_k": k, "rule": rule.to_dict(),
            "best_match_counts": wins, "mean_routing_weight": weight / pairs,
            "mean_pair_contribution": contribution / pairs,
            "document_slot_win_rate": wins.sum(0) / wins.sum(),
            "document_slot_score_share": contribution.sum(0) / total if total else contribution.sum(0) * 0,
            "query_slot_score_share": contribution.sum(1) / total if total else contribution.sum(1) * 0,
            "mean_final_score": total / pairs}


@torch.inference_mode()
def slot_statistics(source, *, queries: Embeddings | None = None, top_k: int = 10, sample: int | None = None,
                    rule=None, base=None) -> SlotStatistics:
    """Statistics of the slots of `source` (Embeddings or an Index; see SlotStatistics).

    queries: optional query Embeddings; adds slot usage over each query's top_k documents of
    `source` (by final score under `rule`). sample: use an evenly spaced subset of this many texts.
    """
    data = _collection(source, sample)
    n, slots = len(data), data.num_slots
    if n == 0:
        raise ValueError("No texts to describe.")
    routing = F.normalize(data.routing.double(), dim=-1)
    value = data.value.double()
    mean_routing = routing.mean(dim=0)
    resultant = mean_routing.norm(dim=-1)
    value_norm = value.norm(dim=-1)
    value_variance = (value - value.mean(dim=0)).pow(2).sum(-1).mean(dim=0)
    unit_value = F.normalize(value, dim=-1)
    within_routing = torch.einsum("nsd,ntd->st", routing, routing) / n
    within_value = torch.einsum("nsd,ntd->st", unit_value, unit_value) / n
    centroid = F.normalize(mean_routing, dim=-1)
    centroid_cosine = centroid @ centroid.T
    per_text = [effective_rank(routing[i]) for i in range(n)]
    per_text_value = [effective_rank(value[i]) for i in range(n)]
    slot_rank = [effective_rank(routing[:, s] - mean_routing[s]) if n > 1 else 0.0 for s in range(slots)]
    off = _off_diagonal(within_routing)
    flat = within_routing.clone()
    flat.fill_diagonal_(-2)
    a, b = divmod(int(flat.argmax()), slots)
    usage = None
    if queries is not None:
        resolved = resolve_rule(rule, base_config(base))
        usage = _usage(queries.without_tokens(), data, top_k, resolved)
    per_slot = []
    for s in range(slots):
        row = {"slot": s, "routing_resultant": float(resultant[s]), "routing_variance": float(1 - resultant[s] ** 2),
               "routing_effective_rank": slot_rank[s], "value_norm_mean": float(value_norm[:, s].mean()),
               "value_norm_std": float(value_norm[:, s].std(unbiased=False)),
               "value_relative_variance": float(value_variance[s] / value_norm[:, s].pow(2).mean().clamp_min(1e-30))}
        if usage:
            row["win_rate"] = float(usage["document_slot_win_rate"][s])
            row["score_share"] = float(usage["document_slot_score_share"][s])
        per_slot.append(row)
    collapse = {"mean_within_text_cosine": float(off.mean()) if len(off) else 1.0,
                "max_within_text_cosine": float(within_routing[a, b]) if slots > 1 else 1.0,
                "most_similar_pair": [a, b] if slots > 1 else [],
                "mean_centroid_cosine": float(_off_diagonal(centroid_cosine).mean()) if slots > 1 else 1.0}
    rank_summary = {"routing_mean": float(np.mean(per_text)), "routing_min": float(np.min(per_text)),
                    "routing_max": float(np.max(per_text)), "value_mean": float(np.mean(per_text_value)),
                    "value_min": float(np.min(per_text_value)), "slots": slots}
    collection_rank = effective_rank(routing.reshape(-1, routing.shape[-1]))
    return SlotStatistics(data.kind, n, slots, tuple(per_slot), within_routing.float(), within_value.float(),
                          centroid_cosine.float(), rank_summary, collection_rank, collapse, usage)


# ---- per-slot search pipeline --------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ShortlistDiagnostics(ResearchObject):
    """A slot-shortlist profile (at one or several slot-hit widths) against exact routing search.

    rows: one per width with
      documents_hit_mean     distinct documents hit by any query slot;
      candidates_mean/min    documents reaching value reranking (at most the rerank width);
      underfilled            queries with fewer candidates than min(rerank, corpus size);
      candidate_recall       share of the exact-routing top `rerank` documents among the candidates;
      result_recall_at_k     share of the exact profile's top_k results the profile also returns;
      top1_agreement         queries whose first result equals the exact profile's.
    per_slot_hit_recall[width]: [query slots] share of the exact-routing top `rerank` documents
      that each query slot hit (mean over queries): which query slots find the right documents.
    """
    json_type = "shortlist_diagnostics"

    profile: dict
    reference: dict
    queries: int
    documents: int
    top_k: int
    rows: tuple
    per_slot_hit_recall: dict
    per_query: tuple | None

    def _payload(self) -> dict:
        return {"profile": self.profile, "reference": self.reference, "queries": self.queries,
                "documents": self.documents, "top_k": self.top_k, "rows": list(self.rows),
                "per_slot_hit_recall": self.per_slot_hit_recall, "per_query": self.per_query}

    def _html(self) -> str:
        from ._html import table_html
        return table_html("Shortlist diagnostics", list(self.rows),
                          note=f"{self.queries} queries, {self.documents} documents; reference: exact routing "
                               f"(rerank {self.reference['rerank']}).")

    def plot(self, **kwargs):
        from .plot import shortlist
        return shortlist(self, **kwargs)


@torch.inference_mode()
def shortlist_diagnostics(index: Index, queries: Embeddings, *, profile="paper", slot_hits=None, top_k: int = 10,
                          backend: str = "torch", batch_size: int = 128, per_query: bool = False) -> ShortlistDiagnostics:
    """Compare a slot-shortlist profile with exact routing search on `queries` (see ShortlistDiagnostics).

    profile: a slot-shortlist profile or its name ("paper", "default"). slot_hits: a width or a list
    of widths to sweep (default: the profile's). The profile's results are produced by the same
    code as `Index.search`, and the reference is `Index.search(profile="exact")` with the same
    rerank width. Exact routing scores every document once per query.
    """
    base = resolve_profile(profile)
    if base.method != SLOT_SHORTLIST:
        raise ValueError("shortlist_diagnostics needs a slot-shortlist profile (e.g. 'paper' or 'default').")
    widths = [base.slot_hits] if slot_hits is None else ([slot_hits] if isinstance(slot_hits, int) else list(slot_hits))
    if queries.kind != "query" or queries.representation_id != index.representation_id:
        raise ValueError("Pass query embeddings from the model this index was built with.")
    queries = queries.to("cpu").without_tokens()
    n = len(index)
    k = min(top_k, n)
    rerank = base.rerank
    reference_profile = EXACT.replace(rerank=rerank).to_dict()
    exact_top, exact_results = [], []
    for i in range(len(queries)):
        query = queries.select(i)
        top = index._exact_candidates(query, rerank, batch_size)
        exact_top.append(set(top))
        exact_results.append([r.document_index for r in index._rerank(query, top, k, batch_size, False, None,
                                                                      reference_profile)])
    rows, per_slot, details = [], {}, []
    for width in widths:
        current = resolve_profile(base, slot_hits=width)
        slot_recall = torch.zeros(index.slots, dtype=torch.float64)
        metrics = []
        for start in range(0, len(queries), 32):
            batch = queries.select(slice(start, start + 32))
            candidates = index._reference_candidates(batch, current, backend, batch_size)
            for offset, (selected, hits) in enumerate(candidates):
                i = start + offset
                results = [r.document_index for r in index._rerank(batch.select(offset), selected, k, batch_size,
                                                                   False, None, current.to_dict())]
                documents_hit = 0
                if hits is not None:
                    hit_ids = hits[1]
                    valid = hit_ids >= 0
                    documents_hit = len(np.unique(hit_ids[valid] // index.slots))
                    for s in range(index.slots):
                        slot_docs = set((hit_ids[s][hit_ids[s] >= 0] // index.slots).tolist())
                        slot_recall[s] += len(slot_docs & exact_top[i]) / max(1, len(exact_top[i]))
                reference = exact_top[i]
                row = {"query": i, "documents_hit": documents_hit, "candidates": len(selected),
                       "underfilled": len(selected) < min(rerank, n),
                       "candidate_recall": len(set(selected) & reference) / max(1, len(reference)),
                       "result_recall_at_k": len(set(results) & set(exact_results[i])) / max(1, k),
                       "top1_agreement": bool(results[:1] == exact_results[i][:1])}
                metrics.append(row)
                if per_query:
                    details.append({"slot_hits": width, **row, "results": results, "exact_results": exact_results[i]})
        count = max(1, len(metrics))
        candidates = [m["candidates"] for m in metrics] or [0]
        rows.append({"slot_hits": width, "shortlist": current.shortlist, "rerank": rerank, "profile": current.name,
                     "documents_hit_mean": sum(m["documents_hit"] for m in metrics) / count,
                     "candidates_mean": sum(candidates) / count, "candidates_min": min(candidates),
                     "candidates_median": float(stats.median(candidates)),
                     "underfilled": sum(m["underfilled"] for m in metrics),
                     "underfill_rate": sum(m["underfilled"] for m in metrics) / count,
                     "candidate_recall": sum(m["candidate_recall"] for m in metrics) / count,
                     "result_recall_at_k": sum(m["result_recall_at_k"] for m in metrics) / count,
                     "top1_agreement": sum(m["top1_agreement"] for m in metrics) / count})
        per_slot[str(width)] = (slot_recall / count).tolist()
    return ShortlistDiagnostics(base.to_dict(), reference_profile, len(queries), n, k, tuple(rows), per_slot,
                                tuple(details) if per_query else None)

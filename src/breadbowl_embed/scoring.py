"""Public bounded late-interaction scoring API and per-pair explanations."""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ._scoring import score_qk, score_qkv
from .config import ScoringConfig
from .embeddings import Embeddings


@dataclass(frozen=True)
class Scores:
    """[queries, documents] matrices: combined = (1 - value_mix) * routing + value_mix * value."""
    combined: torch.Tensor
    routing: torch.Tensor
    value: torch.Tensor

    @property
    def final(self) -> torch.Tensor:
        return self.combined


def validate_pair(queries: Embeddings, documents: Embeddings):
    if queries.kind != "query" or documents.kind != "document":
        raise ValueError("Expected query embeddings followed by document embeddings.")
    if queries.representation_id != documents.representation_id:
        raise ValueError("Incompatible representations: re-encode with the same model and preprocessing.")
    if queries.routing.shape[-1] != documents.routing.shape[-1] or queries.value.shape[-1] != documents.value.shape[-1]:
        raise ValueError("Query/document dimensions differ.")
    if queries.routing.device != documents.routing.device or queries.value.device != documents.value.device:
        raise ValueError("Move queries and documents to the same device before scoring.")


def _check_batches(query_batch_size: int, document_batch_size: int) -> None:
    if query_batch_size <= 0 or document_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")


def _blocks(count: int, size: int):
    return [slice(start, start + size) for start in range(0, count, size)]


def _routing_block(query_routing: torch.Tensor, document_routing: torch.Tensor, config: ScoringConfig):
    return score_qk(query_routing, document_routing, tau_k=config.tau_k, mode=config.qk_mode)


def score_routing(queries: Embeddings, documents: Embeddings | torch.Tensor, *,
                  config: ScoringConfig | None = None, query_batch_size: int = 8,
                  document_batch_size: int = 128) -> torch.Tensor:
    """[queries, documents] routing scores R alone, for documents kept without value slots.

    `documents` is document Embeddings or a routing tensor [documents, slots, d_k]. Same blocks
    and equations as `score`, so with the same batch sizes the result equals
    `score(...).routing` bit for bit. A bare tensor carries no representation identity: the
    caller is responsible for it coming from the same model.
    """
    config = config or ScoringConfig()
    _check_batches(query_batch_size, document_batch_size)
    if queries.kind != "query":
        raise ValueError("Expected query embeddings.")
    if isinstance(documents, Embeddings):
        if documents.kind != "document" or documents.representation_id != queries.representation_id:
            raise ValueError("Incompatible representations: re-encode with the same model and preprocessing.")
        document_routing = documents.routing
    else:
        document_routing = documents
        if not isinstance(document_routing, torch.Tensor) or document_routing.ndim != 3 \
                or not document_routing.is_floating_point():
            raise ValueError("Document routing must be a floating tensor [documents, slots, d_k].")
    if document_routing.shape[-1] != queries.routing.shape[-1]:
        raise ValueError("Query/document dimensions differ.")
    if document_routing.device != queries.routing.device:
        raise ValueError("Move queries and documents to the same device before scoring.")
    if not len(queries) or not len(document_routing):
        return torch.empty((len(queries), len(document_routing)), device=queries.routing.device)
    return torch.cat([torch.cat([_routing_block(queries.routing[qs], document_routing[ds], config)
                                 for ds in _blocks(len(document_routing), document_batch_size)], dim=1)
                      for qs in _blocks(len(queries), query_batch_size)])


def score(queries: Embeddings, documents: Embeddings, *, config: ScoringConfig | None = None,
          query_batch_size: int = 8, document_batch_size: int = 128) -> Scores:
    """Return [queries, documents] scores; block intermediate attention tensors.

    Raw document value magnitudes are preserved before attention aggregation.
    The returned matrix itself still occupies O(queries * documents) memory.
    """
    validate_pair(queries, documents)
    config = config or ScoringConfig()
    _check_batches(query_batch_size, document_batch_size)
    if not len(queries) or not len(documents):
        empty = torch.empty((len(queries), len(documents)), device=queries.routing.device)
        return Scores(empty, empty.clone(), empty.clone())
    qk_rows, value_rows = [], []
    for qs in _blocks(len(queries), query_batch_size):
        routes, values = [], []
        for ds in _blocks(len(documents), document_batch_size):
            routes.append(_routing_block(queries.routing[qs], documents.routing[ds], config))
            values.append(score_qkv(queries.routing[qs], documents.routing[ds], queries.value[qs],
                                    documents.value[ds], tau_v=config.tau_v))
        qk_rows.append(torch.cat(routes, dim=1))
        value_rows.append(torch.cat(values, dim=1))
    routing, value = torch.cat(qk_rows), torch.cat(value_rows)
    return Scores((1 - config.value_mix) * routing + config.value_mix * value, routing, value)


@dataclass(frozen=True)
class Explanation:
    """How one query-document score decomposes over slots.

    slot_similarity      [query slots, doc slots] cosine of routing vectors.
    routing_contribution [query slots] each query slot's term before the mean over query slots:
                         tau_k * log(mean_j exp(cos_ij / tau_k)) for logmeanexp, max_j cos_ij
                         for max. Their mean is routing_score (R).
    value_attention      [query slots, doc slots] softmax(cos / tau_v) over document slots: how
                         much of each document value slot a query slot reads.
    value_slot_cosine    [query slots] cosine between the query value slot and its readout of
                         raw document values. Their mean is value_score (V).
    score                (1 - value_mix) * R + value_mix * V, the ranking score ("final").
    """
    query_index: int
    document_index: int
    slot_similarity: torch.Tensor
    routing_contribution: torch.Tensor
    value_attention: torch.Tensor
    value_slot_cosine: torch.Tensor
    routing_score: float
    value_score: float
    score: float
    settings: dict

    @property
    def best_document_slot(self) -> torch.Tensor:
        """For each query slot, the document slot with the highest routing cosine."""
        return self.slot_similarity.argmax(dim=-1)

    def slot_matches(self) -> list[dict]:
        best = self.best_document_slot
        return [{"query_slot": i, "document_slot": int(best[i]),
                 "cosine": float(self.slot_similarity[i, best[i]]),
                 "routing_contribution": float(self.routing_contribution[i]),
                 "value_attention": float(self.value_attention[i, best[i]]),
                 "value_slot_cosine": float(self.value_slot_cosine[i])}
                for i in range(self.slot_similarity.shape[0])]

    def to_dict(self) -> dict:
        return {"query_index": self.query_index, "document_index": self.document_index,
                "score": self.score, "routing_score": self.routing_score, "value_score": self.value_score,
                "settings": dict(self.settings),
                "slot_similarity": self.slot_similarity.tolist(),
                "routing_contribution": self.routing_contribution.tolist(),
                "value_attention": self.value_attention.tolist(),
                "value_slot_cosine": self.value_slot_cosine.tolist(),
                "slot_matches": self.slot_matches()}


def _promote(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def explain(queries: Embeddings, documents: Embeddings, *,
            config: ScoringConfig | None = None) -> list[list[Explanation]]:
    """Slot-level decomposition for every (query, document) pair: result[i][j].

    Uses the same equations as `score`: normalised routing cosines, log-mean-exp (or max) over
    document slots, softmax(cos / tau_v) readout of raw document values. Meant for inspection of
    a few pairs; memory grows with queries * documents * slots^2.
    """
    validate_pair(queries, documents)
    config = config or ScoringConfig()
    settings = config.to_dict()
    with torch.inference_mode(), torch.autocast(device_type=queries.routing.device.type, enabled=False):
        q = F.normalize(_promote(queries.routing), p=2, dim=-1)
        d = F.normalize(_promote(documents.routing), p=2, dim=-1)
        qv = F.normalize(_promote(queries.value), p=2, dim=-1)
        dv = _promote(documents.value)
        results = []
        for i in range(len(queries)):
            sim = torch.einsum("bmd,cnd->bcmn", q[i:i + 1], d)[0]  # [docs, q slots, d slots]
            if config.qk_mode == "max":
                contribution = sim.max(dim=-1).values
            else:
                log_mass = torch.logsumexp(sim / config.tau_k, dim=-1)
                contribution = config.tau_k * (log_mass - math.log(sim.size(-1)))
            routing = contribution.mean(dim=-1)
            attention = torch.softmax(sim / config.tau_v, dim=-1)
            readout = torch.einsum("cmn,cnv->cmv", attention, dv)
            cosine = (qv[i][None] * F.normalize(readout, p=2, dim=-1)).sum(-1)
            value = cosine.mean(dim=-1)
            final = (1 - config.value_mix) * routing + config.value_mix * value
            results.append([Explanation(i, j, sim[j].cpu(), contribution[j].cpu(), attention[j].cpu(),
                                        cosine[j].cpu(), float(routing[j]), float(value[j]),
                                        float(final[j]), settings)
                            for j in range(len(documents))])
    return results

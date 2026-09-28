"""Public bounded late-interaction scoring API."""
from dataclasses import dataclass
import torch

from ._scoring import score_qk, score_qkv
from .config import ScoringConfig
from .embeddings import Embeddings


@dataclass(frozen=True)
class Scores:
    combined: torch.Tensor
    routing: torch.Tensor
    value: torch.Tensor


def validate_pair(queries: Embeddings, documents: Embeddings):
    if queries.kind != "query" or documents.kind != "document":
        raise ValueError("Expected query embeddings followed by document embeddings.")
    if queries.representation_id != documents.representation_id:
        raise ValueError("Incompatible representations: re-encode with the same model and preprocessing.")
    if queries.routing.shape[-1] != documents.routing.shape[-1] or queries.value.shape[-1] != documents.value.shape[-1]:
        raise ValueError("Query/document dimensions differ.")
    if queries.routing.device != documents.routing.device or queries.value.device != documents.value.device:
        raise ValueError("Move queries and documents to the same device before scoring.")


def score(queries: Embeddings, documents: Embeddings, *, config: ScoringConfig | None = None,
          query_batch_size: int = 8, document_batch_size: int = 128) -> Scores:
    """Return [queries, documents] scores; block intermediate attention tensors.

    Raw document value magnitudes are preserved before attention aggregation.
    The returned matrix itself still occupies O(queries * documents) memory.
    """
    validate_pair(queries, documents)
    config = config or ScoringConfig()
    if query_batch_size <= 0 or document_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if not len(queries) or not len(documents):
        empty = torch.empty((len(queries), len(documents)), device=queries.routing.device)
        return Scores(empty, empty.clone(), empty.clone())
    qk_rows, value_rows = [], []
    for start in range(0, len(queries), query_batch_size):
        q = queries.select(slice(start, start + query_batch_size))
        routes, values = [], []
        for offset in range(0, len(documents), document_batch_size):
            d = documents.select(slice(offset, offset + document_batch_size))
            routes.append(score_qk(q.routing, d.routing, tau_k=config.tau_k, mode=config.qk_mode))
            values.append(score_qkv(q.routing, d.routing, q.value, d.value, tau_v=config.tau_v))
        qk_rows.append(torch.cat(routes, dim=1))
        value_rows.append(torch.cat(values, dim=1))
    routing, value = torch.cat(qk_rows), torch.cat(value_rows)
    return Scores((1 - config.value_mix) * routing + config.value_mix * value, routing, value)

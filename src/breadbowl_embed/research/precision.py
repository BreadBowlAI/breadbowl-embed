"""Representation experiments: storage dtype, compute dtype and batch size on your own texts.

`compare_precision` encodes the same queries and documents under every combination of compute
dtype (float32, bfloat16, ...), encoder batch size and index storage type (float32, float16,
int8), scores every pair, and compares each combination with the reference (the float32 model,
batch 1, float32 storage): score deltas and ranking agreement, in one call.

A compute-dtype variant is made from the loaded weights with `with_compute_dtype`, which equals
loading the release with `from_pretrained(..., dtype=...)` (same cast of every parameter and
floating buffer) and has the same representation identity. It needs the stored-precision
weights in memory: a model loaded as float32 (or "native" float32) can produce bfloat16 and
float16 variants; a bfloat16 model cannot recover float32, so pass such variants via `models=`.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time

import torch
from torch import nn

from ..config import compute_dtype_name, fingerprint
from ..embeddings import Embeddings
from ..retrieval import STORAGE_TYPES, Index
from ..scoring import score
from ._base import ResearchObject
from .rules import agreement

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _floating_dtypes(model) -> set[torch.dtype]:
    return {p.dtype for p in model.network.parameters() if p.is_floating_point()}


def with_compute_dtype(model, dtype):
    """A BreadBowl computing in `dtype`, built from `model`'s weights without reloading.

    Equivalent to `BreadBowl.from_pretrained(<same release>, dtype=dtype)`: parameters are cast
    and floating buffers are cast as `Module.to` does; the representation identity is computed the
    same way. Returns `model` itself when it already computes in that dtype. Peak extra memory is
    the size of the cast copy.
    """
    name = compute_dtype_name(dtype)
    if name == model.compute_dtype:
        return model
    if name == "native":
        raise ValueError("Load the release again for its native dtype.")
    if model.compute_dtype != "native" and _floating_dtypes(model) != {torch.float32}:
        raise ValueError(f"This model computes in {model.compute_dtype}; the stored weights are not in memory. "
                         "Load the release with the dtype you need and pass it via models=.")
    target = _DTYPES[name]
    network = model.network
    memo = {id(p): nn.Parameter(p.detach().to(target) if p.is_floating_point() else p.detach().clone(),
                                requires_grad=p.requires_grad) for p in network.parameters()}
    copy = deepcopy(network, memo)
    for module in copy.modules():  # as from_pretrained: floating buffers are cast too
        for key, buffer in module._buffers.items():
            if buffer is not None and buffer.is_floating_point():
                module._buffers[key] = buffer.to(target)
    source = model.release_id if model.release_id is not None else model.representation_id
    identity = fingerprint({"model": source, "compute_dtype": name})
    return type(model)(copy, model.tokenizer, model.config, representation_id=identity, device=str(model.device),
                       compute_dtype=name, release_id=model.release_id, weights_sha256=model.weights_sha256,
                       loaded_revision=model.loaded_revision)


def _stored(documents: Embeddings, storage: str) -> Embeddings:
    """Documents as an Index of that storage type reads them back for scoring."""
    index = Index(documents.without_tokens(), [str(i) for i in range(len(documents))], storage=storage)
    everything = list(range(len(documents)))
    return Embeddings(index.routing_vectors(everything), index.value_vectors(everything), documents.representation_id,
                      "document")


@dataclass(frozen=True, eq=False)
class PrecisionComparison(ResearchObject):
    """Every (compute dtype, batch size, storage) variant against the reference.

    rows: one per variant: max/mean |delta| of the final score, max |delta| of R and V, Kendall tau
    and Spearman rho of each query's document ranking with the reference (mean over queries),
    top-1 agreement, top-k overlap, share of queries with an identical ranking, and the time the
    variant's encoders took. scores[name]: {"final", "routing", "value"} [queries, documents].
    """
    json_type = "precision_comparison"

    reference: str
    rows: tuple
    scores: dict
    queries: int
    documents: int
    top_k: int

    def table(self) -> list[dict]:
        return list(self.rows)

    def _payload(self) -> dict:
        return {"reference": self.reference, "queries": self.queries, "documents": self.documents,
                "top_k": self.top_k, "rows": list(self.rows), "scores": self.scores}

    def _html(self) -> str:
        from ._html import table_html
        return table_html("Precision comparison", list(self.rows),
                          note=f"Reference: {self.reference}. {self.queries} queries x {self.documents} documents.")

    def plot(self, **kwargs):
        from .plot import precision
        return precision(self, **kwargs)


def compare_precision(model, queries: list[str], documents: list[str], *, storages=STORAGE_TYPES,
                      compute_dtypes=None, batch_sizes=(1,), instruction: str | None = None, top_k: int = 10,
                      models: dict | None = None) -> PrecisionComparison:
    """Score `queries` x `documents` under every representation variant and compare (see module docstring).

    compute_dtypes: dtypes to compare (default: the model's own plus bfloat16 when the model is
    float32). models: {dtype name: BreadBowl} for variants that cannot be derived from `model`.
    batch_sizes: encoder batch sizes (1 = the default, batch-independent). The reference is the
    first compute dtype, batch 1, float32 storage. Cost: one encoding of every text per compute
    dtype and batch size.
    """
    if isinstance(queries, str) or isinstance(documents, str):
        raise TypeError("Pass lists of strings.")
    if not queries or not documents:
        raise ValueError("Give at least one query and one document.")
    models = {compute_dtype_name(k): v for k, v in (models or {}).items()}
    own = model.compute_dtype if model.compute_dtype != "native" else (
        "float32" if _floating_dtypes(model) == {torch.float32} else "native")
    if compute_dtypes is None:
        compute_dtypes = [own] + (["bfloat16"] if own == "float32" else []) + [d for d in models if d != own]
    compute_dtypes = list(dict.fromkeys(compute_dtype_name(d) for d in compute_dtypes))
    batch_sizes = list(dict.fromkeys(batch_sizes))
    if 1 not in batch_sizes:
        batch_sizes.insert(0, 1)
    storages = list(dict.fromkeys(storages))
    if "float32" not in storages:
        storages.insert(0, "float32")
    variants = {}
    for dtype in compute_dtypes:
        if dtype in models:
            variants[dtype] = models[dtype]
        elif dtype in (own, model.compute_dtype):
            variants[dtype] = model
        else:
            variants[dtype] = with_compute_dtype(model, dtype)
    scores, timings = {}, {}
    for dtype, variant in variants.items():
        for batch in batch_sizes:
            start = time.perf_counter()
            encoded_queries = variant.encode_queries(list(queries), batch_size=batch, instruction=instruction)
            encoded_documents = variant.encode_documents(list(documents), batch_size=batch)
            elapsed = time.perf_counter() - start
            for storage in storages:
                name = f"compute={dtype} batch={batch} storage={storage}"
                result = score(encoded_queries, _stored(encoded_documents, storage), config=model.scoring)
                scores[name] = {"final": result.combined, "routing": result.routing, "value": result.value}
                timings[name] = {"compute_dtype": dtype, "batch_size": batch, "storage": storage,
                                 "encode_seconds": elapsed}
    reference = f"compute={compute_dtypes[0]} batch=1 storage=float32"
    k = min(top_k, len(documents))
    rows = []
    for name, result in scores.items():
        base = scores[reference]
        rows.append({"variant": name, **timings[name], **agreement(base["final"], result["final"], k),
                     "max_abs_delta_routing": float((result["routing"] - base["routing"]).abs().max()),
                     "max_abs_delta_value": float((result["value"] - base["value"]).abs().max())})
    return PrecisionComparison(reference, tuple(rows), scores, len(queries), len(documents), k)

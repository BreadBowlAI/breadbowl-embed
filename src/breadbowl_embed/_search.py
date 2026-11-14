"""Per-slot candidate search and the document shortlist of the paper's retrieval pipeline.

`merge_slot_hits`, `rank_documents_from_slot_hits` and `coarse_candidates` are ports of the
research evaluator's functions (fresh_dev_metrics.merge_slot_hits / coarse_candidates and
retrieval_eval._rank_docs_from_slot_hits); keep them numerically identical.
"""
from __future__ import annotations

import numpy as np
import torch


def merge_slot_hits(score_parts, id_parts, k):
    """Merge exact shard top-k into global slot top-k, breaking ties by slot ID."""
    scores = np.concatenate(score_parts, axis=-1)
    ids = np.concatenate(id_parts, axis=-1)
    scores = np.where(ids >= 0, scores, -np.inf)
    order = np.lexsort((np.where(ids >= 0, ids, np.iinfo(np.int64).max), -scores), axis=-1)[..., :k]
    return np.take_along_axis(scores, order, -1), np.take_along_axis(ids, order, -1)


def rank_documents_from_slot_hits(scores, slot_ids, slot_doc_ids, topk_docs):
    """Documents by mean over query slots of their best slot cosine; a missed query slot counts -1."""
    num_query_slots = scores.shape[0]
    if num_query_slots == 0 or topk_docs <= 0:
        return []
    valid = slot_ids >= 0
    if not np.any(valid):
        return []
    flat_slot_ids = slot_ids[valid]
    flat_doc_ids = slot_doc_ids[flat_slot_ids].astype(np.int64, copy=False)
    flat_scores = scores[valid].astype(np.float32, copy=False)
    flat_query_slots = np.broadcast_to(
        np.arange(num_query_slots, dtype=np.int64)[:, None], slot_ids.shape)[valid]
    pair_keys = flat_doc_ids * num_query_slots + flat_query_slots
    order = np.argsort(pair_keys, kind="mergesort")
    pair_keys_sorted = pair_keys[order]
    flat_doc_ids_sorted = flat_doc_ids[order]
    flat_scores_sorted = flat_scores[order]
    unique_pair_starts = np.concatenate(
        ([0], np.nonzero(pair_keys_sorted[1:] != pair_keys_sorted[:-1])[0] + 1))
    max_scores = np.maximum.reduceat(flat_scores_sorted, unique_pair_starts)
    unique_doc_ids = flat_doc_ids_sorted[unique_pair_starts]
    doc_hit_counts = np.bincount(unique_doc_ids)
    doc_score_offsets = np.bincount(unique_doc_ids, weights=max_scores + 1.0,
                                    minlength=doc_hit_counts.shape[0])
    present_doc_ids = np.nonzero(doc_hit_counts)[0]
    if present_doc_ids.size == 0:
        return []
    doc_means = (doc_score_offsets[present_doc_ids] - num_query_slots) / num_query_slots
    top_n = min(topk_docs, present_doc_ids.size)
    if top_n == present_doc_ids.size:
        top_positions = np.argsort(doc_means)[::-1]
    else:
        partition = np.argpartition(doc_means, -top_n)[-top_n:]
        top_positions = partition[np.argsort(doc_means[partition])[::-1]]
    return present_doc_ids[top_positions].astype(int).tolist()


def coarse_candidates(scores, slot_ids, slots_per_doc, budget):
    """Shortlist `budget` documents from one query's per-slot hits ([query slots, k])."""
    valid = slot_ids >= 0
    if not valid.any():
        return []
    documents, inverse = np.unique(slot_ids[valid] // slots_per_doc, return_inverse=True)
    slots = np.full(slot_ids.shape, -1, dtype=np.int64)
    slots[valid] = np.arange(valid.sum())
    compact = rank_documents_from_slot_hits(scores, slots, inverse, budget)
    return documents[compact].tolist()


def search_slots_exact(query_unit: np.ndarray, blocks, k: int):
    """Exact inner-product top-k of every query slot against every document slot.

    query_unit: [queries, slots, d] unit vectors. blocks yields (first_document, [n, slots, d]
    unit float32 arrays). Returns scores/hits [queries, slots, k] with global slot ids
    (document * slots + slot), -1/-inf where fewer than k slots exist. Equal scores at the
    top-k boundary may select tied slots arbitrarily, as with flat FAISS; returned ties are
    ordered by slot id.
    """
    queries, slots, dimension = query_unit.shape
    flat = torch.from_numpy(np.ascontiguousarray(query_unit.reshape(-1, dimension), dtype=np.float32))
    scores = np.full((len(flat), k), -np.inf, dtype=np.float32)
    hits = np.full((len(flat), k), -1, dtype=np.int64)
    for first, block in blocks:
        doc_slots = torch.from_numpy(np.ascontiguousarray(block.reshape(-1, dimension), dtype=np.float32))
        if not len(doc_slots):
            continue
        width = min(k, len(doc_slots))
        sims = flat @ doc_slots.T
        top = torch.topk(sims, width, dim=1)
        block_scores = top.values.numpy()
        block_hits = top.indices.numpy().astype(np.int64) + first * block.shape[1]
        scores, hits = merge_slot_hits([scores, block_scores], [hits, block_hits], k)
    return scores.reshape(queries, slots, k), hits.reshape(queries, slots, k)


def search_slots_faiss(query_unit: np.ndarray, faiss_index, k: int):
    """Same contract as search_slots_exact, through a FAISS inner-product index of all slots."""
    queries, slots, dimension = query_unit.shape
    flat = np.ascontiguousarray(query_unit.reshape(-1, dimension), dtype=np.float32)
    scores, hits = faiss_index.search(flat, k)
    hits = hits.astype(np.int64)
    scores, hits = merge_slot_hits([scores.astype(np.float32)], [hits], k)
    return scores.reshape(queries, slots, k), hits.reshape(queries, slots, k)

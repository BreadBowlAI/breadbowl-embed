"""Run retrieval or reranking for an evaluation and record how the numbers were produced.

`evaluate_retrieval` searches an index with a search profile and scores two stages of the same
candidate lists: "routing" (the candidates ranked by the routing score R alone, QK) and "final"
(ranked by 0.75 R + 0.25 V, QKV), plus statistics of the candidate lists themselves.
`evaluate_rerank` scores supplied candidate lists (for example a BM25 run) the same way and also
evaluates the input order. Every result records the model release, compute dtype, search
profile, batch sizes, library version and data fingerprints.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import datetime
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

from .._version import __version__
from ..embeddings import Embeddings
from ..retrieval import SLOT_SHORTLIST, resolve_profile
from . import metrics as metrics_module
from .corpus import load_for_evaluation
from .data import fingerprint_qrels, fingerprint_queries, write_run
from .metrics import DEFAULT_METRICS, RunMetrics, as_ranking, evaluate_run, evaluated_queries, normalize_metrics
from .stats import bootstrap_ci, paired_test

RESULT_FORMAT = "breadbowl_evaluation_result_v1"
STAGE_DESCRIPTIONS = {
    "input": "the supplied candidate order",
    "routing": "candidates ranked by the routing score R (QK)",
    "final": "candidates ranked by (1 - value_mix) R + value_mix V (QKV), the model's score",
}


@dataclass
class Stage:
    name: str
    run: dict[str, list[tuple[str, float]]]  # query id -> ranked (document id, score)
    metrics: RunMetrics

    def to_dict(self) -> dict:
        return {"description": STAGE_DESCRIPTIONS.get(self.name, ""), **self.metrics.to_dict()}


@dataclass
class EvaluationResult:
    """Settings, per-stage metrics, runs, candidate statistics and timings of one evaluation."""
    settings: dict
    stages: dict[str, Stage]
    candidates: dict | None
    timings: dict

    def mean(self, stage: str = "final") -> dict[str, float]:
        return dict(self.stages[stage].metrics.mean)

    def confidence_interval(self, metric: str, stage: str = "final", **options) -> dict:
        """Percentile bootstrap interval of one metric's mean (options: resamples, confidence, seed)."""
        return bootstrap_ci(self.stages[stage].metrics.values(metric), **options)

    def to_dict(self) -> dict:
        return {"format": RESULT_FORMAT, "settings": self.settings,
                "stages": {name: stage.to_dict() for name, stage in self.stages.items()},
                "candidates": self.candidates, "timings": self.timings}

    def summary(self) -> str:
        names = list(next(iter(self.stages.values())).metrics.metrics)
        width = max(len(n) for n in self.stages)
        lines = [" " * width + "  " + "  ".join(f"{n:>10}" for n in names)]
        for name, stage in self.stages.items():
            lines.append(f"{name:<{width}}  " + "  ".join(f"{100 * stage.metrics.mean[n]:>10.2f}" for n in names))
        return "\n".join(lines)

    def save(self, directory: str | Path, *, confidence_intervals: bool = True) -> Path:
        """Write result.json, per_query.jsonl and runs/<stage>.trec to a new directory."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        payload = self.to_dict()
        if confidence_intervals:
            for name, stage in self.stages.items():
                payload["stages"][name]["confidence_intervals"] = {
                    metric: self.confidence_interval(metric, name) for metric in stage.metrics.metrics}
        (root / "result.json").write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
        with open(root / "per_query.jsonl", "w", encoding="utf-8") as handle:
            for name, stage in self.stages.items():
                for qid, row in stage.metrics.per_query.items():
                    handle.write(json.dumps({"stage": name, "query_id": qid, **row}, ensure_ascii=False) + "\n")
        (root / "runs").mkdir()
        for name, stage in self.stages.items():
            write_run(root / "runs" / f"{name}.trec", stage.run, tag=f"breadbowl-{name}")
        return root


def load_result(directory: str | Path) -> dict:
    """result.json plus per-query values ({stage: {query id: {metric: value}}}) of a saved result."""
    root = Path(directory)
    payload = json.loads((root / "result.json").read_text())
    if payload.get("format") != RESULT_FORMAT:
        raise ValueError(f"{root} does not hold an evaluation result.")
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    with open(root / "per_query.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            stage, qid = row.pop("stage"), row.pop("query_id")
            per_query.setdefault(stage, {})[qid] = row
    payload["per_query"] = per_query
    return payload


def compare(a, b, metric: str = "ndcg@10", *, stage_a: str = "final", stage_b: str = "final", **options) -> dict:
    """Paired comparison (b minus a) of two results on one metric: randomization p-value and bootstrap CI.

    a and b: EvaluationResult objects or saved result directories. Different stages of one result
    can be compared too (e.g. a=b=result, stage_a="routing", stage_b="final").
    """
    def values(result, stage):
        if isinstance(result, EvaluationResult):
            return result.stages[stage].metrics.values(metric), result.settings.get("data", {})
        loaded = load_result(result)
        name = "{}@{}".format(*metrics_module.parse_metric(metric))
        return {qid: row[name] for qid, row in loaded["per_query"][stage].items()}, loaded["settings"].get("data", {})

    first, data_a = values(a, stage_a)
    second, data_b = values(b, stage_b)
    if data_a.get("qrels_sha256") and data_b.get("qrels_sha256") and data_a["qrels_sha256"] != data_b["qrels_sha256"]:
        raise ValueError("The two results were evaluated against different qrels.")
    return {"metric": metric, "stage_a": stage_a, "stage_b": stage_b, **paired_test(first, second, **options)}


# ---- settings record ---------------------------------------------------------------------------
def _model_record(model, source: str | None) -> dict:
    info = model.info() if hasattr(model, "info") else {}
    release = info.get("release", {})
    hub = release.get("huggingface", {})
    if source is not None and Path(source).exists():
        source = f"local directory ({Path(source).name})"  # no local paths in shareable results
    return {"source": source, "release_id": getattr(model, "release_id", None),
            "representation_id": model.representation_id, "compute_dtype": getattr(model, "compute_dtype", None),
            "device": str(getattr(model, "device", "")), "weights_sha256": release.get("weights_sha256"),
            "repository": hub.get("repository"), "revision": hub.get("revision"),
            "loaded_revision": hub.get("loaded_revision"), "step": info.get("provenance", {}).get("step"),
            "scoring": model.scoring.to_dict(),
            "preprocessing": {k: info.get("preprocessing", {}).get(k)
                              for k in ("query_format", "default_instruction", "max_query_length",
                                        "max_document_length", "document_format")}}


def _environment() -> dict:
    import numpy
    import torch
    import transformers
    return {"python": sys.version.split()[0], "platform": platform.platform(terse=True),
            "machine": platform.machine(), "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": numpy.__version__, "threads": torch.get_num_threads()}


def _base_settings(model, *, mode, model_source, metrics, relevance_level, instruction, query_batch_size,
                   queries, qrels, evaluated, skipped, sampled, dataset) -> dict:
    return {"library_version": __version__, "mode": mode,
            "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "model": _model_record(model, model_source),
            "query_encoding": {"batch_size": query_batch_size,
                               "instruction": instruction if instruction is not None else "checkpoint default"},
            "metrics": {"names": list(metrics), "relevance_level": relevance_level,
                        "ndcg_gain": "linear (judged relevance; unjudged and non-positive = 0)",
                        "map_denominator": "all relevant documents of the query (trec_eval map_cut)",
                        "tie_break": "descending score, then ascending document id",
                        "queries_without_positive_judgments": "skipped"},
            "data": {**(dataset or {}), "queries_sha256": fingerprint_queries(queries),
                     "qrels_sha256": fingerprint_qrels(qrels), "judged_queries": len(qrels),
                     "evaluated_queries": len(evaluated), "skipped_without_relevant": len(skipped),
                     "query_sample": sampled},
            "environment": _environment()}


def _select_queries(queries: Mapping[str, str], qrels, max_queries: int | None, seed: int):
    evaluated, skipped = evaluated_queries(qrels)
    if not evaluated:
        raise ValueError("No query in the qrels has a positive relevance judgment.")
    absent = [qid for qid in evaluated if qid not in queries]
    if absent:
        raise ValueError(f"{len(absent)} judged queries have no text (first: {absent[0]!r}).")
    sampled = None
    if max_queries is not None and max_queries < len(evaluated):
        if type(max_queries) is not int or max_queries < 1:
            raise ValueError("max_queries must be a positive integer.")
        evaluated = sorted(random.Random(seed).sample(evaluated, max_queries))
        sampled = {"queries": max_queries, "seed": seed, "method": "uniform without replacement"}
    return evaluated, skipped, sampled


def _encode_queries(model, texts, batch_size, instruction, progress, label):
    encoded, chunk = [], 64
    for start in range(0, len(texts), chunk):
        encoded.append(model.encode_queries(texts[start:start + chunk], batch_size=batch_size, instruction=instruction))
        if progress is not None:
            progress(label, min(start + chunk, len(texts)), len(texts))
    return Embeddings.concat(encoded)


def _stages(runs: dict[str, dict[str, list[tuple[str, float]]]], qrels, names, relevance_level) -> dict[str, Stage]:
    return {name: Stage(name, run, evaluate_run(qrels, {q: [d for d, _ in hits] for q, hits in run.items()}, names,
                                                relevance_level=relevance_level))
            for name, run in runs.items()}


def _candidate_stats(lists: Mapping[str, Sequence[str]], qrels, width: int, relevance_level: int) -> dict:
    sizes = [len(docs) for docs in lists.values()]
    recalls = []
    for qid, docs in lists.items():
        relevant = {d for d, rel in qrels[qid].items() if rel >= relevance_level}
        recalls.append(len(relevant.intersection(docs)) / len(relevant) if relevant else 0.0)
    return {"width": width, "mean_size": math.fsum(sizes) / len(sizes), "min_size": min(sizes),
            "max_size": max(sizes), "queries_below_width": sum(s < width for s in sizes),
            "recall": math.fsum(recalls) / len(recalls), "relevance_level": relevance_level}


# ---- retrieval -------------------------------------------------------------------------------------
def evaluate_retrieval(model, index, queries: Mapping[str, str], qrels: Mapping[str, Mapping[str, int]], *,
                       profile="paper", metrics: Sequence[str] = DEFAULT_METRICS, relevance_level: int = 1,
                       instruction: str | None = None, query_batch_size: int = 1, backend: str = "torch",
                       shortlist_recall: bool = False, max_queries: int | None = None, seed: int = 0,
                       storage: str | None = None, model_source: str | None = None, dataset: dict | None = None,
                       progress: Callable[[str, int, int], None] | None = None) -> EvaluationResult:
    """Search `index` for every judged query and evaluate the routing and final stages.

    index: an Index, an Index directory, or a build directory from `build_index` (which also
    records how the corpus was encoded). profile: "paper" (128/1024/100, the model card's
    setting), "default" (512/1024/100), "exact", or a SearchProfile. Every candidate the profile
    passes to value reranking is kept, so the "routing" and "final" stages rank the same lists.
    shortlist_recall: also measure recall of the larger per-slot shortlist (a second search).
    max_queries: evaluate a seeded uniform sample of the judged queries.
    """
    names = normalize_metrics(metrics)
    profile = resolve_profile(profile)
    loaded, corpus_record = load_for_evaluation(model, index, storage=storage)
    evaluated, skipped, sampled = _select_queries(queries, qrels, max_queries, seed)
    used_qrels = {qid: dict(qrels[qid]) for qid in evaluated}
    settings = _base_settings(model, mode="retrieval", model_source=model_source, metrics=names,
                              relevance_level=relevance_level, instruction=instruction,
                              query_batch_size=query_batch_size, queries={q: queries[q] for q in evaluated},
                              qrels=used_qrels, evaluated=evaluated, skipped=skipped, sampled=sampled, dataset=dataset)
    settings["search"] = {"profile": profile.to_dict(), "backend": backend}
    settings["corpus"] = corpus_record
    timings = {}
    start = time.monotonic()
    encoded = _encode_queries(model, [queries[q] for q in evaluated], query_batch_size, instruction, progress,
                              "queries")
    timings["query_encoding_seconds"] = time.monotonic() - start
    start = time.monotonic()
    hits = []
    for first in range(0, len(evaluated), 64):
        hits.extend(loaded.search(encoded.select(slice(first, first + 64)), top_k=profile.rerank, profile=profile,
                                  backend=backend))
        if progress is not None:
            progress("search", min(first + 64, len(evaluated)), len(evaluated))
    timings["search_seconds"] = time.monotonic() - start
    final = {qid: [(h.id, h.score) for h in results] for qid, results in zip(evaluated, hits, strict=True)}
    routing = {qid: [(h.id, h.routing_score) for h in sorted(results, key=lambda h: (-h.routing_score, h.id))]
               for qid, results in zip(evaluated, hits, strict=True)}
    stages = _stages({"routing": routing, "final": final}, used_qrels, names, relevance_level)
    candidates = _candidate_stats({q: [d for d, _ in final[q]] for q in evaluated}, used_qrels, profile.rerank,
                                  relevance_level)
    if shortlist_recall and profile.method == SLOT_SHORTLIST:
        wide = profile.replace(rerank=profile.shortlist)
        start = time.monotonic()
        lists = {}
        for first in range(0, len(evaluated), 64):
            batch = loaded.search(encoded.select(slice(first, first + 64)), top_k=wide.rerank, profile=wide,
                                  backend=backend)
            for qid, results in zip(evaluated[first:first + 64], batch, strict=True):
                lists[qid] = [h.id for h in results]
        timings["shortlist_search_seconds"] = time.monotonic() - start
        candidates["shortlist"] = _candidate_stats(lists, used_qrels, profile.shortlist, relevance_level)
    n = len(evaluated)
    timings["milliseconds_per_query"] = {"encoding": 1000 * timings["query_encoding_seconds"] / n,
                                         "search": 1000 * timings["search_seconds"] / n}
    return EvaluationResult(settings, stages, candidates, timings)


# ---- reranking -------------------------------------------------------------------------------------
def evaluate_rerank(model, index, candidates: Mapping[str, Sequence[str] | Mapping[str, float]],
                    queries: Mapping[str, str], qrels: Mapping[str, Mapping[str, int]], *, depth: int = 100,
                    metrics: Sequence[str] = DEFAULT_METRICS, relevance_level: int = 1, instruction: str | None = None,
                    query_batch_size: int = 1, missing_documents: str = "error", max_queries: int | None = None,
                    seed: int = 0, storage: str | None = None, model_source: str | None = None,
                    dataset: dict | None = None, candidates_source: str | None = None,
                    progress: Callable[[str, int, int], None] | None = None) -> EvaluationResult:
    """Rerank supplied candidates with the model's stored-vector score; no retrieval.

    candidates: query id -> ranked document ids, or document id -> score (a TREC run; ranked by
    descending score, ties by ascending id). The first `depth` are scored. Their vectors come from
    `index` (an Index, an Index directory or a build directory, e.g. built over the union of the
    candidates). missing_documents: "error", or "drop" to remove candidates the index lacks (the
    count is recorded). Stages: "input" (the supplied order), "routing" (R) and "final" (QKV).
    """
    names = normalize_metrics(metrics)
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be a positive integer.")
    if missing_documents not in {"error", "drop"}:
        raise ValueError("missing_documents must be 'error' or 'drop'.")
    loaded, corpus_record = load_for_evaluation(model, index, storage=storage)
    evaluated, skipped, sampled = _select_queries(queries, qrels, max_queries, seed)
    used_qrels = {qid: dict(qrels[qid]) for qid in evaluated}
    position = {doc_id: i for i, doc_id in enumerate(loaded.ids)}
    lists, input_run, dropped = {}, {}, 0
    for qid in evaluated:
        entry = candidates.get(qid)
        ranking = as_ranking(entry)[:depth] if entry is not None else []
        kept = [doc for doc in ranking if doc in position]
        if len(kept) != len(ranking):
            if missing_documents == "error":
                missing = next(doc for doc in ranking if doc not in position)
                raise ValueError(f"Candidate {missing!r} of query {qid!r} is not in the index "
                                 "(pass missing_documents='drop' to skip such candidates).")
            dropped += len(ranking) - len(kept)
        lists[qid] = kept
        scores = entry if isinstance(entry, Mapping) else None
        input_run[qid] = [(doc, float(scores[doc]) if scores else float(len(kept) - i)) for i, doc in enumerate(kept)]
    settings = _base_settings(model, mode="rerank", model_source=model_source, metrics=names,
                              relevance_level=relevance_level, instruction=instruction,
                              query_batch_size=query_batch_size, queries={q: queries[q] for q in evaluated},
                              qrels=used_qrels, evaluated=evaluated, skipped=skipped, sampled=sampled, dataset=dataset)
    settings["rerank"] = {"depth": depth, "candidates_source": candidates_source, "dropped_candidates": dropped,
                          "queries_without_candidates": sum(1 for q in evaluated if not lists[q]),
                          "input_tie_break": "descending score, then ascending document id"}
    settings["corpus"] = corpus_record
    timings = {}
    start = time.monotonic()
    encoded = _encode_queries(model, [queries[q] for q in evaluated], query_batch_size, instruction, progress,
                              "queries")
    timings["query_encoding_seconds"] = time.monotonic() - start
    start = time.monotonic()
    routing, final = {}, {}
    for i, qid in enumerate(evaluated):
        docs = lists[qid]
        if not docs:
            routing[qid] = final[qid] = []
            continue
        rows = [position[d] for d in docs]
        documents = Embeddings(loaded.routing_vectors(rows), loaded.value_vectors(rows), loaded.representation_id,
                               "document")
        scores = model.score(encoded.select(i), documents)
        r, c = scores.routing[0].tolist(), scores.combined[0].tolist()
        routing[qid] = sorted(zip(docs, r, strict=True), key=lambda pair: (-pair[1], pair[0]))
        final[qid] = sorted(zip(docs, c, strict=True), key=lambda pair: (-pair[1], pair[0]))
        if progress is not None and (i + 1) % 64 == 0:
            progress("rerank", i + 1, len(evaluated))
    timings["scoring_seconds"] = time.monotonic() - start
    n = len(evaluated)
    timings["milliseconds_per_query"] = {"encoding": 1000 * timings["query_encoding_seconds"] / n,
                                         "scoring": 1000 * timings["scoring_seconds"] / n}
    stages = _stages({"input": input_run, "routing": routing, "final": final}, used_qrels, names, relevance_level)
    candidate_stats = _candidate_stats(lists, used_qrels, depth, relevance_level)
    return EvaluationResult(settings, stages, candidate_stats, timings)

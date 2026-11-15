"""Ranking metrics with stated definitions: nDCG@k, MRR@k, Recall@k and MAP@k.

Pure Python; no optional dependency. The definitions follow trec_eval (and so pytrec_eval, which
BEIR and MTEB use):

- nDCG@k: graded. DCG@k = sum over ranks i <= k of gain(d_i) / log2(i + 1), gain = the judged
  relevance (linear gain; unjudged and non-positive judgments give 0). The ideal DCG@k sorts every
  positive judgment of the query by gain. trec_eval's ``ndcg_cut_k``.
- MRR@k: 1 / rank of the first document within the top k whose relevance is >= relevance_level,
  0 if there is none. pytrec_eval's ``recip_rank`` computed on the run cut at k.
- Recall@k: relevant documents (relevance >= relevance_level) in the top k, divided by all relevant
  documents of the query. trec_eval's ``recall_k``.
- MAP@k: the sum, over ranks i <= k holding a relevant document, of precision@i, divided by all
  relevant documents of the query (not by min(k, relevant)). trec_eval's ``map_cut_k``.

Query set: a query is evaluated when its judgments contain at least one positive relevance; the
others are skipped and reported (MTEB drops them too; trec_eval would average them in as zeros).
An evaluated query that the run does not contain, or that has nothing at the binary
relevance_level, scores 0 on the affected metrics and still counts in every mean, so all metrics
average over the same queries.

Ties: a scored run is ranked by descending score; equal scores order by ascending document id
("id", the order `Index.search` uses) or, with tie_break="trec_eval", by descending document id as
trec_eval and pytrec_eval do. A ranked list is evaluated in the order given.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math

METRIC_NAMES = ("ndcg", "mrr", "recall", "map")
DEFAULT_METRICS = ("ndcg@10", "mrr@10", "recall@100", "map@100")
TIE_BREAKS = ("id", "trec_eval")

Qrels = Mapping[str, Mapping[str, int]]
# A run is ranked lists of document ids, or document -> score mappings (ranked here).
Run = Mapping[str, Sequence[str] | Mapping[str, float]]


def parse_metric(name: str) -> tuple[str, int]:
    """"ndcg@10" -> ("ndcg", 10). Names: ndcg, mrr, recall, map; the cutoff is required."""
    base, sep, cutoff = name.strip().lower().partition("@")
    if base not in METRIC_NAMES or not sep or not cutoff.isdigit() or int(cutoff) < 1:
        raise ValueError(f"Unknown metric {name!r}; expected one of {METRIC_NAMES} with a cutoff, e.g. ndcg@10.")
    return base, int(cutoff)


def normalize_metrics(metrics: Iterable[str]) -> tuple[str, ...]:
    names = []
    for name in metrics:
        base, cutoff = parse_metric(name)
        canonical = f"{base}@{cutoff}"
        if canonical not in names:
            names.append(canonical)
    if not names:
        raise ValueError("Give at least one metric.")
    return tuple(names)


def rank(scores: Mapping[str, float], *, tie_break: str = "id") -> list[str]:
    """Document ids by descending score; ties by ascending id ("id") or descending id ("trec_eval")."""
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"tie_break must be one of {TIE_BREAKS}.")
    for doc, value in scores.items():
        if not isinstance(value, (int, float)) or math.isnan(value):
            raise ValueError(f"Score of document {doc!r} is not a number.")
    # Two stable sorts: first the tie order, then by score.
    ordered = sorted(scores, reverse=(tie_break == "trec_eval"))
    return sorted(ordered, key=lambda doc: -float(scores[doc]))


def as_ranking(entry: Sequence[str] | Mapping[str, float], *, tie_break: str = "id") -> list[str]:
    """A ranked list from a ranked list (checked for duplicates) or a score mapping."""
    if isinstance(entry, Mapping):
        return rank(entry, tie_break=tie_break)
    if isinstance(entry, str):
        raise TypeError("A ranking is a sequence of document ids, not one string.")
    ranking = [str(doc) for doc in entry]
    if len(set(ranking)) != len(ranking):
        raise ValueError("A ranking lists a document more than once.")
    return ranking


def _dcg(gains: Iterable[float]) -> float:
    return sum(gain / math.log2(position + 2) for position, gain in enumerate(gains))


def query_metrics(ranking: Sequence[str], judgments: Mapping[str, int], metrics: Sequence[str] = DEFAULT_METRICS,
                  *, relevance_level: int = 1) -> dict[str, float]:
    """Metrics of one ranked list against one query's judgments (see the module docstring)."""
    relevant = {doc for doc, rel in judgments.items() if rel >= relevance_level}
    ideal = sorted((rel for rel in judgments.values() if rel > 0), reverse=True)
    out = {}
    for name in metrics:
        base, k = parse_metric(name)
        top = ranking[:k]
        if base == "ndcg":
            best = _dcg(ideal[:k])
            out[name] = _dcg(max(judgments.get(doc, 0), 0) for doc in top) / best if best > 0 else 0.0
        elif base == "mrr":
            out[name] = next((1.0 / (i + 1) for i, doc in enumerate(top) if doc in relevant), 0.0)
        elif base == "recall":
            out[name] = sum(doc in relevant for doc in top) / len(relevant) if relevant else 0.0
        else:  # map
            hits, total = 0, 0.0
            for i, doc in enumerate(top):
                if doc in relevant:
                    hits += 1
                    total += hits / (i + 1)
            out[name] = total / len(relevant) if relevant else 0.0
    return out


@dataclass
class RunMetrics:
    """Per-query and mean metrics of one run.

    per_query: query id -> metric -> value, for every evaluated query (sorted by query id).
    mean: metric -> mean over the evaluated queries.
    skipped: query ids in the qrels without any positive judgment (not evaluated).
    missing: evaluated query ids the run has no ranking for (scored 0).
    ignored: number of run queries that are not in the evaluated set.
    """
    per_query: dict[str, dict[str, float]]
    mean: dict[str, float]
    metrics: tuple[str, ...]
    relevance_level: int
    tie_break: str
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    ignored: int = 0

    @property
    def queries(self) -> int:
        return len(self.per_query)

    def values(self, metric: str) -> dict[str, float]:
        """query id -> value of one metric (for bootstrap intervals and paired tests)."""
        base, cutoff = parse_metric(metric)
        name = f"{base}@{cutoff}"
        if name not in self.metrics:
            raise KeyError(f"{metric!r} was not computed; computed: {self.metrics}.")
        return {qid: row[name] for qid, row in self.per_query.items()}

    def to_dict(self) -> dict:
        return {"mean": dict(self.mean), "queries": self.queries, "metrics": list(self.metrics),
                "relevance_level": self.relevance_level, "tie_break": self.tie_break,
                "skipped_without_relevant": len(self.skipped), "missing_from_run": len(self.missing),
                "ignored_run_queries": self.ignored}


def evaluated_queries(qrels: Qrels) -> tuple[list[str], list[str]]:
    """(evaluated, skipped) query ids: evaluated ones have at least one positive judgment."""
    evaluated, skipped = [], []
    for qid in sorted(qrels):
        (evaluated if any(rel > 0 for rel in qrels[qid].values()) else skipped).append(qid)
    return evaluated, skipped


def evaluate_run(qrels: Qrels, run: Run, metrics: Iterable[str] = DEFAULT_METRICS, *, relevance_level: int = 1,
                 tie_break: str = "id") -> RunMetrics:
    """Evaluate a run (ranked lists or score mappings per query) against qrels."""
    names = normalize_metrics(metrics)
    if type(relevance_level) is not int or relevance_level < 1:
        raise ValueError("relevance_level must be a positive integer.")
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"tie_break must be one of {TIE_BREAKS}.")
    evaluated, skipped = evaluated_queries(qrels)
    if not evaluated:
        raise ValueError("No query in the qrels has a positive relevance judgment.")
    per_query, missing = {}, []
    for qid in evaluated:
        entry = run.get(qid)
        if entry is None:
            missing.append(qid)
            ranking = []
        else:
            ranking = as_ranking(entry, tie_break=tie_break)
        per_query[qid] = query_metrics(ranking, qrels[qid], names, relevance_level=relevance_level)
    mean = {name: math.fsum(row[name] for row in per_query.values()) / len(per_query) for name in names}
    ignored = sum(1 for qid in run if qid not in per_query)
    return RunMetrics(per_query, mean, names, relevance_level, tie_break, skipped, missing, ignored)

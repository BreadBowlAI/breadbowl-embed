"""Evaluation toolkit: retrieval and reranking metrics, resumable corpus encoding, MTEB adapter.

The core (metrics, data files, statistics, runs) has no dependency beyond the library's own.
Names are imported on first use, so `import breadbowl_embed.evaluation` stays light and the MTEB
adapter needs `mteb` only when it is used (optional extra).

    from breadbowl_embed.evaluation import build_index, evaluate_retrieval, load_beir

See docs/evaluation.md; command line: `python -m breadbowl_embed.evaluation --help`.
"""
from importlib import import_module

_EXPORTS = {
    # metrics
    "DEFAULT_METRICS": "metrics", "RunMetrics": "metrics", "evaluate_run": "metrics", "query_metrics": "metrics",
    "rank": "metrics", "parse_metric": "metrics",
    # data files
    "Dataset": "data", "load_beir": "data", "load_corpus": "data", "load_queries": "data", "load_qrels": "data",
    "read_run": "data", "write_run": "data", "document_text": "data",
    # statistics
    "bootstrap_ci": "stats", "paired_test": "stats",
    # corpus encoding
    "build_index": "corpus", "read_build": "corpus",
    # evaluation runs
    "EvaluationResult": "runner", "evaluate_retrieval": "runner", "evaluate_rerank": "runner", "compare": "runner",
    "load_result": "runner",
    # MTEB (optional extra)
    "BreadBowlMTEB": "mteb_model", "model_meta": "mteb_model",
    # command line
    "main": "cli",
}

__all__ = ["DEFAULT_METRICS", "BreadBowlMTEB", "Dataset", "EvaluationResult", "RunMetrics", "bootstrap_ci",
           "build_index", "compare", "document_text", "evaluate_rerank", "evaluate_retrieval", "evaluate_run",
           "load_beir", "load_corpus", "load_qrels", "load_queries", "load_result", "main", "model_meta",
           "paired_test", "parse_metric", "query_metrics", "rank", "read_build", "read_run", "write_run"]


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

"""Evaluate retrieval and reranking from the command line.

    python -m breadbowl_embed.evaluation index    --model M --data DIR --work-dir W
    python -m breadbowl_embed.evaluation retrieve --model M --data DIR --work-dir W --output OUT
    python -m breadbowl_embed.evaluation rerank   --model M --data DIR --candidates RUN --work-dir W --output OUT
    python -m breadbowl_embed.evaluation metrics  --qrels QRELS --run RUN [--run RUN2 ...]
    python -m breadbowl_embed.evaluation compare  --result-a OUT_A --result-b OUT_B --metric ndcg@10

--data is a BEIR-layout directory (corpus.jsonl, queries.jsonl, qrels/<split>.tsv); --corpus,
--queries and --qrels name the files instead. --work-dir holds the resumable corpus encoding:
rerun the same command after an interruption and finished parts are kept. Results go to a new
--output directory (result.json, per_query.jsonl, runs/*.trec); a summary is printed as JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .._version import __version__

PROG = "python -m breadbowl_embed.evaluation"


def _print(value) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _load_model(args):
    import torch

    from ..model import BreadBowl
    if args.threads:
        torch.set_num_threads(args.threads)
    return BreadBowl.from_pretrained(args.model, device=args.device, dtype=args.dtype, revision=args.revision,
                                     local_files_only=args.local_files_only)


def _metric_list(text: str) -> list[str]:
    return [m for m in (part.strip() for part in text.split(",")) if m]


def _load_data(args, *, corpus: bool):
    from .data import Dataset, load_beir, load_corpus, load_qrels, load_queries
    if args.data:
        dataset = load_beir(args.data, args.split, corpus=corpus and not args.corpus)
        if corpus and args.corpus:
            dataset.corpus = load_corpus(args.corpus)
        return dataset
    if not (args.queries and args.qrels):
        raise SystemExit("Give --data DIR, or --queries and --qrels (and --corpus to encode documents).")
    return Dataset(load_corpus(args.corpus) if corpus and args.corpus else {}, load_queries(args.queries),
                   load_qrels(args.qrels), {"name": Path(args.qrels).stem})


def _index_progress(done: int, total: int, encoded: int, seconds: float) -> None:
    """Documents done, this run's rate and the time left at that rate."""
    rate = encoded / seconds if seconds > 0 else 0
    left = f", about {(total - done) / rate / 60:.0f} min left" if rate else ""
    _status(f"[index] {done}/{total} documents, {rate:.2f} documents/s{left}")


def _query_progress(label: str, done: int, total: int) -> None:
    _status(f"[{label}] {done}/{total}")


def _build(args, model, corpus):
    from .corpus import build_index
    if not corpus:
        raise SystemExit("No corpus to encode: give --data with corpus.jsonl, or --corpus.")
    started = time.monotonic()
    index = build_index(model, corpus, args.work_dir, document_format=args.document_format, storage=args.storage,
                        chunk_size=args.chunk_size, batch_size=args.batch_size, keep_parts=not args.prune_parts,
                        progress=_index_progress)
    _status(f"[index] {len(index)} documents ready in {args.work_dir} ({time.monotonic() - started:.0f} s this run)")
    return index


def cmd_index(args) -> None:
    from .corpus import read_build
    from .data import load_beir_corpus, load_corpus
    if not args.work_dir:
        raise SystemExit("Give --work-dir.")
    if args.corpus:
        corpus = load_corpus(args.corpus)
    elif args.data:
        corpus = load_beir_corpus(args.data)
    else:
        raise SystemExit("Give --data DIR or --corpus FILE.")
    model = _load_model(args)
    _build(args, model, corpus)
    info = read_build(args.work_dir)
    _print({"work_dir": str(args.work_dir), **{k: info[k] for k in ("documents", "compute_dtype", "batch_size",
                                                                   "document_format", "corpus_sha256")},
            "encoding_timing": info["encoding_timing"]})


def _index_source(args, model, corpus):
    if args.index:
        return args.index
    if not args.work_dir:
        raise SystemExit("Give --work-dir (encode the corpus there, resumable) or --index (an existing index).")
    if corpus:
        _build(args, model, corpus)
    return args.work_dir


def _finish(args, result) -> None:
    result.save(args.output)
    _status(result.summary())
    _print({"output": str(args.output), "mode": result.settings["mode"],
            "stages": {name: stage.metrics.mean for name, stage in result.stages.items()},
            "candidates": result.candidates, "queries": result.settings["data"]["evaluated_queries"],
            "model": {k: result.settings["model"][k] for k in ("release_id", "compute_dtype")},
            "timings": result.timings})


def cmd_retrieve(args) -> None:
    from .runner import evaluate_retrieval
    if Path(args.output).exists():
        raise SystemExit(f"{args.output} exists; results go to a new directory.")
    model = _load_model(args)
    dataset = _load_data(args, corpus=not args.index)
    source = _index_source(args, model, dataset.corpus)
    result = evaluate_retrieval(model, source, dataset.queries, dataset.qrels, profile=args.profile,
                                metrics=_metric_list(args.metrics), relevance_level=args.relevance_level,
                                instruction=args.instruction, query_batch_size=args.query_batch_size,
                                backend=args.backend, shortlist_recall=args.shortlist_recall,
                                max_queries=args.max_queries, seed=args.seed,
                                storage=args.storage if not args.index else None, model_source=args.model,
                                dataset=dataset.source, progress=_query_progress)
    _finish(args, result)


def cmd_rerank(args) -> None:
    from .data import read_run
    from .metrics import as_ranking, evaluated_queries
    from .runner import evaluate_rerank
    if Path(args.output).exists():
        raise SystemExit(f"{args.output} exists; results go to a new directory.")
    model = _load_model(args)
    dataset = _load_data(args, corpus=not args.index)
    candidates = read_run(args.candidates)
    corpus = dataset.corpus
    if corpus and not args.index:
        # Encode only what will be scored: the top `depth` candidates of the evaluated queries.
        evaluated, _ = evaluated_queries(dataset.qrels)
        needed = sorted({doc for qid in evaluated if qid in candidates
                         for doc in as_ranking(candidates[qid])[:args.depth] if doc in corpus})
        corpus = {doc: corpus[doc] for doc in needed}
        _status(f"[rerank] {len(needed)} distinct candidate documents to encode")
    source = _index_source(args, model, corpus)
    result = evaluate_rerank(model, source, candidates, dataset.queries, dataset.qrels, depth=args.depth,
                             metrics=_metric_list(args.metrics), relevance_level=args.relevance_level,
                             instruction=args.instruction, query_batch_size=args.query_batch_size,
                             missing_documents=args.missing_documents, max_queries=args.max_queries, seed=args.seed,
                             storage=args.storage if not args.index else None, model_source=args.model,
                             dataset=dataset.source, candidates_source=Path(args.candidates).name,
                             progress=_query_progress)
    _finish(args, result)


def cmd_metrics(args) -> None:
    from .data import load_qrels, read_run
    from .metrics import evaluate_run
    from .stats import bootstrap_ci
    qrels = load_qrels(args.qrels)
    for path in args.run:
        result = evaluate_run(qrels, read_run(path), _metric_list(args.metrics), relevance_level=args.relevance_level,
                              tie_break=args.tie_break)
        row = {"run": Path(path).name, **result.to_dict()}
        if args.ci:
            row["confidence_intervals"] = {m: bootstrap_ci(result.values(m), resamples=args.resamples, seed=args.seed)
                                           for m in result.metrics}
        _print(row)


def cmd_compare(args) -> None:
    from .runner import compare
    if args.result_a and args.result_b:
        _print(compare(args.result_a, args.result_b, args.metric, stage_a=args.stage_a, stage_b=args.stage_b,
                       resamples=args.resamples, seed=args.seed))
        return
    if not (args.qrels and args.run_a and args.run_b):
        raise SystemExit("Give --result-a and --result-b, or --qrels with --run-a and --run-b.")
    from .data import load_qrels, read_run
    from .metrics import evaluate_run
    from .stats import paired_test
    qrels = load_qrels(args.qrels)
    a = evaluate_run(qrels, read_run(args.run_a), [args.metric], relevance_level=args.relevance_level)
    b = evaluate_run(qrels, read_run(args.run_b), [args.metric], relevance_level=args.relevance_level)
    _print({"metric": args.metric, "run_a": Path(args.run_a).name, "run_b": Path(args.run_b).name,
            **paired_test(a.values(args.metric), b.values(args.metric), resamples=args.resamples, seed=args.seed)})


# ---- parser ------------------------------------------------------------------------------------------
def _model_options(p) -> None:
    p.add_argument("--model", required=True, help="Release directory or Hugging Face repo id")
    p.add_argument("--revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default=None, help="native (default), float32, bfloat16 or float16. The model card's "
                                                 "numbers were measured with a bfloat16 backbone")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (default: torch's choice)")
    p.add_argument("--local-files-only", action="store_true")


def _data_options(p, *, corpus: bool = True) -> None:
    p.add_argument("--data", help="BEIR-layout directory: corpus.jsonl, queries.jsonl, qrels/<split>.tsv")
    p.add_argument("--split", default="test", help="qrels split under --data (default test)")
    if corpus:
        p.add_argument("--corpus", help="corpus file (.jsonl with _id/title/text, or .tsv); overrides --data's")
    p.add_argument("--queries", help="queries file (.jsonl with _id/text, or id<TAB>text)")
    p.add_argument("--qrels", help="qrels file (BEIR .tsv with header, or TREC 'qid 0 docid rel')")


def _encoding_options(p) -> None:
    p.add_argument("--work-dir", help="Build directory for the resumable corpus encoding")
    p.add_argument("--document-format", choices=["title_text", "text"], default="title_text",
                   help="title_text: (title + ' ' + text).strip() as MTEB does (default); text: text only")
    p.add_argument("--storage", choices=["float32", "float16", "int8"], default="float32",
                   help="Storage of the merged index (default float32)")
    p.add_argument("--chunk-size", type=int, default=256, help="Documents per resumable part (default 256)")
    p.add_argument("--batch-size", type=int, default=1, help="Documents per forward pass (default 1: each encoding "
                                                             "independent of its neighbours)")
    p.add_argument("--prune-parts", action="store_true", help="Delete the float32 parts once the index is merged")


def _evaluation_options(p) -> None:
    p.add_argument("--output", required=True, help="New directory for result.json, per_query.jsonl and runs/")
    p.add_argument("--metrics", default="ndcg@10,mrr@10,recall@100,map@100",
                   help="Comma-separated: ndcg@k, mrr@k, recall@k, map@k (default %(default)s)")
    p.add_argument("--relevance-level", type=int, default=1,
                   help="Smallest relevance counted as relevant by mrr, recall and map (default 1; the model card "
                        "uses 2 for TREC DL recall)")
    p.add_argument("--instruction", help="Query instruction (default: the checkpoint's)")
    p.add_argument("--query-batch-size", type=int, default=1, help="Queries per forward pass (default 1)")
    p.add_argument("--max-queries", type=int, help="Evaluate a seeded random sample of this many judged queries")
    p.add_argument("--seed", type=int, default=0, help="Seed of --max-queries sampling (default 0)")
    p.add_argument("--index", help="Use an existing index or build directory instead of encoding the corpus")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the evaluation subcommands to `parser` (used by `breadbowl-embed evaluate` and by main())."""
    commands = parser.add_subparsers(dest="evaluation_command", required=True)

    p = commands.add_parser("index", help="Encode a corpus into a resumable build directory")
    _model_options(p)
    _data_options(p)
    _encoding_options(p)
    p.set_defaults(evaluation_handler=cmd_index)

    p = commands.add_parser("retrieve", help="Search the corpus for every judged query and evaluate the routing "
                                             "(QK) and final (QKV) stages")
    _model_options(p)
    _data_options(p)
    _encoding_options(p)
    _evaluation_options(p)
    p.add_argument("--profile", default="paper", help="paper (128/1024/100, the model card's setting; default), "
                                                      "default (512/1024/100) or exact")
    p.add_argument("--backend", choices=["torch", "faiss"], default="torch")
    p.add_argument("--shortlist-recall", action="store_true", help="Also report recall of the per-slot shortlist")
    p.set_defaults(evaluation_handler=cmd_retrieve)

    p = commands.add_parser("rerank", help="Rerank supplied candidates (a TREC run) and evaluate input, routing "
                                           "and final orders")
    _model_options(p)
    _data_options(p)
    _encoding_options(p)
    _evaluation_options(p)
    p.add_argument("--candidates", required=True, help="TREC run with the candidates (e.g. BM25)")
    p.add_argument("--depth", type=int, default=100, help="Candidates per query to rerank (default 100)")
    p.add_argument("--missing-documents", choices=["error", "drop"], default="error",
                   help="Candidates missing from the index: fail (default) or drop them")
    p.set_defaults(evaluation_handler=cmd_rerank)

    p = commands.add_parser("metrics", help="Evaluate TREC run files against qrels (no model)")
    p.add_argument("--qrels", required=True)
    p.add_argument("--run", required=True, action="append", help="TREC run file; repeat for several")
    p.add_argument("--metrics", default="ndcg@10,mrr@10,recall@100,map@100")
    p.add_argument("--relevance-level", type=int, default=1)
    p.add_argument("--tie-break", choices=["id", "trec_eval"], default="id",
                   help="Equal scores: ascending document id (default) or descending, as trec_eval")
    p.add_argument("--ci", action="store_true", help="Add 95%% bootstrap confidence intervals")
    p.add_argument("--resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(evaluation_handler=cmd_metrics)

    p = commands.add_parser("compare", help="Paired significance test between two results or two runs")
    p.add_argument("--result-a", help="Saved result directory (from retrieve or rerank)")
    p.add_argument("--result-b")
    p.add_argument("--stage-a", default="final")
    p.add_argument("--stage-b", default="final")
    p.add_argument("--qrels", help="With --run-a/--run-b: compare two TREC run files")
    p.add_argument("--run-a")
    p.add_argument("--run-b")
    p.add_argument("--metric", default="ndcg@10")
    p.add_argument("--relevance-level", type=int, default=1)
    p.add_argument("--resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(evaluation_handler=cmd_compare)


def run(args) -> int:
    """Run parsed arguments (the handler for `breadbowl-embed evaluate ...`)."""
    args.evaluation_handler(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"breadbowl-embed {__version__}")
    add_arguments(parser)
    return parser


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

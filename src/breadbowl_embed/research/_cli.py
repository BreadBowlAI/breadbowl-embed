"""`breadbowl-embed inspect ...`: the research toolkit from the shell.

Every subcommand prints one JSON object per result to stdout (or writes it with --output);
`inspect pair --html` also writes the self-contained HTML report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

INSPECT_HELP = """Research tools (experimental API): pair reports, attention dumps, traces, occlusion, slot ablation,
scoring rules, slot statistics, shortlist diagnostics and precision comparisons."""


def _emit(result, args) -> None:
    if getattr(args, "output", None):
        result.save_json(args.output, digits=args.digits)
        sys.stdout.write(json.dumps({"type": result.json_type, "output": str(args.output)}) + "\n")
    else:
        sys.stdout.write(result.to_json(digits=args.digits) + "\n")


def _texts(args, kind: str, helpers) -> tuple[list[str], list[str] | None]:
    """Texts from --text (repeatable) or --input/--queries/--documents files, with per-query instructions."""
    items = []
    source = getattr(args, "input", None)
    if source:
        items = helpers["read_items"](source, kind)
    for text in getattr(args, "text", None) or []:
        items.append({"id": str(len(items)), "text": text, "instruction": None, "metadata": {}})
    if not items:
        raise SystemExit("Give --text or --input.")
    return [i["text"] for i in items], items


def _instructions(model, items, args, helpers, kind):
    if kind != "query":
        return None
    return helpers["query_instructions"](model, items, getattr(args, "instruction", None))


def cmd_pair(args, helpers) -> None:
    from .pair import inspect_pair
    model = helpers["load_model"](args)
    pair = inspect_pair(model, args.query, args.document, instruction=args.instruction, rule=args.rule)
    if args.html:
        pair.report(args.html)
    if args.output or not args.html:
        _emit(pair, args)
    else:
        d = pair.decomposition
        sys.stdout.write(json.dumps({"type": "pair_inspection", "html": str(args.html), "final": d.final,
                                     "routing": d.routing, "value": d.value, "residuals": d.residuals()}) + "\n")


def cmd_attention(args, helpers) -> None:
    from .attention import attention_maps
    model = helpers["load_model"](args)
    texts, items = _texts(args, args.kind, helpers)
    maps = attention_maps(model, texts, args.kind, instructions=_instructions(model, items, args, helpers, args.kind),
                          logits=args.logits)
    for item, m in zip(items, maps):
        row = {"id": item["id"], "kind": args.kind, "heads": m.num_heads, "slots": m.num_slots,
               "tokens": m.tokens.to_dict(digits=args.digits)["tokens"], "statistics": m.statistics(),
               "top_tokens": m.top_tokens(args.top_tokens),
               "attention": m.weights if args.per_head else m.matrix()}
        if args.logits:
            row["logits"] = m.logits
        from ._base import to_jsonable
        sys.stdout.write(json.dumps(to_jsonable(row, args.digits), ensure_ascii=False) + "\n")


def cmd_trace(args, helpers) -> None:
    from .capture import trace
    model = helpers["load_model"](args)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    capture = "all" if args.capture == "all" else [c for c in args.capture.split(",") if c]
    traces = trace(model, [args.text], args.kind, instruction=args.instruction, capture=capture, layers=layers)
    traces[0].save(args.output)
    sys.stdout.write(json.dumps({"type": "trace", "output": args.output, "tensors": traces[0].summary()}) + "\n")


def cmd_occlude(args, helpers) -> None:
    from .interventions import occlusion_sweep
    model = helpers["load_model"](args)
    sweep = occlusion_sweep(model, args.query, args.document, side=args.side, unit=args.unit, mode=args.mode,
                            replacement=args.replacement, instruction=args.instruction, rule=args.rule,
                            limit=args.limit)
    _emit(sweep, args)


def _encode_pair_sets(args, helpers):
    model = helpers["load_model"](args)
    queries, q_items = _texts(argparse.Namespace(input=args.queries, text=args.query), "query", helpers)
    documents, _ = _texts(argparse.Namespace(input=args.documents, text=args.document), "document", helpers)
    encoded_queries = model.encode_queries(queries, instructions=_instructions(model, q_items, args, helpers, "query"))
    return model, encoded_queries, model.encode_documents(documents), queries, documents, q_items


def cmd_ablate(args, helpers) -> None:
    from .interventions import slot_ablation
    model, queries, documents, *_ = _encode_pair_sets(args, helpers)
    for q in range(len(queries)):
        _emit(slot_ablation(queries, documents, query=q, mode=args.mode, rule=args.rule, base=model), args)


def cmd_rules(args, helpers) -> None:
    from .rules import PRESETS, compare_rules
    model, queries, documents, *_ = _encode_pair_sets(args, helpers)
    rules = [r for r in args.rules.split(",") if r] if args.rules else list(PRESETS)
    _emit(compare_rules(queries, documents, rules, base=model, top_k=args.top_k), args)


def _require_model(args) -> None:
    if args.model is None:
        raise SystemExit("--model is required for this input.")


def cmd_slots(args, helpers) -> None:
    from .diagnostics import slot_statistics
    if bool(args.index) == bool(args.documents):
        raise SystemExit("Give --index or --documents.")
    model = queries = None
    if args.documents or args.queries:
        _require_model(args)
        model = helpers["load_model"](args)
    if args.index:
        from ..retrieval import Index
        source = (model.load_index(args.index, mmap=True, verify=not args.no_verify) if model is not None
                  else Index.load(args.index, mmap=True, verify=not args.no_verify))
    else:
        texts, _ = _texts(argparse.Namespace(input=args.documents, text=None), "document", helpers)
        source = model.encode_documents(texts)
    if args.queries:
        texts, items = _texts(argparse.Namespace(input=args.queries, text=None), "query", helpers)
        queries = model.encode_queries(texts, instructions=_instructions(model, items, args, helpers, "query"))
    _emit(slot_statistics(source, queries=queries, top_k=args.top_k, sample=args.sample, base=model), args)


def cmd_shortlist(args, helpers) -> None:
    from .diagnostics import shortlist_diagnostics
    if args.encoded_queries:
        from ..embeddings import Embeddings
        from ..retrieval import Index
        queries = Embeddings.load(args.encoded_queries)
        index = Index.load(args.index, representation_id=queries.representation_id, mmap=True,
                           verify=not args.no_verify)
    else:
        _require_model(args)
        model = helpers["load_model"](args)
        index = model.load_index(args.index, mmap=True, verify=not args.no_verify)
        texts, items = _texts(argparse.Namespace(input=args.queries, text=args.query), "query", helpers)
        queries = model.encode_queries(texts, instructions=_instructions(model, items, args, helpers, "query"))
    widths = [int(x) for x in args.slot_hits.split(",")] if args.slot_hits else None
    _emit(shortlist_diagnostics(index, queries, profile=args.profile, slot_hits=widths, top_k=args.top_k,
                                backend=args.backend, per_query=args.per_query), args)


def _split(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def cmd_precision(args, helpers) -> None:
    from .precision import compare_precision
    model = helpers["load_model"](args)
    queries, _ = _texts(argparse.Namespace(input=args.queries, text=args.query), "query", helpers)
    documents, _ = _texts(argparse.Namespace(input=args.documents, text=args.document), "document", helpers)
    result = compare_precision(model, queries, documents, storages=_split(args.storages),
                               compute_dtypes=_split(args.compute_dtypes) if args.compute_dtypes else None,
                               batch_sizes=[int(x) for x in _split(args.batch_sizes)], instruction=args.instruction,
                               top_k=args.top_k)
    if not args.include_scores:
        object.__setattr__(result, "scores", {})
    _emit(result, args)


def add_inspect_parser(commands, model_options, helpers) -> None:
    """Register `inspect` and its subcommands on the main parser's subparsers."""
    inspect = commands.add_parser("inspect", help="Research tools (experimental): pair report, attention, trace, "
                                                  "occlusion, ablation, rules, slot and shortlist diagnostics, "
                                                  "precision", description=INSPECT_HELP)
    sub = inspect.add_subparsers(dest="inspect_command", required=True)

    def common(p, *, output=True):
        model_options(p, required=False)
        p.add_argument("--model", required=False, help="Release directory or Hugging Face repo id")
        if output:
            p.add_argument("--output", type=Path, help="Write the JSON result to this file instead of stdout")
        p.add_argument("--digits", type=int, default=None, help="Round floats in the JSON output")

    def run(handler):
        def wrapped(args):
            if getattr(args, "model", None) is None and handler.__name__ not in ("cmd_slots", "cmd_shortlist"):
                raise SystemExit("--model is required.")
            handler(args, helpers)
        return wrapped

    p = sub.add_parser("pair", help="Decompose one query-document score; JSON and/or a self-contained HTML report")
    common(p)
    p.add_argument("--query", required=True)
    p.add_argument("--document", required=True)
    p.add_argument("--instruction")
    p.add_argument("--rule", help="Scoring-rule preset: model, routing-only, value-only, maxsim, maxsim+value")
    p.add_argument("--html", type=Path, help="Write the HTML report here")
    p.set_defaults(handler=run(cmd_pair))

    p = sub.add_parser("attention", help="Slot x token attention per text (JSON lines)")
    common(p, output=False)
    p.add_argument("--kind", choices=["query", "document"], required=True)
    p.add_argument("--text", action="append", help="A text (repeatable)")
    p.add_argument("--input", help="Texts file (.jsonl or one per line)")
    p.add_argument("--instruction")
    p.add_argument("--per-head", action="store_true", help="Every head instead of the head mean")
    p.add_argument("--logits", action="store_true", help="Include pre-softmax logits")
    p.add_argument("--top-tokens", type=int, default=5)
    p.set_defaults(handler=run(cmd_attention))

    p = sub.add_parser("trace", help="Save every intermediate tensor of one text (Safetensors + JSON)")
    common(p, output=False)
    p.add_argument("--kind", choices=["query", "document"], required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--instruction")
    p.add_argument("--capture", default="all", help="all, or a comma list of hidden,logits,projections,slot_states,"
                                                    "value_parts")
    p.add_argument("--layers", help="Comma list of backbone layer indices to keep (e.g. 0,11,-1)")
    p.add_argument("--output", required=True, help="New directory")
    p.set_defaults(handler=run(cmd_trace))

    p = sub.add_parser("occlude", help="Occlude each word/token/sentence of one side and rescore")
    common(p)
    p.add_argument("--query", required=True)
    p.add_argument("--document", required=True)
    p.add_argument("--instruction")
    p.add_argument("--side", choices=["query", "document"], default="document")
    p.add_argument("--unit", choices=["word", "token", "sentence"], default="word")
    p.add_argument("--mode", choices=["slot-mask", "delete", "replace"], default="slot-mask")
    p.add_argument("--replacement", default="")
    p.add_argument("--limit", type=int)
    p.add_argument("--rule")
    p.set_defaults(handler=run(cmd_occlude))

    for name, handler, text in (("ablate", cmd_ablate, "Ablate each query and document slot in turn and rescore"),
                                ("rules", cmd_rules, "Compare scoring rules (routing-only, value-only, MaxSim, ...)")):
        p = sub.add_parser(name, help=text)
        common(p)
        p.add_argument("--query", action="append")
        p.add_argument("--queries", help="Query file")
        p.add_argument("--document", action="append")
        p.add_argument("--documents", help="Document file")
        p.add_argument("--instruction")
        if name == "ablate":
            p.add_argument("--mode", choices=["drop", "zero"], default="drop")
            p.add_argument("--rule")
        else:
            p.add_argument("--rules", help="Comma list of presets (default: all)")
            p.add_argument("--top-k", type=int, default=10)
        p.set_defaults(handler=run(handler))

    p = sub.add_parser("slots", help="Slot statistics of an index or a document file (collapse, rank, usage)")
    common(p)
    p.add_argument("--index")
    p.add_argument("--documents", help="Document file to encode instead of an index")
    p.add_argument("--queries", help="Query file: adds which slots win in scoring")
    p.add_argument("--instruction")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--sample", type=int, help="Evenly spaced subset of this many documents")
    p.add_argument("--no-verify", action="store_true")
    p.set_defaults(handler=run(cmd_slots))

    p = sub.add_parser("shortlist", help="Slot-shortlist profile vs exact search: recall, per-slot hits, underfill")
    common(p)
    p.add_argument("--index", required=True)
    p.add_argument("--query", action="append")
    p.add_argument("--queries", help="Query file")
    p.add_argument("--encoded-queries", help="Directory saved by 'breadbowl-embed encode --kind query'")
    p.add_argument("--instruction")
    p.add_argument("--profile", default="paper")
    p.add_argument("--slot-hits", help="Comma list of widths to sweep (default: the profile's)")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--backend", choices=["torch", "faiss"], default="torch")
    p.add_argument("--per-query", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.set_defaults(handler=run(cmd_shortlist))

    p = sub.add_parser("precision", help="Storage x compute dtype x batch size: score deltas, ranking agreement")
    common(p)
    p.add_argument("--query", action="append")
    p.add_argument("--queries", help="Query file")
    p.add_argument("--document", action="append")
    p.add_argument("--documents", help="Document file")
    p.add_argument("--instruction")
    p.add_argument("--storages", default="float32,float16,int8")
    p.add_argument("--compute-dtypes", help="Comma list (default: the model's dtype plus bfloat16 for float32)")
    p.add_argument("--batch-sizes", default="1")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--include-scores", action="store_true", help="Include every score matrix in the JSON")
    p.set_defaults(handler=run(cmd_precision))

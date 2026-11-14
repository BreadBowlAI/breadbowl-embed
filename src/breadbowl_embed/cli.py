"""breadbowl-embed: export, encode, index, search, explain and info from the command line.

Inputs: a .jsonl file with one object per line ({"id": ..., "text": ...}; queries may use
"query" and "instruction"; documents may carry "metadata"), or a plain text file with one
text per line (ids number the non-empty lines from 0). Results are printed as JSON lines.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

from . import export as export_module
from ._version import __version__
from .config import ARCHITECTURE_NAME


def _read_items(path: str, kind: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            for number, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("query" if kind == "query" and "query" in row else "text")
                if not isinstance(text, str):
                    raise ValueError(f"{path}:{number + 1}: expected a string 'text'" +
                                     (" or 'query'" if kind == "query" else ""))
                items.append({"id": str(row.get("id", len(items))), "text": text,
                              "instruction": row.get("instruction"), "metadata": row.get("metadata", {})})
        else:
            for line in handle:
                line = line.rstrip("\n")
                if line:
                    items.append({"id": str(len(items)), "text": line, "instruction": None, "metadata": {}})
    return items


def _load_model(args):
    from .model import BreadBowl
    if args.threads:
        torch.set_num_threads(args.threads)
    return BreadBowl.from_pretrained(args.model, device=args.device, dtype=args.dtype,
                                     revision=args.revision, local_files_only=args.local_files_only)


def _print(value) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")


def _query_instructions(model, items, default):
    instructions = [item["instruction"] or default for item in items]
    if all(i is None for i in instructions):
        return None
    fallback = model.config["preprocessing"]["default_instruction"]
    return [fallback if i is None else i for i in instructions]


def cmd_export(args) -> None:
    export_module.run(args)


def cmd_info(args) -> None:
    target = Path(args.target)
    if (target / "index.json").is_file():
        from .retrieval import Index
        index = Index.load(target, mmap=True, verify=args.verify)
        _print({"kind": "index", **index.describe()})
        return
    if (target / "config.json").is_file() and (target / "manifest.json").is_file():
        root = target
    else:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(args.target, revision=args.revision, local_files_only=args.local_files_only,
                                      allow_patterns=["config.json", "manifest.json"]))
    config = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    sizes = {name: (root / name).stat().st_size for name in manifest.get("files", {}) if (root / name).exists()}
    release = dict(config.get("release") or {})
    release.setdefault("architecture", ARCHITECTURE_NAME)
    release["weights_sha256"] = manifest.get("files", {}).get("model.safetensors")
    _print({"kind": "model", "format": config.get("format"), "schema_version": config.get("schema_version"),
            "release_id": manifest.get("representation_id"), "release": release,
            "preprocessing": config["preprocessing"], "scoring": config["scoring"],
            "architecture": config["architecture"], "inference_setup": config.get("inference_setup"),
            "provenance": config.get("provenance", {}), "file_bytes": sizes, "library_version": __version__})
    if args.load:
        args.model = str(root)
        _print({"kind": "loaded", **_load_model(args).info()})


def cmd_release(args) -> None:
    from .model import set_release_metadata
    release = set_release_metadata(args.release_dir, repository=args.repository, revision=args.revision,
                                   license=args.license)
    _print({"release": release})


def cmd_tokenize(args) -> None:
    model = _load_model(args)
    items = _read_items(args.input, args.kind)
    texts = [item["text"] for item in items]
    instructions = _query_instructions(model, items, args.instruction) if args.kind == "query" else None
    for item, tokenized in zip(items, model.tokenize(texts, args.kind, instructions=instructions)):
        _print({"id": item["id"], "token_count": tokenized.token_count, "truncated": tokenized.truncated,
                "untruncated_token_count": tokenized.untruncated_token_count,
                "tokens": [{"id": int(i), "token": t, "text": d, "segment": s, "offsets": o}
                           for i, t, d, s, o in zip(tokenized.input_ids.tolist(), tokenized.tokens,
                                                    tokenized.token_texts, tokenized.segments, tokenized.offsets)]})


def cmd_encode(args) -> None:
    model = _load_model(args)
    items = _read_items(args.input, args.kind)
    texts = [item["text"] for item in items]
    if args.kind == "query":
        encoded = model.encode_queries(texts, batch_size=args.batch_size,
                                       instructions=_query_instructions(model, items, args.instruction))
    else:
        encoded = model.encode_documents(texts, batch_size=args.batch_size)
    encoded.save(args.output)
    (Path(args.output) / "ids.json").write_text(json.dumps([item["id"] for item in items]) + "\n")
    _print({"encoded": len(encoded), "kind": args.kind, "output": args.output,
            "representation_id": model.representation_id})


def cmd_index(args) -> None:
    from .retrieval import Index
    model = _load_model(args)
    items = _read_items(args.input, "document")
    index = None
    for start in range(0, len(items), args.chunk_size):
        chunk = items[start:start + args.chunk_size]
        encoded = model.encode_documents([item["text"] for item in chunk], batch_size=args.batch_size)
        ids, metadata = [item["id"] for item in chunk], [item["metadata"] for item in chunk]
        if index is None:
            index = Index(encoded, ids, scoring=model.scoring, metadata=metadata, storage=args.storage,
                          profile=args.profile)
        else:
            index.add(encoded, ids, metadata=metadata)
        print(f"[index] encoded {min(start + args.chunk_size, len(items))}/{len(items)}", file=sys.stderr, flush=True)
    if index is None:
        raise SystemExit("No documents in the input.")
    index.save(args.output)
    _print({"indexed": len(index), **index.describe(), "output": args.output})


def _profile(args, index):
    from .retrieval import resolve_profile
    return resolve_profile(args.profile or index.profile, args.candidate_k, slot_hits=args.slot_hits,
                           shortlist=args.shortlist)


def cmd_search(args) -> None:
    model = _load_model(args)
    index = model.load_index(args.index, mmap=args.mmap, verify=not args.no_verify)
    if args.queries:
        items = _read_items(args.queries, "query")
    else:
        items = [{"id": str(i), "text": q, "instruction": None, "metadata": {}} for i, q in enumerate(args.query)]
    if not items:
        raise SystemExit("Give --query or --queries.")
    queries = model.encode_queries([item["text"] for item in items], batch_size=args.batch_size,
                                   instructions=_query_instructions(model, items, args.instruction))
    profile = _profile(args, index)
    results = index.search(queries, top_k=args.top_k, profile=profile, explain=args.explain, backend=args.backend)
    for item, hits in zip(items, results):
        rows = [hit.to_dict() for hit in hits]
        for row in rows:
            row.pop("profile")  # once per query below
        _print({"query_id": item["id"], "query": item["text"], "profile": profile.to_dict(), "results": rows})


def _top_tokens(encoded, item: int, text: str, count: int) -> list[dict]:
    """For each slot, the tokens it attends to most (mean over heads)."""
    weights = encoded.attention[item].mean(dim=0)  # [slots, tokens]
    out = []
    for slot in range(weights.shape[0]):
        order = torch.argsort(weights[slot], descending=True)[:count].tolist()
        out.append({"slot": slot, "tokens": [
            {"token": encoded.tokens[item][t], "display": encoded.token_texts[item][t],
             "weight": float(weights[slot, t]), "offsets": encoded.offsets[item][t],
             "text": None if encoded.offsets[item][t] is None else text[slice(*encoded.offsets[item][t])]}
            for t in order]})
    return out


def cmd_explain(args) -> None:
    model = _load_model(args)
    queries = model.encode_queries([args.query], instruction=args.instruction, return_attention=True, return_tokens=True)
    documents = model.encode_documents([args.document], return_attention=True, return_tokens=True)
    explanation = model.explain(queries, documents)[0][0].to_dict()
    explanation["query_slots"] = _top_tokens(queries, 0, args.query, args.top_tokens)
    explanation["document_slots"] = _top_tokens(documents, 0, args.document, args.top_tokens)
    explanation["query_truncated"] = queries.truncated[0]
    explanation["document_truncated"] = documents.truncated[0]
    _print(explanation)


def _model_options(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    if required:
        parser.add_argument("--model", required=True, help="Release directory or Hugging Face repo id")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=None, help="native (default), float32, bfloat16 or float16")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (default: torch's choice)")
    parser.add_argument("--local-files-only", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="breadbowl-embed", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"breadbowl-embed {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("export", help="Training checkpoint + base model -> release directory")
    export_module.add_arguments(p)
    p.set_defaults(handler=cmd_export)

    p = commands.add_parser("info", help="Describe a release (directory or repo id) or an index directory")
    p.add_argument("target")
    p.add_argument("--load", action="store_true", help="Also load the weights and report parameters")
    p.add_argument("--verify", action="store_true", help="Verify index shard checksums")
    _model_options(p, required=False)
    p.set_defaults(handler=cmd_info)

    p = commands.add_parser("release", help="Record publish-time facts (repository, revision, licence) in a "
                                            "local schema-2 release before uploading it")
    p.add_argument("release_dir")
    export_module.add_release_arguments(p, license_default=None)
    p.set_defaults(handler=cmd_release)

    p = commands.add_parser("tokenize", help="Show how texts are tokenised (ids, display strings, offsets, "
                                             "truncation) without running the model")
    _model_options(p)
    p.add_argument("--kind", choices=["query", "document"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--instruction", help="Query instruction (default: the checkpoint's)")
    p.set_defaults(handler=cmd_tokenize)

    p = commands.add_parser("encode", help="Encode queries or documents and save routing/value slots")
    _model_options(p)
    p.add_argument("--kind", choices=["query", "document"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="New directory")
    p.add_argument("--instruction", help="Query instruction (default: the checkpoint's)")
    p.add_argument("--batch-size", type=int, default=1, help="Texts per forward pass (default 1: reproducible)")
    p.set_defaults(handler=cmd_encode)

    p = commands.add_parser("index", help="Encode documents into a persistent index")
    _model_options(p)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="New directory")
    p.add_argument("--storage", choices=["float32", "float16", "int8"], default="float32")
    p.add_argument("--batch-size", type=int, default=1, help="Texts per forward pass (default 1: reproducible)")
    p.add_argument("--chunk-size", type=int, default=4096, help="Documents per stored shard")
    p.add_argument("--profile", default="default", help="Default search profile stored with the index: "
                                                        "default (512/1024/100), paper (128/1024/100) or exact")
    p.set_defaults(handler=cmd_index)

    p = commands.add_parser("search", help="Search an index")
    _model_options(p)
    p.add_argument("--index", required=True)
    p.add_argument("--query", action="append", default=[])
    p.add_argument("--queries", help="Query file (.jsonl or one per line)")
    p.add_argument("--instruction")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--profile", help="default (slot hits 512, shortlist 1024, rerank 100), paper (128/1024/100, "
                                     "the reported setting) or exact (every document's routing). "
                                     "Default: the profile stored with the index")
    p.add_argument("--candidate-k", type=int, help="Override: documents reaching value reranking")
    p.add_argument("--slot-hits", type=int, help="Override: hits per query slot (slot-shortlist profiles)")
    p.add_argument("--shortlist", type=int, help="Override: shortlisted documents (slot-shortlist profiles)")
    p.add_argument("--backend", choices=["torch", "faiss"], default="torch")
    p.add_argument("--explain", action="store_true")
    p.add_argument("--batch-size", type=int, default=1, help="Texts per forward pass (default 1: reproducible)")
    p.add_argument("--mmap", action="store_true", help="Memory-map the index shards")
    p.add_argument("--no-verify", action="store_true", help="Skip shard checksums")
    p.set_defaults(handler=cmd_search)

    p = commands.add_parser("explain", help="Slot-level explanation of one query-document score")
    _model_options(p)
    p.add_argument("--query", required=True)
    p.add_argument("--document", required=True)
    p.add_argument("--instruction")
    p.add_argument("--top-tokens", type=int, default=5)
    p.set_defaults(handler=cmd_explain)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CPU timings of a release: load, single-query encoding, document throughput, index search.

  python tools/benchmark_cpu.py --release /path/to/release --output timings.json \
      [--threads 0 16] [--dtype float32] [--index-documents 100000]

Query latency: one query per call to encode_queries (tokenisation included), distinct queries,
warm-up calls dropped. Document throughput: passages long enough to be cut at the 384-token limit,
encoded with each of --document-batch-sizes (default 1 and 8). Search: synthetic unit routing / random value vectors (search cost does
not depend on the values), one query per call, per profile.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch

WORDS = ("retrieval model passage query vector slot routing value index corpus ranking score document "
         "evidence answer question history science river mountain city language network energy").split()


def queries(count: int) -> list[str]:
    return [f"what is the {WORDS[i % len(WORDS)]} of {WORDS[(i * 7) % len(WORDS)]} number {i}" for i in range(count)]


def passages(count: int) -> list[str]:
    return [" ".join(WORDS[(i + j * 3) % len(WORDS)] + ("," if j % 9 == 8 else "") for j in range(500)) + f" {i}."
            for i in range(count)]


def summary(seconds: list[float]) -> dict:
    ordered = sorted(seconds)
    return {"n": len(ordered), "p50_ms": 1000 * statistics.median(ordered),
            "p90_ms": 1000 * ordered[int(0.9 * (len(ordered) - 1))], "mean_ms": 1000 * statistics.fmean(ordered),
            "min_ms": 1000 * ordered[0], "max_ms": 1000 * ordered[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[0], help="0 = torch's default")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--queries", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--documents", type=int, default=64)
    parser.add_argument("--document-batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--index-documents", type=int, default=0)
    parser.add_argument("--search-queries", type=int, default=20)
    parser.add_argument("--search-only", action="store_true", help="Skip the encoding measurements")
    args = parser.parse_args()
    from breadbowl_embed import BreadBowl, Embeddings, Index

    report = {"machine": {"cpu_count": os.cpu_count(), "platform": platform.platform(),
                          "processor": platform.processor(), "load_average_start": os.getloadavg()},
              "torch": torch.__version__, "dtype": args.dtype, "runs": []}
    started = time.perf_counter()
    model = BreadBowl.from_pretrained(args.release, dtype=args.dtype)
    report["load_seconds"] = time.perf_counter() - started
    texts = passages(args.documents)
    lengths = [len(model.tokenizer(t)["input_ids"]) for t in texts]
    limit = model.config["preprocessing"]["max_document_length"]
    assert min(lengths) > limit, "benchmark passages must reach the token limit"
    for threads in [] if args.search_only else args.threads:
        if threads:
            torch.set_num_threads(threads)
        run = {"threads": torch.get_num_threads()}
        sample = queries(args.queries + args.warmup)
        times = []
        for i, text in enumerate(sample):
            began = time.perf_counter()
            model.encode_queries([text])
            if i >= args.warmup:
                times.append(time.perf_counter() - began)
        run["query_encode_batch1"] = summary(times)
        for batch_size in args.document_batch_sizes:
            model.encode_documents(texts[:batch_size], batch_size=batch_size)  # warm-up
            began = time.perf_counter()
            model.encode_documents(texts, batch_size=batch_size)
            elapsed = time.perf_counter() - began
            run[f"documents_batch{batch_size}_384_tokens"] = {"documents": len(texts), "seconds": elapsed,
                                                              "documents_per_second": len(texts) / elapsed}
        report["runs"].append(run)
        print(json.dumps(run), flush=True)
    if args.index_documents:
        if args.search_only and args.threads[0]:
            torch.set_num_threads(args.threads[0])
        generator = torch.Generator().manual_seed(0)
        slots, d_k = model.config["architecture"]["slot_extractor"]["num_slots"], model.config["architecture"]["heads"]["d_k"]
        d_v = model.config["architecture"]["heads"]["d_v"]
        routing = torch.nn.functional.normalize(torch.randn(args.index_documents, slots, d_k, generator=generator), dim=-1)
        documents = Embeddings(routing, torch.randn(args.index_documents, slots, d_v, generator=generator),
                               model.representation_id, "document")
        index = Index(documents, [str(i) for i in range(args.index_documents)], scoring=model.scoring)
        del documents, routing
        encoded = model.encode_queries(queries(args.search_queries))
        search = {"documents": args.index_documents, "threads": torch.get_num_threads()}
        for profile in ("default", "paper", "exact"):
            began = time.perf_counter()
            index.search(encoded.select([0]), top_k=10, profile=profile)  # warm-up (builds caches)
            search[profile + "_first_call_ms"] = 1000 * (time.perf_counter() - began)
            times = []
            for i in range(len(encoded)):
                began = time.perf_counter()
                index.search(encoded.select([i]), top_k=10, profile=profile)
                times.append(time.perf_counter() - began)
            search[profile] = summary(times)
            print(json.dumps({profile: search[profile]}), flush=True)
        report["search_float32_storage"] = search
    report["machine"]["load_average_end"] = os.getloadavg()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

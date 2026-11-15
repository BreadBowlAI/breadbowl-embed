"""Evaluate retrieval on a BEIR-layout dataset: routing (QK) and final (QKV) stages.

    python examples/evaluate_beir.py DATA_DIR WORK_DIR OUTPUT_DIR

DATA_DIR holds corpus.jsonl, queries.jsonl and qrels/test.tsv. Encoding the corpus is slow on CPU
(see docs/evaluation.md); it is written to WORK_DIR in parts, so rerunning the script after an
interruption continues where it stopped. BREADBOWL_MODEL selects the release (directory or repo id).
"""
import os
import sys

from breadbowl_embed import BreadBowl
from breadbowl_embed.evaluation import build_index, compare, evaluate_retrieval, load_beir

data_dir, work_dir, output_dir = sys.argv[1:4]
model = BreadBowl.from_pretrained(os.environ.get("BREADBOWL_MODEL", "dotproductx/breadbowl-embed-v1.3"),
                                  dtype="bfloat16")  # the model card's numbers used a bfloat16 backbone
data = load_beir(data_dir, split="test")
build_index(model, data.corpus, work_dir, chunk_size=256,
            progress=lambda done, total, encoded, seconds: print(f"{done}/{total} documents", flush=True))
# Passing the build directory (not the returned Index) records how the corpus was encoded.
result = evaluate_retrieval(model, work_dir, data.queries, data.qrels, profile="paper",
                            metrics=["ndcg@10", "mrr@10", "recall@100"], dataset=data.source)
print(result.summary())                                   # one row per stage, metrics x 100
print(result.candidates)                                  # candidate-list size and recall
print(result.confidence_interval("ndcg@10"))              # bootstrap over queries
print(compare(result, result, "ndcg@10", stage_a="routing", stage_b="final"))  # QKV minus QK, paired
result.save(output_dir)                                   # result.json, per_query.jsonl, runs/*.trec

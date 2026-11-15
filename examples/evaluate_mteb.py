"""Run an MTEB retrieval task with BreadBowl's own score (needs the extra: python -m pip install ".[mteb]").

    python examples/evaluate_mteb.py [TASK]

The adapter implements MTEB's search protocol, so MTEB ranks documents by 0.75 R + 0.25 V from
`Index.search` rather than by cosine or MaxSim. The default task, NanoQuoraRetrieval (5,046 short
documents, 50 queries), is small enough for a CPU. BREADBOWL_MODEL selects the release.
"""
import os
import sys

import mteb

from breadbowl_embed.evaluation import BreadBowlMTEB

task_name = sys.argv[1] if len(sys.argv) > 1 else "NanoQuoraRetrieval"
model = BreadBowlMTEB.from_pretrained(os.environ.get("BREADBOWL_MODEL", "dotproductx/breadbowl-embed-v1.3"),
                                      dtype="bfloat16", profile="paper", cache_dir="mteb-encodings")
results = mteb.evaluate(model, mteb.get_tasks(tasks=[task_name]), cache=None)
for task_result in results.task_results:
    for split, rows in task_result.scores.items():
        for row in rows:
            print(task_result.task_name, split, {k: row[k] for k in ("ndcg_at_10", "recall_at_100", "mrr_at_10")})
print(model.records)  # what was indexed and searched, with timings

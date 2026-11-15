"""Inspect one query-document score with the research toolkit.

BREADBOWL_MODEL=/path/to/release python examples/research.py [report.html]

Prints the score decomposition, where the slots read, which words carry the score
(attention-weighted) and what happens when sentences are removed (causal), then writes the
self-contained HTML report.
"""
import os
import sys

from breadbowl_embed import BreadBowl
from breadbowl_embed import research

model = BreadBowl.from_pretrained(os.environ.get("BREADBOWL_MODEL", "dotproductx/breadbowl-embed-v1.3"))
query = "How many hearts does an octopus have?"
document = ("Octopuses have three hearts. Two branchial hearts pump blood through the gills, while a single "
            "systemic heart circulates it through the rest of the body. Their blood is blue because it carries "
            "oxygen with copper-based hemocyanin rather than iron-based hemoglobin.")

# 1. Decompose the score: R, V and the final score split exactly over the 16 x 16 slot pairs.
pair = research.inspect_pair(model, query, document)
d = pair.decomposition
print(f"final {d.final:.6f} = 0.75 * R {d.routing:.6f} + 0.25 * V {d.value:.6f}")
print("recombination residuals:", {k: f"{v:.1e}" for k, v in d.residuals().items()})
for row in d.top_pairs(3):
    print(f"  query slot {row['query_slot']:2d} x document slot {row['document_slot']:2d}: "
          f"contribution {row['contribution']:.4f}, cosine {row['similarity']:.3f}")

# 2. Slot attention: how much each query slot reads the instruction rather than the query text.
mass = pair.query.segment_mass()
print("instruction + template mass per query slot:",
      [round(float(a + b), 3) for a, b in zip(mass["instruction"], mass["template"])])

# 3. Attention-weighted attribution (bookkeeping along the pooling weights, not causal).
words = sorted(pair.attribution("document").word_weights(), key=lambda w: -w["weight"])[:5]
print("document words carrying the most attribution:", [(w["word"], round(w["weight"], 4)) for w in words])

# 4. Causal checks: delete each sentence and re-encode; hide each word from the slots.
for unit in research.occlusion_sweep(model, query, document, unit="sentence", mode="delete").top(3):
    print(f"  deleting {unit['text'].strip()[:48]!r}: {unit['delta']:+.4f}")
masked = research.occlusion_sweep(model, query, document, unit="word", mode="slot-mask")
print("words whose hiding from the slots costs most:", [(u["text"], round(u["delta"], 4)) for u in masked.top(4)])

# 5. Other scoring rules on the same vectors, without touching model.scoring.
others = ["The Pacific is Earth's largest ocean.", "An octopus has eight arms and a beak."]
queries, documents = model.encode_queries([query]), model.encode_documents([document] + others)
for row in research.compare_rules(queries, documents, base=model).table():
    print(f"  {row['rule']:>13}: top-1 agrees {row['top1_agreement']:.0%}, Kendall tau {row['kendall_tau']:+.2f}")

path = sys.argv[1] if len(sys.argv) > 1 else "breadbowl-pair-report.html"
pair.report(path)
print("HTML report:", path)

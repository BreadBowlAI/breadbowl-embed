"""BREADBOWL_MODEL=/path/to/release python examples/rerank.py"""
import os
from breadbowl_embed import BreadBowl

model = BreadBowl.from_pretrained(os.environ.get("BREADBOWL_MODEL", "dotproductx/breadbowl-embed-v1.2-preview"))
passages = ["Octopuses have three hearts.", "The Pacific is Earth's largest ocean.",
            "Plants use sunlight during photosynthesis."]
for hit in model.rerank("How many hearts does an octopus have?", passages, batch_size=2):
    print(hit.to_dict(), passages[hit.document_index])

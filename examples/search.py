"""Cache documents once and reload a small late-interaction index."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from breadbowl_embed import BreadBowl, Index

model = BreadBowl.from_pretrained(os.environ.get("BREADBOWL_MODEL", "dotproductx/breadbowl-embed-v1.2-preview"))
passages = ["Octopuses have three hearts.", "The Pacific is Earth's largest ocean.",
            "Plants use sunlight during photosynthesis."]
index = model.index(model.encode_documents(passages, batch_size=2), ids=["octopus", "ocean", "plants"])
queries = model.encode_queries(["How many hearts does an octopus have?"])
hits = index.search(queries, top_k=2, candidate_k=len(passages))
with TemporaryDirectory() as temporary:
    path = Path(temporary) / "index"
    index.save(path)
    assert Index.load(path).search(queries, top_k=2, candidate_k=len(passages)) == hits
for hit in hits[0]:
    print(hit.to_dict())

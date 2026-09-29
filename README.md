# BreadBowl Embed

A small Python toolkit for trying BreadBowl's routing–value embedding architecture. Encode passages once, reuse their routing and value slots, and score queries with late interaction.

This repository contains the model implementation and local research tools. BreadBowl's hosted service is a separate product.

**Research preview:** the step-4000 KL+replay weights are available on [Hugging Face](https://huggingface.co/dotproductx/breadbowl-embed-v1.2-preview). Anonymous download and CPU inference have been verified from a fresh cache. Install the toolkit from this repository; it is not yet published on PyPI.

## Install

Use Python 3.12 or newer. Clone the public repository:

```bash
git clone https://github.com/BreadBowlAI/breadbowl-embed.git
cd breadbowl-embed
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

CPU is the default. The current release was smoke-tested with Python 3.12, PyTorch 2.11 and Transformers 5.4 on CPU; the preview pins its core dependencies to the tested versions. CUDA can be requested with `device="cuda"`, but this release has not been validated on GPU. Start with a small batch. The Qwen backbone can use its PyTorch fallback without optional accelerated kernels.

## Rerank a few passages

The example downloads the public weights automatically. Optionally set `BREADBOWL_MODEL` to a local exported model directory.

```python
import os
from breadbowl_embed import BreadBowl

model = BreadBowl.from_pretrained(
    os.environ.get("BREADBOWL_MODEL", "dotproductx/breadbowl-embed-v1.2-preview")
)
passages = [
    "Octopuses have three hearts.",
    "The Pacific is Earth's largest ocean.",
    "Plants use sunlight during photosynthesis.",
]
for hit in model.rerank("How many hearts does an octopus have?", passages):
    print(hit.id, hit.score, passages[hit.document_index])
```

Scores rank candidates; they are not probabilities. Query instructions and formatting are loaded from the checkpoint. Long passages are truncated at the model's configured token limit (384 for this checkpoint); split long documents into passages before encoding. Queries have a 128-token limit, including their instruction.

## Encode once, search repeatedly

```python
from breadbowl_embed import Index

vectors = model.encode_documents(passages, batch_size=2)
index = model.index(vectors, ids=["octopus", "ocean", "plants"])
queries = model.encode_queries(["How many hearts does an octopus have?"])
hits = index.search(queries, top_k=2, candidate_k=3)[0]
for hit in hits:
    print(hit.to_dict())

# Save to a new directory; existing directories are never overwritten.
index.save("my-index")
restored = Index.load("my-index")
assert restored.search(queries, top_k=2, candidate_k=3) == [hits]
```

The example index scans **every document's routing slots** and then reranks `candidate_k` documents with the combined routing/value score. It is intended for small experiments. It is not an ANN index or the paper's per-slot shortlist retrieval pipeline. Reducing `candidate_k` can change the final results; set it to the corpus size to rerank every document.

The index keeps all document slots in memory and scores in batches. It stores IDs and optional metadata, not your original text. Keep your own ID-to-text mapping. At 16 routing slots and 16 value slots of 256 float32 dimensions each, raw vectors use 32 KiB per passage before metadata and runtime overhead.

## What the architecture returns

Each item has two tensors, shaped `[items, slots, dimensions]`:

- `routing`: normalized slots used to identify useful document slots.
- `value`: raw value slots used to read query-dependent information from the document.

Routing scores use the checkpoint's log-mean-exp aggregation. Value scoring attends over document routing slots, combines the corresponding **raw document values**, normalizes that readout, and compares it to query values. Normalizing document values before this read changes the model. The loader and scorer preserve this behavior.

For the step-4000 checkpoint, the final score is `0.75 * routing + 0.25 * value`, with routing temperature 0.02 and value-attention temperature 0.1. `model.score(queries, documents)` exposes all three matrices. These matrices occupy memory proportional to the number of query–document pairs, even though intermediate calculations are batched.

Use `vectors.save("new-directory")` and `Embeddings.load(...)` to cache representations without an index. Saved files use Safetensors and checksum manifests. Model and dtype identifiers prevent accidental mixing of incompatible cached representations. These identifiers are compatibility checks, not signatures from a trusted publisher. Keep the default `dtype="native"` to preserve the checkpoint's mixed parameter dtypes.

## Examples and development

- [Reranking script](examples/rerank.py)
- [Search and persistence script](examples/search.py)
- [Try the model notebook](notebooks/try_breadbowl.ipynb)
- [Release format and export instructions](docs/releasing.md)
- [Validation scope](docs/validation.md)

```bash
python -m pip install '.[dev]'
python -m pytest
```

Weights belong in the Hugging Face repository; library code and examples belong here. The toolkit has no API server, authentication, billing, tenant management, or deployment stack.

## License

The toolkit uses the [Apache 2.0 license](LICENSE). The released model is based on Qwen/Qwen3.5-0.8B-Base; its pinned source revision is recorded in the model configuration. See the model card for model-specific provenance and evaluation details.

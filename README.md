# BreadBowl Embed

**Encode once. Retrieve and rerank from the same stored representation.**

BreadBowl represents each passage as 16 routing–value slots. Routing vectors find relevant information; the query then reads a weighted combination of stored values to refine the ranking. This lets future queries use cached document representations without another backbone pass over the candidate text or a separate cross-encoder.

This repository provides the Python toolkit for trying and inspecting that architecture. See the [model card and evaluation](https://huggingface.co/dotproductx/breadbowl-embed-v1.2-preview) for the research motivation and results.

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

CPU is the default. On a supported Apple Silicon Mac, use `device="mps"` in `BreadBowl.from_pretrained(...)` to encode on the Apple GPU. CPU and MPS have passed small inference tests with Python 3.12, PyTorch 2.11 and Transformers 5.4; MPS was tested with native precision and float32, with automatic CPU fallback disabled. CUDA can be requested with `device="cuda"`, but has not been validated. These are functionality checks, not speed benchmarks. The high-level encoders return CPU float32 tensors; the convenience reranking and index APIs score on CPU.

The direct runtime dependencies are **PyTorch, Transformers, Hugging Face Hub, and Safetensors**, pinned to the tested preview versions. There is **no FAISS or vector-database dependency**. The Qwen backbone can use its PyTorch implementation without optional accelerated kernels.

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

Passing passage strings to `rerank` encodes them on each call. For repeated queries, encode once and pass the cached representations instead:

```python
from breadbowl_embed import Index

vectors = model.encode_documents(passages, batch_size=2)
cached_hits = model.rerank("How many hearts does an octopus have?", vectors)
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

The included index performs exact document-routing retrieval followed by value reranking:

1. Encode the query into routing and value slots.
2. Scan **every document's routing slots** in batches on CPU.
3. Keep the best `candidate_k` documents by routing score.
4. Read those candidates' stored values and compute `0.75 * routing + 0.25 * value` for this checkpoint.
5. Return the best `top_k` results after reranking.

Every query scans the corpus, so routing retrieval cost grows linearly with document count. This is a reference implementation for small experiments, with no FAISS or ANN backend. It does not implement the paper's per-slot shortlist candidate-retrieval pipeline. Reducing `candidate_k` can change the final results; set it to the corpus size to rerank every document. The API requires `candidate_k >= top_k >= 1`.

The index keeps all document slots in memory and scores in batches. It stores IDs and optional metadata, not your original text. Keep your own ID-to-text mapping. At 16 routing slots and 16 value slots of 256 float32 dimensions each, raw vectors use 32 KiB per passage before metadata and runtime overhead.

## What the architecture returns

Each item has two tensors, shaped `[items, slots, dimensions]`:

- `routing`: normalized slots used to identify useful document slots.
- `value`: raw value slots used to read query-dependent information from the document.

Routing scores use the checkpoint's log-mean-exp aggregation. Value scoring attends over document routing slots, combines the corresponding **raw document values**, normalizes that readout, and compares it to query values. Normalizing document values before this read changes the model. The loader and scorer preserve this behavior.

For the step-4000 checkpoint, the final score is `0.75 * routing + 0.25 * value`, with routing temperature 0.02 and value-attention temperature 0.1. `model.score(queries, documents)` exposes all three matrices. These matrices occupy memory proportional to the number of query–document pairs, even though intermediate calculations are batched.

Use `vectors.save("new-directory")` and `Embeddings.load(...)` to cache representations without an index. Saved files use Safetensors and checksum manifests. Model and dtype identifiers prevent accidental mixing of incompatible cached representations. These identifiers are compatibility checks, not signatures from a trusted publisher. Keep the default `dtype="native"` to preserve the checkpoint's mixed parameter dtypes.

## Access the slots and scores directly

You can use the model's tensors in your own retrieval code:

```python
print(vectors.routing.shape)  # [3, 16, 256]
print(vectors.value.shape)    # [3, 16, 256]

scores = model.score(queries, vectors)
print(scores.routing)   # [number of queries, number of vectors]
print(scores.value)
print(scores.combined)
```

`vectors.routing` contains L2-normalized routing vectors. `vectors.value` contains value vectors without normalization. Both are ordinary PyTorch tensors. **Preserve document-value magnitudes before attention aggregation**: normalizing each document value first changes the scorer. Representations can be saved with `vectors.save(...)` and loaded with `Embeddings.load(...)`.

For research access to the underlying model, `model.network` is the actual PyTorch module. A direct forward pass exposes unnormalized routing vectors, values, slot-extraction attention, hidden slot states, and learned slot queries:

```python
import torch

prep = model.config["preprocessing"]
batch = model.tokenizer(
    [prep["document_format"].format(text=passages[0])],
    padding=True,
    truncation=True,
    max_length=prep["max_document_length"],
    return_tensors="pt",
)
with torch.inference_mode():
    raw = model.network(
        input_ids=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
    )

raw.routing       # unnormalized routing-head outputs
raw.value         # value-head outputs
raw.attention     # slot-extraction attention over input tokens
raw.slot_states   # hidden slot representations
raw.slot_queries  # learned slot queries
```

The direct forward pass leaves tensors on the model device and bypasses the high-level encoding helpers. For query inputs, use the checkpoint's query format and instruction, or call `encode_queries` to apply them automatically. The attention shown here is the encoder's slot-extraction attention, not query-to-document attention from reranking.

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

Weights belong in the Hugging Face repository; library code and examples belong here. The architecture, inference/scoring code, exact document-routing index, persistence, exporter, tests, and examples are open. The complete training/data pipeline and the paper's full evaluation and candidate-retrieval pipeline are not yet included. The toolkit has no API server, authentication, billing, tenant management, or deployment stack.

## License

The toolkit uses the [Apache 2.0 license](LICENSE). The released model is based on Qwen/Qwen3.5-0.8B-Base; its pinned source revision is recorded in the model configuration. See the model card for model-specific provenance and evaluation details.

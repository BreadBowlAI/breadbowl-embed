# BreadBowl Embed

**Encode once. Retrieve and rerank from the same stored representation.**

BreadBowl represents each passage as 16 routing–value slots. Routing vectors find relevant information; the query then reads a weighted combination of stored values to refine the ranking. This lets future queries use cached document representations without another backbone pass over the candidate text or a separate cross-encoder.

This repository provides the Python toolkit for trying and inspecting that architecture. See the [model card and evaluation](https://huggingface.co/dotproductx/breadbowl-embed-v1.2-preview) for the research motivation and results.

This repository contains the model implementation and local research tools, and the client SDKs for BreadBowl's hosted API. The hosted service itself is a separate product.

**Research preview:** the step-4000 KL+replay weights (v1.2-preview) are available on [Hugging Face](https://huggingface.co/dotproductx/breadbowl-embed-v1.2-preview). Anonymous download and CPU inference have been verified from a fresh cache. The v1.3 (step 5152) release format, exporter and parity gate are in this repository; its weights are not published yet. Install the toolkit from this repository; it is not yet published on PyPI.

## Which package do I need?

| Package | Use it to | Where |
|---|---|---|
| `breadbowl-embed` (Python, `import breadbowl_embed`) | Run the model yourself: encode, score, index and search locally with PyTorch. The rest of this README is about this package. | repository root ([`src/breadbowl_embed`](src/breadbowl_embed)) |
| `breadbowl` (Python, `import breadbowl`) | Call the hosted BreadBowl Embed API (`bb-embed` v1.3) over HTTP. Depends only on `httpx`; no model, no PyTorch. | [`sdks/python`](sdks/python) |
| `@breadbowl/embed` (TypeScript/JavaScript) | The same HTTP client for Node 18+, Bun, Deno and browsers. No runtime dependencies. | [`sdks/typescript`](sdks/typescript) |

The types of both clients are generated from the API's OpenAPI contract, [`sdks/openapi/bb-embed.v1.yaml`](sdks/openapi/bb-embed.v1.yaml). Each SDK's README has its install and test commands.

## Install

Use Python 3.12 or newer. Clone the public repository:

```bash
git clone https://github.com/BreadBowlAI/breadbowl-embed.git
cd breadbowl-embed
python -m venv .venv
source .venv/bin/activate
python -m pip install .            # add '.[faiss]' for the optional FAISS search backend
```

CPU is the default. On a supported Apple Silicon Mac, use `device="mps"` in `BreadBowl.from_pretrained(...)` to encode on the Apple GPU. CUDA can be requested with `device="cuda"`, but has not been validated. The high-level encoders return CPU float32 tensors; the convenience reranking and index APIs score on CPU.

The direct runtime dependencies are **PyTorch, Transformers, Hugging Face Hub, Safetensors and NumPy**. The allowed versions are bounded by the two stacks the test suite and release checks ran on (PyTorch 2.11–2.14, Transformers 5.4–5.17; see [validation](docs/validation.md)). FAISS is optional. The Qwen backbone uses its PyTorch implementation without optional accelerated kernels.

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

Scores rank candidates; they are not probabilities. Query instructions and formatting are loaded from the checkpoint (`"{instruction}\nQuery: {query}"`, default instruction "Retrieve relevant documents."). Long passages are truncated at the model's configured token limit (384); split long documents into passages before encoding. Queries have a 128-token limit, including their instruction. Pass `instruction="..."` (one for all queries) or `instructions=[...]` (one per query) to `encode_queries` to change the instruction.

**Batching.** The encoders run one text per forward pass by default (`batch_size=1`), so a text's slots never depend on which other texts were encoded with it; repeated encodings are bit-identical at a fixed thread count (in float32, different thread counts can differ by ~1e-6). Larger batches pad shorter texts and use different matrix kernels; that changes results slightly (in float32 by up to ~6e-7 in score, in bfloat16 by up to ~7e-3 and enough to reorder close candidates), even for texts of equal length, and on CPU it was not faster for full-length passages ([validation](docs/validation.md)). Pass a larger `batch_size` only when throughput on an accelerator matters more than batch-independent results.

**Threads.** PyTorch uses one thread per physical core by default. Several encoders on one machine, or a busy host, oversubscribe the cores and latency can collapse; give each process a share, e.g. `torch.set_num_threads(physical_cores // 2)` for two processes (`--threads` on the command line). On a loaded 16-core host, 8 threads answered single queries faster than 16.

## See the tokens without running the model

```python
for t in model.tokenize(["How many hearts does an octopus have?"], "query"):
    print(t.token_count, t.truncated, t.untruncated_token_count)
    for token, shown, offset, segment in zip(t.tokens, t.token_texts, t.offsets, t.segments):
        print(repr(token), repr(shown), offset, segment)       # 'Ġhearts' ' hearts' (8, 15) text
```

`tokenize(texts, "query" | "document", instruction=..., instructions=...)` returns one `TokenizedText` per text: the exact model input (`formatted`) and where the text sits in it (`span`), `input_ids` after truncation, `token_count`, `truncated`, `untruncated_token_count`, vocabulary `tokens`, display strings `token_texts`, `offsets` into your text and `segments` ("text", "template" or "special"). The encoders use the same tokenizer call, so these equal what `return_tokens=True` reports. Display strings are the decoded text each token completes: joined they reproduce the decoded input, and a token that ends inside a multi-byte character (byte-level BPE splits emoji and many CJK characters) shows `""` while the token that completes it shows the whole character, never `"�"`.

## Encode once, search repeatedly

```python
from breadbowl_embed import Index

vectors = model.encode_documents(passages)
index = model.index(vectors, ids=["octopus", "ocean", "plants"])            # storage="float32" | "float16" | "int8"
queries = model.encode_queries(["How many hearts does an octopus have?"])
hits = index.search(queries, top_k=2)[0]                                    # profile="default" | "paper" | "exact"
for hit in hits:
    print(hit.id, hit.score, hit.routing_score, hit.value_score, hit.profile["name"])

# Save to a new directory; existing directories are never overwritten.
index.save("my-index")
restored = model.load_index("my-index")    # refuses an index built by a different model
assert restored.search(queries, top_k=2) == [hits]
```

Search profiles decide which documents reach value reranking. Every result records the profile name and parameters it was produced with, and the index manifest records the index's default profile.

| profile | method | slot hits | shortlist | rerank |
|---|---|---|---|---|
| `default` | per-slot shortlist | 512 | 1024 | 100 |
| `paper` | per-slot shortlist | 128 | 1024 | 100 |
| `exact` | exact routing of every document (v0.1 behaviour) | – | – | 100 |

The per-slot shortlist is the paper's retrieval pipeline: every query slot retrieves its top `slot_hits` document routing slots by cosine; documents are shortlisted by the mean, over query slots, of their best hit (a query slot that did not hit the document counts −1); exact routing rescoring keeps `rerank` documents; their stored values rerank them by `0.75 * routing + 0.25 * value`. `paper` reproduces the reported evaluation settings. At 128 hits per slot, many queries on a large corpus shortlist fewer than 100 documents, so the rerank list is underfilled; `default` searches 512 hits per slot to fill it. `exact` scores every document's routing; with an exact backend it is the upper bound, at linear cost per query. Override parameters with `candidate_k=`, or pass `SearchProfile(...)`; a changed named profile is recorded as `custom`. The API requires `rerank >= top_k >= 1`.

The slot search is exact (PyTorch matrix products over all stored routing slots, in blocks); `backend="faiss"` uses a flat FAISS inner-product index instead (optional extra). The first slot search keeps a float32 copy of the unit routing vectors in memory (16 KiB per passage); set `index.cache_routing = False` to read storage block by block on every search instead. Results are deterministic: final ties order by id. CPU timings are in [validation](docs/validation.md).

Storage: `float32` keeps 32,768 bytes of slots per passage, `float16` 16,384, `int8` 8,256 (per-vector symmetric fixed point, scale = largest magnitude / 127; value slots keep a float32 scale, routing slots are compared by cosine so their scale is dropped). An index is a directory of `.npy` shards plus `index.json` holding the model's representation id, storage type, representation settings, scoring settings, default profile, ids and metadata; shard checksums are verified on load (`verify=False` skips them, `mmap=True` memory-maps). `Index.add(...)` appends documents as a new shard. Indexes saved by v0.1 still load and keep their exact profile. The index stores ids and optional metadata, not your text.

## What the architecture returns

`encode_queries` and `encode_documents` return an `Encoding` (also exported as `Embeddings`) with named fields:

- `routing`: `[items, 16, 256]` L2-normalised routing slots, used to find relevant document slots.
- `value`: `[items, 16, 256]` raw value slots; the query reads document values through attention.
- with `return_attention=True`, `attention`: per text, `[4 heads, 16 slots, tokens]` slot-extraction attention over that text's tokens (padding removed).
- with `return_tokens=True`, per text: `input_ids`, `tokens` (vocabulary strings), `token_texts` (display strings, see above), `offsets` ((start, end) character offsets into the text you passed, or `None` for tokens of the query instruction/template, so a UI can highlight spans) and `truncated`.

Routing scores use the checkpoint's log-mean-exp aggregation (temperature 0.02). Value scoring attends over document routing slots (softmax of cosines / 0.1), combines the corresponding **raw document values**, normalises that readout, and compares it to query values. Normalising document values before this read changes the model. The final score is `0.75 * routing + 0.25 * value`. `model.score(queries, documents)` returns `[queries, documents]` matrices `combined` (alias `final`), `routing` and `value`. For documents kept without value slots, `model.score_routing(queries, documents_or_routing_tensor)` returns the routing scores alone, through the same blocks, equal bit for bit to `score(...).routing`.

## Explain a score

```python
docs = model.encode_documents(passages, return_attention=True, return_tokens=True)
e = model.explain(queries, docs)[0][0]          # query 0, document 0
e.slot_similarity        # [16, 16] routing cosines, query slot x document slot
e.routing_contribution   # [16] per query slot: 0.02 * log(mean_j exp(cos_ij / 0.02)); mean == e.routing_score
e.value_attention        # [16, 16] softmax(cos / 0.1) over document slots
e.value_slot_cosine      # [16] cosine of each query value slot with its readout; mean == e.value_score
e.score                  # 0.75 * e.routing_score + 0.25 * e.value_score
e.slot_matches()         # per query slot: best document slot, cosine, contributions
```

`index.search(..., explain=True)` attaches the slot matches to each result and, for per-slot shortlist profiles, the slot hits (query slot, document slot, rank, cosine) that put the document on the shortlist.

## Command line

```bash
breadbowl-embed info  /path/to/release                    # identity, provenance, settings (or an index directory)
breadbowl-embed index --model /path/to/release --input docs.jsonl --output my-index --storage int8
breadbowl-embed search --model /path/to/release --index my-index --query "octopus hearts" --explain
breadbowl-embed explain --model /path/to/release --query "octopus hearts" --document "Octopuses have three hearts."
breadbowl-embed encode --model /path/to/release --kind query --input queries.txt --output query-slots
breadbowl-embed tokenize --model /path/to/release --kind document --input docs.jsonl
breadbowl-embed export --checkpoint /path/to/model.pt --output /path/to/new-release --base-model-path /path/to/base
breadbowl-embed release /path/to/new-release --repository org/name --revision v1.3   # before uploading
```

Inputs are `.jsonl` (`{"id", "text"}`; queries may use `"query"` and `"instruction"`, documents `"metadata"`) or plain text with one item per line. Output is JSON lines.

## Models and formats

A release directory holds `config.json` (architecture, text formats, default instruction, token limits, scoring settings, base model id and revision, source checkpoint sha256 and step, exporter versions), `manifest.json` (file checksums and representation identity), `model.safetensors`, `backbone/config.json` and `tokenizer/`. No pickle, optimizer state or training paths. Loading verifies every checksum.

Schema 2 configs add a `release` section: architecture name (`qkv-slot-encoder`), licence, weights sha256, and the Hugging Face repository and revision name the release is published under. The repository and revision are set at publish time (`breadbowl-embed release ...`, or `--repository`/`--revision` on export); a revision recorded this way is a name such as a tag, because a file cannot contain the commit hash of the commit that adds it. `model.info()` reports all of these plus `loaded_revision`, the commit actually loaded when the model came from the Hub, the release identity (`release_id`) and the compute dtype. Release metadata is outside the representation identity: setting it, or re-exporting a schema-1 release as schema 2, keeps the identity. Schema 1 releases (v1.2-preview, the first v1.3 export) still load; their `info()` shows the architecture name and the weights sha256 from the manifest, and no licence or repository.

- **v1.3 (step 5152)**: float32 weights (3.0 GB). The backbone is the text model only: the Qwen3.5 vision tower is not exported because a text-only forward pass is numerically identical. The training-only `<jepa_mask>` token is not exported, so literal "<jepa_mask>" in user text is ordinary text. `dtype="bfloat16"` or `"float16"` loads a cast copy, casting each tensor while it is read (about 2.1 GiB peak for bfloat16 on CPU); float32 loads memory-map the weights file. The compute dtype is part of the representation identity, so representations from different dtypes are never mixed.
- **v1.2-preview (step 4000)**: loads unchanged (mixed bf16/float32 weights, multimodal backbone config, sentinel kept).

Use `vectors.save("new-directory")` and `Embeddings.load(...)` to cache representations without an index. Saved files use Safetensors and checksum manifests. Model and dtype identifiers prevent accidental mixing of incompatible cached representations. These identifiers are compatibility checks, not signatures from a trusted publisher.

For research access, `model.network` is the PyTorch module. A direct forward pass exposes unnormalised routing vectors, values, slot-extraction attention, hidden slot states and learned slot queries; use `model.format_texts(...)` for the exact model inputs.

## Examples and development

- [Reranking script](examples/rerank.py)
- [Search and persistence script](examples/search.py)
- [Try the model notebook](notebooks/try_breadbowl.ipynb)
- [Release format, export and parity gate](docs/releasing.md)
- [Validation scope, parity numbers and CPU timings](docs/validation.md)

```bash
python -m pip install '.[dev]'
python -m pytest          # release gates in tests/test_release_parity.py skip unless their files are given
```

Weights belong in the Hugging Face repository; library code and examples belong here. The architecture, inference/scoring code, retrieval pipeline, persistence, exporter, tests and examples are open. The training/data pipeline and the paper's evaluation harness are not included. The toolkit has no API server, authentication, billing, tenant management, or deployment stack.

## License

The toolkit uses the [Apache 2.0 license](LICENSE). The released model is based on Qwen/Qwen3.5-0.8B-Base; its pinned source revision is recorded in the model configuration. See the model card for model-specific provenance and evaluation details.

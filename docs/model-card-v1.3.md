---
license: apache-2.0
library_name: breadbowl-embed
pipeline_tag: sentence-similarity
base_model: Qwen/Qwen3.5-0.8B-Base
base_model_relation: finetune
language:
- en
tags:
- multi-vector
- late-interaction
- retrieval
- reranking
- research
datasets:
- lightonai/embeddings-pre-training-curated
- lightonai/embeddings-fine-tuning
- sentence-transformers/embedding-training-data
- allenai/scirepeval
- qiaojin/PubMedQA
model-index:
- name: breadbowl-embed-v1.3
  results:
  - task:
      type: text-retrieval
      name: Passage retrieval and reranking
    dataset:
      type: BeIR/msmarco
      name: "MS MARCO passage dev (BEIR, 6,980 queries, full 8,841,823-passage corpus)"
      split: dev
    metrics:
    - type: ndcg_at_10
      name: "nDCG@10, QKV (0.75 routing + 0.25 value), slot-hit width 128"
      value: 41.71
    - type: ndcg_at_10
      name: "nDCG@10, QK (routing only), slot-hit width 128"
      value: 39.30
    - type: recall_at_100
      name: "Recall@100, slot-hit width 128, relevance >= 1"
      value: 84.46
  - task:
      type: text-retrieval
      name: Passage retrieval and reranking
    dataset:
      type: trec-dl-2019-passage
      name: "TREC DL 2019 passage (43 judged queries, same 8,841,823-passage corpus)"
      split: test
    metrics:
    - type: ndcg_at_10
      name: "nDCG@10, QKV (0.75 routing + 0.25 value), slot-hit width 128"
      value: 74.63
    - type: ndcg_at_10
      name: "nDCG@10, QK (routing only), slot-hit width 128"
      value: 71.56
    - type: recall_at_100
      name: "Recall@100, slot-hit width 128, relevance >= 2"
      value: 57.01
  - task:
      type: text-retrieval
      name: Passage retrieval and reranking
    dataset:
      type: trec-dl-2020-passage
      name: "TREC DL 2020 passage (54 judged queries, same 8,841,823-passage corpus)"
      split: test
    metrics:
    - type: ndcg_at_10
      name: "nDCG@10, QKV (0.75 routing + 0.25 value), slot-hit width 128"
      value: 73.14
    - type: ndcg_at_10
      name: "nDCG@10, QK (routing only), slot-hit width 128"
      value: 68.73
    - type: recall_at_100
      name: "Recall@100, slot-hit width 128, relevance >= 2"
      value: 63.71
---

# BreadBowl-Embed v1.3

**Encode once. Retrieve and rerank from the same stored representation.**

BreadBowl-Embed represents every query and passage as **16 routing–value slots**. Each slot pairs a routing vector, used to find candidates, with a value vector that a query can read. Passages are encoded once. At search time the query's routing vectors retrieve candidates from an index of passage routing vectors, and the same routing similarities then decide which of each candidate's stored value vectors the query reads to rerank it. Reranking uses only cached vectors: there is no further backbone pass over passage text and no separate cross-encoder.

v1.3 continues the line of [v1.2-preview](https://huggingface.co/dotproductx/breadbowl-embed-v1.2-preview) (the step-4000 checkpoint). It was trained for 1,024 more updates with the full text backbone unfrozen and labels from a larger cross-encoder teacher. The architecture, scoring rule and input limits are unchanged. This is a research release.

[Source code and toolkit](https://github.com/BreadBowlAI/breadbowl-embed) · [v1.2-preview](https://huggingface.co/dotproductx/breadbowl-embed-v1.2-preview)

## What it is, and how it differs

| | Single-vector embedding | ColBERT-style late interaction | BreadBowl-Embed |
| --- | --- | --- | --- |
| Vectors stored per passage | 1 | 1 per token; grows with passage length | 16 routing + 16 value, whatever the passage length |
| How vectors are formed | Pooling of token states | Token states projected one by one | 16 learned slot queries cross-attend over the token states |
| Relevance score | Cosine of two vectors | Sum over query tokens of the best-matching passage token (MaxSim) | Routing score R plus a value score V, both from the stored slots |
| Second-stage reranking | Usually a separate model | Rescoring from stored token vectors | Value readout from the same stored slots |

The two scores, for a query q and a passage d (16 slots each, routing vectors L2-normalised):

- **Routing score R (QK).** Take the 16 × 16 cosine similarities between query and passage routing vectors. For each query slot, take a smooth maximum over passage slots (log-mean-exp, temperature 0.02), then average over the 16 query slots. As the temperature goes to zero this approaches mean MaxSim. R alone is a fixed-size late-interaction score. The index searches the individual routing vectors; R then orders the retrieved candidates.
- **Value score V.** Turn the same 16 × 16 similarities into attention over passage slots (softmax, temperature 0.1). Each query slot gets a weighted sum of the passage's **raw** value vectors. V is the cosine between that readout and the query slot's own value vector, averaged over the 16 query slots.
- **Final score (QKV)** = 0.75 R + 0.25 V. QK and QKV rank the same candidates; only the ordering changes.

Reranking from cached vectors is not unique to this design: ColBERT-style models also rescore from stored vectors. What is specific here is that the vectors indexed for retrieval also define the attention that reads the stored values, and that each passage costs a fixed number of vectors. A comparison with single-vector and ColBERT-style models under one protocol has not been run yet (see [Limitations](#limitations)).

## Quick start

Requires **Python 3.12+**. Install the toolkit from GitHub; it is not published on PyPI:

```bash
python -m pip install "git+https://github.com/BreadBowlAI/breadbowl-embed.git"
```

Rerank a few passages:

```python
from breadbowl_embed import BreadBowl

model = BreadBowl.from_pretrained("dotproductx/breadbowl-embed-v1.3")
passages = [
    "Octopuses have three hearts.",
    "The Pacific is Earth's largest ocean.",
    "Plants use sunlight during photosynthesis.",
]
query = "How many hearts does an octopus have?"

for hit in model.rerank(query, passages):
    print(f"{hit.score:.4f}", passages[hit.document_index])
```

Passing strings to `rerank` encodes them on every call. To encode passages once and reuse them:

```python
documents = model.encode_documents(passages)                 # cache these
queries = model.encode_queries([query])                       # default instruction applied

hits = model.rerank(query, documents)    # only the query passes through the backbone
scores = model.score(queries, documents) # R, V and the combined score for every pair
```

`encode_queries` takes an optional `instruction="..."` (one for every query) or `instructions=[...]` (one per query). All results on this card use the checkpoint's default instruction, `Retrieve relevant documents.`

Index a corpus and search it with the evaluated retrieval pipeline:

```python
from breadbowl_embed import Index

index = model.index(documents, ids=["octopus", "ocean", "plants"])   # storage="float32" (default), "float16" or "int8"
results = index.search(queries, top_k=2, candidate_k=3)[0]           # profile="default" (default), "paper" or "exact"
for hit in results:
    print(hit.id, hit.routing_score, hit.value_score, hit.score)

index.save("my-index")              # creates a new directory; never overwrites
restored = Index.load("my-index")
checked = model.load_index("my-index")   # also refuses an index built by another model or dtype
```

`Index.search` runs the per-slot shortlist pipeline of the [evaluation](#protocol): each query routing vector retrieves its `slot_hits` nearest passage routing vectors; hit passages are shortlisted by the mean, over query slots, of their best hit (a query slot without a hit in the passage counts −1); the best `shortlist` passages get the exact R; the best `rerank` of those by R also get V; the result is ranked by 0.75 R + 0.25 V. Named search profiles:

| Profile | Slot hits per query slot | Shortlist | Rerank | Note |
| --- | ---: | ---: | ---: | --- |
| `default` | 512 | 1,024 | 100 | Fills the rerank list (see the width-512 rows under [Results](#results-slot-hit-width-128-the-primary-setting)) |
| `paper` | 128 | 1,024 | 100 | The setting of every result on this card |
| `exact` | – | – | 100 | Exact R for every passage, then the same reranking; cost linear in corpus size |

`candidate_k=` overrides the rerank width; a profile with changed parameters is recorded as `custom`, and every result records the profile that produced it. The slot search is exact: PyTorch matrix products over all stored routing vectors on CPU, in blocks. `backend="faiss"` uses a flat FAISS inner-product index instead (optional extra: `pip install "breadbowl-embed[faiss] @ git+https://github.com/BreadBowlAI/breadbowl-embed.git"`). In the release validation (20 queries, 20 documents), `paper` and `default` returned the same passages in the same order as the evaluation's own search code, with scores within 1.2e-7. Stored slots can be float32 (32,768 bytes per passage), float16 (16,384) or int8 (8,256, the same per-vector scaling as the int8 row [below](#storage-and-serving-cost)); queries stay float32. An index is a directory of checksummed `.npy` shards plus a manifest; `Index.add` appends passages and `mmap=True` memory-maps the shards on load. It stores IDs and optional metadata, not passage text, and has no metadata filtering and no approximate-search backend.

**Precision.** The released weights are float32, and the default `dtype=None` (also spelled `"native"`) loads them as stored. The results on this card were measured with a bfloat16 backbone under bfloat16 autocast on GPU; loading with `dtype="bfloat16"` matches the evaluated weights. float32 and bfloat16 encodings of the same text differ: in the release validation, final scores moved by up to 8.8e-3 (about 1e-2) and some rankings among 20 passages changed. CPU and GPU bfloat16 kernels also differ, so a CPU run does not reproduce the evaluation bit for bit. The compute dtype is part of the model identity (`model.representation_id`), so encodings, saved embeddings and indexes produced under different dtypes refuse to mix: encode the corpus and the queries with the same `dtype`. The encoders return CPU float32 tensors whatever the compute dtype. BreadBowl's hosted API also computes in bfloat16.

**Batching.** The encoders run one text per forward pass by default (`batch_size=1`), so a text's slots do not depend on which other texts were encoded with it, and repeated encodings are bit-identical at a fixed thread count. Larger batches pad shorter texts and use different matrix kernels, which changes results even for texts of equal length: in the release validation, batches of 8 moved final scores by up to 6.3e-7 in float32 (rankings unchanged) and up to 6.6e-3 in bfloat16 (rankings changed). On CPU, batch 1 was also faster for full-length passages. Pass a larger `batch_size` only when accelerator throughput matters more than batch-independent results.

CPU is the default device. With the v1.2 weights, the toolkit passed small inference tests on CPU and on Apple-silicon MPS (`device="mps"`). CUDA can be requested with `device="cuda"` but has not been validated in the toolkit. Scores rank candidates; they are not probabilities. CPU timings are under [Storage and serving cost](#cpu-timings-toolkit).

## Outputs and shapes

| Returned by | Field | Shape and type | Meaning |
| --- | --- | --- | --- |
| `encode_queries`, `encode_documents` (`Embeddings`) | `routing` | `[items, 16, 256]`, float32, CPU | L2-normalised routing vectors |
| | `value` | `[items, 16, 256]`, float32, CPU | Raw value vectors. Do not normalise passage values: their magnitudes enter the readout before normalisation. |
| | `representation_id`, `kind` | `str` | Compatibility check between queries, passages and indexes; `"query"` or `"document"` |
| `model.score(queries, documents)` (`Scores`) | `routing`, `value`, `combined` | `[queries, passages]` each | R, V and 0.75 R + 0.25 V |
| `rerank`, `Index.search` (`SearchResult`) | `id`, `score`, `routing_score`, `value_score`, `document_index`, `metadata` | scalars | One ranked passage |
| `Index.search` only | `profile`; `explanation` with `explain=True` | `dict` | Name and parameters of the search profile; slot matches and, for per-slot shortlist profiles, the slot hits that shortlisted the passage |

`Embeddings.save(...)` and `Embeddings.load(...)` persist routing and value tensors in Safetensors with a checksum manifest.

**Direct forward pass.** `model.network` is the PyTorch module. A forward pass returns unnormalised routing vectors and the encoder's internal tensors:

```python
import torch

prep = model.config["preprocessing"]
batch = model.tokenizer(
    [prep["document_format"].format(text=t) for t in passages],
    padding=True, truncation=True,
    max_length=prep["max_document_length"], return_tensors="pt",
)
with torch.inference_mode():
    raw = model.network(
        input_ids=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
    )

raw.routing       # [batch, 16, 256]  unnormalised routing-head outputs
raw.value         # [batch, 16, 256]  value-head outputs
raw.attention     # [batch, 4, 16, tokens]  slot-extraction attention: per head, per slot, over input tokens
raw.slot_states   # [batch, 16, 1024]  slot states read by both heads
raw.slot_queries  # [16, 1024]  learned slot queries
```

`raw.attention` covers the padded batch length (at most 128 tokens for queries, 384 for passages). Each row sums to 1 over tokens and padding positions get zero weight. It shows how the encoder pooled tokens into slots. It is not the query-to-passage attention used in scoring.

**Explanation tensors.** The query-to-passage quantities behind R and V can be computed from the public tensors today. For one query and one passage:

```python
import math
import torch
import torch.nn.functional as F

cfg = model.scoring                                   # tau_k=0.02, tau_v=0.1, value_mix=0.25
q_route, q_value = queries.routing[0], queries.value[0]          # [16, 256] each
d_route, d_value = documents.routing[0], documents.value[0]      # [16, 256] each

C = q_route @ d_route.T                                           # [16, 16] routing cosines
routing_per_slot = cfg.tau_k * (torch.logsumexp(C / cfg.tau_k, dim=-1) - math.log(C.shape[-1]))  # [16]
A = torch.softmax(C / cfg.tau_v, dim=-1)              # [16 query slots, 16 passage slots]; rows sum to 1
readout = A @ d_value                                 # [16, 256], built from raw passage values
value_per_slot = (F.normalize(q_value, dim=-1) * F.normalize(readout, dim=-1)).sum(-1)  # [16]

R, V = routing_per_slot.mean(), value_per_slot.mean()
S = (1 - cfg.value_mix) * R + cfg.value_mix * V      # equals model.score(queries, documents).combined[0, 0]
```

Row `i` of `A` says which passage slots query slot `i` read; `value_per_slot[i]` says how well that readout agreed with the query slot's value vector. These are attention weights and per-slot terms of the score, not a causal attribution.

The v1.3 toolkit adds per-passage token detail to encoding:

```python
documents = model.encode_documents(passages, return_attention=True, return_tokens=True)

documents.attention[0]   # [4 heads, 16 slots, tokens in passage 0], padding removed
documents.tokens[0]      # token strings of passage 0
documents.token_texts[0] # display strings: the decoded text each token completes
documents.offsets[0]     # (start, end) character offsets into passages[0]
documents.truncated[0]   # True if the 384-token limit cut the passage

# Which tokens of passage 0 fed the slots that query slot 0 read (A from above, heads averaged):
token_weights = A[0] @ documents.attention[0].mean(0)   # [tokens]
```

and a convenience wrapper for the explanation tensors above, one per query–passage pair:

```python
explanation = model.explain(queries, documents)[0][0]   # query 0, passage 0

explanation.slot_similarity        # [16 query slots, 16 passage slots], C above
explanation.routing_contribution   # [16], routing_per_slot above; mean is R
explanation.value_attention        # [16 query slots, 16 passage slots], A above
explanation.value_slot_cosine      # [16], value_per_slot above; mean is V
explanation.routing_score, explanation.value_score, explanation.score   # R, V, S
```

## Architecture and scoring

| Component | This checkpoint |
| --- | --- |
| Backbone | Qwen3.5-0.8B-Base, revision `a9a407bcae463285164cc9133995c515379cebe5`, fine-tuned |
| Slot extractor | 16 learned slot queries; one multi-head cross-attention over the final token states; 4 heads, total width 512; output projection and layer norm give 16 slot states |
| Routing head | Linear, 256 dimensions per slot |
| Value head | Linear, 256 dimensions per slot, plus a residual MLP (hidden width 512, SiLU) |
| Query input | `"{instruction}\nQuery: {query}"`, default instruction `Retrieve relevant documents.`; 128 tokens including the instruction |
| Passage input | Text as given; 384 tokens |
| Routing score R | Log-mean-exp over passage slots, temperature 0.02, averaged over query slots |
| Value score V | Softmax attention over passage slots, temperature 0.1; cosine of readout and query value, averaged |
| Final score | `0.75 × R + 0.25 × V` |

Queries and passages are encoded independently by the same encoder. Longer inputs are truncated; split long documents into passages before encoding. A single-vector cosine search does not reproduce this scorer.

## Evaluation

### Protocol

- **Checkpoint**: stage 2c after 1,024 updates (training step 5152), SHA-256 `c416e3b264f0c756ea96fff7a6a0eec93859747c23d6677ecca0afd02cf9084c`. It was chosen on a development panel by a rule written before training (below), before any test set was scored. Each test set was scored once.
- **Corpus**: the BEIR MS MARCO passage corpus, all 8,841,823 passages, encoded once.
- **Queries**: the 6,980 BEIR MS MARCO dev queries; the 43 judged TREC DL 2019 and 54 judged TREC DL 2020 passage queries.
- **Retrieval**: each of the 16 query routing vectors retrieves its 128 nearest passage routing vectors by exact flat inner-product search (slot-hit width 128). Hits are grouped by passage, up to 1,024 passages are kept by a coarse score, R is computed exactly for those, and the top 100 by R form the candidate list. **QK** ranks that list by R. **QKV** reranks the same list by 0.75 R + 0.25 V. No relevant passages are inserted.
- **Metrics**: graded nDCG@10 (pytrec_eval) and recall@100, multiplied by 100. Recall counts relevance ≥ 1 on MS MARCO dev and ≥ 2 on DL19/DL20.
- **Cross-encoder references** rescore exactly the same width-128 candidate lists: Qwen3-Reranker-0.6B (revision `e61197ed`) and Qwen3-Reranker-4B (revision `22e68366`), 512-token limit, instruction `Retrieve relevant documents.`
- **v1.2-preview rows** come from an earlier run of the same evaluator, spec and settings on the step-4000 checkpoint. One difference: v1.3's corpus was encoded on four GPUs and v1.2's on two, so passages were batched with different neighbours.

### Results (slot-hit width 128, the primary setting)

nDCG@10 × 100:

| System | MS MARCO dev | DL19 | DL20 |
| --- | ---: | ---: | ---: |
| v1.3, QKV (0.75 R + 0.25 V) | 41.71 | 74.63 | 73.14 |
| v1.3, QK (routing only) | 39.30 | 71.56 | 68.73 |
| Qwen3-Reranker-0.6B on v1.3's candidates | 42.36 | 72.80 | 69.44 |
| Qwen3-Reranker-4B on v1.3's candidates | 44.62 | 75.29 | 75.30 |
| v1.2-preview (step 4000), QKV | 34.27 | 68.99 | 59.57 |
| v1.2-preview (step 4000), QK | 31.92 | 64.87 | 56.21 |

Candidate lists (shared by QK, QKV and both cross-encoders at a given width):

| | MS MARCO dev | DL19 | DL20 |
| --- | ---: | ---: | ---: |
| v1.3 recall@100, width 128 | 84.46 | 57.01 | 63.71 |
| v1.3 mean candidates, width 128 | 72.00 | 67.67 | 65.89 |
| v1.3 queries with fewer than 100 candidates, width 128 | 4,964 of 6,980 | 31 of 43 | 43 of 54 |
| v1.3 recall@100, width 512 | 88.55 | 66.63 | 73.22 |
| v1.3 mean candidates, width 512 | 99.84 | 99.98 | 100 |
| v1.3 queries with fewer than 100 candidates, width 512 | 101 of 6,980 | 1 of 43 | 0 of 54 |
| v1.2-preview recall@100, width 128 | 82.01 | 58.99 | 65.24 |
| v1.2-preview mean candidates, width 128 | 92.07 | 83.93 | 93.50 |
| v1.2-preview queries with fewer than 100 candidates, width 128 | 2,161 of 6,980 | 19 of 43 | 18 of 54 |

At width 128 most queries get fewer than 100 candidates, because different query slots often hit the same passages. Recall@100 at width 128 therefore measures lists that average 65.89–72.00 passages, and on DL19 and DL20 it is lower than v1.2's. Widening the search to 512 hits per slot (from the same search) fills the lists: recall@100 rises to 88.55, 66.63 and 73.22, while nDCG@10 is unchanged to two decimals except MS MARCO dev QKV (41.70 at width 512, 41.71 at width 128). Width 128 stays the reported setting because it was fixed before evaluation.

Reading the table: on the same candidates QKV is above QK on all three sets. It is below the 0.6B cross-encoder on MS MARCO dev and above it on DL19 and DL20, and below the 4B cross-encoder on all three. DL19 and DL20 have 43 and 54 queries, so their differences are noisy. MS MARCO is in the training data (see [Training](#training-lineage-and-data)), so none of these results is zero-shot.

### Development panel

5,000 held-out development queries over nine prepared sources, each with 100 saved candidates mined by an earlier checkpoint of this lineage. Every system ranks the same lists, so this measures ranking on fixed candidates, not retrieval. Source-macro nDCG@10 × 100:

| System | QK | QKV / score |
| --- | ---: | ---: |
| v1.2-preview (step 4000) | 60.37 | 63.39 |
| v1.3 (step 5152) | 67.14 | 69.17 |
| Qwen3-Reranker-0.6B on the same lists | — | 69.80 |

This panel chose the checkpoint and its sources overlap the training data; it is not a test set. The selection rule, fixed before training: report the last scored checkpoint unless an earlier one scores higher by more than its paired bootstrap interval. Training was extended in blocks up to 1,024 updates, continuing only while the development score kept rising; that extension was decided after seeing development scores, not planned from the start. 1,024 minus 896 updates was +0.04 QKV [−0.17, +0.25] (10,000 paired bootstrap replicates), so the development score had levelled off.

## Storage and serving cost

Measured on the MS MARCO evaluation's stored passage vectors, through the evaluator's own search and scoring code. With float32 vectors the measurement reproduced the evaluation exactly (same top documents for all 7,077 queries; largest score difference 5.96e-08).

| Stored passage vectors | Bytes per passage | MS MARCO dev QK / QKV / R@100 | DL19 QK / QKV / R@100 | DL20 QK / QKV / R@100 |
| --- | ---: | --- | --- | --- |
| float32 | 32,768 | 39.30 / 41.71 / 84.46 | 71.56 / 74.63 / 57.01 | 68.73 / 73.14 / 63.71 |
| int8, per-vector scale | 8,256 | 39.31 / 41.70 / 84.47 | 71.56 / 74.64 / 57.01 | 68.86 / 73.14 / 63.74 |
| float16 | 16,384 | not yet measured | not yet measured | not yet measured |
| int4, per-vector scale | 4,160 | not yet measured | not yet measured | not yet measured |

Bytes cover the 16 routing and 16 value vectors of 256 dimensions, excluding index structures and metadata. int8 and int4 scale each vector by its largest magnitude; the scale is stored for value vectors only, since routing vectors are compared by cosine. Query vectors stay float32. At float32, 32 KiB per passage is eight times the 4 KiB of a single 1,024-dimensional float32 vector.

**Exact serving latency.** One query at a time (first 1,000 MS MARCO dev queries plus all 97 DL queries; 20 warm-up queries dropped), on the evaluation host with four NVIDIA GeForce RTX 5090 GPUs. The routing vectors of all 8,841,823 passages were held in float16 flat indexes across the four GPUs (72,432,214,016 bytes). The value vectors of the 100 reranked passages were read from local disk, because both vector sets did not fit in host memory at float16.

| Stage, milliseconds per query | Median | Mean | p95 |
| --- | ---: | ---: | ---: |
| Query encoding (batch 1, tokenisation included) | 33.18 | 33.39 | 34.02 |
| Slot search (exact, all passages) | 121.90 | 121.70 | 123.75 |
| Shortlist | 0.56 | 0.59 | 0.87 |
| Fetch routing vectors | 3.57 | 5.32 | 15.72 |
| Routing rescoring | 0.58 | 0.60 | 0.78 |
| Fetch value vectors (disk read) | 11.19 | 10.74 | 16.05 |
| Value reranking | 0.61 | 0.63 | 0.71 |
| **Total** | **172.37** | **172.96** | **187.87** |

Accuracy of the served rankings: MS MARCO dev (those 1,000 queries) QK 39.16, QKV 41.88, R@100 85.83; DL19 71.78 / 74.70 / 57.02; DL20 68.73 / 73.15 / 63.48. Exact search dominates the time. For reference, the cross-encoder runs above, on the same GPUs, rescored 466.93–486.15 query–passage pairs per second per GPU (Qwen3-Reranker-0.6B) and 94.27–111.52 (Qwen3-Reranker-4B). That is batch throughput, not per-query latency, so it does not compare directly with the table.

Not yet measured: float16 and int4 storage accuracy, and approximate search (an IVF-PQ index over all routing vectors in front of the same exact rescoring).

### CPU timings (toolkit)

Measured with the toolkit's `tools/benchmark_cpu.py` on one 16-core / 32-thread desktop CPU (AMD Ryzen 9 7950X3D), torch 2.14.1. Other processes raised the load average from 4 to 13 during the runs, so these are not quiet-machine numbers.

| Measurement | bfloat16, 8 threads | bfloat16, 16 threads | float32, 8 threads | float32, 16 threads |
| --- | ---: | ---: | ---: | ---: |
| Load (`from_pretrained`, checksums included) | 3.4 s | | 3.8 s | |
| One query per call, p50 (p90), tokenisation included | 70 ms (71 ms) | 76 ms (78 ms) | 141 ms (150 ms) | 157 ms (174 ms) |
| Passages per second, 384 tokens each, batch 1 (default) | 2.67 | 2.78 | 1.38 | 1.83 |
| Passages per second, 384 tokens each, batch 8 | 1.58 | 1.91 | 0.92 | 1.10 |

In short: about 70–76 ms per query and 2.7 passages per second in bfloat16, about half that speed in float32 (141–157 ms, 1.4–1.8 passages per second). bfloat16 is faster on this CPU (it has AVX-512 BF16) and changes scores (see **Precision** and **Batching** under [Quick start](#quick-start)). Most of the encoding time is spent in the backbone's 18 linear-attention layers, which run Transformers' PyTorch reference implementation on CPU. PyTorch's default is one thread per physical core; on this loaded host 8 threads answered single queries faster than 16, so give each encoder process a share of the cores (for example half the physical cores for two processes).

Loading peaks at about 2.15 GiB of resident memory in bfloat16 (each tensor is cast while it is read) and 2.57 GiB in float32, most of it the memory-mapped weights file.

Search over 100,000 synthetic passages (float32 storage, 16 threads, one pre-encoded query per call, top 10): `default` p50 177 ms, `paper` 120 ms, `exact` 409 ms. The first per-slot shortlist search builds a float32 cache of unit routing vectors (960 ms and 1.6 GB for 100,000 passages). The synthetic vectors measure cost, not accuracy.

## Training lineage and data

| Stage | Trained | Data | Teacher |
| --- | --- | --- | --- |
| Weak contrastive, 1,000 updates | Backbone, slot extractor, linear routing and value heads | LightOn curated pre-training pairs: 29,939,781 training pairs from 34 subsets (MS MARCO 2.13% of rows); 16.384M pair presentations | — |
| Distillation pilot, 1,600 updates | Slot extractor and value modules (residual value MLP added); backbone and routing head frozen | 20,000 queries: 10,000 LightOn, 10,000 PubMedQA-derived | Qwen3-Reranker-0.6B |
| KL distillation + retrieval replay to step 4000 (= v1.2-preview) | Same restricted modules | 500K-query pool from nine sources; 1.024M KL query presentations, 8.192M replay pair presentations | Qwen3-Reranker-0.6B |
| Ranking-focused continuation, 128 updates (step 4128) | Same restricted modules; backbone frozen | Same pool; loss 0.1 × KL + 0.9 × 1.2226 × top-order | Qwen3-Reranker-0.6B |
| **Stage 2c, 1,024 updates (step 5152 = v1.3)** | **Full text backbone** (learning rate 1e-6), routing projection (2.51e-6), slot extractor and value modules | 262,144 KL queries; 493,366 replay queries; nine sources (below) | **Qwen3-Reranker-4B** |

Stage 2c loss: 0.1 × KL + 0.9 × 1.2226 × top-order + 0.25 × InfoNCE, 256 queries per update, retrieval replay on every 8th update. Unlike v1.2, the KL and top-order losses also train the routing path. The KL stream excludes the development panel's queries.

Stage 2c data by source:

| Source | KL queries | Replay queries |
| --- | ---: | ---: |
| MS MARCO (LightOn fine-tuning collection) | 93,012 | 175,000 |
| Natural Questions (LightOn) | 34,535 | 65,000 |
| FEVER (LightOn) | 31,762 | 60,000 |
| StackExchange duplicate question titles | 26,511 | 50,000 |
| HotpotQA (LightOn) | 21,265 | 40,000 |
| TriviaQA (LightOn) | 21,262 | 40,000 |
| SciRepEval search | 13,322 | 25,000 |
| PubMedQA question–abstract pairs | 13,290 | 25,000 |
| SQuAD v2 (LightOn) | 7,185 | 13,366 |
| **Total** | **262,144** | **493,366** |

Sources on the Hub: [lightonai/embeddings-pre-training-curated](https://huggingface.co/datasets/lightonai/embeddings-pre-training-curated) (weak stage), [lightonai/embeddings-fine-tuning](https://huggingface.co/datasets/lightonai/embeddings-fine-tuning), [sentence-transformers/embedding-training-data](https://huggingface.co/datasets/sentence-transformers/embedding-training-data) (StackExchange duplicate titles), [allenai/scirepeval](https://huggingface.co/datasets/allenai/scirepeval) (search) and [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA). The training data is English.

**Data terms.** The training mixture includes MS MARCO and other sources that carry their own terms. LightOn states that the sources in its curated pre-training collection keep their original licences, and MS MARCO's published terms, for example, describe use for non-commercial research. The Apache-2.0 licence on these weights does not relicense any training source. Review the terms of the training sources before using this model for your purpose.

## Intended use

- Research on retrieval and reranking from cached multi-vector representations.
- Reranking candidate passages using only stored vectors, and comparing that against routing alone or a cross-encoder on the same candidates.
- Inspecting slot-extraction attention and query-to-passage slot attention.
- English passages of up to 384 tokens and queries of up to 128 tokens.

Not intended for: production search without your own evaluation on your data; non-English text; long documents without splitting; treating scores as calibrated probabilities; uses that the training data's terms do not allow.

## Limitations

- **Narrow test evidence.** For this checkpoint only MS MARCO dev, DL19 and DL20 have been run. BEIR has not been run, and neither have same-protocol baselines (single-vector retrievers, ColBERT-style models, or another retriever followed by a cross-encoder).
- **Not state of the art on BEIR.** v1.2-preview scored 51.54 macro nDCG@10 on 15 BEIR tasks (QKV), against a published 55.52 for Qwen3-Embedding-0.6B; protocols differ. v1.3's BEIR score is unknown.
- **In-domain test sets.** MS MARCO training queries and passages are part of the training data, and query overlap with MS MARCO dev and TREC DL has not been fully audited. Do not read these results as zero-shot.
- **Shortlist underfill.** At the reported width of 128, 4,964 of 6,980 MS MARCO dev queries, 31 of 43 DL19 queries and 43 of 54 DL20 queries had fewer than 100 candidates, more than for v1.2. Recall@100 at width 128 is measured on these shorter lists.
- **Small query sets and one run.** DL19 and DL20 have 43 and 54 queries; one training seed; the development panel informed many earlier decisions.
- **Remaining gap to a larger cross-encoder.** On identical candidates, Qwen3-Reranker-4B scores above QKV on MS MARCO dev, DL19 and DL20, and Qwen3-Reranker-0.6B scores above it on MS MARCO dev.
- **Storage.** 32,768 bytes per passage at float32; int8 (8,256 bytes) is measured above. float16, int4 and approximate search are not yet measured. The GPU latency figures cover exact search on four GPUs; the CPU figures cover the toolkit on one 16-core CPU.
- **English only.** Training data is English. Other languages have not been evaluated.
- **Toolkit scope.** The toolkit's `Index` runs the evaluated per-slot shortlist pipeline (`paper` profile) with exact slot search on CPU or flat FAISS, over in-memory or memory-mapped shards; it has no approximate search, metadata filtering or server. The training pipeline and the paper's evaluation harness are not in the toolkit.
- **Truncation.** Queries over 128 tokens (including the instruction) and passages over 384 tokens are truncated.

## Citation

<!-- TODO: replace with the final citation. -->

```bibtex
@misc{breadbowl_embed,
  title  = {BreadBowl-Embed: Encode Once, Retrieve and Rerank},
  author = {TODO},
  year   = {TODO},
  note   = {TODO}
}
```

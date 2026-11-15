---
license: apache-2.0
library_name: breadbowl-embed
pipeline_tag: sentence-similarity
base_model: Qwen/Qwen3.5-0.8B-Base
base_model_relation: finetune
language: [en]
tags: [multi-vector, late-interaction, retrieval, reranking, research]
datasets:
- lightonai/embeddings-pre-training-curated
- lightonai/embeddings-fine-tuning
- sentence-transformers/embedding-training-data
- allenai/scirepeval
- qiaojin/PubMedQA
model-index:
- name: breadbowl-embed-v1.3
  results:
  - task: {type: text-retrieval, name: Passage retrieval and reranking}
    dataset: {type: BeIR/msmarco, name: "MS MARCO passage dev (BEIR, 6,980 queries, full 8,841,823-passage corpus)", split: dev}
    metrics:
    - {type: ndcg_at_10, name: "nDCG@10, QKV (0.75 routing + 0.25 value), slot-hit width 128", value: 41.71}
    - {type: ndcg_at_10, name: "nDCG@10, QK (routing only), slot-hit width 128", value: 39.30}
    - {type: recall_at_100, name: "Recall@100, slot-hit width 128, relevance >= 1", value: 84.46}
  - task: {type: text-retrieval, name: Passage retrieval and reranking}
    dataset: {type: trec-dl-2019-passage, name: "TREC DL 2019 passage (43 judged queries, same 8,841,823-passage corpus)", split: test}
    metrics:
    - {type: ndcg_at_10, name: "nDCG@10, QKV (0.75 routing + 0.25 value), slot-hit width 128", value: 74.63}
    - {type: ndcg_at_10, name: "nDCG@10, QK (routing only), slot-hit width 128", value: 71.56}
    - {type: recall_at_100, name: "Recall@100, slot-hit width 128, relevance >= 2", value: 57.01}
  - task: {type: text-retrieval, name: Passage retrieval and reranking}
    dataset: {type: trec-dl-2020-passage, name: "TREC DL 2020 passage (54 judged queries, same 8,841,823-passage corpus)", split: test}
    metrics:
    - {type: ndcg_at_10, name: "nDCG@10, QKV (0.75 routing + 0.25 value), slot-hit width 128", value: 73.14}
    - {type: ndcg_at_10, name: "nDCG@10, QK (routing only), slot-hit width 128", value: 68.73}
    - {type: recall_at_100, name: "Recall@100, slot-hit width 128, relevance >= 2", value: 63.71}
---

# BreadBowl-Embed v1.3

**Encode once. Retrieve and rerank from the same stored representation.**

An English retrieval model that stores each passage once, and uses the same stored vectors to find candidates and to rerank them.

![Diagram. A query or passage goes through the backbone, Qwen3.5 0.8B, and becomes 16 slots, stored once per passage. Each slot has a routing vector (256-d, finds candidates) and a value vector (256-d, reranks). Score equals 0.75 times routing plus 0.25 times value. No second model at rerank time.](assets/architecture.png)

## How it works

- A 0.8B-parameter text backbone reads the text once and pools it into **16 slots**.
- Each slot holds a **routing** vector and a **value** vector, 256 numbers each.
- **Find:** the query's routing vectors search the stored passage routing vectors.
- **Rerank:** the same routing match decides which stored value vectors the query reads.
- **Score** = 0.75 × routing + 0.25 × value. No cross-encoder, no second pass over the passage.
- Every passage costs the same 16 + 16 vectors, however long it is.

## Quick start

Python 3.12 or newer. Install the library from GitHub:

```bash
pip install "git+https://github.com/BreadBowlAI/breadbowl-embed.git"
```

```python
from breadbowl_embed import BreadBowl

model = BreadBowl.from_pretrained("dotproductx/breadbowl-embed-v1.3")
ids = ["octopus", "ocean", "plants"]
passages = ["Octopuses have three hearts.",
            "The Pacific is Earth's largest ocean.",
            "Plants use sunlight during photosynthesis."]

index = model.index(model.encode_documents(passages), ids=ids)  # encode once, keep
query = model.encode_queries(["How many hearts does an octopus have?"])
for hit in index.search(query, top_k=2)[0]:                     # retrieve, then rerank
    print(hit.id, round(hit.score, 3))
```

More examples (rerank a list, save an index, explain a score) are in the [library README](https://github.com/BreadBowlAI/breadbowl-embed).

## Results

![Bar chart of nDCG@10 on MS MARCO dev, TREC DL19 and TREC DL20. v1.3 is higher than v1.2-preview on all three. The values are in the table below.](assets/benchmarks.png)

| nDCG@10 | MS MARCO dev | TREC DL19 | TREC DL20 |
| --- | ---: | ---: | ---: |
| **v1.3** | **41.71** | **74.63** | **73.14** |
| v1.2-preview | 34.27 | 68.99 | 59.57 |
| Qwen3-Reranker-0.6B, same candidates | 42.36 | 72.80 | 69.44 |
| Qwen3-Reranker-4B, same candidates | 44.62 | 75.29 | 75.30 |

- **Setup:** full MS MARCO corpus (8.8 million passages), `paper` search profile, each test set scored once.
- **Cross-encoder rows** rerank v1.3's own candidate lists, so only the ordering differs.
- **Storage:** 32,768 bytes per passage in float32. int8 needs 8,256 bytes and scores almost the same.
- **Speed:** 172 ms per query end to end (median), exact search over the whole corpus on four GPUs.
- **New in v1.3:** 1,024 more training updates, with the full backbone unfrozen and a larger teacher.

## Look inside

![Real model output for the query "How many hearts does an octopus have?". Three rows, one per passage slot, each showing the passage with words shaded by attention. In all three rows the darkest words are "hearts" and the full stop after it. Slot 12 also weights "Octopuses" and "three".](assets/attention.png)

Every score can be taken apart. `model.explain(queries, documents)` gives the slot-to-slot attention.
`encode_documents(..., return_attention=True, return_tokens=True)` shows which words fed each slot.
The figure above comes from [`scripts/attention_figure.py`](scripts/attention_figure.py).

## Search profiles

| Profile | Use it for |
| --- | --- |
| `default` | Everyday search. 512 hits per slot, so the 100-passage rerank list fills up. |
| `paper` | Matching the results above. 128 hits per slot. |
| `exact` | Small collections. Scores every passage; cost grows with collection size. |

Pick one with `index.search(query, profile="paper")`.

## Limits and licence

- **English only.** Queries are cut at 128 tokens (instruction included), passages at 384. Split long documents.
- **Small, in-domain tests.** MS MARCO is in the training data. DL19 and DL20 have only 43 and 54 queries. One training run. BEIR has not been run for v1.3.
- **Not ahead of a large cross-encoder.** Qwen3-Reranker-4B scores higher on all three sets.
- **Cost.** On a 16-core desktop CPU (bfloat16): about 70 ms per query, 2.7 passages per second. CUDA is not yet validated in the library.
- **Precision.** Results were measured in bfloat16. Load with `dtype="bfloat16"` to match; the float32 default scores slightly differently.
- Scores rank candidates; they are not probabilities. This is a research release.
- **Licence.** Weights: Apache-2.0. Base model: [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base).
- **Training data keeps its own terms.** MS MARCO's terms, for example, describe non-commercial research use. The Apache-2.0 licence does not relicense any training source. Check those terms before you use the model.

## Links

- Code: [github.com/BreadBowlAI/breadbowl-embed](https://github.com/BreadBowlAI/breadbowl-embed)
- Full details (evaluation, storage, speed, training data, citation): [model-details-v1.3.md](https://github.com/BreadBowlAI/breadbowl-embed/blob/main/docs/model-details-v1.3.md)
- Previous version: [v1.2-preview](https://huggingface.co/dotproductx/breadbowl-embed-v1.2-preview)

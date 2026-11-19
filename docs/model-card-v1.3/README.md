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

**One model does the work of two.** It encodes each passage once into 16 slots, then finds *and* reranks candidates from those same stored vectors. No cross-encoder. No second pass over the text.

![Two stacks compared. The usual stack: retriever, vector index, candidate texts, then a cross-encoder (Qwen3-Reranker-0.6B, on v1.3's candidates) that runs a full pass per candidate: 150 ms to rerank, nDCG@10 42.4, 72.8 and 69.4. BreadBowl-Embed: Slot Encoder (the only model), routing index, stored routing and value slots, then attention over stored values with no model run: 2.9 ms to rerank, nDCG@10 41.7, 74.6 and 73.1.](assets/one-vs-two.png)

*Same candidates for both (v1.3's, 72 per query on average), one RTX 5090. The cross-encoder is Qwen3-Reranker-0.6B; its time comes from measured batch throughput. ▼ significantly below v1.3.*

- **As good as two models.** Level with Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B on MS MARCO and DL19, and **5.5 nDCG@10 ahead on DL20**.
- **50× faster rerank.** 2.9 ms against 150 ms on the same candidates.
- **Ahead of its first teacher on DL20.** 3.7 points above Qwen3-Reranker-0.6B on the same candidates; level on MS MARCO and DL19. No candidate text is read at query time.
- **6 wins, 9 level, 0 losses** against released models of similar size.
- **BEIR:** 55.1 mean over 15 tasks, 3.6 above v1.2-preview.

📄 **Paper:** [BreadBowl-Embed: Encode Once, Retrieve and Rerank](paper/breadbowl-embed-v1.3.pdf)

## How it works

![Diagram. A query or passage goes through the backbone, Qwen3.5 0.8B, and becomes 16 slots, stored once per passage. Each slot has a routing vector (256-d, finds candidates) and a value vector (256-d, reranks). Score equals 0.75 times routing plus 0.25 times value. No second model at rerank time.](assets/architecture.png)

- **Encode once.** A 0.8B backbone reads each passage once, at index time, into 16 slots. At search time only the query runs through the model.
- **Find.** Each slot holds a **routing** vector and a **value** vector. The query's routing vectors search the stored routing vectors.
- **Rerank.** The same routing match decides which stored value vectors the query reads. That is a few small matrix products, not a model pass.
- **Score** = 0.75 × routing + 0.25 × value. The toolkit splits every score into slot-pair parts you can inspect.
- **Fixed size.** 16 + 16 vectors per passage, however long it is. At int4: 4,160 bytes, for at most 0.12 points.

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

| nDCG@10, full MS MARCO corpus | Models | MS MARCO dev | DL19 | DL20 |
| --- | ---: | ---: | ---: | ---: |
| **v1.3** | **1** | **41.7** | **74.6** | **73.1** |
| Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B | 2 | 41.9 | 72.0 | 67.6 ▼ |
| Jina v5 text-small + Qwen3-Reranker-0.6B | 2 | 42.0 | 72.6 | 66.8 ▼ |
| Qwen3-Embedding-0.6B alone | 1 | 38.0 ▼ | 66.8 ▼ | 67.6 ▼ |
| v1.3 + Qwen3-Reranker-4B | 2 | 44.6 ▲ | 75.3 | 75.3 |

▼ / ▲ significantly below / above v1.3 (95% paired bootstrap); unmarked rows are level with it. Every setting is in the paper.

## Limits

- **English only.** MS MARCO is in the training data. DL19 and DL20 have 43 and 54 queries. Trained once, not repeated over seeds.
- **Not top of every chart.** Qwen3-Reranker-4B, five times larger, is ahead on MS MARCO. On BEIR, the Jina v5 paper reports higher means for Qwen3-Embedding-0.6B and Jina v5 text-small, under other settings. On argument retrieval the value score falls short of the cross-encoders.
- **CPU is slow.** About 70 ms per query and 2.7 passages per second on a 16-core desktop CPU (measured with the library). CUDA is not yet validated in the library.
- **Precision.** Results were measured in bfloat16; load with `dtype="bfloat16"` to match.
- Scores rank candidates; they are not probabilities. This is a research release.
- **Licence.** Weights: Apache-2.0. Base model: [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base). Training data keeps its own terms (MS MARCO's describe non-commercial research use); check them before you use the model.

## Links

- Paper: [paper/breadbowl-embed-v1.3.pdf](paper/breadbowl-embed-v1.3.pdf)
- Code: [github.com/BreadBowlAI/breadbowl-embed](https://github.com/BreadBowlAI/breadbowl-embed)
- Full details (BEIR per task, storage, speed, training data, citation): [model-details-v1.3.md](https://github.com/BreadBowlAI/breadbowl-embed/blob/main/docs/model-details-v1.3.md)
- Previous version: [v1.2-preview](https://huggingface.co/dotproductx/breadbowl-embed-v1.2-preview)

# Evaluation

`breadbowl_embed.evaluation` measures retrieval and reranking quality on your own data, with the same search pipeline and score as the model card. It runs on CPU, records how each number was produced, and does not need any package beyond the library's own (the MTEB adapter needs the optional `mteb`).

| | |
|---|---|
| Data | BEIR layout (`corpus.jsonl`, `queries.jsonl`, `qrels/<split>.tsv`), TREC qrels, TREC run files, or plain Python mappings |
| Metrics | nDCG@k, MRR@k, Recall@k, MAP@k (trec_eval definitions), per query and averaged |
| Stages | `routing`: candidates ranked by the routing score R (QK); `final`: the same candidates ranked by 0.75 R + 0.25 V (QKV); in rerank mode also `input`, the supplied order |
| Candidates | size of each query's candidate list, how many queries got fewer than the rerank width, recall of the list; optionally recall of the larger per-slot shortlist |
| Uncertainty | bootstrap confidence intervals over queries; paired randomization test and paired bootstrap interval for comparing two runs or two stages |
| Outputs | `result.json` (settings, means, intervals, candidate statistics, timings), `per_query.jsonl`, `runs/<stage>.trec` |
| Encoding | resumable: the corpus is written in parts in the library's `Index` format, and a rerun continues where it stopped |

## Reproduce a card-style number on your own data

The model card's protocol, in toolkit terms: search profile `paper` (128 slot hits per query slot, shortlist 1,024, rerank 100); **QK** is the `routing` stage, **QKV** the `final` stage, and **recall@100** is the recall of the 100-candidate list (the same for both stages); nDCG@10 is graded; recall counts relevance ≥ 1 on MS MARCO dev and ≥ 2 on TREC DL (`--relevance-level 2`; nDCG is graded and does not use it). The card used a bfloat16 backbone, so pass `--dtype bfloat16`.

```bash
python -m breadbowl_embed.evaluation retrieve \
  --model dotproductx/breadbowl-embed-v1.3 --dtype bfloat16 --threads 8 \
  --data path/to/beir/dataset --split test \
  --work-dir path/to/work --output path/to/result \
  --profile paper --metrics ndcg@10,mrr@10,recall@100 --relevance-level 1
```

- `--work-dir` receives the corpus encoding (`build.json`, `parts/part-*/`, `index-float32/`). Interrupt at any time and rerun the same command: finished parts are kept, a half-written part is redone, and the run refuses to continue with a different model, dtype, batch size, document format, chunk size, library version or corpus. `python -m breadbowl_embed.evaluation index ...` encodes without evaluating, e.g. overnight.
- `--output` must be a new directory. The command prints the stage means, candidate statistics and timings as JSON, and a table on stderr.
- `--index DIR` evaluates an existing index directory (from `breadbowl-embed index`) or build directory instead of encoding. An index directory does not record how its corpus was encoded (batch size, title handling), so prefer a build directory.
- `--max-queries N --seed S` evaluates a seeded random sample of the judged queries. The corpus is always encoded in full; searching a smaller corpus is a different, easier task.
- `--document-format title_text` (default) encodes `(title + " " + text).strip()`, MTEB's rule; `text` ignores titles. MS MARCO passages have no titles, so both give the same text there.
- `--profile default` (512/1,024/100) fills the candidate lists; `exact` scores every document's routing (linear in corpus size).

In Python:

```python
from breadbowl_embed import BreadBowl
from breadbowl_embed.evaluation import build_index, compare, evaluate_retrieval, load_beir

model = BreadBowl.from_pretrained("dotproductx/breadbowl-embed-v1.3", dtype="bfloat16")
data = load_beir("path/to/beir/dataset", split="test")
build_index(model, data.corpus, "path/to/work")                  # resumable
result = evaluate_retrieval(model, "path/to/work", data.queries, data.qrels, profile="paper",
                            metrics=["ndcg@10", "recall@100"], dataset=data.source)
print(result.summary())                       # routing and final rows
print(result.confidence_interval("ndcg@10"))  # final stage, 95 % percentile bootstrap
print(compare(result, result, "ndcg@10", stage_a="routing", stage_b="final"))  # QKV - QK, paired
result.save("path/to/result")
```

### Reranking supplied candidates

To rerank another system's candidates (for example BM25) with the stored-vector score, give a TREC run:

```bash
python -m breadbowl_embed.evaluation rerank --model ... --dtype bfloat16 \
  --data path/to/beir/dataset --candidates bm25.trec --depth 100 \
  --work-dir path/to/rerank-work --output path/to/rerank-result
```

Only the documents among the top `--depth` candidates of the judged queries are encoded. Stages: `input` (the run's order, by descending score, ties by ascending document id), `routing` and `final`. Candidates that are not in the corpus fail the run unless `--missing-documents drop` (the count is recorded). `evaluate_rerank(...)` is the Python equivalent and also takes candidates as `{query id: [document ids]}`.

### Comparing runs

```bash
python -m breadbowl_embed.evaluation compare --result-a path/to/result-a --result-b path/to/result-b --metric ndcg@10
python -m breadbowl_embed.evaluation compare --result-a path/to/result --result-b path/to/result --stage-a routing --metric ndcg@10
python -m breadbowl_embed.evaluation compare --qrels qrels.tsv --run-a a.trec --run-b b.trec --metric ndcg@10
python -m breadbowl_embed.evaluation metrics --qrels qrels.tsv --run a.trec --run b.trec --ci
```

`compare` reports both means, the mean difference (b − a) with a 95 % paired percentile bootstrap interval, a two-sided paired randomization (sign-flip) test p-value, and how many queries each side won (10,000 resamples, seed 0 by default; p is (1 + extreme draws) / (1 + resamples), so it is never 0). Both runs must cover the same queries, and two saved results must have been evaluated against the same qrels (checked by fingerprint).

## What each result records

`result.json` → `settings`:

- `library_version`, `mode` (retrieval or rerank), `created`.
- `model`: `release_id` (identity of the weights and encoding configuration), `representation_id` (adds the compute dtype), `compute_dtype`, `device`, `weights_sha256`, Hub repository/revision/loaded commit when known, training step, scoring settings, query/document formats and token limits. A local model directory is recorded by name only, so results can be shared without local paths.
- `query_encoding`: batch size, instruction. `corpus`: the build directory's batch size, document format, chunk size, corpus fingerprint, library version and encoding throughput (or, for a bare index directory, that these are unknown).
- `search`: the profile's name and parameters, the slot-search backend. `rerank` mode: depth, candidate source, dropped candidates.
- `metrics`: names, relevance level and the conventions below. `data`: dataset name and split, fingerprints of queries and qrels, judged/evaluated/skipped query counts, the query sample if any. `environment`: Python, PyTorch, Transformers, NumPy, thread count, platform.

`timings` holds query-encoding and search (or scoring) seconds and milliseconds per query; the build directory holds encoding seconds per part.

## Metric conventions

- **nDCG@k**: linear gain = judged relevance (unjudged and non-positive judgments give 0), discount log2(rank + 1), ideal DCG from all of the query's positive judgments. trec_eval `ndcg_cut_k`.
- **MRR@k**: 1 / rank of the first document with relevance ≥ the relevance level within the top k, else 0.
- **Recall@k**: relevant documents in the top k / all relevant documents of the query. trec_eval `recall_k`.
- **MAP@k**: sum of precision at each relevant rank ≤ k, divided by all relevant documents of the query (not min(k, relevant)). trec_eval `map_cut_k`.
- **Queries**: a query is evaluated when it has at least one positive judgment; others are skipped and counted. An evaluated query that the run lacks scores 0, and a query with no document at the binary relevance level scores 0 on MRR, recall and MAP, so every mean is over the same queries. trec_eval/pytrec_eval would also average in judged queries without positives (as zeros); MTEB drops queries without any judgment but keeps those judged all-zero.
- **Ties**: equal scores are ordered by ascending document id, the order `Index.search` returns. trec_eval, pytrec_eval, ranx and MTEB order ties by descending id; `evaluate_run(..., tie_break="trec_eval")` and `metrics --tie-break trec_eval` reproduce that. Exact ties occur mainly between identical documents, which get identical vectors.
- **Cross-checks** (`tests/test_evaluation_crosscheck.py`, skipped unless the packages are installed): on random graded qrels and runs (40 queries, partial runs, a query without positives), per-query nDCG@1/10/100, Recall@10/100, MAP@10/100 and MRR@10/100 equal pytrec_eval-terrier 0.5.11 within 1e-12 at relevance levels 1 and 2, with and without tied scores (ties with `tie_break="trec_eval"`), and equal ranx 0.3.21 within 1e-12 at level 1 without ties. The hand-computed cases are in `tests/test_evaluation.py`.

## What differs from the model card's setup

| | Model card | Typical run with this toolkit |
|---|---|---|
| Hardware | four NVIDIA RTX 5090 GPUs | CPU |
| Compute dtype | bfloat16 backbone under GPU autocast | whatever you load (`--dtype`; native float32 by default). CPU and GPU bfloat16 kernels differ, and float32 and bfloat16 encodings differ (final scores by up to about 1e-2 in the release validation), so rankings among close candidates can change |
| Corpus | the full BEIR MS MARCO corpus, 8,841,823 passages | your corpus. Fewer documents make retrieval easier and change how full the candidate lists are; numbers on different corpora are not comparable |
| Corpus batching | encoded in batches on four GPUs | batch 1 by default (each encoding independent of its neighbours) |
| Slot search | exact FAISS flat inner product on GPU | exact PyTorch matrix products on CPU (or flat FAISS); the release validation (20 queries, 20 documents) found the same documents in the same order |
| Profile | slot-hit width 128, shortlist 1,024, rerank 100 (`paper`) | `paper` by default in `retrieve`; the library's own default is `default` (512) |
| Metrics | the research evaluator with pytrec_eval | this toolkit's metrics, cross-checked against pytrec_eval (above) |
| Queries | default instruction `Retrieve relevant documents.` | the checkpoint's default instruction unless `--instruction` is given |

So the card's MS MARCO dev and TREC DL numbers can be approached only by encoding the full 8.8M-passage corpus, which is not practical on one CPU (below). On your own data, the same commands give numbers measured with the same pipeline and score, labelled with everything that differs.

## CPU cost

Measured with this toolkit on one AMD Ryzen 9 7950X3D (16 cores), PyTorch 2.14.1, 8 threads, batch 1, v1.3 weights. Other processes raised the load average to between 15 and 31 during most of these runs, so a quiet machine will be somewhat faster (the model card's CPU table, on the same CPU at a load of 4 to 13, has 2.67 passages/s at 384 tokens in bfloat16).

| Encoding, documents per second (12 documents each) | bfloat16 | float32 (native) |
|---|---:|---:|
| 16 tokens | 10.4 | 4.1 |
| 64 tokens | 6.8 | 2.65 |
| 128 tokens | 5.1 | 1.85 |
| 384 tokens (the limit) | 2.3 | 0.79 |
| One short query, p50 | 95 ms | 215 ms |

End to end, bfloat16: NanoQuoraRetrieval (5,046 documents of 55 characters on average, 50 queries) took 565 s to encode (8.9 documents/s, in 20 resumable parts), 98 ms per query to encode and 5.6 ms (`paper`) or 9.0 ms (`default`) per query to search; the `retrieve` command on the finished build took 14 s including loading the model, and the encoding run peaked at 3.2 GB of resident memory.

Estimates (documents ÷ the bfloat16 rates above, not measured runs):

| Corpus | Encoding time |
|---|---|
| 5,000 documents of 384 tokens | about 40 minutes |
| 50,000 documents of 128 tokens | about 3 hours |
| 1,000,000 documents of 64 tokens | about 2 days |
| MS MARCO, 8,841,823 passages (mostly under 128 tokens) | about 15 to 20 days |

Disk: the float32 parts take 32,768 bytes per document and the merged index as much again at float32 (8,256 bytes at int8), checked before encoding starts; `--prune-parts` deletes the parts after merging. Memory: the merged index is loaded into memory, and the first per-slot search adds a float32 copy of the unit routing vectors (16,384 bytes per document), so about 48 KiB per document at float32 storage; the full MS MARCO corpus would need about 400 GiB. Search time grows linearly with the corpus (the model card measured 120 ms per query for `paper` over 100,000 synthetic documents on 16 threads).

## MTEB

`BreadBowlMTEB` lets `mteb` evaluate the model with its own score. MTEB ranks encoder models by cosine or MaxSim over vectors returned from `encode`; BreadBowl's 0.75 R + 0.25 V over 16 routing/value slots is neither, so the adapter implements MTEB's search protocol (`mteb.models.SearchProtocol`: `index`, `search`, `mteb_model_meta`) instead of the encoder protocol. `index` encodes the task corpus with MTEB's text rule (`(title + " " + text).strip()`) into an `Index`; `search` encodes the queries with the checkpoint's instruction template and returns each query's documents and scores from `Index.search`. For reranking tasks (MTEB passes candidate lists), it scores exactly those candidates. MTEB then computes its own metrics (with pytrec_eval) from these scores.

```bash
python -m pip install ".[mteb]"     # from a checkout; "breadbowl-embed[mteb]" once the package is on PyPI
```

```python
import mteb
from breadbowl_embed.evaluation import BreadBowlMTEB

model = BreadBowlMTEB.from_pretrained("dotproductx/breadbowl-embed-v1.3", dtype="bfloat16",
                                      profile="paper", cache_dir="mteb-encodings")
results = mteb.evaluate(model, mteb.get_tasks(tasks=["NanoQuoraRetrieval"]), cache=None)
```

- `profile`, `slot_hits`, `shortlist`, `candidate_k`: the search profile (default `default`). At most `candidate_k` (100) documents are returned per query, so MTEB's recall@1000 is computed on that list; raising `candidate_k` reranks more candidates by V and can change the top 10.
- `stage="routing"` returns R instead of the final score, over the same candidates (QK).
- `cache_dir`: encode each task's corpus resumably under this directory (one build per task, subset, split, model and batch size); without it, encodings live in memory.
- `query_batch_size`, `document_batch_size` (1), `storage` (`float32`), `instruction` (the checkpoint's). Per-query instructions that a task supplies are appended to the query text, as MTEB does for models without instruction support. Conversational (multi-turn) queries are refused.
- `model.records` lists every `index` and `search` call with document/query counts and seconds.

**Metadata.** `mteb_model_meta` is an `mteb.ModelMeta`: name (the release's Hub repository when recorded, otherwise `local/breadbowl-embed-<release id>`, or `name=`), revision (loaded Hub commit, recorded revision or release id), `model_type=["late-interaction"]`, `similarity_fn_name="custom"` (on mteb releases that have it; otherwise unset), `use_instructions=True`, `max_tokens=384`, `embed_dim=256` (per routing or value vector; each text has 16 of each), parameter count and memory at the compute dtype, `languages=["eng-Latn"]`, `license="apache-2.0"`, `adapted_from` the base model, `framework=["PyTorch"]`, `output_dtypes=None` (MTEB receives scores, not embeddings), and the settings above (profile, candidate width, stage, dtype, storage, batch sizes, instruction, library version) as `experiment_kwargs` and `loader_kwargs`, so they appear in MTEB's result files and `meta.load_model()` restores them. Fields an older mteb release does not know are left out. `training_datasets` lists the MTEB tasks of the fine-tuning sources the model card names (MS MARCO, Natural Questions, FEVER, HotpotQA and their MTEB variants); the weak pre-training stage's 34-subset collection has not been mapped to MTEB tasks, so treat MTEB's zero-shot flag for this model as unreliable.

**Versions.** The protocol was taken from the source of mteb 2.27.14, the newest release on 2026-11-15. `tests/test_evaluation_mteb.py` (protocol checks, a hand-made retrieval task and a hand-made reranking task run through `mteb.evaluate`, MTEB's metrics compared with the toolkit's, and the metadata loader round trip) passes with mteb 2.0.5, 2.10.14 and 2.27.14; 2.0.5 and 2.10.14 were installed next to 2.27.14's dependencies rather than their own pins. Releases in between were not run; on this evidence the adapter supports `mteb>=2.0.5,<3`. With the v1.3 weights in bfloat16 on CPU, mteb 2.27.14 also ran a hand-made 20-passage task and NanoQuoraRetrieval end to end; MTEB's numbers equalled the toolkit's own evaluation of the same encoded index to MTEB's five decimals (nDCG@10, Recall@10 and MRR@10 on the hand-made task; nDCG@1, nDCG@10, MRR@10, MAP@100, Recall@10 and Recall@100 on NanoQuoraRetrieval). No other MTEB task has been run.

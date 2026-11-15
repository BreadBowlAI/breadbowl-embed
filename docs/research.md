# Research toolkit (`breadbowl_embed.research`)

Tools for looking inside BreadBowl Embed and testing what you see: slot-attention maps, an exact decomposition of every score, token attribution, interventions that re-run the model or the scorer, slot and search-pipeline diagnostics, precision experiments, an offline HTML report, notebook displays, matplotlib figures and JSON export.

> **Stability.** `breadbowl_embed.research` is **experimental**: names, signatures and JSON fields may change between minor versions (0.2 → 0.3). The core API (`BreadBowl`, `Embeddings`, `score`, `score_routing`, `explain`, `Index`, `SearchProfile`, the CLI commands other than `inspect`) is stable. The research package never changes core behaviour: it reuses the library's own tokenizer call, forward pass and scorer, and its defaults reproduce the core numbers exactly (tested bit for bit, below).

```bash
pip install "breadbowl-embed[viz]"   # once the package is published on PyPI; matplotlib is only needed for .plot()
python -m pip install "breadbowl-embed[viz] @ git+https://github.com/BreadBowlAI/breadbowl-embed.git"   # works now
```

## Quick start

```python
from breadbowl_embed import BreadBowl, research

model = BreadBowl.from_pretrained("dotproductx/breadbowl-embed-v1.3")
pair = research.inspect_pair(model, "How many hearts does an octopus have?",
                             "Octopuses have three hearts. Two branchial hearts pump blood ...")
pair.decomposition.final, pair.decomposition.routing, pair.decomposition.value   # == model.score(...)
pair.report("pair.html")       # self-contained HTML: works offline, light and dark, keyboard-usable
pair                           # in a notebook: a static summary
pair.save_json("pair.json")    # every research object exports JSON
```

From the shell:

```bash
breadbowl-embed inspect pair --model dotproductx/breadbowl-embed-v1.3 \
  --query "How many hearts does an octopus have?" --document "Octopuses have three hearts. ..." --html pair.html
```

## What each tool answers

| Tool | Question | Model passes | Returns |
|---|---|---|---|
| `trace(model, texts, kind)` | What did the encoder compute? Backbone token states, slot attention before and after the softmax, projections, slot states before and after the layer norm, raw routing, value parts, learned slot queries | one per text | `TraceSet` of `Trace` |
| `attention_maps(model, texts, kind)` | Which tokens, words and character spans did each slot pool from, per head or averaged? How focused is each slot? Does it read the query instruction or the query text? | one per text | `AttentionMap` |
| `decompose(queries, documents)` | How does R, V and the final score split over the 16 × 16 slot pairs? | none (stored vectors) | `ScoreDecomposition` |
| `attribute(decomposition, attention)` | Which tokens carry the score through which slots (attention-weighted, not causal)? | none | `TokenAttribution` |
| `inspect_pair(model, query, document)` | All of the above for one pair, plus the HTML report | two | `PairInspection` |
| `slot_ablation`, `ablate_slots` | What happens to scores and rankings without a query or document slot? | none | `SlotAblation`, `Scores` |
| `occlude`, `occlusion_sweep` | What happens when a span is deleted, replaced, or hidden from the slots? | one per occlusion (`delete`/`replace`); one in total (`slot-mask`) | `Occlusion`, `OcclusionSweep` |
| `rescore`, `compare_rules`, `ScoringRule` | Would another rule (R only, V only, MaxSim, other temperatures) rank differently? | none | `Scores`, `RuleComparison` |
| `slot_statistics` | Do slots collapse onto each other? How much does each vary across texts? How many directions do they use? Which slots win in scoring? | none | `SlotStatistics` |
| `shortlist_diagnostics` | How close is a slot-shortlist profile to exact search? Which query slots find the exact top documents? How many queries underfill the rerank list? | none | `ShortlistDiagnostics` |
| `compare_precision` | How much do float16/int8 storage, bfloat16 compute and batching move scores and rankings on my texts? | one per text per compute dtype and batch size | `PrecisionComparison` |

Every result object has `to_dict(digits=None)`, `to_json()`, `save_json(path)` and `_repr_html_()` (a static notebook view, no scripts). Most have `.plot()` (needs the `viz` extra). JSON output carries `"type"`, `"schema": 1` and `"library_version"`; tensors become nested lists, non-finite floats become `null`, and `digits=` rounds floats so saved results diff cleanly.

## Worked example

Real output of [`examples/research.py`](../examples/research.py) with the v1.3 weights (float32, CPU):

```text
final 0.817236 = 0.75 * R 0.824890 + 0.25 * V 0.794275
recombination residuals: {'final': '-4.1e-08', 'routing': '-2.3e-08', 'value': '-3.7e-08'}
  query slot  7 x document slot 11: contribution 0.0346, cosine 0.867
  query slot  0 x document slot 11: contribution 0.0342, cosine 0.862
  query slot 11 x document slot 11: contribution 0.0333, cosine 0.871
instruction + template mass per query slot: [0.032, 0.008, 0.15, 0.034, 0.022, 0.022, 0.034, 0.011, 0.003, 0.007, 0.018, 0.01, 0.047, 0.046, 0.004, 0.026]
document words carrying the most attribution: [('.', 0.1322), ('hearts', 0.12), ('.', 0.0768), ('the', 0.0441), ('Octopuses', 0.0425)]
  deleting 'Octopuses have three hearts.': -0.4415
  deleting 'Two branchial hearts pump blood through the gill': -0.0074
  deleting 'Their blood is blue because it carries oxygen wi': +0.0267
words whose hiding from the slots costs most: [('.', -0.0215), ('hearts', -0.0213), ('while', -0.002), ('hearts', -0.002)]
```

How to read it:

- The decomposition is exact: the slot-pair parts recombine to the scorer's numbers within 4e-8. Most of the score runs through one document slot (11) that every query slot matches.
- For this query, every query slot puts at least 84% of its attention on the query text and at most 16% on the instruction and template.
- The attribution and the occlusions agree on "hearts" and on the end of the first sentence. The large weight on "." is typical of a causal backbone: a sentence-final token's state summarises the sentence before it, so a slot reading "." reads that sentence. Attribution follows the pooling weights, so it shows where the slots read, not which words matter; the occlusions are the causal test. Deleting the answer sentence lowers the score by 0.44, deleting another sentence changes it by 0.03 or less.

## Tracing: intermediate tensors

```python
traces = research.trace(model, ["Octopuses have three hearts."], "document", capture="all", layers=[0, 11, -1])
t = traces[0]
t.hidden_states         # [T, 1024]  backbone output tokens, as the slot extractor reads them
t.layer_states[11]      # [T, 1024]  output of decoder layer 11 (requested layers only)
t.attention_logits      # [4, 16, T] pre-softmax scaled dot products
t.attention             # [4, 16, T] slot attention, equal to encode_documents(..., return_attention=True)
t.slot_query_projection # [4, 16, 128]; t.key_projection, t.value_projection: [4, T, 128]
t.slot_readout          # [16, 512]  attention readout, heads concatenated, before the output projection
t.slot_states_pre_norm  # [16, 1024] after the output projection, before the layer norm
t.slot_states           # [16, 1024] after the layer norm: the input of both heads
t.routing_raw           # [16, 256]  routing head output before L2 normalisation; t.routing is normalised
t.value_linear, t.value_adapter   # [16, 256] the two summands of t.value
t.slot_queries          # [16, 1024] learned slot queries (shared by all texts)
traces.encoding()       # the Embeddings the encoder returns, bit for bit
t.save("trace-dir")     # tensors.safetensors + trace.json
```

`capture` selects any of `"hidden"`, `"logits"`, `"projections"`, `"slot_states"`, `"value_parts"` (default: all but projections; `"all"` for every one); attention, raw and normalised routing and value are always kept. `layers` adds decoder-layer outputs (negative indices count from the end; each costs T × 1024 floats per text).

**dtype and device.** The model runs in its own compute dtype on its own device; temporary forward hooks copy tensors out and are removed when the call returns. Every captured tensor is returned as a float32 CPU copy with padding removed. bfloat16 and float16 values are exactly representable in float32, so the copies are lossless. The logits are recomputed from the captured projections with the extractor's own operations in the model's dtype, so `torch.softmax(t.attention_logits.to(model_dtype), -1)` reproduces `t.attention` exactly. Copies are ordinary tensors (not inference-mode tensors), so they can feed probes trained with autograd. `batch_size` means what it means for the encoders (default 1, each text independent of the others).

`run_heads(model.network, hidden_states, attention_mask)` runs only the slot extractor and both heads on given backbone states; with the backbone's own states and mask it reproduces the forward pass exactly.

## Attention maps

```python
[m] = research.attention_maps(model, ["How many hearts does an octopus have?"], "query")
m.matrix()                    # [16, T] head mean; m.matrix(head=2) one head; m.weights is [4, 16, T]
m.top_tokens(5)               # per slot: index, token, display string, segment, offsets, text, weight
m.entropy(), m.entropy(normalized=True), m.effective_tokens(), m.coverage(0.9), m.head_divergence()
m.segment_mass()              # {"instruction", "template", "text", "special"} -> [16]: what each slot reads
m.slots_reading("instruction", threshold=0.5)
m.word_matrix()               # [16, words] attention mass per word of the caller's text
m.phrase_mass("octopus"), m.span_mass(13, 24), m.character_spans(slot=3)
m.tokens                      # TokenView: segments, offsets into your text and the formatted input, words
```

Token offsets come from the same offset mapping the encoders report, so they refer to your original text, also after truncation and for queries without the instruction prefix. The `TokenView` adds the `"instruction"` segment (tokens that overlap the instruction inside `"{instruction}\nQuery: {query}"`), the offsets of every token in the formatted input, and a word segmentation (runs of Unicode word characters, joined by internal apostrophes or hyphens, and single punctuation marks; a token counts towards the first word it overlaps). `AttentionMap.from_encoding(model, encoding, i, text)` wraps the attention of an existing `encode_*(..., return_attention=True)` result.

The head mean is a summary convention: the model concatenates the four heads and projects them; it does not average them. Rows sum to 1 up to the compute dtype's rounding (about 1e-7 in float32 and up to about 3e-3 in bfloat16).

## Score decomposition

For query slots *i* and document slots *j* with routing cosines *s<sub>ij</sub>*:

- **R.** With *p<sub>ij</sub>* = softmax<sub>j</sub>(*s<sub>ij</sub>* / τ<sub>k</sub>) and *H<sub>i</sub>* the entropy of row *i*, log-mean-exp is exactly τ<sub>k</sub> log mean<sub>j</sub> exp(*s<sub>ij</sub>*/τ<sub>k</sub>) = Σ<sub>j</sub> *p<sub>ij</sub> s<sub>ij</sub>* + τ<sub>k</sub>(*H<sub>i</sub>* − log 16). So R = Σ<sub>ij</sub> *w<sub>ij</sub> s<sub>ij</sub>* + Σ<sub>i</sub> *e<sub>i</sub>*, with pair weights *w<sub>ij</sub>* = *p<sub>ij</sub>*/16 (they sum to 1, and *w<sub>ij</sub>* = ∂R/∂*s<sub>ij</sub>*) and smoothing terms *e<sub>i</sub>* ≤ 0 that belong to query slots, not to pairs.
- **V.** With attention *A<sub>ij</sub>* = softmax<sub>j</sub>(*s<sub>ij</sub>*/τ<sub>v</sub>), readout *r<sub>i</sub>* = Σ<sub>j</sub> *A<sub>ij</sub> v<sub>j</sub>* over raw document values and unit query values *u<sub>i</sub>*: V = Σ<sub>ij</sub> *A<sub>ij</sub>* (*u<sub>i</sub>* · *v<sub>j</sub>*) / (|*r<sub>i</sub>*| · 16).
- **Final** = 0.75 R + 0.25 V = Σ<sub>ij</sub> *c<sub>ij</sub>* + 0.75 Σ<sub>i</sub> *e<sub>i</sub>*.

```python
[[d]] = research.decompose(queries.select([0]), documents.select([0]), base=model)
d.final, d.routing, d.value          # the numbers model.score returns, bit for bit
d.residuals()                        # total minus recombined parts, per component (~1e-7)
d.pair_contribution("final")         # [16, 16] c_ij; also "routing" and "value"
d.routing_weight, d.routing_entropy_term, d.value_attention, d.value_alignment, d.value_readout_norm
d.query_slot_contribution(), d.document_slot_contribution(), d.unattributed()
d.top_pairs(10), d.slot_table()
```

**Invariant (tested).** `final`, `routing` and `value` are taken from one `score` call with the same settings, so they equal `model.score(...)` exactly. The parts are computed in float64 from the same normalised vectors and recombine to them within float32 rounding: residuals of 1e-9 to 7e-8 on the v1.3 weights (float32 and bfloat16), below 2e-6 in every unit test. `rule=` decomposes under another scoring rule (below).

## Token attribution

```python
a = pair.attribution("document", "final")   # or research.attribute(d, attention_map, side=..., component=...)
a.weights                  # [T] per token; a.by_slot is [16, T]
a.top_tokens(10), a.word_weights(), a.segment_totals(), a.span_total(0, 28)
float(a.weights.sum()) + a.unattributed == a.total   # within the decomposition residual
```

Each slot's exact contribution is spread over the tokens that slot was pooled from, in proportion to its slot-extraction attention (each row renormalised to sum to 1 in float64, so bfloat16 rounding does not leak mass). On the query side the smoothing terms are included in the slot contributions; on the document side they are reported as `unattributed`. **This is attention-weighted attribution: a bookkeeping of the score along the model's own pooling weights, not a causal claim.** The backbone has already mixed information across tokens before the slots pool them, so high attribution means "a slot read this position", not "this word caused the score". Use the interventions below for causal questions.

## Interventions

```python
ablation = research.slot_ablation(queries, documents, query=0)       # each slot of either side removed in turn
ablation.effects()                    # mean/max |Δ final|, top-1 change, Kendall tau with the baseline order
research.ablate_slots(queries, documents, query_slots=[3], document_slots=[11], mode="zero")

research.occlude(model, query, document, [(0, 28)])                       # delete characters 0-28, re-encode
research.occlude(model, query, document, [(0, 28)], mode="replace", replacement="...")
research.occlude(model, query, document, [(21, 27)], mode="slot-mask")    # hide the tokens from the slots only
research.occlude(model, query, document, side="query", mode="slot-mask",
                 tokens=pair.query.tokens.segment_mask("instruction").nonzero().flatten().tolist())
sweep = research.occlusion_sweep(model, query, document, unit="word", mode="slot-mask")  # or "token", "sentence"
sweep.top(10), sweep.token_delta()
```

- `drop` removes a slot, so log-mean-exp, the value softmax and the mean over query slots run over the remaining slots; `zero` keeps it with zero routing and value vectors.
- `delete` and `replace` re-encode the edited text: the backbone's context changes too (other tokens' states and, in a causal backbone, positions shift). `slot-mask` keeps the backbone states and hides the span's tokens from the slot extractor only (its attention mask), isolating the pooling step; the backbone runs once for a whole sweep. Neither isolates "the information in the span".

**Scoring rules.** `ScoringRule.preset(name, model.scoring)` with `"model"` (0.75 R + 0.25 V), `"routing-only"`, `"value-only"`, `"maxsim"` (mean over query slots of the best routing cosine: a ColBERT-style MaxSim over the same 16 vectors) and `"maxsim+value"`; `rule.replace(tau_k=0.05)` for anything else. `rescore`, `decompose(rule=...)`, `inspect_pair(rule=...)` and `compare_rules` use the library's own `score` with a fresh configuration; `model.scoring` is never modified. Absolute scores of different rules are on different scales: compare rankings.

## Collection diagnostics

```python
stats = research.slot_statistics(index_or_documents, queries=queries, top_k=10, sample=5000)
stats.per_slot            # resultant length / variance / effective rank of routing across texts, value norms,
                          # and with queries: win rate and score share of each document slot
stats.collapse            # mean and max within-text cosine between slots, the most similar pair
stats.within_text_routing_cosine, stats.centroid_cosine, stats.per_text_effective_rank
stats.usage               # which query slot matches which document slot over each query's top_k documents
```

Within-text cosine near 1 for two slots means they encode the same direction for the same text (collapse); a routing resultant near 1 means a slot is nearly the same for every text; effective rank (exp of the entropy of normalised singular values) counts the directions in use, per text (at most 16) and per slot across the collection. Singular values below 1e-6 are treated as zero.

```python
diag = research.shortlist_diagnostics(index, queries, profile="paper", slot_hits=[16, 32, 64, 128, 256, 512])
diag.rows                 # per width: documents hit, candidates (mean/min/median), underfilled queries,
                          # candidate recall and result recall@k against exact search, top-1 agreement
diag.per_slot_hit_recall  # per width, per query slot: share of the exact top documents that slot hit
```

The per-slot shortlist keeps `slot_hits` document slots per query slot; when many hits fall on the same documents, fewer than `rerank` (100) documents reach value reranking. Such queries are counted as **underfilled**: at the `paper` width of 128, many queries on a large corpus are underfilled, which is why the `default` profile uses 512. The profile's results come from the same code as `Index.search` (tested equal), and the reference is the `exact` profile with the same rerank width. Exact routing scores every document once per query.

## Precision experiments

```python
result = research.compare_precision(model, queries, documents, batch_sizes=(1, 8))
result.rows                # one per (compute dtype, batch size, storage)
```

Every combination of compute dtype (default: float32 plus bfloat16 when the model is float32), encoder batch size and index storage (float32, float16, int8) is encoded and scored, and compared with the float32, batch-1, float32-storage reference: max and mean |Δ| of the final score, max |Δ| of R and V, Kendall tau and Spearman rho of each query's ranking, top-1 agreement, top-k overlap, the share of queries with an identical ranking, and encoder time. A bfloat16 variant of a float32 model is made from the loaded weights by `with_compute_dtype` (equal to `from_pretrained(..., dtype="bfloat16")`, same representation identity; 2 bytes per parameter extra, 1.4 GiB for v1.3); a model loaded in bfloat16 cannot recover float32 weights, so pass other variants with `models={"float32": model_fp32}`.

Measured on 5 queries × 10 short passages (v1.3, CPU):

| variant | max \|Δ final\| | mean \|Δ final\| | Kendall tau | identical rankings |
|---|---|---|---|---|
| float32, batch 1, float16 storage | 1.4e-5 | 3.0e-6 | 1.00 | 5/5 |
| float32, batch 1, int8 storage | 3.8e-4 | 1.0e-4 | 1.00 | 5/5 |
| float32, batch 8, float32 storage | 3.0e-7 | 1.0e-7 | 1.00 | 5/5 |
| bfloat16, batch 1, float32 storage | 4.2e-3 | 1.1e-3 | 0.99 | 4/5 |
| bfloat16, batch 8, float32 storage | 3.9e-3 | 1.0e-3 | 1.00 | 5/5 |

These are a handful of texts; run it on your own data before drawing conclusions.

## Visualisation

- **HTML report** (`pair.report(path)`, `breadbowl-embed inspect pair --html`): one file with inline CSS, JavaScript and SVG; no network access, fonts or CDNs. Score summary; both texts; token heatmaps of the query and the document (attribution of the final score, R or V, or one slot's attention; head mean or one head; optionally through the selected slot only); the 16 × 16 slot-pair matrix (contribution, R or V contribution, routing weight, value attention, cosine; select a cell to select both slots); static tables of query slots, document slots, the largest pair contributions and the most attributed words; and the caveats below. Light and dark themes follow the system and can be toggled. Every control works from the keyboard: slot pickers with arrow keys, the matrix with arrow keys and Enter, token panels with left/right arrows (values are announced in a live region). All user text is escaped in the HTML and embedded as JSON that the script only assigns with `textContent`. Without JavaScript the tables and a static attribution view remain.
- **Notebooks:** every research object renders as a static HTML summary (no scripts, so it also shows in untrusted notebooks): shaded tokens, SVG heatmaps with tooltips, tables. See [`notebooks/research_toolkit.ipynb`](../notebooks/research_toolkit.ipynb).
- **matplotlib** (`pip install "breadbowl-embed[viz]"`): `.plot()` on `AttentionMap`, `ScoreDecomposition` (`kind=`), `TokenAttribution`, `OcclusionSweep`, `SlotStatistics`, `ShortlistDiagnostics`, `PrecisionComparison`, or the functions in `breadbowl_embed.research.plot`; each returns the Axes and accepts `ax=`.
- **JSON:** `to_json()` / `save_json()` on every object.

## Command line

```bash
M=dotproductx/breadbowl-embed-v1.3
breadbowl-embed inspect pair      --model $M --query "..." --document "..." --html pair.html [--output pair.json] [--rule maxsim]
breadbowl-embed inspect attention --model $M --kind query --text "..." [--input file.jsonl] [--per-head] [--logits]
breadbowl-embed inspect trace     --model $M --kind document --text "..." --layers 0,-1 --output trace-dir
breadbowl-embed inspect occlude   --model $M --query "..." --document "..." --unit word --mode slot-mask
breadbowl-embed inspect ablate    --model $M --query "..." --documents docs.jsonl
breadbowl-embed inspect rules     --model $M --queries queries.jsonl --documents docs.jsonl
breadbowl-embed inspect slots     --index my-index [--model $M --queries queries.jsonl] [--sample 5000]
breadbowl-embed inspect shortlist --model $M --index my-index --queries queries.jsonl --slot-hits 16,64,128,512
breadbowl-embed inspect precision --model $M --queries queries.jsonl --documents docs.jsonl --batch-sizes 1,8
```

Each prints one JSON object per result (JSON lines for `attention`), or writes it with `--output`; `--digits N` rounds floats. Model options (`--dtype`, `--device`, `--threads`, `--revision`, `--local-files-only`) are those of the other commands.

## Caveats

- **Attention is not explanation.** Slot attention shows how token states were pooled; the backbone has already mixed information across tokens, and a causal backbone concentrates sentence summaries on later tokens such as sentence-final punctuation. Attribution inherits this. Check claims with occlusion and slot ablation.
- **Precision and batching.** bfloat16 changes final scores by up to about 1e-2 against float32, and encoding several texts per forward pass moves bfloat16 scores by up to about 7e-3 ([validation](validation.md)). Every research tool encodes one text per forward pass unless you pass `batch_size`; compare like with like, and record the compute dtype (every result does).
- **Truncation.** Queries are cut at 128 tokens including the instruction, documents at 384. Truncated parts appear in the texts but have no tokens; occlusion units inside them are skipped.
- **CPU cost** (measured, v1.3, 8 threads on a shared 16-core host, so ±30%):

| | float32 | bfloat16 |
|---|---|---|
| `inspect_pair`, short passage / 384-token passage | 0.6 s / 2.5 s | 0.4 s / 0.8 s |
| `trace` of a 384-token passage, every capture and all 24 layers | 1.1 s | 0.6 s |
| `occlusion_sweep`, 239 words of a 384-token passage, `slot-mask` | 2.6 s | 1.5 s |
| `occlusion_sweep`, 44 words, `delete` (one re-encoding each) | 13.5 s | 9.4 s |
| `occlusion_sweep`, sentences of a 384-token passage, `delete` | 15.6 s | 10.4 s |
| `compare_precision`, 5 × 10 texts, float32 + bfloat16, batch 1 and 8 | 9.2 s | – |
| HTML report, `slot_ablation`, `compare_rules`, `decompose` (few pairs) | < 0.03 s | < 0.03 s |
| `slot_statistics`, 5,000 of 20,000 stored passages (no model pass) | 2.7–4.2 s | same |
| `shortlist_diagnostics`, 20,000 passages, 20 queries, widths 128 and 512 (no model pass) | 4.8–10 s | same |

## Testing

The unit tests (`tests/test_research_*.py`) run on tiny random-weight models without downloads. `tests/test_research_real_model.py` runs the same invariants on real weights when `BREADBOWL_TEST_MODEL` points at a release (optionally `BREADBOWL_TEST_DTYPE=bfloat16`): traces equal the encoders bit for bit (including a truncated passage), attention rows are distributions and softmax(logits) reproduces them exactly, the decomposition recombines to `score` within 1e-6, removing the answer sentence costs far more than removing an unrelated one, and the precision variants stay within their expected ranges.

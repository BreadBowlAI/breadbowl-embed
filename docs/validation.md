# Validation

## v1.3 (step 5152)

**Parity with the research implementation** (`step5152-parity.json`, `step5152-parity-bfloat16.json`). The checkpoint (sha256 `c416e3b2…084c`) was exported with the default settings (text-only backbone, sentinel removed; schema 2) and compared on CPU with the research loader (`qkv.inference_artifacts.load_inference_artifacts`, embedded configuration) using `tools/verify_export.py`: 20 queries and 20 documents, including multilingual text, a single-character document, two custom instructions, a query cut at the 128-token limit and a document cut at the 384-token limit. Both sides encode in batches of 8, and again one text at a time (the toolkit's default).

| quantity (max absolute difference) | float32 | bfloat16 (both sides cast) |
|---|---|---|
| input ids | identical | identical |
| routing (normalised), value, slot attention, raw routing, batches of 8 | 0.0 | 0.0 |
| routing (normalised), value, one text at a time | 0.0 | 0.0 |
| final score / R / V | 1.2e-7 / 1.2e-7 / 8.9e-8 | 1.2e-7 / 1.2e-7 / 6.0e-8 |
| rankings of all 20 documents for all 20 queries | identical | identical |

`explain` identities held to 6e-8 (final = 0.75 R + 0.25 V; mean routing contribution = R; mean value-slot cosine = V; explain = score). `score_routing` equals `score(...).routing` bit for bit; `tokenize` gives the encoders' and the research input ids; display strings join to the decoded input. Search: the `exact` profile matched the research brute-force top 10, and the per-slot shortlist at 128/1024/100 (`paper`), 512/1024/100 (`default`) and a narrow 16/6/4 returned the same documents, in the same order, as the research evaluator's own search code (FAISS flat slot search, `merge_slot_hits`, `coarse_candidates`, routing rescoring, value reranking), with scores within 1.2e-7. On these 20 documents (320 slots) width 128 shortlisted 13.85 documents per query on average and width 512 all 20.

Score differences of ~1e-7 come from L2-normalising routing twice (once when encoding, once in the scorer); the tensors themselves are bit-identical.

A text containing the literal string "<jepa_mask>" is the one deliberate difference: the research tokenizer reads it as the training sentinel (15 tokens), the release as text (18 tokens). It is reported in the parity file and not gated.

**Loading in reduced precision.** Loading the release with `dtype="bfloat16"` and comparing with the float32 reference gives final-score differences up to 8.8e-3 and changes some rankings among the 20 documents; float32 is the reference precision for this release.

**Batch context** (real export, 8 threads; the same 20 queries and 20 documents encoded one at a time and in batches of 8):

| largest change from batching | float32 | bfloat16 |
|---|---|---|
| query routing / value | 4.5e-7 / 2.4e-6 | 6.6e-3 / 6.3e-2 |
| document routing / value | 1.3e-6 / 3.5e-5 | 7.5e-3 / 6.3e-2 |
| final score, queries batched | 1.9e-7 | 3.0e-3 |
| final score, documents batched | 5.8e-7 | 4.2e-3 |
| final score, both batched | 6.3e-7 (rankings unchanged) | 6.6e-3 (rankings change) |
| 8 passages of exactly 384 tokens (no padding), batched vs alone | not bit-identical (8e-7) | not bit-identical (1.6e-2 in value) |
| documents per second, 384 tokens, batch 1 / batch 8 | 1.36 / 0.94 | 2.50 / 1.67 |

The research model shows the same effect on its own (final scores at batch 8 vs alone: 6.2e-7 float32, 6.6e-3 bfloat16). Even equal-length batches, which need no padding, change results, because batched matrix kernels accumulate differently; sorting texts into equal-length buckets would therefore still make results depend on batch composition, and on this CPU it would not help speed, since batch 1 was fastest. So the default `batch_size` is 1: each text's slots are then independent of other texts, and repeated encodings are bit-identical at a fixed thread count. Across thread counts bfloat16 encodings were identical (4, 8, 16 threads) and float32 encodings identical between 8 and 16 threads but differed by up to 2e-6 at 4 threads. The parity reports compare both batch 8 (as before) and batch 1.

**Load memory** (peak resident memory of a process that loads the release and encodes one query and one document; torch 2.14.1):

| | before | now |
|---|---|---|
| float32 (native) | 3.54 GiB peak, 2.92 GiB after | 2.57 GiB peak, mostly the memory-mapped weights file |
| bfloat16 | 4.96 GiB peak, 2.17 GiB after | 2.15 GiB peak = steady (1.99 GiB on torch 2.11 / transformers 5.4) |

The network is built without initialising weights that are replaced at once; float32 weights are memory-mapped (shared, reclaimable page cache); a cast reads and converts one tensor at a time in 64 MiB chunks, then casts floating buffers as `Module.to` does, so the cast model is bit-identical to loading float32 and calling `.to(dtype)`.

**Threads.** PyTorch's default is one thread per physical core (16 here). With other processes on the machine (load average 4–11), 8 threads answered single queries faster than 16 (below), and a worker sharing the host settled on half the physical cores. Several encoder processes on one host should split the cores rather than each take the default.

**CPU timings** (`tools/benchmark_cpu.py`, this version; AMD Ryzen 9 7950X3D, 16 cores / 32 threads; other processes raised the load average from 4 to 13 during the runs, so these are not quiet-machine numbers; torch 2.14.1):

| | float32, 8 threads | float32, 16 threads (default) | bfloat16, 8 threads | bfloat16, 16 threads |
|---|---|---|---|---|
| load (`from_pretrained`, checksums included) | 3.8 s | | 3.4 s | |
| one query per call, p50 (p90), tokenisation included | 141 ms (150) | 157 ms (174) | 70 ms (71) | 76 ms (78) |
| documents per second, 384 tokens, batch 1 (default) | 1.38 | 1.83 | 2.67 | 2.78 |
| documents per second, 384 tokens, batch 8 | 0.92 | 1.10 | 1.58 | 1.91 |

bfloat16 is about twice as fast on this CPU (it has AVX-512 BF16) but changes scores by up to 8.8e-3 against float32 and, batched, depends on batch composition (above).

Search over 100,000 synthetic documents (float32 storage, 16 threads, one query per call, top 10): `default` p50 177 ms, `paper` 120 ms, `exact` 409 ms. The first slot-shortlist search builds a float32 unit-routing cache (960 ms here; 1.6 GB for 100k documents). Search cost does not depend on the vector values, but shortlist sizes do, so these are cost measurements, not accuracy ones.

Encoding a batch of 8 × 384 tokens took 7.0 s, of which 4.5 s were the 18 linear-attention (Gated DeltaNet) layers running Transformers' PyTorch reference implementation, 2.0 s the MLPs and 0.4 s the 6 full-attention layers. The accelerated kernels Transformers suggests are GPU kernels.

**Dependency stacks.** The suite passes on both stacks below (release gates skipped without files; FAISS tests skipped on the first, which has no FAISS); the wheel was installed non-editable into a separate environment for the first. The v1.3 release loads on both with the same representation identity; encodings differ by at most 2.1e-5 (values) and 4.1e-7 (final scores), with identical rankings.

| | lower bound | upper bound |
|---|---|---|
| Python | 3.12 | 3.12 |
| torch (CPU) | 2.11.0 | 2.14.1 |
| transformers | 5.4.0 | 5.17.0 |
| huggingface-hub | 1.8.0 | 1.33.0 |
| safetensors | 0.7.0 | 0.8.0 |
| numpy | 2.5.4 | 1.26.4 |

Intermediate versions and mixed combinations (e.g. torch 2.11 with transformers 5.17) were not run.

**Release format.** The schema-2 re-export of step 5152 differs from the first (schema-1) export only in `config.json` (and so the manifest): `model.safetensors`, the backbone config and the tokenizer files are byte-identical, and the representation identity is unchanged (`b1f90b86…`).

**Unit tests** (no downloads, tiny models): scoring equations and gradients, explain identities, `score_routing` bit-identity, tokenisation (byte-level display strings with split multi-byte characters, offsets, segments, truncation counts, equality with the encoders), batch-1 independence, the Safetensors reader and cast-while-loading, encoder attention/tokens/offsets/instructions, export of a tiny multimodal Qwen3.5 research-style checkpoint (text-only and sentinel-free export reproduces the research network bit for bit; the v1.2 layout still exports and loads; schema 2 release metadata keeps the identity; no training paths or optimizer state), index storage round trips (float32/float16/int8, mmap), quantisation error bounds, identity refusal, v0.1 index directories, incremental `add`, per-slot shortlist against its definition and against brute force, FAISS against the PyTorch backend, CLI.

**Research toolkit** (`breadbowl_embed.research`, 0.2.0; v1.3 release on CPU, float32 and bfloat16, `tests/test_research_real_model.py`): `trace` reproduced the encoders bit for bit (routing, value, attention, token offsets and truncation, including a passage cut at 384 tokens); softmax of the recomputed pre-softmax logits equalled the returned attention bit for bit; attention rows summed to 1 within 1.2e-7 (float32) and 2.7e-3 (bfloat16); the slot-pair decomposition took its totals from `score` (identical) and its parts recombined to them within 6.5e-8; token attribution summed to the score within the same residual; deleting the sentence that answers the query lowered the final score by 0.44, deleting other sentences changed it by at most 0.03. Unit tests on tiny random models cover the same identities plus slot ablation against manual rescoring, scoring-rule presets against their definitions, shortlist diagnostics against `Index.search`, `with_compute_dtype` against loading in that dtype, report escaping and self-containment, JSON export and the `inspect` CLI. CPU timings are in [research.md](research.md#caveats).

## v1.2-preview (step 4000)

The public Hub revision `d55da971` loads with this version and has the recorded representation identifier. On each of the two stacks above, v0.1.0a1 and this version give identical scores for the README example. Absolute scores differ from `hub-verification.json` by up to 2.3e-3 (final) and 6.9e-3 (value) because the v1.2 backbone runs in bfloat16 and CPU bfloat16 kernels differ between machines and library versions; rankings match.

Earlier record: the step-4000 checkpoint was exported locally and compared on CPU against the original research implementation using two queries and four documents (`step4000-parity.json`); public Hub download and loading were verified from a fresh cache (`hub-verification.json`), and Apple Silicon MPS passed a small inference test with PyTorch 2.11.

## Scope

These are numerical fidelity and functionality checks, not a benchmark rerun: they do not establish paper-level accuracy, GPU behaviour or memory use at scale.

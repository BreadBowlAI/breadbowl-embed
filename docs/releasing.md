# Preparing a model release

Keep the code repository separate from the Hugging Face weight repository. Release only an inference artifact, never the research checkpoint directory or training configuration wholesale.

From an installed checkout, convert a trusted full training checkpoint:

```bash
breadbowl-embed export --checkpoint /path/to/checkpoint_step_5152/model.pt --output /path/to/new-release \
  --base-model-path /path/to/base-model --local-files-only --expected-sha256 <sha256 of model.pt>
```

(`breadbowl-export` takes the same arguments.) The exporter reads `model.pt` memory-mapped and uses the model, data and training configuration embedded in it (not an external YAML). It reads only the base model's config and tokenizer, never its weights, and writes Safetensors with the checkpoint's parameter dtypes (float32 for step 5152). LoRA checkpoints must first be merged and validated; this exporter rejects them.

By default two training-only parts are removed, and both removals are recorded under `provenance.removed` in `config.json`:

- the vision tower of the multimodal Qwen3.5 backbone (100,592,896 parameters for Qwen3.5-0.8B). For text input `Qwen3_5Model` runs exactly `Qwen3_5TextModel` over `language_model`, so the release keeps only that text model;
- the `<jepa_mask>` sentinel: its tokenizer entry, its embedding row (the last row, after the base vocabulary) and its learned offset. No other token id or embedding changes. The loader no longer adds the token (`inference_setup.stage` is `inference`).

`--keep-vision-tower --keep-sentinel` reproduces the v1.2-preview layout.

A runnable model repository contains:

```text
config.json            # schema 2: architecture, preprocessing, score settings, release (architecture name,
                       # licence, weights sha256, Hugging Face repository/revision), provenance (step,
                       # checkpoint sha256, base model id/revision, removed parts, exporter versions)
manifest.json          # file checksums and representation identity
model.safetensors      # full inference parameters
backbone/config.json   # configuration only
tokenizer/             # tokenizer assets
```

Keep the model card, license, paper, and validation report alongside those assets. The raw optimizer state, dataset locations, training credentials, and original checkpoint are not release files.

## Parity gate

Before uploading, compare the release with the research implementation. This needs the research checkout and environment; the toolkit itself never imports it:

```bash
python tools/verify_export.py all \
  --research-src /path/to/research/src --checkpoint /path/to/model.pt \
  --base-model /path/to/base-model --release /path/to/new-release \
  --dtype float32 --output /path/to/parity.json
```

The reference phase loads the checkpoint with `qkv.inference_artifacts.load_inference_artifacts` (embedded configs), encodes 20 queries and 20 documents (varied lengths and scripts, one query and one document over their token limits, custom instructions) in batches of 8, and scores them with the research scorer. It also runs the research evaluator's search (FAISS flat slot search, `merge_slot_hits`, `coarse_candidates`, routing rescoring, value reranking) at 128/1024/100, 512/1024/100 and a narrow 16/6/4. It also encodes every text on its own (batch 1). The compare phase loads the release through `breadbowl_embed`, encodes the same texts with the same batching and requires identical input ids, routing/value/attention and final/R/V scores within 1e-5, the same within 1e-5 for the toolkit's default (one text per forward pass) against the research model's batch-1 encodings, identical rankings, `explain` identities, `score_routing` bit-identical to `score(...).routing`, `tokenize` equal to the encoders' and the research input ids, and search results identical to the research pipeline for every profile width. The same gate runs from pytest when its files are supplied (`tests/test_release_parity.py`).

`tools/benchmark_cpu.py` records CPU load time, single-query latency, document throughput and search latency.

## Upload

Record where the release will be published (schema-2 releases; this updates `config.json` and the manifest checksums, never the representation identity):

```bash
breadbowl-embed release /path/to/new-release --repository <namespace>/<repository> --revision <tag>
```

The revision is a name (a tag or branch): a file cannot hold the hash of the commit that adds it. Loaders report the resolved commit as `info()["release"]["huggingface"]["loaded_revision"]`. The licence defaults to Apache-2.0 at export (`--license` changes it).

With Hugging Face CLI authentication configured, upload the reviewed inference directory:

```bash
hf upload <namespace>/<repository> /path/to/new-release .
```

After uploading, verify loading by immutable Hub revision from a clean cache, then update the model card and README to announce weight availability. Until that check passes, keep the current preview wording. Do not claim `pip install breadbowl-embed` works until the package has actually been published on PyPI; installation from this Git checkout works now.

The manifest uses SHA-256 to detect accidental modification. It does not establish publisher authenticity. Pin a trusted Hub commit for reproducible experiments.

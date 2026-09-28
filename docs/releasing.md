# Preparing a model release

Keep the code repository separate from the Hugging Face weight repository. Release only an inference artifact, never the research checkpoint directory or training configuration wholesale.

From an installed checkout, convert a trusted full training checkpoint:

```bash
breadbowl-export --checkpoint /path/to/checkpoint_step_4000 --output /path/to/new-release
```

The exporter reads `model.pt`, resolves the pinned backbone config and tokenizer, preserves mixed parameter dtypes, and writes Safetensors. It does not need to download original backbone weights. Use `--base-model-path` and `--local-files-only` with an existing config/tokenizer directory for offline export. LoRA checkpoints must first be merged and validated; this exporter rejects them.

A runnable model repository contains:

```text
config.json            # architecture, preprocessing, score settings, source checksum
manifest.json          # file checksums and representation identity
model.safetensors      # full inference parameters
backbone/config.json   # configuration only
tokenizer/             # tokenizer assets
```

Keep the model card, license, paper, and validation report alongside those assets. The raw optimizer state, dataset locations, training credentials, and original checkpoint are not release files. The exporter includes model parameters required for strict checkpoint compatibility, including any learned sentinel token.

Before uploading, compare against the research implementation:

```bash
python tools/verify_export.py \
  --checkpoint /path/to/checkpoint_step_4000/model.pt \
  --release /path/to/new-release \
  --reference-source /path/to/research/src \
  --output /path/to/parity.json
```

Then run the notebook with `BREADBOWL_MODEL` pointing at the release. This smoke test checks numerical fidelity and API usability; it does not reproduce the paper's benchmark suite.

With Hugging Face CLI authentication configured, upload the reviewed inference directory:

```bash
hf upload dotproductx/breadbowl-embed-v1.2-preview /path/to/new-release .
```

After uploading, verify loading by immutable Hub revision from a clean cache, then update the model card and README to announce weight availability. Until that check passes, keep the current preview wording. Do not claim `pip install breadbowl-embed` works until the package has actually been published on PyPI; installation from this Git checkout works now.

The manifest uses SHA-256 to detect accidental modification. It does not establish publisher authenticity. Pin a trusted Hub commit for reproducible experiments.

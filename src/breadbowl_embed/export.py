"""Convert a trusted research checkpoint to the public inference format.

Two parts of a research checkpoint are not needed to encode text and are removed by default:

* The vision tower. Qwen3.5 checkpoints carry the multimodal Qwen3_5Model; for text-only inputs
  its forward pass is exactly Qwen3_5TextModel over `language_model` (no pixel inputs, so no
  M-RoPE position offsets). The export keeps only the text model (`text_only=True`).
* The `<jepa_mask>` training sentinel. JEPA/candidate-distill training adds it to the tokenizer,
  appends one embedding row and learns an offset for it. Normal text never produces the token, so
  removing it leaves every other token id and embedding unchanged; it also stops literal
  "<jepa_mask>" text in user input from being read as the training mask (`keep_sentinel=False`).

`text_only=False, keep_sentinel=True` reproduces the v1.2-preview layout.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import tempfile

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from ._version import __version__
from .architecture import QKVRetriever, resolve_jepa_sentinel_token
from .config import (DEFAULT_LICENSE, FORMAT, SCHEMA_VERSION, ScoringConfig, fingerprint, release_section,
                     sha256_file)
from .model import BreadBowl

_TEXT_PREFIX = "backbone.language_model."


def _resolve_source(backbone: dict, base_model_path: str | None, local_files_only: bool) -> str:
    source = base_model_path or backbone.get("source_repo") or os.path.expandvars(backbone["pretrained"])
    if "$" in source:
        raise ValueError("Specify --base-model-path; checkpoint refers to an unavailable training-machine path.")
    if not Path(source).is_dir():
        source = snapshot_download(source, revision=backbone.get("revision"), local_files_only=local_files_only,
                                   allow_patterns=["config.json", "tokenizer*", "special_tokens_map.json",
                                                   "added_tokens.json", "vocab.json", "merges.txt", "*.jinja"])
    return source


def from_checkpoint(checkpoint_path: str | Path, *, base_model_path: str | None = None,
                    local_files_only: bool = False, text_only: bool = True, keep_sentinel: bool = False,
                    expected_sha256: str | None = None, license: str | None = DEFAULT_LICENSE,
                    repository: str | None = None, revision: str | None = None) -> BreadBowl:
    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "model.pt"
    checkpoint_sha256 = sha256_file(path)
    if expected_sha256 is not None and checkpoint_sha256 != expected_sha256.lower():
        raise ValueError(f"Checkpoint sha256 {checkpoint_sha256} differs from the expected {expected_sha256}.")
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    architecture = deepcopy(checkpoint["model_config"])
    train = checkpoint["train_config"]
    if train.get("use_lora", False):
        raise ValueError("Merge and validate LoRA weights before public export; this exporter handles full checkpoints.")
    loss = train["loss"]
    if "value_mix" in loss:
        mixture = loss["value_mix"]
    else:
        alpha, beta = loss["alpha_qk"], loss["beta_qkv"]
        if abs(alpha + beta - 1) > 1e-6:
            raise ValueError("Public export requires a normalized score mixture.")
        mixture = beta
    scoring = ScoringConfig(qk_mode=loss["qk_mode"], tau_k=loss["tau_k"], tau_v=loss["tau_v"], value_mix=mixture)
    fields = ("query_format", "document_format", "default_instruction", "max_query_length", "max_document_length")
    preprocessing = {key: checkpoint["data_config"][key] for key in fields}
    backbone = architecture["backbone"]
    source = _resolve_source(backbone, base_model_path, local_files_only)
    trust = backbone.get("trust_remote_code", False)
    backbone_config = AutoConfig.from_pretrained(source, trust_remote_code=trust, local_files_only=local_files_only)
    base_tokens = len(AutoTokenizer.from_pretrained(source, trust_remote_code=trust, local_files_only=local_files_only))
    state = dict(checkpoint["model"])
    removed = {}

    if text_only and getattr(backbone_config, "text_config", None) is not None and any(
            key.startswith(_TEXT_PREFIX) for key in state):
        other = [key for key in state if key.startswith("backbone.") and not key.startswith(_TEXT_PREFIX)]
        removed["non_text_backbone"] = {"modules": sorted({key.split(".")[1] for key in other}),
                                        "parameters": int(sum(state[key].numel() for key in other))}
        state = {("backbone." + key[len(_TEXT_PREFIX):] if key.startswith(_TEXT_PREFIX) else key): value
                 for key, value in state.items() if key not in other}
        backbone_config = deepcopy(backbone_config.text_config)
    text_config = getattr(backbone_config, "text_config", None) or backbone_config

    sentinel = resolve_jepa_sentinel_token(train)
    setup = {"stage": train.get("stage", "weak"), "use_lora": False}
    if "jepa" in train:
        setup["jepa"] = {k: v for k, v in train["jepa"].items() if k in {"mask_token_strategy", "mask_token_text"}}
    if sentinel is not None and not keep_sentinel:
        embedding_keys = [key for key in state if key.startswith("backbone.") and key.endswith("embed_tokens.weight")]
        if len(embedding_keys) != 1:
            raise ValueError("Expected exactly one backbone token embedding.")
        rows = state[embedding_keys[0]].shape[0]
        if rows != base_tokens + 1:
            raise ValueError(f"Expected the sentinel to be the only added embedding row ({base_tokens} + 1), found {rows}.")
        state[embedding_keys[0]] = state[embedding_keys[0]][:base_tokens]
        delta = state.pop("jepa_mask_embedding_delta", None)
        text_config.vocab_size = base_tokens
        removed["jepa_sentinel"] = {"token": sentinel, "token_id": base_tokens, "embedding_rows": 1,
                                    "learned_offset": delta is not None}
        # A stage without a sentinel: the loader must not add the token back.
        setup = {"stage": "inference", "use_lora": False}

    with tempfile.TemporaryDirectory() as temporary:
        backbone_config.save_pretrained(temporary)
        network, tokenizer = QKVRetriever.from_config(
            architecture, train_config=setup, backbone_init="config", backbone_source_override=temporary,
            tokenizer_source_override=source, local_files_only_override=True, enable_gradient_checkpointing=False,
        )
    network.load_state_dict(state, strict=True, assign=True)
    backbone["pretrained"] = "backbone"
    backbone["local_files_only"] = True
    provenance = {"step": checkpoint.get("step"), "checkpoint_sha256": checkpoint_sha256,
                  "base_model": {"repo": backbone.get("source_repo"), "revision": backbone.get("revision")},
                  "backbone_class": type(network.backbone).__name__,
                  "parameter_dtypes": sorted({str(p.dtype).removeprefix("torch.") for p in network.parameters()}),
                  "removed": removed,
                  "exported_with": {"breadbowl_embed": __version__, "torch": torch.__version__.split("+")[0],
                                    "transformers": transformers.__version__}}
    config = {"format": FORMAT, "schema_version": SCHEMA_VERSION, "architecture": architecture,
              "preprocessing": preprocessing, "scoring": scoring.to_dict(), "inference_setup": setup,
              "routing_normalized": True, "value_normalized": False,
              # weights_sha256 is filled in by save_pretrained once the weights are written.
              "release": release_section(license=license, repository=repository, revision=revision),
              "provenance": provenance}
    return BreadBowl(network, tokenizer, config, representation_id=fingerprint(config))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint directory or model.pt")
    parser.add_argument("--output", required=True, help="New directory for the inference-only release")
    parser.add_argument("--base-model-path", help="Local base-model directory (config and tokenizer are read)")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--expected-sha256", help="Refuse a checkpoint with a different sha256")
    parser.add_argument("--keep-vision-tower", action="store_true",
                        help="Keep the full multimodal backbone (v1.2-preview layout)")
    parser.add_argument("--keep-sentinel", action="store_true",
                        help="Keep the <jepa_mask> training token, its embedding row and offset")
    add_release_arguments(parser, license_default=DEFAULT_LICENSE)


def add_release_arguments(parser: argparse.ArgumentParser, *, license_default: str | None) -> None:
    parser.add_argument("--repository", help="Hugging Face repository id the release will be published to")
    parser.add_argument("--revision", help="Hugging Face revision name (tag or branch) it will be published under; "
                                           "a commit hash cannot be known before the upload")
    parser.add_argument("--license", default=license_default, help=f"Licence (default: {license_default})")


def run(args) -> None:
    model = from_checkpoint(args.checkpoint, base_model_path=args.base_model_path,
                            local_files_only=args.local_files_only, text_only=not args.keep_vision_tower,
                            keep_sentinel=args.keep_sentinel, expected_sha256=args.expected_sha256,
                            license=args.license, repository=args.repository, revision=args.revision)
    model.save_pretrained(args.output)
    print(f"Exported inference artifact to {args.output}. Verify numerical parity before publishing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

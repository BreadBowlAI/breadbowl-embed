"""Convert a trusted research checkpoint to the public inference format."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import torch
from huggingface_hub import snapshot_download

from .architecture import QKVRetriever
from .config import FORMAT, ScoringConfig, fingerprint, sha256_file
from .model import BreadBowl


def from_checkpoint(checkpoint_path: str | Path, *, base_model_path: str | None = None,
                    local_files_only: bool = False) -> BreadBowl:
    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "model.pt"
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
    setup = {"stage": train.get("stage", "weak"), "use_lora": False}
    if "jepa" in train:
        setup["jepa"] = {k: v for k, v in train["jepa"].items() if k in {"mask_token_strategy", "mask_token_text"}}
    backbone = architecture["backbone"]
    source = base_model_path or backbone.get("source_repo") or os.path.expandvars(backbone["pretrained"])
    if "$" in source:
        raise ValueError("Specify --base-model-path; checkpoint refers to an unavailable training-machine path.")
    if not Path(source).is_dir():
        source = snapshot_download(source, revision=backbone.get("revision"), local_files_only=local_files_only,
                                   allow_patterns=["config.json", "tokenizer*", "special_tokens_map.json",
                                                   "added_tokens.json", "vocab.json", "merges.txt", "*.jinja"])
    # Config/tokenizer only: do not download the original untrained backbone weights.
    network, tokenizer = QKVRetriever.from_config(
        architecture, train_config=setup, backbone_init="config", backbone_source_override=source,
        tokenizer_source_override=source, local_files_only_override=local_files_only,
        enable_gradient_checkpointing=False,
    )
    network.load_state_dict(checkpoint["model"], strict=True, assign=True)
    backbone["pretrained"] = "backbone"
    backbone["local_files_only"] = True
    config = {"format": FORMAT, "schema_version": 1, "architecture": architecture,
              "preprocessing": preprocessing, "scoring": scoring.to_dict(), "inference_setup": setup,
              "routing_normalized": True, "value_normalized": False,
              "provenance": {"step": checkpoint.get("step"), "checkpoint_sha256": sha256_file(path)}}
    return BreadBowl(network, tokenizer, config, representation_id=fingerprint(config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New directory for the inference-only release")
    parser.add_argument("--base-model-path")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    model = from_checkpoint(args.checkpoint, base_model_path=args.base_model_path,
                            local_files_only=args.local_files_only)
    model.save_pretrained(args.output)
    print(f"Exported inference artifact to {args.output}. Verify numerical parity before publishing.")


if __name__ == "__main__":
    main()

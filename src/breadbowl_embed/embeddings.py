"""Named routing/value tensors with explicit representation identity."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file, save_file

from .config import canonical_json, sha256_file


@dataclass(frozen=True)
class Embeddings:
    routing: torch.Tensor
    value: torch.Tensor
    representation_id: str
    kind: Literal["query", "document"]

    def __post_init__(self):
        if self.kind not in {"query", "document"} or not self.representation_id:
            raise ValueError("Embeddings need a representation ID and query/document kind.")
        if self.routing.ndim != 3 or self.value.ndim != 3:
            raise ValueError("Tensors must have shape [items, slots, dimensions].")
        if self.routing.shape[:2] != self.value.shape[:2]:
            raise ValueError("Routing and value item/slot counts differ.")
        if min(*self.routing.shape[1:], self.value.shape[-1]) <= 0:
            raise ValueError("Slot counts and dimensions must be positive.")
        for tensor in (self.routing, self.value):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError("Embeddings must contain finite floating-point tensors.")

    def __len__(self):
        return self.routing.shape[0]

    def select(self, indices) -> "Embeddings":
        if isinstance(indices, int):
            indices = [indices]
        return Embeddings(self.routing[indices], self.value[indices], self.representation_id, self.kind)

    def to(self, device) -> "Embeddings":
        return Embeddings(self.routing.to(device), self.value.to(device), self.representation_id, self.kind)

    def save(self, directory: str | Path) -> None:
        """Create a new directory. Refuse overwrite; manifest is written last."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        tensors = root / "embeddings.safetensors"
        save_file({"routing": self.routing.detach().cpu().contiguous().clone(),
                   "value": self.value.detach().cpu().contiguous().clone()}, str(tensors))
        manifest = {"format": "breadbowl_slots_v1", "representation_id": self.representation_id,
                    "kind": self.kind, "sha256": sha256_file(tensors),
                    "routing_shape": list(self.routing.shape), "value_shape": list(self.value.shape)}
        (root / "manifest.json.tmp").write_text(canonical_json(manifest) + "\n")
        os.replace(root / "manifest.json.tmp", root / "manifest.json")

    @classmethod
    def load(cls, directory: str | Path) -> "Embeddings":
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("format") != "breadbowl_slots_v1":
            raise ValueError("Unsupported slot artifact format.")
        path = root / "embeddings.safetensors"
        if sha256_file(path) != manifest["sha256"]:
            raise ValueError("Slot artifact checksum mismatch.")
        tensors = load_file(str(path))
        if set(tensors) != {"routing", "value"}:
            raise ValueError("Unexpected slot tensors.")
        for name in tensors:
            if list(tensors[name].shape) != manifest[name + "_shape"]:
                raise ValueError("Slot shape differs from manifest.")
        return cls(tensors["routing"], tensors["value"], manifest["representation_id"], manifest["kind"])

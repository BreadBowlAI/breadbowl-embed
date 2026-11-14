"""Named routing/value tensors with explicit representation identity."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file, save_file

from .config import canonical_json, sha256_file

# Optional per-item fields, filled only when an encoder is asked for them.
TOKEN_FIELDS = ("attention", "input_ids", "tokens", "offsets", "truncated", "token_texts")


def _positions(indices, count: int) -> list[int]:
    if isinstance(indices, int):
        indices = [indices]
    if isinstance(indices, slice):
        return list(range(count))[indices]
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    positions = [int(i) for i in indices]
    return [i + count if i < 0 else i for i in positions]


@dataclass(frozen=True)
class Embeddings:
    """Routing and value slots for N texts, plus optional token-level detail.

    routing:   [N, slots, d_k] L2-normalised routing vectors.
    value:     [N, slots, d_v] raw value vectors (magnitudes are part of the model).
    attention: per text, [heads, slots, tokens] slot-extraction attention over that text's
               tokens (padding removed); only when encoded with return_attention=True.
    input_ids, tokens, token_texts, offsets, truncated: per text, only with return_tokens=True.
               `tokens` are vocabulary strings; `token_texts` are display strings (see
               TokenizedText); `offsets` are (start, end) character offsets into the original text
               passed by the caller, or None for tokens that come only from the query
               instruction/template; `truncated` says whether the token limit cut the text.
    """
    routing: torch.Tensor
    value: torch.Tensor
    representation_id: str
    kind: Literal["query", "document"]
    attention: tuple | None = None
    input_ids: tuple | None = None
    tokens: tuple | None = None
    offsets: tuple | None = None
    truncated: tuple | None = None
    token_texts: tuple | None = None

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
        for name in TOKEN_FIELDS:
            items = getattr(self, name)
            if items is None:
                continue
            if not isinstance(items, tuple):
                object.__setattr__(self, name, tuple(items))
            if len(getattr(self, name)) != len(self):
                raise ValueError(f"`{name}` must have one entry per item.")
        if self.attention is not None:
            for item in self.attention:
                if item.ndim != 3 or item.shape[1] != self.routing.shape[1]:
                    raise ValueError("Attention entries must be [heads, slots, tokens].")
            if self.input_ids is not None and any(a.shape[-1] != len(i) for a, i in
                                                  zip(self.attention, self.input_ids)):
                raise ValueError("Attention and token counts differ.")

    def __len__(self):
        return self.routing.shape[0]

    @property
    def num_slots(self) -> int:
        return self.routing.shape[1]

    def _extras(self, positions: list[int] | None = None) -> dict:
        out = {}
        for name in TOKEN_FIELDS:
            items = getattr(self, name)
            out[name] = None if items is None else (
                items if positions is None else tuple(items[i] for i in positions))
        return out

    def select(self, indices) -> "Embeddings":
        positions = _positions(indices, len(self))
        index = torch.tensor(positions, dtype=torch.long)
        return Embeddings(self.routing[index], self.value[index], self.representation_id, self.kind,
                          **self._extras(positions))

    def to(self, device) -> "Embeddings":
        return Embeddings(self.routing.to(device), self.value.to(device), self.representation_id,
                          self.kind, **self._extras())

    def without_tokens(self) -> "Embeddings":
        """Only routing and value: what an index or a saved artifact keeps."""
        return Embeddings(self.routing, self.value, self.representation_id, self.kind)

    @staticmethod
    def concat(items: list["Embeddings"]) -> "Embeddings":
        if not items:
            raise ValueError("Nothing to concatenate.")
        first = items[0]
        if any(e.representation_id != first.representation_id or e.kind != first.kind for e in items):
            raise ValueError("Cannot concatenate embeddings from different models or kinds.")
        extras = {}
        for name in TOKEN_FIELDS:
            parts = [getattr(e, name) for e in items]
            extras[name] = None if any(p is None for p in parts) else tuple(x for p in parts for x in p)
        return Embeddings(torch.cat([e.routing for e in items]), torch.cat([e.value for e in items]),
                          first.representation_id, first.kind, **extras)

    def save(self, directory: str | Path) -> None:
        """Create a new directory. Refuse overwrite; manifest is written last.

        Only routing and value are saved; token-level fields are not persisted.
        """
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


# The name used in the v1.3 documentation: one typed result with named fields.
Encoding = Embeddings

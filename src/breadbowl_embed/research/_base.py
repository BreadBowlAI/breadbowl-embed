"""Shared behaviour of research result objects: JSON export and notebook display."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from .._version import __version__

# Version of the JSON layout written by `to_dict`. It changes when a field changes meaning.
JSON_SCHEMA = 1


def to_jsonable(value, digits: int | None = None):
    """Plain JSON types from tensors, arrays, dataclasses' payloads and research objects.

    Non-finite floats become None (JSON has no NaN or infinity). `digits` rounds floats, which
    makes saved results easier to diff; None keeps every float as computed.
    """
    if isinstance(value, ResearchObject):
        return value.to_dict(digits=digits)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype in (torch.bfloat16, torch.float16):
            value = value.float()
        return to_jsonable(value.tolist(), digits)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist(), digits)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return round(value, digits) if digits is not None else value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v, digits) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


class ResearchObject:
    """Base of every result object in `breadbowl_embed.research`.

    Subclasses implement `_payload()` (their fields as plain data) and `_html()` (a static HTML
    summary without scripts, for notebooks). Everything else is shared: `to_dict`, `to_json`,
    `save_json` and `_repr_html_`.
    """
    json_type: ClassVar[str] = "object"

    def _payload(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def _html(self) -> str:
        from ._html import generic_html
        return generic_html(self)

    def to_dict(self, *, digits: int | None = None) -> dict:
        """Plain-JSON form: {"type", "schema", "library_version", ...fields}."""
        return {"type": self.json_type, "schema": JSON_SCHEMA, "library_version": __version__,
                **to_jsonable(self._payload(), digits)}

    def to_json(self, *, digits: int | None = None, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(digits=digits), ensure_ascii=False, indent=indent, allow_nan=False)

    def save_json(self, path: str | Path, *, digits: int | None = None, indent: int | None = 1) -> Path:
        """Write `to_json()` to `path` (UTF-8) and return the path."""
        path = Path(path)
        path.write_text(self.to_json(digits=digits, indent=indent) + "\n", encoding="utf-8")
        return path

    def _repr_html_(self) -> str:
        return self._html()


def as_float(value) -> float:
    return float(value.item() if isinstance(value, torch.Tensor) else value)

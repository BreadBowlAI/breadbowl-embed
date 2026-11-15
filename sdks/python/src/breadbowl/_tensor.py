"""The contract's tensor object, decoded without numpy (and with it, when installed)."""

from __future__ import annotations

import base64
import binascii
import math
import struct
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ._errors import TensorDecodeError

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

__all__ = ["Tensor", "NestedFloats"]

NestedFloats = Union[float, List[Any]]

_ITEMSIZE = {"float16": 2, "float32": 4}
_STRUCT = {"float16": "e", "float32": "f"}
_NUMPY_DTYPE = {"float16": "<f2", "float32": "<f4"}
_AXES = ("slot", "dim", "head", "token", "query_slot", "document_slot")


def _product(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _round_to(dtype: str, value: float) -> float:
    """Round a Python float to the nearest ``dtype`` value (what the server's bytes would hold)."""
    fmt = "<" + _STRUCT[dtype]
    try:
        return float(struct.unpack(fmt, struct.pack(fmt, value))[0])
    except (OverflowError, struct.error) as exc:
        raise TensorDecodeError(f"value {value!r} is not representable as {dtype}") from exc


def _nest(flat: Sequence[float], shape: Sequence[int]) -> NestedFloats:
    if len(shape) == 0:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    step = _product(shape[1:])
    return [_nest(flat[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


def _flatten_checked(data: Any, shape: Sequence[int], path: str = "data") -> List[float]:
    if len(shape) == 0:
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise TensorDecodeError(f"{path} must be a number, got {type(data).__name__}")
        if not math.isfinite(float(data)):
            raise TensorDecodeError(f"{path} is not finite")
        return [float(data)]
    if not isinstance(data, list):
        raise TensorDecodeError(f"{path} must be a list of length {shape[0]}, got {type(data).__name__}")
    if len(data) != shape[0]:
        raise TensorDecodeError(f"{path} has length {len(data)}, shape says {shape[0]}")
    out: List[float] = []
    for i, item in enumerate(data):
        out.extend(_flatten_checked(item, shape[1:], f"{path}[{i}]"))
    return out


class Tensor:
    """A dense row-major tensor as the API sends it: ``{encoding, dtype, shape, axes, data}``.

    Decoding is lazy and cached. ``to_list()`` always works (float16 is decoded with
    ``struct``'s ``'e'`` format); ``to_numpy()`` needs ``pip install "breadbowl[numpy]"``.
    Values from ``encoding: float`` are rounded to ``dtype`` so both encodings decode to the
    same numbers.
    """

    __slots__ = ("encoding", "dtype", "shape", "axes", "byte_order", "data", "_flat")

    def __init__(
        self,
        *,
        encoding: str,
        dtype: str,
        shape: Sequence[int],
        axes: Sequence[str],
        data: Any,
        byte_order: Optional[str] = None,
    ) -> None:
        self.encoding = encoding
        self.dtype = dtype
        self.shape: Tuple[int, ...] = tuple(shape)
        self.axes: Tuple[str, ...] = tuple(axes)
        self.byte_order = byte_order
        self.data = data
        self._flat: Optional[List[float]] = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Tensor:
        if not isinstance(value, Mapping):
            raise TensorDecodeError(f"tensor must be an object, got {type(value).__name__}")
        missing = [k for k in ("encoding", "dtype", "shape", "axes", "data") if k not in value]
        if missing:
            raise TensorDecodeError(f"tensor is missing {', '.join(missing)}")
        return cls(
            encoding=value["encoding"],
            dtype=value["dtype"],
            shape=value["shape"],
            axes=value["axes"],
            data=value["data"],
            byte_order=value.get("byte_order"),
        )

    @classmethod
    def from_list(
        cls,
        values: Any,
        *,
        axes: Sequence[str],
        dtype: str = "float32",
        encoding: str = "base64",
    ) -> Tensor:
        """Build a tensor from nested lists (useful for tests and for round-tripping data)."""
        shape: List[int] = []
        probe = values
        while isinstance(probe, list):
            shape.append(len(probe))
            probe = probe[0] if probe else None
        if dtype not in _ITEMSIZE:
            raise TensorDecodeError(f"unsupported dtype {dtype!r}")
        flat = [_round_to(dtype, v) for v in _flatten_checked(values, shape)]
        if encoding == "base64":
            raw = struct.pack(f"<{len(flat)}{_STRUCT[dtype]}", *flat)
            return cls(
                encoding="base64",
                dtype=dtype,
                shape=shape,
                axes=axes,
                data=base64.b64encode(raw).decode("ascii"),
                byte_order="little",
            )
        if encoding == "float":
            return cls(encoding="float", dtype=dtype, shape=shape, axes=axes, data=_nest(flat, shape))
        raise TensorDecodeError(f"unsupported encoding {encoding!r}")

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "encoding": self.encoding,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "axes": list(self.axes),
        }
        if self.byte_order is not None:
            out["byte_order"] = self.byte_order
        out["data"] = self.data
        return out

    # ------------------------------------------------------------------ properties
    @property
    def size(self) -> int:
        return _product(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def itemsize(self) -> int:
        return _ITEMSIZE.get(self.dtype, 0)

    def axis(self, name: str) -> int:
        """Position of a named axis, e.g. ``attention.axis("token") == 2``."""
        try:
            return self.axes.index(name)
        except ValueError:
            raise KeyError(f"tensor has no axis {name!r}; axes are {list(self.axes)}") from None

    # ------------------------------------------------------------------ validation and decoding
    def _check_header(self) -> None:
        if self.dtype not in _ITEMSIZE:
            raise TensorDecodeError(f"unsupported dtype {self.dtype!r}")
        if self.encoding not in ("base64", "float"):
            raise TensorDecodeError(f"unsupported encoding {self.encoding!r}")
        if not self.shape or any(isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in self.shape):
            raise TensorDecodeError(f"invalid shape {list(self.shape)}")
        if len(self.axes) != len(self.shape):
            raise TensorDecodeError(f"axes {list(self.axes)} do not match shape {list(self.shape)}")
        if self.encoding == "base64" and self.byte_order not in (None, "little"):
            raise TensorDecodeError(f"unsupported byte_order {self.byte_order!r}")

    def _bytes(self) -> bytes:
        if not isinstance(self.data, str):
            raise TensorDecodeError("base64 tensor data must be a string")
        try:
            raw = base64.b64decode(self.data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise TensorDecodeError(f"tensor data is not valid base64: {exc}") from exc
        expected = self.size * _ITEMSIZE[self.dtype]
        if len(raw) != expected:
            raise TensorDecodeError(
                f"tensor data has {len(raw)} bytes; shape {list(self.shape)} of {self.dtype} needs {expected}"
            )
        return raw

    def validate(self) -> Tensor:
        """Check header, length and nesting against ``shape``; returns ``self``."""
        self.flat()
        return self

    def flat(self) -> List[float]:
        """All values in row-major order as Python floats (decoded once, then cached)."""
        if self._flat is None:
            self._check_header()
            if self.encoding == "base64":
                raw = self._bytes()
                n = self.size
                self._flat = [float(x) for x in struct.unpack(f"<{n}{_STRUCT[self.dtype]}", raw)]
            else:
                self._flat = [_round_to(self.dtype, v) for v in _flatten_checked(self.data, self.shape)]
        return self._flat

    def to_list(self) -> NestedFloats:
        """Nested Python lists matching ``shape`` (pure Python, no numpy needed)."""
        return _nest(self.flat(), self.shape)

    def to_bytes(self) -> bytes:
        """Little-endian ``dtype`` bytes, whatever the wire encoding was."""
        self._check_header()
        if self.encoding == "base64":
            return self._bytes()
        flat = self.flat()
        return struct.pack(f"<{len(flat)}{_STRUCT[self.dtype]}", *flat)

    def to_numpy(self, dtype: Optional[str] = None) -> np.ndarray:
        """A numpy array of ``shape`` with the tensor's dtype (or ``dtype``, e.g. ``"float32"``)."""
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - exercised only without numpy
            raise ImportError('Tensor.to_numpy() needs numpy: pip install "breadbowl[numpy]"') from exc
        self._check_header()
        if self.encoding == "base64":
            arr = np.frombuffer(self._bytes(), dtype=_NUMPY_DTYPE[self.dtype]).reshape(self.shape)
        else:
            arr = np.asarray(self.flat(), dtype=_NUMPY_DTYPE[self.dtype]).reshape(self.shape)
        return arr.astype(dtype or arr.dtype.newbyteorder("="), copy=True)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        arr = self.to_numpy()
        return arr if dtype is None else arr.astype(dtype)

    def __getitem__(self, index: Union[int, Tuple[int, ...]]) -> NestedFloats:
        """``t[i, j]`` → float for a full index, nested lists for a prefix."""
        idx = index if isinstance(index, tuple) else (index,)
        if len(idx) > self.ndim:
            raise IndexError(f"too many indices for a tensor of shape {list(self.shape)}")
        offset = 0
        for k, i in enumerate(idx):
            dim = self.shape[k]
            if i < 0:
                i += dim
            if not 0 <= i < dim:
                raise IndexError(f"index {idx[k]} out of range for axis {k} ({self.axes[k]}) of size {dim}")
            offset = offset * dim + i
        rest = self.shape[len(idx) :]
        step = _product(rest)
        flat = self.flat()
        if not rest:
            return flat[offset]
        return _nest(flat[offset * step : (offset + 1) * step], rest)

    def __len__(self) -> int:
        return self.shape[0]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={list(self.shape)}, axes={list(self.axes)}, dtype={self.dtype!r}, "
            f"encoding={self.encoding!r})"
        )

"""Read a Safetensors file tensor by tensor, optionally casting floats in bounded chunks.

Plain reads instead of a memory map: mapped pages count toward the process's resident memory,
and casting after a full float32 load holds both copies. Peak memory is the result plus one chunk.
"""
from __future__ import annotations

import json
import math
import os
import struct

import torch

_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
           "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
           "BOOL": torch.bool}
CHUNK_BYTES = 64 * 1024 * 1024
_MAX_HEADER = 100 * 1024 * 1024


def _read_into(handle, target: torch.Tensor) -> None:
    view = memoryview(target.view(torch.uint8).numpy()).cast("B")
    filled = 0
    while filled < len(view):
        count = handle.readinto(view[filled:])
        if not count:
            raise ValueError("Safetensors file ended early.")
        filled += count


def read_safetensors(path, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """All tensors of a Safetensors file; floating ones cast to `dtype` when given."""
    tensors = {}
    with open(path, "rb", buffering=0) as handle:
        size = os.fstat(handle.fileno()).st_size
        header_length = struct.unpack("<Q", handle.read(8))[0]
        if header_length > min(size - 8, _MAX_HEADER):
            raise ValueError("Invalid Safetensors header length.")
        header = json.loads(handle.read(header_length))
        base = 8 + header_length
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            stored = _DTYPES.get(meta.get("dtype"))
            if stored is None:
                raise ValueError(f"Unsupported Safetensors dtype for {name}: {meta.get('dtype')!r}.")
            shape, (start, end) = list(meta["shape"]), meta["data_offsets"]
            count = math.prod(shape)
            itemsize = torch.empty(0, dtype=stored).element_size()
            if start < 0 or end - start != count * itemsize or base + end > size:
                raise ValueError(f"Safetensors entry {name} lies outside the file or has the wrong size.")
            target = dtype if dtype is not None and stored.is_floating_point and stored != dtype else stored
            tensor = torch.empty(shape, dtype=target)
            handle.seek(base + start)
            if target == stored:
                _read_into(handle, tensor.view(-1))
            else:
                flat, step = tensor.view(-1), max(1, CHUNK_BYTES // itemsize)
                buffer = torch.empty(min(step, count), dtype=stored)
                for first in range(0, count, step):
                    part = buffer[:min(step, count - first)]
                    _read_into(handle, part)
                    flat[first:first + len(part)] = part.to(target)
            tensors[name] = tensor
    return tensors

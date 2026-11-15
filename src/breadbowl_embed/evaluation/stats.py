"""Uncertainty over queries: bootstrap confidence intervals and a paired significance test.

Both resample queries (the unit of evaluation), use NumPy only and are deterministic for a given
seed. Resampling is done in blocks, so memory stays bounded for large query sets.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

_BLOCK = 1000


def _array(values: Sequence[float] | Mapping[str, float]) -> np.ndarray:
    data = np.asarray(list(values.values()) if isinstance(values, Mapping) else list(values), dtype=np.float64)
    if data.ndim != 1 or not len(data):
        raise ValueError("Need at least one per-query value.")
    if not np.isfinite(data).all():
        raise ValueError("Per-query values must be finite.")
    return data


def _check(resamples: int, confidence: float) -> None:
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive integer.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")


def _bootstrap_means(data: np.ndarray, resamples: int, rng: np.random.Generator) -> np.ndarray:
    means = np.empty(resamples)
    for start in range(0, resamples, _BLOCK):
        count = min(_BLOCK, resamples - start)
        means[start:start + count] = data[rng.integers(0, len(data), size=(count, len(data)))].mean(axis=1)
    return means


def bootstrap_ci(values: Sequence[float] | Mapping[str, float], *, resamples: int = 10_000,
                 confidence: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap interval of the mean over queries (queries resampled with replacement)."""
    _check(resamples, confidence)
    data = _array(values)
    means = _bootstrap_means(data, resamples, np.random.default_rng(seed))
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return {"mean": float(math.fsum(data) / len(data)), "low": float(low), "high": float(high),
            "queries": len(data), "confidence": confidence, "resamples": resamples, "seed": seed,
            "method": "percentile bootstrap over queries"}


def paired_test(a: Mapping[str, float], b: Mapping[str, float], *, resamples: int = 10_000,
                confidence: float = 0.95, seed: int = 0) -> dict:
    """Compare two systems on the same queries (b minus a).

    p_value: two-sided paired randomization test (each query's pair of values is swapped with
    probability 1/2; Monte Carlo with `resamples` draws, p = (1 + extreme draws) / (1 + resamples)).
    low/high: percentile bootstrap interval of the mean difference, queries resampled in pairs.
    Both runs must cover exactly the same query ids (evaluate them against the same qrels).
    """
    _check(resamples, confidence)
    if set(a) != set(b):
        only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
        raise ValueError(f"The runs cover different queries ({only_a} only in the first, {only_b} only in the "
                         "second); evaluate both against the same qrels.")
    keys = sorted(a)
    first, second = _array([a[k] for k in keys]), _array([b[k] for k in keys])
    diff = second - first
    observed = float(math.fsum(diff) / len(diff))
    rng = np.random.default_rng(seed)
    extreme = 0
    tolerance = 1e-12 * max(1.0, abs(observed))
    for start in range(0, resamples, _BLOCK):
        count = min(_BLOCK, resamples - start)
        signs = rng.integers(0, 2, size=(count, len(diff)), dtype=np.int8) * 2 - 1
        flipped = np.abs((signs * diff).mean(axis=1))
        extreme += int((flipped >= abs(observed) - tolerance).sum())
    means = _bootstrap_means(diff, resamples, rng)
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return {"queries": len(diff), "mean_a": float(math.fsum(first) / len(first)),
            "mean_b": float(math.fsum(second) / len(second)), "difference": observed,
            "low": float(low), "high": float(high), "p_value": (1 + extreme) / (1 + resamples),
            "b_better": int((diff > 0).sum()), "a_better": int((diff < 0).sum()), "ties": int((diff == 0).sum()),
            "confidence": confidence, "resamples": resamples, "seed": seed,
            "test": "two-sided paired randomization (sign flip)",
            "interval": "percentile bootstrap of the mean difference"}

"""Small numeric helpers (no SciPy): rankings, rank correlations, overlap, effective rank."""
from __future__ import annotations

import math

import numpy as np
import torch


def _array(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().double().numpy()
    return np.asarray(x, dtype=np.float64)


def ranking(scores) -> list[int]:
    """Indices by descending score; ties keep index order (as a stable sort)."""
    values = _array(scores)
    return np.argsort(-values, kind="stable").tolist()


def kendall_tau(a, b) -> float:
    """Kendall's tau-b between two score vectors (ties handled); 1.0 for fewer than two items.

    O(n^2) time, O(n) memory.
    """
    x, y = _array(a), _array(b)
    n = len(x)
    if n < 2:
        return 1.0
    concordance = 0.0
    ties_x = ties_y = 0
    for i in range(n - 1):
        dx = np.sign(x[i] - x[i + 1:])
        dy = np.sign(y[i] - y[i + 1:])
        concordance += float((dx * dy).sum())
        ties_x += int((dx == 0).sum())
        ties_y += int((dy == 0).sum())
    pairs = n * (n - 1) // 2
    denominator = math.sqrt((pairs - ties_x) * (pairs - ties_y))
    if denominator == 0:
        return 1.0 if ties_x == ties_y == pairs else 0.0
    return concordance / denominator


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start
        while end + 1 < len(x) and sorted_x[end + 1] == sorted_x[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2.0
        start = end + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman's rho (average ranks for ties); 1.0 for fewer than two items or constant input."""
    x, y = _array(a), _array(b)
    if len(x) < 2:
        return 1.0
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    if denominator == 0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float((rx * ry).sum() / denominator)


def top_k_overlap(a: list[int], b: list[int], k: int) -> float:
    """|top-k of a ∩ top-k of b| / k."""
    if k <= 0:
        return 1.0
    return len(set(a[:k]) & set(b[:k])) / k


def effective_rank(matrix: torch.Tensor, atol: float = 1e-6) -> float:
    """exp(entropy of the normalised singular values) (Roy and Vetterli, 2007).

    Singular values at or below `atol` are treated as zero (the measure is scale-free, so
    rounding noise of a constant matrix would otherwise look full-rank); 0 for a zero matrix.
    """
    s = torch.linalg.svdvals(matrix.double())
    s = s[s > atol]
    total = float(s.sum())
    if total <= 0:
        return 0.0
    p = s / total
    return math.exp(float(-torch.xlogy(p, p).sum()))

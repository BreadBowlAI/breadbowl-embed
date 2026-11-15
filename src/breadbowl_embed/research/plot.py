"""Matplotlib figures for research objects (optional: pip install 'breadbowl-embed[viz]').

Every function takes the research object, an optional matplotlib Axes (`ax=`) and returns the
Axes it drew on. Magnitudes use one blue ramp; signed quantities a blue (positive) / red
(negative) diverging map around a neutral grey.
"""
from __future__ import annotations

import math

import torch

SEQUENTIAL = ("#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281")
DIVERGING = ("#b8322f", "#e34948", "#f0efec", "#2a78d6", "#184f95")
INK, MUTED = "#0b0b0b", "#898781"


def _pyplot():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError("Plotting needs matplotlib: pip install 'breadbowl-embed[viz]'") from exc
    return plt, LinearSegmentedColormap


def _colormaps():
    _, cmap = _pyplot()
    return (cmap.from_list("breadbowl_sequential", SEQUENTIAL), cmap.from_list("breadbowl_diverging", DIVERGING))


def _axes(ax, size=(8, 4)):
    plt, _ = _pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=size, layout="constrained")
    return ax


def _label(text: str, limit: int = 14) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _heatmap(ax, matrix: torch.Tensor, diverging: bool, colorbar_label: str):
    sequential, diverging_map = _colormaps()
    m = matrix.detach().double().cpu().numpy()
    if diverging:
        bound = float(abs(m).max()) or 1.0
        image = ax.imshow(m, cmap=diverging_map, vmin=-bound, vmax=bound, aspect="auto", interpolation="nearest")
    else:
        image = ax.imshow(m, cmap=sequential, vmin=0, aspect="auto", interpolation="nearest")
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label(colorbar_label)
    bar.outline.set_visible(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return image


def attention(attention_map, *, head=None, ax=None, max_tokens: int = 96, segments=None):
    """Slots x tokens heatmap of an AttentionMap (heads averaged unless `head`); first max_tokens tokens."""
    matrix = attention_map.matrix(head)
    view = attention_map.tokens
    keep = list(range(len(view)))
    if segments is not None:
        keep = [t for t in keep if view.segments[t] in segments]
    keep = keep[:max_tokens]
    ax = _axes(ax, (max(6.0, 0.16 * len(keep) + 2), 4.2))
    _heatmap(ax, matrix[:, keep], False, "attention" + ("" if head is None else f" (head {head})"))
    ax.set_xticks(range(len(keep)), [_label(view.token_texts[t]) for t in keep], rotation=90, fontsize=7)
    ax.set_yticks(range(attention_map.num_slots))
    ax.set_ylabel("slot")
    ax.set_title(f"{view.kind} slot attention" + (f" (first {len(keep)} tokens)" if len(keep) < len(view) else ""),
                 loc="left", fontsize=10)
    return ax


def slot_pairs(decomposition, *, kind: str = "contribution", ax=None):
    """Query slot x document slot heatmap: kind "contribution", "routing_contribution",
    "value_contribution", "routing_weight", "value_attention" or "similarity"."""
    matrices = {"contribution": (decomposition.pair_contribution("final"), True),
                "routing_contribution": (decomposition.routing_pair_contribution, True),
                "value_contribution": (decomposition.value_pair_contribution, True),
                "routing_weight": (decomposition.routing_weight, False),
                "value_attention": (decomposition.value_attention, False),
                "similarity": (decomposition.similarity, True)}
    if kind not in matrices:
        raise ValueError(f"kind must be one of {sorted(matrices)}.")
    matrix, diverging = matrices[kind]
    ax = _axes(ax, (5.6, 4.6))
    _heatmap(ax, matrix, diverging, kind.replace("_", " "))
    ax.set_xlabel("document slot")
    ax.set_ylabel("query slot")
    ax.set_title(f"final {decomposition.final:.4f} = {1 - decomposition.value_mix:.2f}·R {decomposition.routing:.4f}"
                 f" + {decomposition.value_mix:.2f}·V {decomposition.value:.4f}", loc="left", fontsize=10)
    return ax


def _signed_bars(ax, labels, values, title, xlabel):
    colors = [DIVERGING[3] if v >= 0 else DIVERGING[1] for v in values]
    positions = list(range(len(values)))[::-1]
    ax.barh(positions, values, color=colors, height=0.7)
    ax.set_yticks(positions, labels, fontsize=8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def token_bars(attribution, *, k: int = 20, ax=None):
    """The k tokens with the largest |attribution|, as signed bars."""
    top = attribution.top_tokens(k)
    ax = _axes(ax, (6, 0.28 * len(top) + 1.2))
    labels = [f"#{t['index']} {_label(t['display'].strip() or repr(t['display']), 18)}" for t in top]
    return _signed_bars(ax, labels, [t["weight"] for t in top],
                        f"{attribution.side} tokens, {attribution.component} (attention-weighted, not causal)",
                        "attribution")


def occlusion(sweep, *, k: int = 20, ax=None, component: str = "final"):
    """Units whose occlusion changed the score most (support = baseline - occluded)."""
    delta = sweep.delta(component)
    order = sorted(range(len(sweep.units)), key=lambda i: -abs(float(delta[i])) if math.isfinite(float(delta[i]))
                   else 0.0)[:k]
    ax = _axes(ax, (6, 0.28 * len(order) + 1.2))
    labels = [_label(sweep.units[i]["text"], 22) for i in order]
    return _signed_bars(ax, labels, [-float(delta[i]) for i in order],
                        f"occlusion ({sweep.mode}, {sweep.unit}s of the {sweep.side})",
                        f"support: {component} score drop when occluded")


def slot_statistics(stats, *, ax=None):
    """Within-text routing cosine between slots (collapse check)."""
    ax = _axes(ax, (5.6, 4.6))
    _heatmap(ax, stats.within_text_routing_cosine, True, "mean within-text cosine")
    ax.set_xlabel("slot")
    ax.set_ylabel("slot")
    ax.set_title(f"{stats.count} {stats.kind}s; per-text effective rank "
                 f"{stats.per_text_effective_rank['routing_mean']:.2f} of {stats.slots}", loc="left", fontsize=10)
    return ax


def shortlist(diagnostics, *, ax=None):
    """Underfill rate and recall against exact routing, per slot-hit width."""
    rows = diagnostics.rows
    widths = [r["slot_hits"] for r in rows]
    ax = _axes(ax, (6, 3.6))
    series = (("underfill_rate", "underfilled queries", DIVERGING[1]),
              ("candidate_recall", "candidate recall vs exact", DIVERGING[3]),
              ("result_recall_at_k", f"result recall@{diagnostics.top_k}", "#1baf7a"))
    for key, label, color in series:
        ax.plot(widths, [r[key] for r in rows], marker="o", linewidth=2, markersize=5, color=color, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths, [str(w) for w in widths])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("slot hits per query slot")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{diagnostics.queries} queries, {diagnostics.documents} documents", loc="left", fontsize=10)
    return ax


def precision(comparison, *, ax=None):
    """Largest final-score change of every variant against the reference (log scale)."""
    rows = [r for r in comparison.rows if r["variant"] != comparison.reference]
    ax = _axes(ax, (7, 0.3 * len(rows) + 1.4))
    values = [max(r["max_abs_delta"], 1e-12) for r in rows]
    positions = list(range(len(rows)))[::-1]
    ax.barh(positions, values, color=DIVERGING[3], height=0.7)
    ax.set_yticks(positions, [r["variant"] for r in rows], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("max |Δ final score| vs reference")
    ax.set_title(f"reference: {comparison.reference}", loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax

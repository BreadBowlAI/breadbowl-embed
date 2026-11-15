"""Draw which passage words a few slots attend to, from a real model run.

    pip install "git+https://github.com/BreadBowlAI/breadbowl-embed.git" matplotlib
    python attention_figure.py                    # writes ../assets/attention.png

What it shows:
  1. Encode one query and one passage (CPU, float32, one text per pass).
  2. `model.explain` gives the query-to-passage slot attention (16 x 16).
     Summed over query slots, it says which passage slots the query reads.
  3. For the 3 passage slots read most, take each slot's attention over the
     passage tokens (`return_attention=True`, mean over the 4 heads), and add
     token weights up into words. Punctuation stays separate.

Deterministic: fixed text, fixed thread count, batch size 1. Weights are
attention, not a causal explanation of the score.
"""

import argparse
import re
from pathlib import Path

import torch
from matplotlib.patches import FancyBboxPatch

import figstyle as st
from breadbowl_embed import BreadBowl

QUERY = "How many hearts does an octopus have?"
PASSAGE = ("Octopuses have three hearts. Two pump blood through the gills, "
           "and the third pumps it to the rest of the body.")
SLOTS_SHOWN = 3


def word_units(text):
    """Character spans of words and of single punctuation marks."""
    return [m.span() for m in re.finditer(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']", text)]


def word_weights(attention, offsets, units):
    """Add token weights into word units, split by character overlap."""
    out = torch.zeros(attention.shape[0], len(units))
    for t, span in enumerate(offsets):
        if span is None:
            continue
        a, b = span
        hits = [(k, min(b, ue) - max(a, us)) for k, (us, ue) in enumerate(units)
                if min(b, ue) > max(a, us)]
        total = sum(n for _, n in hits)
        for k, n in hits:
            out[:, k] += attention[:, t] * n / total
    return out


def compute(model_id, threads):
    torch.manual_seed(0)
    torch.set_num_threads(threads)
    model = BreadBowl.from_pretrained(model_id)  # float32 as stored; CPU
    docs = model.encode_documents([PASSAGE], return_attention=True, return_tokens=True)
    queries = model.encode_queries([QUERY])
    exp = model.explain(queries, docs)[0][0]
    read = exp.value_attention.sum(0)  # [16 passage slots]
    slots = torch.argsort(read, descending=True)[:SLOTS_SHOWN].tolist()
    units = word_units(PASSAGE)
    weights = word_weights(docs.attention[0].mean(0), docs.offsets[0], units)
    rows = [(s, weights[s].tolist()) for s in slots]
    return rows, units, float(exp.score)


def layout(ax, units, x0, x1, size):
    """Place word units left to right, wrapping at x1. Returns (x, width, line)."""
    renderer = ax.figure.canvas.get_renderer()
    space = 12
    placed, x, line = [], x0, 0
    for k, (a, b) in enumerate(units):
        word = PASSAGE[a:b]
        t = st.text(ax, 0, 0, word, size)
        width = t.get_window_extent(renderer).width
        t.remove()
        spaced = k > 0 and PASSAGE[units[k - 1][1]:a] != ""
        gap = space if spaced else 3
        if k and x + gap + width + 12 > x1 and spaced:
            x, line = x0, line + 1
        elif k:
            x += gap
        placed.append((x, width + 12, line))
        x += width + 12
    return placed


def draw(rows, units, out):
    W = 1200
    size, line_h, row_gap = 26, 50, 30
    x_text, x_end = 190, 1160
    probe, ax = st.canvas(W, 100)
    placed = layout(ax, units, x_text, x_end, size)
    lines = max(p[2] for p in placed) + 1
    row_h = lines * line_h
    top = 215
    H = top + len(rows) * (row_h + row_gap) + 90
    probe.clf()

    fig, ax = st.canvas(W, H)
    st.text(ax, 40, 62, "What the query reads in the passage", 32, weight="bold")
    st.text(ax, 40, 104, f"Query: {QUERY}", 24, color=st.INK)
    st.text(ax, 40, 142, "Each row is one of the 3 passage slots this query reads most.", 24,
            color=st.INK_2)
    st.text(ax, 40, 174, "Shading shows which words that slot pools. Darker = more attention.", 24,
            color=st.INK_2)

    peak = max(max(w) for _, w in rows)
    for r, (slot, weights) in enumerate(rows):
        y = top + r * (row_h + row_gap)
        st.text(ax, 40, y + line_h / 2, f"Slot {slot}", 24, weight="bold", va="center")
        for (a, b), (x, width, line), w in zip(units, placed, weights):
            fill = st.blend(st.RAMP, w / peak)
            yy = y + line * line_h
            ax.add_patch(FancyBboxPatch((x, yy + 4), width, line_h - 8,
                                        boxstyle="round,pad=0,rounding_size=6",
                                        facecolor=fill, edgecolor="none"))
            ink = "#ffffff" if st.contrast("#ffffff", fill) > st.contrast(st.INK, fill) else st.INK
            st.text(ax, x + width / 2, yy + line_h / 2, PASSAGE[a:b], size, color=ink,
                    ha="center", va="center_baseline")

    # legend: less -> more
    ly = H - 52
    st.text(ax, 40, ly + 8, "less", 24, color=st.INK_2, va="center")
    steps = 120
    for i in range(steps):
        ax.add_patch(FancyBboxPatch((100 + i * 2, ly - 6), 2.5, 26, boxstyle="square,pad=0",
                                    facecolor=st.blend(st.RAMP, i / (steps - 1)),
                                    edgecolor="none"))
    st.text(ax, 100 + steps * 2 + 14, ly + 8, "more", 24, color=st.INK_2, va="center")
    st.save(fig, out)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="dotproductx/breadbowl-embed-v1.3")
    parser.add_argument("--out", type=Path, default=here.parent / "assets" / "attention.png")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    rows, units, score = compute(args.model, args.threads)
    print(f"score {score:.4f}")
    for slot, weights in rows:
        top = sorted(zip(weights, (PASSAGE[a:b] for a, b in units)), reverse=True)[:5]
        print(f"slot {slot:2d}:", ", ".join(f"{word} {w:.3f}" for w, word in top))
    draw(rows, units, args.out)


if __name__ == "__main__":
    main()

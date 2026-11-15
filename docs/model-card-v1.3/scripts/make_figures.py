"""Draw the architecture diagram, the benchmark chart and the social image.

    python make_figures.py                      # writes ../assets/*.png
    python make_figures.py --og path/to/og.png  # also writes the 1200x630 social image

Numbers: nDCG@10 x 100, `paper` search profile (128 slot hits), full
8,841,823-passage MS MARCO corpus. Source: the v1.3 model card results table.
"""

import argparse
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath

import figstyle as st

RESULTS = [  # (test set, v1.2-preview, v1.3)
    ("MS MARCO dev", 34.27, 41.71),
    ("TREC DL19", 68.99, 74.63),
    ("TREC DL20", 59.57, 73.14),
]


def box(ax, x0, y0, x1, y1, color=st.BOX, radius=14, **kw):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=color, edgecolor="none", **kw))


def arrow(ax, x0, y0, x1, y1, color=st.MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=26,
                                 linewidth=3, color=color, shrinkA=0, shrinkB=0))


def slot_square(ax, x, y, size, gap=3):
    half = (size - gap) / 2
    ax.add_patch(Rectangle((x, y), half, size, facecolor=st.ROUTING, edgecolor="none"))
    ax.add_patch(Rectangle((x + half + gap, y), half, size, facecolor=st.VALUE, edgecolor="none"))


def architecture(path):
    W, H = 1200, 520
    fig, ax = st.canvas(W, H)
    mid = 205

    # Text -> backbone
    box(ax, 30, mid - 55, 190, mid + 55)
    st.text(ax, 110, mid - 8, "Query or", 26, weight="bold", ha="center")
    st.text(ax, 110, mid + 26, "passage", 26, weight="bold", ha="center")
    arrow(ax, 194, mid, 236, mid)
    box(ax, 236, mid - 65, 434, mid + 65)
    st.text(ax, 335, mid - 4, "Backbone", 28, weight="bold", ha="center")
    st.text(ax, 335, mid + 34, "Qwen3.5 0.8B", 24, color=st.INK_2, ha="center")
    arrow(ax, 438, mid, 486, mid)

    # Stored-once container
    ax.add_patch(FancyBboxPatch((455, 45), 720, 315, boxstyle="round,pad=0,rounding_size=18",
                                facecolor="none", edgecolor=st.BASELINE, linewidth=2.5,
                                linestyle=(0, (6, 5))))
    label = st.text(ax, 480, 54, " Stored once per passage ", 24, color=st.INK_2, va="center")
    label.set_backgroundcolor(st.SURFACE)

    # 16 slots, each half routing (blue) and half value (orange)
    size, gap = 28, 11
    gx, gy = 492, mid - 72
    st.text(ax, gx + 72, gy - 20, "16 slots", 26, weight="bold", ha="center")
    for r in range(4):
        for c in range(4):
            slot_square(ax, gx + c * (size + gap), gy + 6 + r * (size + gap), size)

    # Two lanes
    lanes = [("Routing vectors", "256-d · find candidates", st.ROUTING, mid - 60),
             ("Value vectors", "256-d · rerank", st.VALUE, mid + 60)]
    for title, sub, color, cy in lanes:
        arrow(ax, 646, mid, 712, cy)
        box(ax, 716, cy - 46, 1155, cy + 46, radius=12)
        ax.add_patch(FancyBboxPatch((736, cy - 18), 36, 36, boxstyle="round,pad=0,rounding_size=6",
                                    facecolor=color, edgecolor="none"))
        st.text(ax, 792, cy - 6, title, 26, weight="bold")
        st.text(ax, 792, cy + 28, sub, 24, color=st.INK_2)

    # Score
    arrow(ax, 815, 362, 815, 404)
    box(ax, 455, 408, 1175, 488, color="#e6eefa")
    st.text(ax, 815, 458, "Score = 0.75 × routing + 0.25 × value", 30, weight="bold", ha="center")
    st.text(ax, 30, 438, "No second model", 24, color=st.INK_2)
    st.text(ax, 30, 470, "at rerank time.", 24, color=st.INK_2)

    st.save(fig, path)


def bar(ax, x0, y, x1, thickness, color, radius=7):
    """Horizontal bar, square at the baseline, rounded at the data end."""
    t, b = y, y + thickness
    r = min(radius, (x1 - x0) / 2, thickness / 2)
    verts = [(x0, t), (x1 - r, t), (x1, t), (x1, t + r), (x1, b - r), (x1, b), (x1 - r, b),
             (x0, b), (x0, t)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO,
             MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none"))


def benchmarks(path):
    W, H = 1200, 560
    fig, ax = st.canvas(W, H)
    st.text(ax, 40, 62, "nDCG@10 on three test sets", 32, weight="bold")
    st.text(ax, 40, 102, "Higher is better. Same corpus and search settings for both.", 24,
            color=st.INK_2)

    # Legend
    lx, ly = 40, 150
    for name, color in (("v1.2-preview", st.OLD), ("v1.3", st.NEW)):
        ax.add_patch(FancyBboxPatch((lx, ly - 22), 26, 26, boxstyle="round,pad=0,rounding_size=4",
                                    facecolor=color, edgecolor="none"))
        t = st.text(ax, lx + 38, ly, name, 24, color=st.INK)
        lx += 38 + (175 if name.startswith("v1.2") else 80)

    x0, x100 = 255, 1080
    scale = (x100 - x0) / 100
    top, step, thick, gap = 200, 112, 38, 4
    bottom = top + step * len(RESULTS) - 30

    for v in (0, 25, 50, 75, 100):
        x = x0 + v * scale
        ax.plot([x, x], [top - 14, bottom], color=st.GRID if v else st.BASELINE,
                linewidth=1.2 if v else 2.4, solid_capstyle="butt", zorder=0)
        st.text(ax, x, bottom + 36, str(v), 24, color=st.MUTED, ha="center")

    for i, (name, old, new) in enumerate(RESULTS):
        y = top + i * step
        st.text(ax, x0 - 22, y + thick + gap / 2, name, 26, ha="right", va="center")
        for j, (value, color, ink, weight) in enumerate(
                ((old, st.OLD, st.INK_2, "normal"), (new, st.NEW, st.INK, "bold"))):
            yy = y + j * (thick + gap)
            bar(ax, x0, yy, x0 + value * scale, thick, color)
            st.text(ax, x0 + value * scale + 12, yy + thick / 2, f"{value:.2f}", 24, color=ink,
                    weight=weight, va="center_baseline")

    st.save(fig, path)


def og(path):
    W, H = 1200, 630
    fig, ax = st.canvas(W, H)
    st.text(ax, 72, 130, "BreadBowl-Embed v1.3", 60, weight="bold")
    st.text(ax, 72, 200, "Encode once. Retrieve and rerank", 36, color=st.INK_2)
    st.text(ax, 72, 248, "from the same stored representation.", 36, color=st.INK_2)

    # slot motif, top right
    size, gap = 28, 12
    for r in range(4):
        for c in range(4):
            slot_square(ax, 962 + c * (size + gap), 72 + r * (size + gap), size)

    tiles = [(name, new, old) for name, old, new in RESULTS]
    tx = 72
    for name, new, old in tiles:
        box(ax, tx, 330, tx + 330, 520, color=st.BOX, radius=18)
        st.text(ax, tx + 30, 420, f"{new:.2f}", 68, weight="bold")
        st.text(ax, tx + 30, 462, name, 28)
        st.text(ax, tx + 30, 498, f"v1.2-preview: {old:.2f}", 22, color=st.INK_2)
        tx += 354
    st.text(ax, 72, 578, "nDCG@10 · full MS MARCO corpus · Apache-2.0 weights", 24,
            color=st.INK_2)
    st.save(fig, path, radius=0, border=False)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=here.parent / "assets")
    parser.add_argument("--og", type=Path, help="also write the 1200x630 social image here")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    architecture(args.out / "architecture.png")
    benchmarks(args.out / "benchmarks.png")
    if args.og:
        og(args.og)


if __name__ == "__main__":
    main()

"""Shared look for the BreadBowl-Embed card figures.

Every figure is a PNG with its own light background and rounded corners,
so it reads the same on light and dark pages. Sizes are in pixels.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

FONT = "DejaVu Sans"

# Colours (colour-blind-safe; checked with a CVD validator)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BOX = "#f0efec"
BORDER = "#d9d8d1"

ROUTING = "#2a78d6"  # blue
VALUE = "#eb6834"  # orange
OLD = "#86b6ef"  # v1.2-preview, light blue
NEW = "#1c5cab"  # v1.3, dark blue
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

PT = 72 / 100  # points per pixel at dpi 100


def canvas(width, height):
    """A figure whose data coordinates are pixels, origin top-left."""
    plt.rcParams["font.family"] = FONT
    plt.rcParams["svg.fonttype"] = "none"
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return fig, ax


def text(ax, x, y, s, size, color=INK, weight="normal", ha="left", va="baseline", **kw):
    return ax.text(x, y, s, fontsize=size * PT, color=color, fontweight=weight,
                   ha=ha, va=va, fontfamily=FONT, **kw)


def save(fig, path, radius=28, border=True):
    """Save, then give the image rounded corners and a hairline border."""
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    if radius:
        round_corners(path, radius, border)
    shrink(path)


def round_corners(path, radius, border):
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    s = 4  # supersample for smooth edges
    mask = Image.new("L", (w * s, h * s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w * s - 1, h * s - 1), radius * s, fill=255)
    img.putalpha(mask.resize((w, h), Image.LANCZOS))
    if border:
        line = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
        ImageDraw.Draw(line).rounded_rectangle(
            (s, s, w * s - 1 - s, h * s - 1 - s), radius * s, outline=BORDER, width=2 * s)
        img = Image.alpha_composite(img, line.resize((w, h), Image.LANCZOS))
    img.save(path)


def shrink(path):
    """Quantise to a 256-colour palette; these figures are flat colour."""
    img = Image.open(path)
    img.quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(
        path, optimize=True)


def blend(colors, t):
    """Linear interpolation along a list of hex colours, t in [0, 1]."""
    t = min(max(t, 0.0), 1.0) * (len(colors) - 1)
    i = min(int(t), len(colors) - 2)
    f = t - i
    a = [int(colors[i][k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(colors[i + 1][k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * f):02x}" for x, y in zip(a, b))


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def luminance(hex_color):
    def channel(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hex_color[k:k + 2], 16) for k in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)

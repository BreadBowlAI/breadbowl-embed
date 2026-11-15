"""Static HTML for notebooks (`_repr_html_`): no scripts, every user string escaped.

Colours follow one scheme everywhere: magnitude (attention) is a single blue ramp; signed
quantities (contributions, attribution, deltas) are diverging blue (positive) / red (negative)
around a neutral midpoint. Fills are translucent so they read on light and dark notebook themes.
"""
from __future__ import annotations

import html
import math

import torch

BLUE = (42, 120, 214)
RED = (227, 73, 72)
MAX_ALPHA = 0.72
STYLE = """<style>
.bbr{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:13px;line-height:1.45;max-width:100%}
.bbr h4{margin:.6em 0 .3em;font-size:14px}
.bbr .note{opacity:.75;font-size:12px;margin:.2em 0 .6em}
.bbr table{border-collapse:collapse;margin:.3em 0}
.bbr th,.bbr td{padding:2px 8px;text-align:right;border-bottom:1px solid rgba(128,128,128,.25);
  font-variant-numeric:tabular-nums}
.bbr th:first-child,.bbr td:first-child{text-align:left}
.bbr .tokens{white-space:pre-wrap;word-break:break-word;line-height:1.9;font-family:ui-monospace,monospace;
  font-size:12px;border:1px solid rgba(128,128,128,.3);border-radius:4px;padding:6px 8px;max-width:60em}
.bbr .tok{border-radius:3px}
.bbr .seg-instruction,.bbr .seg-template{text-decoration:underline dashed rgba(128,128,128,.8);
  text-underline-offset:3px}
.bbr .seg-special{opacity:.6}
.bbr .kv{display:flex;gap:1.5em;flex-wrap:wrap;margin:.3em 0}
.bbr .kv b{font-variant-numeric:tabular-nums}
.bbr svg text{font-size:9px;fill:currentColor}
</style>"""


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def fmt(value, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "–"
        if value != 0 and (abs(value) < 10 ** -digits or abs(value) >= 1e5):
            return f"{value:.{max(1, digits - 1)}e}"
        return f"{value:.{digits}f}"
    if isinstance(value, (list, tuple)) and len(value) <= 4:
        return ", ".join(fmt(v, digits) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}={fmt(v, digits)}" for k, v in value.items())
    return str(value)


def fill(weight: float, scale: float, diverging: bool = False) -> str:
    """CSS rgba fill for a value relative to `scale` (the largest magnitude shown)."""
    if not scale or not math.isfinite(weight):
        return "transparent"
    alpha = min(1.0, abs(weight) / scale) * MAX_ALPHA
    rgb = RED if diverging and weight < 0 else BLUE
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.3f})"


def wrap(body: str, title: str | None = None, note: str | None = None) -> str:
    head = f"<h4>{esc(title)}</h4>" if title else ""
    foot = f'<div class="note">{esc(note)}</div>' if note else ""
    return f'{STYLE}<div class="bbr">{head}{body}{foot}</div>'


def table(rows: list[dict], columns: list[str] | None = None, digits: int = 4, limit: int = 200) -> str:
    if not rows:
        return '<div class="note">(empty)</div>'
    columns = columns or list(rows[0])
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{esc(fmt(row.get(c), digits))}</td>" for c in columns) + "</tr>"
                   for row in rows[:limit])
    more = f'<div class="note">{len(rows) - limit} more rows not shown.</div>' if len(rows) > limit else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{more}"


def table_html(title: str, rows: list[dict], note: str | None = None) -> str:
    return wrap(table(rows), title, note)


def tokens_html(view, weights=None, *, diverging: bool = False, label: str = "weight") -> str:
    """The model input token by token (display strings), optionally shaded by per-token weights."""
    values = None if weights is None else [float(w) for w in weights]
    scale = max((abs(v) for v in values if math.isfinite(v)), default=0.0) if values else 0.0
    parts = []
    for t, (shown, segment) in enumerate(zip(view.token_texts, view.segments)):
        title = f"#{t} {view.tokens[t]!r} [{segment}]"
        style = ""
        if values is not None:
            title += f" {label}={fmt(values[t])}"
            style = f' style="background:{fill(values[t], scale, diverging)}"'
        parts.append(f'<span class="tok seg-{esc(segment)}" title="{esc(title)}"{style}>{esc(shown)}</span>')
    suffix = '<span class="seg-special"> …[truncated]</span>' if view.truncated else ""
    return f'<div class="tokens">{"".join(parts)}{suffix}</div>'


def svg_heatmap(matrix: torch.Tensor, *, diverging: bool, row_label: str, col_label: str, cell: int = 14,
                max_width: int = 900, title: str = "value") -> str:
    """A rows x columns heatmap as inline SVG; every cell has a native tooltip."""
    m = matrix.detach().double().cpu()
    rows, columns = m.shape
    width = min(cell, max(2, max_width // max(1, columns)))
    finite = m[torch.isfinite(m)]
    scale = float(finite.abs().max()) if finite.numel() else 0.0
    left, top = 28, 14
    cells = []
    for i in range(rows):
        for j in range(columns):
            v = float(m[i, j])
            cells.append(f'<rect x="{left + j * width}" y="{top + i * cell}" width="{width}" height="{cell}" '
                         f'fill="{fill(v, scale, diverging)}"><title>{esc(row_label)} {i}, {esc(col_label)} {j}: '
                         f'{esc(title)} {fmt(v)}</title></rect>')
    labels = [f'<text x="{left - 4}" y="{top + i * cell + cell * 0.72}" text-anchor="end">{i}</text>'
              for i in range(rows)]
    if width >= 10:
        labels += [f'<text x="{left + j * width + width / 2}" y="{top - 4}" text-anchor="middle">{j}</text>'
                   for j in range(columns)]
    total_w, total_h = left + columns * width + 4, top + rows * cell + 4
    frame = (f'<rect x="{left}" y="{top}" width="{columns * width}" height="{rows * cell}" fill="none" '
             'stroke="currentColor" stroke-opacity=".25"/>')
    return (f'<svg role="img" aria-label="{esc(row_label)} by {esc(col_label)} heatmap" width="{total_w}" '
            f'height="{total_h}" viewBox="0 0 {total_w} {total_h}">{"".join(cells)}{frame}{"".join(labels)}</svg>'
            f'<div class="note">rows: {esc(row_label)}; columns: {esc(col_label)}; '
            f'{"blue positive, red negative" if diverging else "stronger colour = larger"}; max |{esc(title)}| {fmt(scale)}</div>')


def generic_html(obj) -> str:
    payload = obj._payload()
    rows = [{"field": k, "value": v} for k, v in payload.items()
            if isinstance(v, (int, float, str, bool)) or v is None]
    return table_html(type(obj).__name__, rows)


def token_view_html(view) -> str:
    counts = {s: view.segments.count(s) for s in dict.fromkeys(view.segments)}
    info = f'<div class="kv">{view.token_count} tokens · {esc(fmt(counts))} · truncated: {fmt(view.truncated)}</div>'
    return wrap(info + tokens_html(view), f"{view.kind.capitalize()} tokens",
                "Dashed underline: instruction/template tokens. Hover a token for its vocabulary string.")


def attention_html(attention) -> str:
    stats = attention.statistics()
    top = attention.top_tokens(3)
    rows = [{"slot": s["slot"], "entropy": s["entropy"], "eff. tokens": s["effective_tokens"],
             "cover 90%": s["coverage_90"], **{k: v for k, v in s["segment_mass"].items() if
                                               any(r["segment_mass"][k] > 0 for r in stats)},
             "top tokens": " | ".join(t["display"].strip() or repr(t["display"]) for t in top[s["slot"]])}
            for s in stats]
    body = svg_heatmap(attention.matrix(), diverging=False, row_label="slot", col_label="token", cell=12,
                       title="attention") + table(rows)
    best = int(torch.tensor([s["coverage_90"] for s in stats]).argmin())
    body += f"<h4>Text shaded by slot {best} (most focused slot), heads averaged</h4>" + tokens_html(
        attention.tokens, attention.slot(best), label="attention")
    return wrap(body, f"Slot attention: {attention.tokens.kind}, {attention.num_heads} heads x {attention.num_slots} "
                      f"slots x {attention.num_tokens} tokens",
                "Heads averaged. Attention shows how tokens were pooled into slots; it is not an explanation.")


def decomposition_html(d) -> str:
    kv = (f'<div class="kv">final <b>{fmt(d.final, 6)}</b> = {fmt(1 - d.value_mix, 2)} × R <b>{fmt(d.routing, 6)}</b> '
          f'+ {fmt(d.value_mix, 2)} × V <b>{fmt(d.value, 6)}</b> · residual {fmt(d.residuals()["final"])}'
          f" · rule {esc(d.rule.name)}</div>")
    body = kv + svg_heatmap(d.pair_contribution("final"), diverging=True, row_label="query slot",
                            col_label="document slot", title="contribution")
    rows = [{k: r[k] for k in ("query_slot", "best_document_slot", "best_cosine", "routing_contribution",
                               "value_slot_cosine", "final_contribution")} for r in d.slot_table()]
    return wrap(body + table(rows), "Score decomposition",
                "Pair contributions sum, with the routing smoothing terms, to the final score.")


def attribution_html(a) -> str:
    kv = (f'<div class="kv">{esc(a.component)} total <b>{fmt(a.total, 6)}</b> · on tokens '
          f'<b>{fmt(float(a.weights.sum()), 6)}</b> · unattributed {fmt(a.unattributed, 6)}</div>')
    body = kv + tokens_html(a.tokens, a.weights, diverging=True, label="attribution") + table(
        [{k: t[k] for k in ("index", "display", "segment", "weight", "top_slot")} for t in a.top_tokens(8)])
    return wrap(body, f"Token attribution ({a.side}, {a.component})", a.note)


def pair_html(p) -> str:
    d = p.decomposition
    body = (f'<div class="kv">final <b>{fmt(d.final, 6)}</b> · R <b>{fmt(d.routing, 6)}</b> · '
            f'V <b>{fmt(d.value, 6)}</b> · model {esc(str(p.model["representation_id"])[:12])}… '
            f'({esc(p.model["compute_dtype"])})</div>')
    body += "<h4>Query (shaded by attention-weighted attribution of the final score)</h4>"
    body += tokens_html(p.query.tokens, p.attribution("query").weights, diverging=True, label="attribution")
    body += "<h4>Document (shaded by attention-weighted attribution of the final score)</h4>"
    body += tokens_html(p.document.tokens, p.attribution("document").weights, diverging=True, label="attribution")
    body += "<h4>Slot pairs (contribution to the final score)</h4>"
    body += svg_heatmap(d.pair_contribution("final"), diverging=True, row_label="query slot",
                        col_label="document slot", title="contribution")
    return wrap(body, "Query–document pair",
                "Attribution is attention-weighted bookkeeping, not causal. pair.report(path) writes the "
                "interactive report.")


def trace_html(trace) -> str:
    rows = [{"tensor": name, **{k: v for k, v in entry.items()}} for name, entry in trace.summary().items()]
    for row in rows:
        row["shape"] = " x ".join(str(s) for s in row["shape"])
    return wrap(tokens_html(trace.tokens) + table(rows, ["tensor", "shape", "norm_mean", "norm_min", "norm_max"]),
                f"Trace ({trace.kind}, network {trace.network_dtype}, copies in float32)")


def trace_set_html(traces) -> str:
    rows = [{"item": i, "tokens": t.tokens.token_count, "truncated": t.tokens.truncated,
             "text": t.tokens.text[:80]} for i, t in enumerate(traces)]
    return table_html(f"{len(traces)} traces ({traces.kind})", rows)


def slot_statistics_html(s) -> str:
    kv = (f'<div class="kv">{s.count} {esc(s.kind)}s · within-text routing cosine mean '
          f'<b>{fmt(s.collapse["mean_within_text_cosine"])}</b>, max <b>{fmt(s.collapse["max_within_text_cosine"])}</b>'
          f' (slots {esc(fmt(s.collapse["most_similar_pair"]))}) · per-text effective rank '
          f'<b>{fmt(s.per_text_effective_rank["routing_mean"], 2)}</b> of {s.slots}</div>')
    body = kv + table(list(s.per_slot)) + "<h4>Within-text routing cosine between slots</h4>" + svg_heatmap(
        s.within_text_routing_cosine, diverging=True, row_label="slot", col_label="slot", title="cosine")
    return wrap(body, "Slot statistics")


def sweep_html(sweep) -> str:
    delta = sweep.token_delta()
    body = (f'<div class="kv">baseline final <b>{fmt(sweep.baseline["final"], 6)}</b> · {len(sweep.units)} '
            f'{esc(sweep.unit)}s occluded one at a time ({esc(sweep.mode)})</div>')
    body += tokens_html(sweep.tokens, -delta, diverging=True, label="support (score drop when occluded)")
    body += table([{k: r[k] for k in ("index", "text", "start", "end", "delta")} for r in sweep.top(8)])
    return wrap(body, f"Occlusion sweep ({sweep.side})",
                "Shaded by support = baseline - occluded score. Blue: occluding the unit lowered the score (it "
                "supported the score); red: occluding it raised the score.")

"""Self-contained HTML report for one query-document pair.

One file with inline CSS, JavaScript and SVG: no network access, no external fonts or scripts.
User text is escaped in the HTML and embedded as JSON that the script only ever assigns with
textContent. The tables are static HTML, so the decomposition stays readable without
JavaScript; the token heatmaps and the slot-pair matrix are interactive (mouse and keyboard).
"""
from __future__ import annotations

from datetime import datetime, timezone
from importlib import resources
import json

from .._version import __version__
from ._html import esc, fmt, tokens_html
from .decomposition import COMPONENTS

MATRICES = {
    "contribution": ("contribution to the final score", True,
                     "c_ij = (1 - m) w_ij s_ij + m A_ij (u_i · v_j) / (|r_i| S_q); with the routing smoothing "
                     "terms these sum to the final score."),
    "routing_contribution": ("contribution to R", True, "w_ij s_ij; with the smoothing terms these sum to R."),
    "value_contribution": ("contribution to V", True, "A_ij (u_i · v_j) / (|r_i| S_q); these sum to V."),
    "routing_weight": ("routing weight", False,
                       "w_ij = softmax_j(s_ij / tau_k) / S_q: the weight each pair carries in R (sums to 1)."),
    "value_attention": ("value attention", False,
                        "A_ij = softmax_j(s_ij / tau_v): how much of document value slot j query slot i reads."),
    "similarity": ("routing cosine", True, "s_ij: cosine of query routing slot i and document routing slot j."),
}
CAVEATS = (
    "Slot attention shows how the encoder pooled token states into slots. It is not an explanation of the "
    "score on its own: the backbone has already mixed information across tokens before pooling.",
    "Token attribution distributes each slot's exact contribution over that slot's tokens by its attention "
    "weights. It is attention-weighted bookkeeping, not a causal claim; test causal effects with occlusion "
    "and slot ablation (breadbowl_embed.research.occlusion_sweep / slot_ablation).",
    "The head average is a summary: the model concatenates the heads and projects them, it does not average them.",
    "Scores depend on the compute dtype and on batching: bfloat16 differs from float32 by up to ~1e-2 in final "
    "score, and encoding several texts per forward pass moves bfloat16 scores by up to ~7e-3. This report "
    "encodes each text alone.",
    "Truncated texts are scored on their first tokens only (128 query tokens including the instruction, 384 "
    "document tokens); the rest of the text is shown but was not read.",
)


def _asset(name: str) -> str:
    return resources.files("breadbowl_embed.research").joinpath("_static", name).read_text(encoding="utf-8")


def _script_json(value) -> str:
    """JSON safe to embed inside <script>: no '<', '>', '&' or line separators survive literally."""
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for raw, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), (" ", "\\u2028"),
                         (" ", "\\u2029")):
        text = text.replace(raw, escaped)
    return text


def _rounded(matrix, digits: int = 6):
    return [[round(float(v), digits) for v in row] for row in matrix.tolist()]


def _side(attention, digits: int = 5) -> dict:
    view = attention.tokens
    return {"tokens": [[shown, segment, token, t] for t, (shown, segment, token) in
                       enumerate(zip(view.token_texts, view.segments, view.tokens))],
            "attention": [[[round(float(v), digits) for v in row] for row in head] for head in attention.weights.tolist()],
            "truncated": view.truncated}


def report_data(pair) -> dict:
    d = pair.decomposition
    matrices = {
        "contribution": d.pair_contribution("final"), "routing_contribution": d.routing_pair_contribution,
        "value_contribution": d.value_pair_contribution, "routing_weight": d.routing_weight,
        "value_attention": d.value_attention, "similarity": d.similarity}
    top = d.top_pairs(1)[0]
    return {"slots": {"query": pair.query.num_slots, "document": pair.document.num_slots},
            "heads": pair.document.num_heads,
            "query": _side(pair.query), "document": _side(pair.document),
            "contributions": {"query": {c: [float(v) for v in d.query_slot_contribution(c)] for c in COMPONENTS},
                              "document": {c: [float(v) for v in d.document_slot_contribution(c)] for c in COMPONENTS}},
            "matrices": {name: {"label": MATRICES[name][0], "diverging": MATRICES[name][1], "note": MATRICES[name][2],
                                "data": _rounded(m)} for name, m in matrices.items()},
            "defaults": {"query_slot": top["query_slot"], "document_slot": top["document_slot"]}}


def _table(rows: list[dict], columns: list[tuple[str, str]], text_columns=()) -> str:
    head = "".join(f'<th scope="col" class="{"text" if key in text_columns else ""}">{esc(label)}</th>'
                   for key, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            css = "text" if key in text_columns else ""
            if isinstance(value, float) and key not in text_columns:
                css += " pos" if value > 0 and key.endswith("contribution") else (
                    " neg" if value < 0 and key.endswith("contribution") else "")
            cells.append(f'<td class="{css.strip()}">{esc(fmt(value, 5) if not isinstance(value, str) else value)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _top_tokens_text(entries) -> str:
    return " · ".join(f"{(t['display'].strip() or repr(t['display']))} ({t['weight']:.2f})" for t in entries)


def render_pair_report(pair, *, title: str | None = None) -> str:
    d = pair.decomposition
    title = title or "BreadBowl pair report"
    css, script = _asset("report.css"), _asset("report.js")
    q_stats, d_stats = pair.query.statistics(), pair.document.statistics()
    q_top, d_top = pair.query.top_tokens(4), pair.document.top_tokens(4)
    q_rows = []
    for row, stat, top in zip(d.slot_table(), q_stats, q_top):
        q_rows.append({**row, "entropy": stat["entropy"], "coverage_90": stat["coverage_90"],
                       "instruction": stat["segment_mass"]["instruction"] + stat["segment_mass"]["template"],
                       "text": stat["segment_mass"]["text"], "top": _top_tokens_text(top)})
    doc_final, doc_routing, doc_value = (d.document_slot_contribution(c) for c in ("final", "routing", "value"))
    d_rows = [{"slot": s, "final_contribution": float(doc_final[s]), "routing_contribution": float(doc_routing[s]),
               "value_contribution": float(doc_value[s]), "value_norm": float(d.document_value_norm[s]),
               "entropy": stat["entropy"], "coverage_90": stat["coverage_90"], "top": _top_tokens_text(top)}
              for s, (stat, top) in enumerate(zip(d_stats, d_top))]
    pairs = d.top_pairs(12)
    words = sorted(pair.attribution("document").word_weights(), key=lambda w: -abs(w["weight"]))[:12]
    residual = d.residuals()
    model = pair.model
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    unattributed = d.unattributed("final")
    query_view, doc_view = pair.query.tokens, pair.document.tokens
    instruction = (f'<p><span class="label-chip">instruction</span>{esc(query_view.instruction)}</p>'
                   if query_view.instruction is not None else "")
    static_tokens = (
        "<h3>Query, shaded by attention-weighted attribution of the final score</h3>"
        + tokens_html(query_view, pair.attribution("query").weights, diverging=True, label="attribution")
        + "<h3>Document, shaded by attention-weighted attribution of the final score</h3>"
        + tokens_html(doc_view, pair.attribution("document").weights, diverging=True, label="attribution"))
    matrix_options = "".join(f'<option value="{esc(k)}">{esc(v[0])}</option>' for k, v in MATRICES.items())
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="breadbowl-embed {esc(__version__)}">
<title>{esc(title)}</title>
<style>{css}</style>
</head>
<body>
<header>
<div class="topbar">
<div>
<h1>{esc(title)}</h1>
<div class="meta">model {esc(model["representation_id"])} · release {esc(model.get("release_id"))} · compute
{esc(model["compute_dtype"])} on {esc(model["device"])} · rule {esc(d.rule.name)} (tau_k {fmt(d.rule.tau_k, 3)}, tau_v
{fmt(d.rule.tau_v, 3)}, value_mix {fmt(d.rule.value_mix, 3)}, {esc(d.rule.qk_mode)}) · breadbowl-embed
{esc(__version__)} · {esc(generated)}</div>
</div>
<button type="button" id="theme" aria-pressed="false">Dark theme</button>
</div>
</header>
<main>
<section class="card" aria-labelledby="h-scores">
<h2 id="h-scores">Score</h2>
<div class="scores">
<div class="score"><div class="label">final</div><div class="value">{fmt(d.final, 6)}</div></div>
<div class="score"><div class="label">routing R</div><div class="value">{fmt(d.routing, 6)}</div></div>
<div class="score"><div class="label">value V</div><div class="value">{fmt(d.value, 6)}</div></div>
</div>
<p class="formula">final = {fmt(1 - d.value_mix, 2)} × R + {fmt(d.value_mix, 2)} × V. The slot-pair contributions
sum to {fmt(float(d.pair_contribution("final").sum()), 6)}; the routing smoothing terms add {fmt(unattributed, 6)}
(they belong to query slots, not to document slots). Recombination residuals (score − parts): final
{fmt(residual["final"], 2)}, R {fmt(residual["routing"], 2)}, V {fmt(residual["value"], 2)}.</p>
</section>

<section class="card" aria-labelledby="h-texts">
<h2 id="h-texts">Texts</h2>
<div class="texts">
<div><h3>Query ({query_view.token_count} tokens{", truncated" if query_view.truncated else ""})</h3>{instruction}
<div class="textbox">{esc(query_view.text)}</div></div>
<div><h3>Document ({doc_view.token_count} tokens{", truncated" if doc_view.truncated else ""})</h3>
<div class="textbox">{esc(doc_view.text)}</div></div>
</div>
</section>

<section class="card" aria-labelledby="h-tokens">
<h2 id="h-tokens">Tokens</h2>
<div id="controls-js" class="controls" hidden>
<label>Show <select id="view">
<option value="attribution-final">attribution of the final score</option>
<option value="attribution-routing">attribution of R</option>
<option value="attribution-value">attribution of V</option>
<option value="attention">slot attention of the selected slot</option>
</select></label>
<label>Heads <select id="head"><option value="mean">mean of heads</option></select></label>
<label><input type="checkbox" id="only-slot"> attribution through the selected slot only</label>
</div>
<div id="interactive" class="panels" hidden>
<div class="panel">
<h3>Query slots</h3>
<div id="query-slots" class="slotpicker" role="radiogroup" aria-label="Query slot"></div>
<div id="query-legend" class="legend"></div>
<div id="query-tokens" class="tokens" tabindex="0" role="group"
 aria-label="Query tokens. Use the left and right arrow keys to read token values."></div>
<div id="query-readout" class="readout" aria-live="polite"></div>
</div>
<div class="panel">
<h3>Document slots</h3>
<div id="document-slots" class="slotpicker" role="radiogroup" aria-label="Document slot"></div>
<div id="document-legend" class="legend"></div>
<div id="document-tokens" class="tokens" tabindex="0" role="group"
 aria-label="Document tokens. Use the left and right arrow keys to read token values."></div>
<div id="document-readout" class="readout" aria-live="polite"></div>
</div>
</div>
<noscript>{static_tokens}<p class="meta">Enable JavaScript for per-slot and per-head views.</p></noscript>
<p class="meta">Dashed underline: query instruction and template tokens. Attribution: blue raises the score, red lowers
it. Selecting a cell in the slot-pair matrix selects both slots.</p>
</section>

<section class="card" aria-labelledby="h-matrix">
<h2 id="h-matrix">Slot pairs</h2>
<div class="controls"><label>Matrix <select id="matrix-kind">{matrix_options}</select></label></div>
<div id="matrix-legend" class="legend"></div>
<div id="matrix-focus" class="matrix-wrap" tabindex="0" role="group"
 aria-label="Query slot by document slot matrix. Arrow keys move, Enter selects the pair.">
<svg id="matrix" class="matrix" aria-hidden="true"></svg>
</div>
<div id="matrix-readout" class="readout" aria-live="polite"></div>
<p id="matrix-note" class="meta"></p>
<h3>Largest slot-pair contributions to the final score</h3>
{_table(pairs, [("query_slot", "query slot"), ("document_slot", "document slot"), ("contribution", "contribution"),
                ("similarity", "cosine"), ("routing_weight", "routing weight"), ("value_attention", "value attention")])}
</section>

<section class="card" aria-labelledby="h-qslots">
<h2 id="h-qslots">Query slots</h2>
{_table(q_rows, [("query_slot", "slot"), ("final_contribution", "final contribution"),
                 ("routing_contribution", "R contribution"), ("routing_entropy_term", "smoothing term"),
                 ("value_slot_cosine", "value cosine"), ("best_document_slot", "best doc slot"),
                 ("best_cosine", "best cosine"), ("instruction", "instruction+template mass"),
                 ("text", "query-text mass"), ("entropy", "attention entropy"), ("coverage_90", "tokens for 90%"),
                 ("top", "most attended tokens")], text_columns=("top",))}
</section>

<section class="card" aria-labelledby="h-dslots">
<h2 id="h-dslots">Document slots</h2>
{_table(d_rows, [("slot", "slot"), ("final_contribution", "final contribution"),
                 ("routing_contribution", "R contribution"), ("value_contribution", "V contribution"),
                 ("value_norm", "value norm"), ("entropy", "attention entropy"), ("coverage_90", "tokens for 90%"),
                 ("top", "most attended tokens")], text_columns=("top",))}
<h3>Document words carrying the most attribution (final score)</h3>
{_table(words, [("word", "word"), ("start", "start"), ("end", "end"), ("weight", "attribution")],
        text_columns=("word",))}
</section>

<section class="card caveats" aria-labelledby="h-caveats">
<h2 id="h-caveats">How to read this</h2>
<ul>{"".join(f"<li>{esc(c)}</li>" for c in CAVEATS)}</ul>
</section>
</main>
<footer>Generated by breadbowl-embed {esc(__version__)} (breadbowl_embed.research, experimental API).</footer>
<script type="application/json" id="report-data">{_script_json(report_data(pair))}</script>
<script>{script}</script>
</body>
</html>
"""

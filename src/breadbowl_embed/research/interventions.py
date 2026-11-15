"""Causal checks: remove slots, occlude text, and rescore.

Slot ablation works on stored representations and re-runs only the scorer:
  mode "drop"  removes the slot, so log-mean-exp, the value softmax and the mean over query slots
               run over the remaining slots;
  mode "zero"  keeps the slot but zeroes its routing and value vectors (cosine 0 with everything,
               no value read from it).

Occlusion re-runs the model on one side of a pair:
  mode "delete"     removes the characters of the span(s) from the text and re-encodes it;
  mode "replace"    replaces them with `replacement` and re-encodes;
  mode "slot-mask"  keeps the text and the backbone's token states, and hides the span's tokens
                    from the slot extractor only (attention mask), so the slots cannot read them.
                    Cheap: the backbone runs once; only the extractor and heads re-run per unit.
"delete"/"replace" change the backbone's context as well (other tokens' states change; a causal
backbone also shifts positions); "slot-mask" isolates the pooling step. Neither is the effect
of the information in the span alone. Scores are compared with the unmodified pair under the
same scoring rule; the model is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import torch
import torch.nn.functional as F

from ..embeddings import Embeddings
from ..scoring import Scores, score
from ._base import ResearchObject
from ._metrics import kendall_tau, ranking
from .capture import run_heads, trace
from .rules import ScoringRule, base_config, resolve_rule

ABLATION_MODES = ("drop", "zero")
OCCLUSION_MODES = ("delete", "replace", "slot-mask")
UNITS = ("word", "token", "sentence")
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)|\n+")


def _scores_dict(scores: Scores, row: int = 0) -> dict[str, torch.Tensor]:
    return {"final": scores.combined[row], "routing": scores.routing[row], "value": scores.value[row]}


def _without_slots(embeddings: Embeddings, slots, mode: str) -> Embeddings:
    slots = sorted({int(s) % embeddings.num_slots for s in slots})
    if not slots:
        return embeddings.without_tokens()
    if mode == "drop":
        keep = [s for s in range(embeddings.num_slots) if s not in slots]
        if not keep:
            raise ValueError("Cannot drop every slot.")
        return Embeddings(embeddings.routing[:, keep], embeddings.value[:, keep], embeddings.representation_id,
                          embeddings.kind)
    if mode == "zero":
        routing, value = embeddings.routing.clone(), embeddings.value.clone()
        routing[:, slots] = 0
        value[:, slots] = 0
        return Embeddings(routing, value, embeddings.representation_id, embeddings.kind)
    raise ValueError(f"mode must be one of {ABLATION_MODES}.")


def ablate_slots(queries: Embeddings, documents: Embeddings, *, query_slots=(), document_slots=(),
                 mode: str = "drop", rule=None, base=None) -> Scores:
    """Scores with the given query slots and/or document slots removed (every query/document)."""
    config = resolve_rule(rule, base_config(base)).config()
    return score(_without_slots(queries, query_slots, mode), _without_slots(documents, document_slots, mode),
                 config=config)


@dataclass(frozen=True, eq=False)
class SlotAblation(ResearchObject):
    """One query against a set of documents, with each single slot ablated in turn.

    baseline[c]:        [documents] scores of component c ("final", "routing", "value").
    query_slot[c]:      [query slots, documents] scores with that query slot ablated.
    document_slot[c]:   [document slots, documents] scores with that document slot ablated in
                        every document.
    `effects()` summarises each ablation: mean and max |delta| of the final score and, with
    several documents, whether the top document changes and Kendall's tau with the baseline order.
    """
    json_type = "slot_ablation"

    query_index: int
    mode: str
    rule: ScoringRule
    baseline: dict
    query_slot: dict
    document_slot: dict

    def delta(self, side: str = "query", component: str = "final") -> torch.Tensor:
        """[slots, documents] ablated minus baseline."""
        table = self.query_slot if side == "query" else self.document_slot
        return table[component] - self.baseline[component][None]

    def effects(self) -> list[dict]:
        base = self.baseline["final"]
        base_rank = ranking(base)
        rows = []
        for side, table in (("query", self.query_slot), ("document", self.document_slot)):
            for slot, scores in enumerate(table["final"]):
                delta = (scores - base).double()
                rows.append({"side": side, "slot": slot, "mean_delta": float(delta.mean()),
                             "max_abs_delta": float(delta.abs().max()),
                             "top1_changed": ranking(scores)[0] != base_rank[0] if len(base) > 1 else False,
                             "kendall_tau": kendall_tau(base, scores)})
        return rows

    def _payload(self) -> dict:
        return {"query_index": self.query_index, "mode": self.mode, "rule": self.rule.to_dict(),
                "baseline": self.baseline, "query_slot": self.query_slot, "document_slot": self.document_slot,
                "effects": self.effects()}

    def _html(self) -> str:
        from ._html import table_html
        return table_html(f"Slot ablation ({self.mode})", self.effects(),
                          note="Change of the final score when one slot is ablated (scorer re-run only).")


def slot_ablation(queries: Embeddings, documents: Embeddings, *, query: int = 0, mode: str = "drop", rule=None,
                  base=None) -> SlotAblation:
    """Ablate every query slot and every document slot in turn for one query (see SlotAblation)."""
    if mode not in ABLATION_MODES:
        raise ValueError(f"mode must be one of {ABLATION_MODES}.")
    resolved = resolve_rule(rule, base_config(base))
    config = resolved.config()
    q = queries.select([query]).without_tokens()
    d = documents.without_tokens()
    baseline = _scores_dict(score(q, d, config=config))
    per_query = [_scores_dict(score(_without_slots(q, [s], mode), d, config=config)) for s in range(q.num_slots)]
    per_document = [_scores_dict(score(q, _without_slots(d, [s], mode), config=config)) for s in range(d.num_slots)]
    stack = lambda rows: {c: torch.stack([r[c] for r in rows]) for c in ("final", "routing", "value")}  # noqa: E731
    return SlotAblation(query, mode, resolved, baseline, stack(per_query), stack(per_document))


# ---- occlusion -----------------------------------------------------------------------------------
def _merge_spans(spans, length: int) -> list[tuple[int, int]]:
    cleaned = []
    for span in spans:
        start, end = int(span[0]), int(span[1])
        if not 0 <= start < end <= length:
            raise ValueError(f"Span ({start}, {end}) is empty or outside the text (length {length}).")
        cleaned.append((start, end))
    merged: list[list[int]] = []
    for start, end in sorted(cleaned):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def edit_text(text: str, spans, replacement: str = "") -> str:
    """`text` with every span (merged if overlapping) replaced by `replacement`."""
    out, position = [], 0
    for start, end in _merge_spans(spans, len(text)):
        out.append(text[position:start])
        out.append(replacement)
        position = end
    out.append(text[position:])
    return "".join(out)


class _Occluder:
    """Shared state for occluding one side of a pair: the other side's encoding and the baseline."""

    def __init__(self, model, query: str, document: str, side: str, instruction, rule):
        if side not in ("query", "document"):
            raise ValueError("side must be 'query' or 'document'.")
        self.model, self.side, self.instruction = model, side, instruction
        self.rule = resolve_rule(rule, base_config(model)) if rule is not None else ScoringRule.from_config(model.scoring)
        self.config = self.rule.config()
        self.texts = {"query": query, "document": document}
        traced = trace(model, [self.texts[side]], side, instruction=instruction if side == "query" else None,
                       capture=("hidden",))
        self.trace = traced[0]
        self.encoded = traced.encoding().without_tokens()
        other = "document" if side == "query" else "query"
        if other == "query":
            self.other = model.encode_queries([query], instruction=instruction)
        else:
            self.other = model.encode_documents([document])
        self.baseline = self._score(self.encoded)

    @property
    def view(self):
        return self.trace.tokens

    def _score(self, encoded: Embeddings) -> dict[str, float]:
        pair = (encoded, self.other) if self.side == "query" else (self.other, encoded)
        scores = score(*pair, config=self.config)
        return {"final": float(scores.combined[0, 0]), "routing": float(scores.routing[0, 0]),
                "value": float(scores.value[0, 0])}

    @torch.inference_mode()
    def slot_mask(self, token_indices) -> dict[str, float]:
        network = self.model.network
        weight = network.routing_head.weight
        hidden = self.trace.hidden_states.to(device=weight.device, dtype=weight.dtype)[None]
        mask = torch.ones(1, hidden.shape[1], dtype=torch.long, device=weight.device)
        mask[0, list(token_indices)] = 0
        if not bool(mask.any()):
            raise ValueError("Cannot hide every token from the slot extractor.")
        routing, value, _ = run_heads(network, hidden, mask)
        encoded = Embeddings(F.normalize(routing.float(), dim=-1).cpu(), value.float().cpu(),
                             self.model.representation_id, self.side)
        return self._score(encoded)

    def reencode(self, text: str) -> dict[str, float]:
        if self.side == "query":
            encoded = self.model.encode_queries([text], instruction=self.instruction)
        else:
            encoded = self.model.encode_documents([text])
        return self._score(encoded)


@dataclass(frozen=True, eq=False)
class Occlusion(ResearchObject):
    """Scores of a pair before and after occluding span(s) of one side."""
    json_type = "occlusion"

    side: str
    mode: str
    spans: tuple
    occluded_text: str | None
    masked_tokens: tuple
    baseline: dict
    occluded: dict
    rule: ScoringRule

    @property
    def delta(self) -> dict[str, float]:
        return {k: self.occluded[k] - self.baseline[k] for k in self.baseline}

    def _payload(self) -> dict:
        return {"side": self.side, "mode": self.mode, "spans": self.spans, "occluded_text": self.occluded_text,
                "masked_tokens": self.masked_tokens, "baseline": self.baseline, "occluded": self.occluded,
                "delta": self.delta, "rule": self.rule.to_dict()}

    def _html(self) -> str:
        from ._html import table_html
        rows = [{"component": k, "baseline": self.baseline[k], "occluded": self.occluded[k], "delta": self.delta[k]}
                for k in self.baseline]
        return table_html(f"Occlusion ({self.mode}, {self.side})", rows)


def occlude(model, query: str, document: str, spans=None, *, side: str = "document", mode: str = "delete",
            replacement: str = "", tokens=None, instruction: str | None = None, rule=None) -> Occlusion:
    """Occlude character span(s) of one side of a pair and rescore (see the module docstring).

    spans: [(start, end), ...] character offsets into the query text (not its instruction) or the
    document text. tokens: token indices to hide instead (slot-mask mode only; can address the
    query instruction, e.g. from `TokenView.segment_mask("instruction")`).
    """
    if mode not in OCCLUSION_MODES:
        raise ValueError(f"mode must be one of {OCCLUSION_MODES}.")
    if (spans is None) == (tokens is None):
        raise ValueError("Give either spans or tokens.")
    if tokens is not None and mode != "slot-mask":
        raise ValueError("Token indices apply to the slot-mask mode only.")
    occluder = _Occluder(model, query, document, side, instruction, rule)
    text = occluder.texts[side]
    if mode == "slot-mask":
        if tokens is None:
            merged = _merge_spans(spans, len(text))
            tokens = sorted({t for s, e in merged for t in occluder.view.tokens_in_span(s, e)})
            spans_out = tuple(merged)
        else:
            tokens = sorted({int(t) % len(occluder.view) for t in tokens})
            spans_out = ()
        if not tokens:
            raise ValueError("The span(s) cover no model token (truncated away?).")
        return Occlusion(side, mode, spans_out, None, tuple(tokens), occluder.baseline, occluder.slot_mask(tokens),
                         occluder.rule)
    merged = _merge_spans(spans, len(text))
    edited = edit_text(text, merged, replacement if mode == "replace" else "")
    if not edited.strip():
        raise ValueError("Occlusion would leave an empty text.")
    return Occlusion(side, mode, tuple(merged), edited, (), occluder.baseline, occluder.reencode(edited),
                     occluder.rule)


@dataclass(frozen=True, eq=False)
class OcclusionSweep(ResearchObject):
    """Occlude each unit (word, token or sentence) of one side in turn.

    units: [{index, start, end, text, tokens}]; scores[c]: [units] component c after occluding
    that unit; delta = scores - baseline (negative: the unit supported the score).
    """
    json_type = "occlusion_sweep"

    side: str
    unit: str
    mode: str
    tokens: object  # TokenView
    units: tuple
    baseline: dict
    scores: dict
    rule: ScoringRule

    def delta(self, component: str = "final") -> torch.Tensor:
        return self.scores[component] - self.baseline[component]

    def top(self, k: int = 10, component: str = "final") -> list[dict]:
        """Units whose occlusion lowered the score most."""
        delta = self.delta(component)
        order = torch.argsort(delta, stable=True)[:k].tolist()
        return [{**self.units[i], "delta": float(delta[i])} for i in order]

    def token_delta(self, component: str = "final") -> torch.Tensor:
        """[tokens] each token's unit delta (0 for tokens in no unit), for heatmaps."""
        out = torch.zeros(len(self.tokens), dtype=torch.float64)
        for unit, delta in zip(self.units, self.delta(component).tolist()):
            for t in unit["tokens"]:
                out[t] = delta
        return out

    def _payload(self) -> dict:
        return {"side": self.side, "unit": self.unit, "mode": self.mode, "rule": self.rule.to_dict(),
                "baseline": self.baseline, "units": [{**u, "delta": {c: float(self.delta(c)[i]) for c in self.scores}}
                                                     for i, u in enumerate(self.units)],
                "scores": self.scores}

    def _html(self) -> str:
        from ._html import sweep_html
        return sweep_html(self)

    def plot(self, **kwargs):
        from .plot import occlusion
        return occlusion(self, **kwargs)


def _units(view, unit: str) -> list[dict]:
    if unit == "word":
        spans = [(w.start, w.end) for w in view.words]
    elif unit == "sentence":
        spans = [(m.start(), m.end()) for m in _SENTENCE.finditer(view.text) if m.group().strip()]
    elif unit == "token":
        spans = sorted({o for o in view.offsets if o is not None})
    else:
        raise ValueError(f"unit must be one of {UNITS}.")
    out = []
    for start, end in spans:
        tokens = view.tokens_in_span(start, end)
        if tokens:  # words cut off by truncation have no tokens
            out.append({"index": len(out), "start": start, "end": end, "text": view.text[start:end],
                        "tokens": tokens})
    return out


def occlusion_sweep(model, query: str, document: str, *, side: str = "document", unit: str = "word",
                    mode: str = "slot-mask", replacement: str = "", instruction: str | None = None, rule=None,
                    limit: int | None = None) -> OcclusionSweep:
    """Occlude every word/token/sentence of one side in turn (see OcclusionSweep).

    mode "slot-mask" (default) runs the backbone once; "delete"/"replace" re-encode the text once
    per unit (on CPU about one document forward pass each). limit: occlude only the first units.
    """
    if mode not in OCCLUSION_MODES:
        raise ValueError(f"mode must be one of {OCCLUSION_MODES}.")
    occluder = _Occluder(model, query, document, side, instruction, rule)
    units = _units(occluder.view, unit)[:limit]
    text = occluder.texts[side]
    rows = []
    undefined = {c: float("nan") for c in occluder.baseline}  # the unit is the whole text
    for u in units:
        if mode == "slot-mask":
            hides_all = len(u["tokens"]) == len(occluder.view)
            rows.append(undefined if hides_all else occluder.slot_mask(u["tokens"]))
        else:
            edited = edit_text(text, [(u["start"], u["end"])], replacement if mode == "replace" else "")
            rows.append(occluder.reencode(edited) if edited.strip() else undefined)
    scores = {c: torch.tensor([r[c] for r in rows], dtype=torch.float64) for c in ("final", "routing", "value")}
    return OcclusionSweep(side, unit, mode, occluder.view, tuple(units), occluder.baseline, scores, occluder.rule)

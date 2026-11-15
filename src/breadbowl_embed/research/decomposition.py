"""Exact decomposition of a query-document score into slot pairs, and token attribution built on it.

With S_q query slots, S_d document slots, routing cosines s_ij, temperatures tau_k, tau_v and
value mix m (model defaults 0.02, 0.1, 0.25):

Routing R (log-mean-exp). With p_ij = softmax_j(s_ij / tau_k) and H_i the entropy of row i,

    tau_k * log mean_j exp(s_ij / tau_k) = sum_j p_ij s_ij + tau_k (H_i - log S_d)

exactly, so R = sum_ij w_ij s_ij + sum_i e_i, with pair weights w_ij = p_ij / S_q (they sum to
1 over all pairs; w_ij is also dR/ds_ij) and per-query-slot smoothing terms
e_i = tau_k (H_i - log S_d) / S_q <= 0. With qk_mode "max", w is one-hot and e = 0.

Value V. With attention A_ij = softmax_j(s_ij / tau_v), readout r_i = sum_j A_ij v_j over the raw
document values and the unit query value u_i, cos(u_i, r_i) = sum_j A_ij (u_i . v_j) / |r_i|, so
V = sum_ij A_ij (u_i . v_j) / (|r_i| S_q): an exact sum over slot pairs at the fixed readout norm.

Final = (1 - m) R + m V = sum_ij c_ij + (1 - m) sum_i e_i with pair contributions
c_ij = (1 - m) w_ij s_ij + m A_ij (u_i . v_j) / (|r_i| S_q).

`routing`, `value` and `final` of a ScoreDecomposition are the numbers `score` returns for the
same inputs and settings (one `score` call over all pairs); the parts are computed in float64
and recombine to them up to float32 rounding (`residuals()`, ~1e-7).

Token attribution distributes a slot's contribution over the tokens that slot was pooled from,
in proportion to its slot-extraction attention. Because attention rows sum to 1 the token
weights add up to the slot contributions; it is attention-weighted attribution, a bookkeeping
of the score along the model's own pooling weights, not a causal claim. Use the interventions
module (occlusion, slot ablation) to test causal effects.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ..embeddings import Embeddings
from ..scoring import score, validate_pair
from ._base import ResearchObject
from .attention import AttentionMap
from .rules import ScoringRule, base_config, resolve_rule

COMPONENTS = ("final", "routing", "value")


def _promote(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


@dataclass(frozen=True, eq=False)
class ScoreDecomposition(ResearchObject):
    """How one (query, document) score splits into slot pairs. Tensors are float64.

    similarity               [S_q, S_d] routing cosines s_ij.
    routing_weight           [S_q, S_d] w_ij: the weight each slot pair carries in R (sums to 1).
    routing_pair_contribution [S_q, S_d] w_ij s_ij.
    routing_entropy_term     [S_q] e_i, the log-mean-exp smoothing term of each query slot.
    value_attention          [S_q, S_d] A_ij (rows sum to 1).
    value_alignment          [S_q, S_d] u_i . v_j: raw document value j projected on query value i.
    value_readout_norm       [S_q] |r_i|.
    value_pair_contribution  [S_q, S_d] A_ij (u_i . v_j) / (|r_i| S_q).
    value_slot_cosine        [S_q] cos(u_i, r_i); their mean is V.
    document_value_norm      [S_d] |v_j|.
    routing, value, final    the scores `score` returns for this pair (floats).
    """
    json_type = "score_decomposition"

    query_index: int
    document_index: int
    rule: ScoringRule
    similarity: torch.Tensor
    routing_weight: torch.Tensor
    routing_pair_contribution: torch.Tensor
    routing_entropy_term: torch.Tensor
    value_attention: torch.Tensor
    value_alignment: torch.Tensor
    value_readout_norm: torch.Tensor
    value_pair_contribution: torch.Tensor
    value_slot_cosine: torch.Tensor
    document_value_norm: torch.Tensor
    routing: float
    value: float
    final: float

    @property
    def value_mix(self) -> float:
        return self.rule.value_mix

    @property
    def score(self) -> float:
        return self.final

    def pair_contribution(self, component: str = "final") -> torch.Tensor:
        """[S_q, S_d] contribution of each slot pair to the component ("final", "routing" or "value")."""
        if component == "routing":
            return self.routing_pair_contribution
        if component == "value":
            return self.value_pair_contribution
        if component == "final":
            return (1 - self.value_mix) * self.routing_pair_contribution + self.value_mix * self.value_pair_contribution
        raise ValueError(f"component must be one of {COMPONENTS}.")

    def unattributed(self, component: str = "final") -> float:
        """The part of the component not carried by any slot pair (the routing smoothing terms)."""
        entropy = float(self.routing_entropy_term.sum())
        return {"routing": entropy, "value": 0.0, "final": (1 - self.value_mix) * entropy}[component]

    def total(self, component: str = "final") -> float:
        return {"routing": self.routing, "value": self.value, "final": self.final}[component]

    def query_slot_contribution(self, component: str = "final") -> torch.Tensor:
        """[S_q] each query slot's share of the component, smoothing term included (sums to it)."""
        out = self.pair_contribution(component).sum(dim=1)
        if component == "routing":
            out = out + self.routing_entropy_term
        elif component == "final":
            out = out + (1 - self.value_mix) * self.routing_entropy_term
        return out

    def document_slot_contribution(self, component: str = "final") -> torch.Tensor:
        """[S_d] each document slot's share (pairs only; add `unattributed` to reach the total)."""
        return self.pair_contribution(component).sum(dim=0)

    def recombined(self, component: str = "final") -> float:
        return float(self.pair_contribution(component).sum()) + self.unattributed(component)

    def residuals(self) -> dict[str, float]:
        """total - recombined parts, per component (float32 rounding of `score`, ~1e-7)."""
        return {c: self.total(c) - self.recombined(c) for c in COMPONENTS}

    def top_pairs(self, k: int = 10, component: str = "final") -> list[dict]:
        """The k slot pairs with the largest absolute contribution to the component."""
        contribution = self.pair_contribution(component)
        order = torch.argsort(contribution.abs().flatten(), descending=True, stable=True)[:k].tolist()
        columns = contribution.shape[1]
        return [{"query_slot": i // columns, "document_slot": i % columns,
                 "contribution": float(contribution.flatten()[i]),
                 "similarity": float(self.similarity.flatten()[i]),
                 "routing_weight": float(self.routing_weight.flatten()[i]),
                 "value_attention": float(self.value_attention.flatten()[i])} for i in order]

    def slot_table(self) -> list[dict]:
        """One row per query slot: its terms of R and V and where it reads in the document."""
        best = self.similarity.argmax(dim=1)
        final = self.query_slot_contribution("final")
        rows = []
        for i in range(self.similarity.shape[0]):
            j = int(best[i])
            rows.append({"query_slot": i, "best_document_slot": j, "best_cosine": float(self.similarity[i, j]),
                         "routing_contribution": float(self.query_slot_contribution("routing")[i]),
                         "routing_entropy_term": float(self.routing_entropy_term[i]),
                         "value_slot_cosine": float(self.value_slot_cosine[i]),
                         "value_contribution": float(self.query_slot_contribution("value")[i]),
                         "value_readout_norm": float(self.value_readout_norm[i]),
                         "final_contribution": float(final[i])})
        return rows

    def _payload(self) -> dict:
        return {"query_index": self.query_index, "document_index": self.document_index, "rule": self.rule.to_dict(),
                "final": self.final, "routing": self.routing, "value": self.value, "residuals": self.residuals(),
                "unattributed": {c: self.unattributed(c) for c in COMPONENTS},
                "similarity": self.similarity, "routing_weight": self.routing_weight,
                "routing_pair_contribution": self.routing_pair_contribution,
                "routing_entropy_term": self.routing_entropy_term, "value_attention": self.value_attention,
                "value_alignment": self.value_alignment, "value_readout_norm": self.value_readout_norm,
                "value_pair_contribution": self.value_pair_contribution, "value_slot_cosine": self.value_slot_cosine,
                "document_value_norm": self.document_value_norm,
                "query_slot_contribution": {c: self.query_slot_contribution(c) for c in COMPONENTS},
                "document_slot_contribution": {c: self.document_slot_contribution(c) for c in COMPONENTS},
                "slots": self.slot_table()}

    def _html(self) -> str:
        from ._html import decomposition_html
        return decomposition_html(self)

    def plot(self, **kwargs):
        """Slot-pair heatmap (needs the `viz` extra); see research.plot.slot_pairs."""
        from .plot import slot_pairs
        return slot_pairs(self, **kwargs)


def _pair(q: torch.Tensor, d: torch.Tensor, qv: torch.Tensor, dv: torch.Tensor, rule: ScoringRule) -> dict:
    """Float64 parts of one pair; q, d already unit (float32 normalisation, as the scorer)."""
    q, d, qv, dv = q.double(), d.double(), qv.double(), dv.double()
    rows, columns = q.shape[0], d.shape[0]
    sim = q @ d.T
    if rule.qk_mode == "max":
        p = F.one_hot(sim.argmax(dim=1), columns).double()
        entropy_term = torch.zeros(rows, dtype=torch.float64)
    else:
        p = torch.softmax(sim / rule.tau_k, dim=1)
        entropy = -torch.xlogy(p, p).sum(dim=1)
        entropy_term = rule.tau_k * (entropy - math.log(columns)) / rows
    weight = p / rows
    attention = torch.softmax(sim / rule.tau_v, dim=1)
    unit_qv = F.normalize(qv, p=2, dim=-1)
    readout = attention @ dv
    readout_norm = readout.norm(dim=-1)
    alignment = unit_qv @ dv.T
    safe = torch.where(readout_norm > 0, readout_norm, torch.ones_like(readout_norm))
    value_pair = torch.where(readout_norm[:, None] > 0, attention * alignment / (safe[:, None] * rows),
                             torch.zeros_like(attention))
    cosine = (unit_qv * F.normalize(readout, p=2, dim=-1)).sum(-1)
    return {"similarity": sim, "routing_weight": weight, "routing_pair_contribution": weight * sim,
            "routing_entropy_term": entropy_term, "value_attention": attention, "value_alignment": alignment,
            "value_readout_norm": readout_norm, "value_pair_contribution": value_pair,
            "value_slot_cosine": cosine, "document_value_norm": dv.norm(dim=-1)}


def decompose(queries: Embeddings, documents: Embeddings, *, rule=None, base=None) -> list[list[ScoreDecomposition]]:
    """Decompose every (query, document) score: result[i][j] (see the module docstring).

    rule: None (the base settings), a preset name ("routing-only", "maxsim", ...), a ScoringRule or
    a ScoringConfig. base: a BreadBowl model or ScoringConfig (default: library defaults; pass the
    model to use its settings). Memory and time grow with queries x documents x slots^2: meant for
    inspecting a few pairs.
    """
    validate_pair(queries, documents)
    resolved = resolve_rule(rule, base_config(base))
    totals = score(queries, documents, config=resolved.config())
    with torch.inference_mode():
        q = F.normalize(_promote(queries.routing), p=2, dim=-1).cpu()
        d = F.normalize(_promote(documents.routing), p=2, dim=-1).cpu()
        qv, dv = _promote(queries.value).cpu(), _promote(documents.value).cpu()
        out = []
        for i in range(len(queries)):
            row = []
            for j in range(len(documents)):
                parts = _pair(q[i], d[j], qv[i], dv[j], resolved)
                row.append(ScoreDecomposition(i, j, resolved, routing=float(totals.routing[i, j]),
                                              value=float(totals.value[i, j]), final=float(totals.combined[i, j]),
                                              **parts))
            out.append(row)
    return out


@dataclass(frozen=True, eq=False)
class TokenAttribution(ResearchObject):
    """Attention-weighted attribution of a score component to the tokens of one side.

    by_slot[s, t] = (slot s's contribution) x (slot s's attention on token t, the row renormalised
    to sum to 1 in float64); weights = sum over slots. weights.sum() + unattributed == total (up
    to the decomposition's residual). This follows the model's own pooling weights; it is not a
    causal attribution.
    """
    json_type = "token_attribution"

    side: str
    component: str
    tokens: object  # TokenView
    slot_contribution: torch.Tensor
    by_slot: torch.Tensor
    weights: torch.Tensor
    total: float
    unattributed: float
    head: int | None
    note: str = "attention-weighted attribution (slot contribution x slot-extraction attention); not causal"

    def top_tokens(self, k: int = 10, *, sign: str = "abs") -> list[dict]:
        """The k tokens with the largest |weight| (sign="abs"), or largest/smallest ("positive"/"negative")."""
        key = {"abs": self.weights.abs(), "positive": self.weights, "negative": -self.weights}[sign]
        order = torch.argsort(key, descending=True, stable=True)[:k].tolist()
        return [{**self.tokens.describe_token(t), "weight": float(self.weights[t]),
                 "top_slot": int(self.by_slot[:, t].abs().argmax())} for t in order]

    def word_weights(self) -> list[dict]:
        """Attribution summed per word of the caller's text (as AttentionMap.word_matrix)."""
        sums = [0.0] * len(self.tokens.words)
        for t, word in enumerate(self.tokens.token_words):
            if word is not None:
                sums[word] += float(self.weights[t])
        return [{"index": w.index, "word": w.text, "start": w.start, "end": w.end, "weight": sums[w.index]}
                for w in self.tokens.words]

    def segment_totals(self) -> dict[str, float]:
        """Attribution per segment (instruction, template, text, special)."""
        out: dict[str, float] = {}
        for segment, weight in zip(self.tokens.segments, self.weights.tolist()):
            out[segment] = out.get(segment, 0.0) + weight
        return out

    def span_total(self, start: int, end: int) -> float:
        """Attribution of the tokens overlapping characters [start, end) of the caller's text."""
        return float(self.weights[self.tokens.tokens_in_span(start, end)].sum())

    def _payload_compact(self) -> dict:
        """Without the per-slot matrix and the token view (for objects that already hold the tokens)."""
        return {"side": self.side, "component": self.component, "head": self.head, "note": self.note,
                "total": self.total, "unattributed": self.unattributed, "slot_contribution": self.slot_contribution,
                "weights": self.weights, "segments": self.segment_totals()}

    def _payload(self) -> dict:
        return {**self._payload_compact(), "by_slot": self.by_slot, "tokens": self.tokens,
                "words": self.word_weights()}

    def _html(self) -> str:
        from ._html import attribution_html
        return attribution_html(self)

    def plot(self, **kwargs):
        from .plot import token_bars
        return token_bars(self, **kwargs)


def attribute(decomposition: ScoreDecomposition, attention: AttentionMap, *, side: str = "document",
              component: str = "final", head: int | None = None) -> TokenAttribution:
    """Token attribution of one side of a decomposed pair (see TokenAttribution)."""
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}.")
    if side == "document":
        contribution = decomposition.document_slot_contribution(component)
        unattributed = decomposition.unattributed(component)
    elif side == "query":
        contribution = decomposition.query_slot_contribution(component)
        unattributed = 0.0
    else:
        raise ValueError("side must be 'query' or 'document'.")
    weights = attention.matrix(head).double()
    if weights.shape[0] != contribution.shape[0]:
        raise ValueError("The attention map and the decomposition have different slot counts.")
    # Rows sum to 1 only up to the compute dtype's rounding (~3e-3 in bfloat16); renormalise so the
    # token weights add up to the slot contributions exactly.
    weights = weights / weights.sum(dim=-1, keepdim=True)
    by_slot = contribution[:, None] * weights
    return TokenAttribution(side, component, attention.tokens, contribution, by_slot, by_slot.sum(0),
                            decomposition.total(component), unattributed, head)

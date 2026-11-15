"""Helpers that turn the raw model data into something you can show a person.

``token_attribution`` decomposes a pair's score exactly over query slots, slot pairs and tokens
using only what ``score``/``search`` return with ``explain=True`` and ``return_attention=True``:

* Per query slot ``i`` the score share is
  ``s[i] = (routing_weight · routing_contribution[i] + value_weight · value_slot_cosine[i]) / n_query_slots``,
  so ``Σ_i s[i] = score`` (up to float16 rounding of the tensors).
* ``s[i]`` is spread over document slots ``j`` by a row-stochastic coupling ``C[i, j]``:
  ``value`` uses ``value_attention`` (what the value readout reads), ``routing`` uses
  ``softmax_j(slot_similarity[i, j] / tau_k)`` (the weights of the log-mean-exp smooth maximum,
  i.e. its gradient), and ``score`` (default) splits each slot's routing part by the routing
  coupling and its value part by the value coupling. ``P[i, j]`` sums to ``s[i]`` per row.
* Slot→token attention rows sum to 1, so query token ``t`` gets ``Σ_i s[i] · A_q[i, t]`` and
  document token ``u`` gets ``Σ_j (Σ_i P[i, j]) · A_d[j, u]``. Each side sums to the total.

Weights can be negative when a slot's contribution is negative. Template and special tokens
receive weight too (the model attends to them); ``*_spans()`` keep only the caller's text.

Offsets and spans: a byte-level token that ends inside a multi-byte character reports the whole
character's offsets, as does the token that completes it, so neighbouring tokens' offsets can
overlap without being identical (``[4, 6)`` and ``[5, 6)``). A token's ``text`` is a display
string (the tokens' texts concatenate to the decoded text); its offsets are not. Everything here
that returns spans (``token_spans``, ``query_spans()``, ``document_spans()``) merges overlapping
offsets into one span (union range, summed weight, every token position), so spans never overlap
and, for a complete tokenization, tile the text. Highlight with those merged spans; per-token
weights stay available unmerged in ``query_tokens``/``document_tokens``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from ._models import Explanation, ScoringParameters, Token
from ._tensor import Tensor

__all__ = [
    "TokenWeight",
    "SpanWeight",
    "TokenAttribution",
    "TokenSpan",
    "token_attribution",
    "token_spans",
    "attention_matrix",
]

# The contract's documented defaults, used only when no ScoringParameters are passed.
_DEFAULT_SCORING = {"tau_k": 0.02, "tau_v": 0.1, "routing_weight": 0.75, "value_weight": 0.25}


@dataclass(frozen=True)
class TokenSpan:
    """A run of the caller's text covered by input tokens (offsets are code points).

    Tokens whose offsets overlap (a character split across byte-level tokens) are merged into
    one span, so spans never overlap. ``positions``/``tokens`` are the merged tokens."""

    start: int
    end: int
    text: str
    positions: Tuple[int, ...]
    tokens: Tuple[Token, ...]


def _merge(items: Iterable[Tuple[int, int, int]]) -> List[Tuple[int, int, List[int]]]:
    """Merge ``(start, end, position)`` ranges that overlap into ``(start, end, positions)``.

    Two ranges overlap when one starts before the other ends; identical ranges and a range nested
    in another (``[4, 6)`` and ``[5, 6)``) merge too. Touching ranges (``[0, 4)``, ``[4, 6)``) stay
    separate. The result is sorted, non-overlapping and keeps every position."""
    merged: List[Tuple[int, int, List[int]]] = []
    for start, end, position in sorted(items):
        if merged:
            m_start, m_end, positions = merged[-1]
            if start < m_end or (start == end and m_start <= start <= m_end):
                merged[-1] = (m_start, max(m_end, end), sorted(positions + [position]))
                continue
        merged.append((start, end, [position]))
    return merged


def _check_offsets(text: str, position: int, start: int, end: int) -> None:
    if not 0 <= start <= end <= len(text):
        raise ValueError(
            f"token {position} offsets [{start}, {end}) fall outside a text of {len(text)} code "
            "points; is this the exact string that was sent?"
        )


def token_spans(text: str, tokens: Sequence[Token]) -> List[TokenSpan]:
    """The input tokens' coverage of ``text`` as merged, non-overlapping spans in text order.

    Template and special tokens are skipped. Overlapping offsets (a character split across
    byte-level tokens) are merged, so the spans' ``text`` concatenates to ``text`` for a complete
    tokenization. Python strings index by code point, the API's unit, so slicing is direct; every
    offset is checked against ``text``."""
    items: List[Tuple[int, int, int]] = []
    for position, token in enumerate(tokens):
        if token.start is None or token.end is None:
            continue
        _check_offsets(text, position, token.start, token.end)
        items.append((token.start, token.end, position))
    return [
        TokenSpan(start, end, text[start:end], tuple(positions), tuple(tokens[p] for p in positions))
        for start, end, positions in _merge(items)
    ]


@dataclass(frozen=True)
class TokenWeight:
    """One token's weight, unmerged.

    ``text`` is the slice of the caller's text covered by this token's offsets. A character split
    across byte-level tokens appears in each of their slices, so do not concatenate these; use the
    merged spans. ``token.text`` is the token's display string."""

    position: int
    token: Optional[Token]
    weight: float
    text: Optional[str] = None

    @property
    def span(self) -> Optional[Tuple[int, int]]:
        return self.token.span if self.token is not None else None


@dataclass(frozen=True)
class SpanWeight:
    """Weight of one merged character span: tokens whose offsets overlap are combined (union
    range, summed weight, every position), so spans never overlap."""

    start: int
    end: int
    weight: float
    positions: Tuple[int, ...]
    text: Optional[str] = None


def _spans(weights: Optional[List[TokenWeight]], text: Optional[str]) -> List[SpanWeight]:
    if weights is None:
        return []
    by_position = {w.position: w for w in weights}
    items = [(w.span[0], w.span[1], w.position) for w in weights if w.span is not None]
    return [
        SpanWeight(
            start,
            end,
            math.fsum(by_position[p].weight for p in positions),
            tuple(positions),
            text[start:end] if text is not None else None,
        )
        for start, end, positions in _merge(items)
    ]


@dataclass(frozen=True)
class TokenAttribution:
    total: float
    """``Σ_i s[i]``; equals the result's ``score`` (up to float32 rounding)."""
    query_slot_scores: List[float]
    """``s[i]``: each query slot's share of the score."""
    slot_pairs: List[List[float]]
    """``P[i][j]``: share of the score flowing from query slot ``i`` to document slot ``j``."""
    document_slot_weights: List[float]
    """``Σ_i P[i][j]``."""
    query_tokens: Optional[List[TokenWeight]]
    """One weight per query token, unmerged (``None`` without query attention)."""
    document_tokens: Optional[List[TokenWeight]]
    """One weight per document token, unmerged (``None`` without document attention, e.g. search)."""
    query_text: Optional[str] = field(default=None, repr=False)
    document_text: Optional[str] = field(default=None, repr=False)

    def query_spans(self) -> List[SpanWeight]:
        """Merged, non-overlapping weights per span of the query text (input tokens only), in text
        order. Use these to highlight; each span's weight is the sum of its tokens' weights."""
        return _spans(self.query_tokens, self.query_text)

    def document_spans(self) -> List[SpanWeight]:
        """Merged, non-overlapping weights per span of the document text (see ``query_spans``)."""
        return _spans(self.document_tokens, self.document_text)


def attention_matrix(attention: Tensor, head: Optional[int] = None) -> List[List[float]]:
    """Slot→token attention as ``[slots][tokens]``: one head, or the mean over heads."""
    if attention.ndim != 3:
        raise ValueError(f"attention must have shape [heads, slots, tokens], got {list(attention.shape)}")
    heads, slots, tokens = attention.shape
    flat = attention.flat()
    if head is not None:
        if not -heads <= head < heads:
            raise IndexError(f"head {head} out of range for {heads} heads")
        h = head % heads
        base = h * slots * tokens
        return [flat[base + s * tokens : base + (s + 1) * tokens] for s in range(slots)]
    out = [[0.0] * tokens for _ in range(slots)]
    for h in range(heads):
        base = h * slots * tokens
        for s in range(slots):
            row = out[s]
            off = base + s * tokens
            for t in range(tokens):
                row[t] += flat[off + t]
    return [[v / heads for v in row] for row in out]


def _matrix(t: Tensor, rows: int, cols: int, name: str) -> List[List[float]]:
    if list(t.shape) != [rows, cols]:
        raise ValueError(f"{name} has shape {list(t.shape)}, expected [{rows}, {cols}]")
    flat = t.flat()
    return [flat[r * cols : (r + 1) * cols] for r in range(rows)]


def _softmax_rows(m: List[List[float]], tau: float) -> List[List[float]]:
    out = []
    for row in m:
        top = max(row)
        exps = [math.exp((v - top) / tau) for v in row]
        z = math.fsum(exps)
        out.append([e / z for e in exps])
    return out


def _token_weights(
    slot_weights: List[float],
    attention: Optional[Tensor],
    tokens: Optional[Sequence[Token]],
    text: Optional[str],
    head: Optional[int],
    side: str,
) -> Optional[List[TokenWeight]]:
    if attention is None:
        return None
    a = attention_matrix(attention, head)
    if len(a) != len(slot_weights):
        raise ValueError(f"{side} attention has {len(a)} slots, the explanation has {len(slot_weights)}")
    n_tokens = len(a[0]) if a else 0
    if tokens is not None and len(tokens) != n_tokens:
        raise ValueError(f"{side} attention covers {n_tokens} tokens but {len(tokens)} tokens were returned")
    weights = [math.fsum(slot_weights[s] * a[s][t] for s in range(len(a))) for t in range(n_tokens)]
    out = []
    for t, w in enumerate(weights):
        tok = tokens[t] if tokens is not None else None
        piece = None
        if text is not None and tok is not None and tok.start is not None and tok.end is not None:
            _check_offsets(text, t, tok.start, tok.end)
            piece = text[tok.start : tok.end]
        out.append(TokenWeight(t, tok, w, piece))
    return out


def token_attribution(
    explanation: Explanation,
    *,
    query: Any = None,
    document: Any = None,
    scoring: Optional[ScoringParameters] = None,
    coupling: str = "score",
    query_text: Optional[str] = None,
    document_text: Optional[str] = None,
    head: Optional[int] = None,
) -> TokenAttribution:
    """Decompose a pair score over slot pairs and query/document tokens (see module docs).

    ``query``/``document`` are the ``TextEncoding`` objects of the response (``response.query``,
    ``result.document``); their ``attention`` (and ``tokens``) must have been requested. Either
    side may be ``None``. ``scoring`` supplies the weights and ``tau_k`` (``ScoreResponse.scoring``
    or the index's ``search_profile.scoring``); without it the contract defaults are used.
    ``head`` picks one attention head instead of the mean over heads.
    """
    if coupling not in ("score", "value", "routing"):
        raise ValueError("coupling must be 'score', 'value' or 'routing'")
    params = dict(_DEFAULT_SCORING)
    if scoring is not None:
        params.update(
            tau_k=scoring.tau_k,
            tau_v=scoring.tau_v,
            routing_weight=scoring.routing_weight,
            value_weight=scoring.value_weight,
        )
    sim_t = explanation.slot_similarity
    if sim_t.ndim != 2:
        raise ValueError(f"slot_similarity must be 2-D, got shape {list(sim_t.shape)}")
    n_q, n_d = sim_t.shape
    sim = _matrix(sim_t, n_q, n_d, "slot_similarity")
    # Explanation tensors are float32, so value_attention rows sum to 1 to float32 precision.
    value_att = _matrix(explanation.value_attention, n_q, n_d, "value_attention")
    r = explanation.routing_contribution.flat()
    v = explanation.value_slot_cosine.flat()
    if len(r) != n_q or len(v) != n_q:
        raise ValueError("routing_contribution and value_slot_cosine must have one value per query slot")

    rw, vw = params["routing_weight"], params["value_weight"]
    routing_part = [rw * r[i] / n_q for i in range(n_q)]
    value_part = [vw * v[i] / n_q for i in range(n_q)]
    slot_scores = [routing_part[i] + value_part[i] for i in range(n_q)]

    routing_cpl = _softmax_rows(sim, params["tau_k"]) if coupling in ("score", "routing") else None
    if coupling == "score":
        assert routing_cpl is not None
        pairs = [
            [routing_part[i] * routing_cpl[i][j] + value_part[i] * value_att[i][j] for j in range(n_d)]
            for i in range(n_q)
        ]
    else:
        cpl = value_att if coupling == "value" else routing_cpl
        assert cpl is not None
        pairs = [[slot_scores[i] * cpl[i][j] for j in range(n_d)] for i in range(n_q)]
    doc_slots = [math.fsum(pairs[i][j] for i in range(n_q)) for j in range(n_d)]

    q_att = getattr(query, "attention", None) if query is not None else None
    d_att = getattr(document, "attention", None) if document is not None else None
    return TokenAttribution(
        total=math.fsum(slot_scores),
        query_slot_scores=slot_scores,
        slot_pairs=pairs,
        document_slot_weights=doc_slots,
        query_tokens=_token_weights(
            slot_scores, q_att, getattr(query, "tokens", None), query_text, head, "query"
        ),
        document_tokens=_token_weights(
            doc_slots, d_att, getattr(document, "tokens", None), document_text, head, "document"
        ),
        query_text=query_text,
        document_text=document_text,
    )

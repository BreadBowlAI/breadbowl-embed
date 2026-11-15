from __future__ import annotations

import math

import pytest

from breadbowl import (
    Explanation,
    ScoreResponse,
    ScoringParameters,
    SearchResponse,
    Tensor,
    TextEncoding,
    Token,
    attention_matrix,
    token_attribution,
    token_spans,
)
from tests.conftest import example, expanded

SCORING = ScoringParameters("logmeanexp", tau_k=0.02, tau_v=0.1, routing_weight=0.75, value_weight=0.25)


def softmax(row, tau):  # type: ignore[no-untyped-def]
    e = [math.exp(v / tau) for v in row]
    return [x / sum(e) for x in e]


def make_explanation() -> Explanation:
    """2 query slots × 3 document slots, consistent with the contract's formulas."""
    sim = [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1]]
    va = [softmax(r, SCORING.tau_v) for r in sim]
    rc = [SCORING.tau_k * (math.log(sum(math.exp(v / SCORING.tau_k) for v in r)) - math.log(3)) for r in sim]
    vc = [0.5, -0.2]
    return Explanation(
        slot_similarity=Tensor.from_list(sim, axes=["query_slot", "document_slot"]),
        routing_contribution=Tensor.from_list(rc, axes=["query_slot"], encoding="float"),
        value_attention=Tensor.from_list(va, axes=["query_slot", "document_slot"]),
        value_slot_cosine=Tensor.from_list(vc, axes=["query_slot"], encoding="float"),
    )


def tokens(spec):  # type: ignore[no-untyped-def]
    return [Token(id=i, text=t, segment=seg, start=s, end=e) for i, (t, seg, s, e) in enumerate(spec)]


QUERY_TEXT = "héllo world"  # non-ASCII: offsets are code points
QUERY_TOKENS = tokens(
    [
        ("Query:", "template", None, None),
        ("h", "input", 0, 1),
        ("é", "input", 1, 2),
        ("llo", "input", 2, 5),
        (" world", "input", 5, 11),
        ("<eos>", "special", None, None),
    ]
)
# Two heads, 2 slots, 6 tokens; each (head, slot) row sums to 1.
Q_ATT = [
    [[0.1, 0.4, 0.1, 0.1, 0.2, 0.1], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
    [[0.1, 0.2, 0.1, 0.3, 0.2, 0.1], [0.2, 0.0, 0.0, 0.0, 0.8, 0.0]],
]
DOC_TEXT = "ab"
# Two byte-level tokens share the span of one character ("a").
DOC_TOKENS = tokens([("a1", "input", 0, 1), ("a2", "input", 0, 1), ("b", "input", 1, 2)])
D_ATT = [[[0.5, 0.25, 0.25], [0.0, 0.0, 1.0], [0.2, 0.2, 0.6]]]


def encodings():  # type: ignore[no-untyped-def]
    q = TextEncoding(6, False, QUERY_TOKENS, Tensor.from_list(Q_ATT, axes=["head", "slot", "token"]))
    d = TextEncoding(
        3, False, DOC_TOKENS, Tensor.from_list(D_ATT, axes=["head", "slot", "token"], encoding="float")
    )
    return q, d


def test_explanation_reproduces_scores() -> None:
    exp = make_explanation()
    r = exp.routing_score()
    v = exp.value_score()
    assert v == pytest.approx(0.15, abs=1e-6)
    assert exp.score(SCORING) == pytest.approx(0.75 * r + 0.25 * v)


@pytest.mark.parametrize("coupling", ["score", "value", "routing"])
def test_attribution_is_an_exact_decomposition(coupling: str) -> None:
    exp = make_explanation()
    q, d = encodings()
    att = token_attribution(
        exp,
        query=q,
        document=d,
        scoring=SCORING,
        coupling=coupling,
        query_text=QUERY_TEXT,
        document_text=DOC_TEXT,
    )
    score = exp.score(SCORING)
    assert att.total == pytest.approx(score, abs=1e-6)
    for i, row in enumerate(att.slot_pairs):
        assert math.fsum(row) == pytest.approx(att.query_slot_scores[i], abs=1e-6)
    assert math.fsum(att.document_slot_weights) == pytest.approx(score, abs=1e-6)
    assert att.query_tokens is not None and att.document_tokens is not None
    assert math.fsum(w.weight for w in att.query_tokens) == pytest.approx(score, abs=1e-5)
    assert math.fsum(w.weight for w in att.document_tokens) == pytest.approx(score, abs=1e-5)


def test_attribution_maps_tokens_to_text_and_merges_spans() -> None:
    q, d = encodings()
    att = token_attribution(
        make_explanation(),
        query=q,
        document=d,
        scoring=SCORING,
        query_text=QUERY_TEXT,
        document_text=DOC_TEXT,
    )
    assert att.query_tokens is not None and att.document_tokens is not None
    assert [w.text for w in att.query_tokens] == [None, "h", "é", "llo", " world", None]
    spans = att.query_spans()
    assert [(s.start, s.end, s.text) for s in spans] == [
        (0, 1, "h"),
        (1, 2, "é"),
        (2, 5, "llo"),
        (5, 11, " world"),
    ]
    doc_spans = att.document_spans()
    assert [(s.start, s.end, s.positions) for s in doc_spans] == [(0, 1, (0, 1)), (1, 2, (2,))]
    assert doc_spans[0].weight == pytest.approx(att.document_tokens[0].weight + att.document_tokens[1].weight)


def test_head_selection() -> None:
    q, _ = encodings()
    att_h0 = token_attribution(make_explanation(), query=q, scoring=SCORING, head=0)
    att_mean = token_attribution(make_explanation(), query=q, scoring=SCORING)
    assert att_h0.query_tokens is not None and att_mean.query_tokens is not None
    assert att_h0.query_tokens[0].weight != att_mean.query_tokens[0].weight
    assert att_mean.document_tokens is None
    assert attention_matrix(q.attention, head=1)[1] == pytest.approx([0.2, 0.0, 0.0, 0.0, 0.8, 0.0])  # type: ignore[arg-type]
    assert attention_matrix(q.attention)[1] == pytest.approx([0.1, 0, 0, 0, 0.9, 0])  # type: ignore[arg-type]
    with pytest.raises(IndexError):
        attention_matrix(q.attention, head=2)  # type: ignore[arg-type]


def test_shape_mismatches_are_reported() -> None:
    q, _ = encodings()
    bad = TextEncoding(5, False, QUERY_TOKENS[:5], q.attention)
    with pytest.raises(ValueError, match="6 tokens but 5"):
        token_attribution(make_explanation(), query=bad)
    wrong_slots = TextEncoding(
        6, False, QUERY_TOKENS, Tensor.from_list([Q_ATT[0][:1]], axes=["head", "slot", "token"])
    )
    with pytest.raises(ValueError, match="1 slots"):
        token_attribution(make_explanation(), query=wrong_slots)
    with pytest.raises(ValueError, match="coupling"):
        token_attribution(make_explanation(), coupling="magic")


def test_token_spans() -> None:
    spans = token_spans(QUERY_TEXT, QUERY_TOKENS)
    assert [(s.positions, s.text) for s in spans] == [
        ((1,), "h"),
        ((2,), "é"),
        ((3,), "llo"),
        ((4,), " world"),
    ]
    assert spans[0].tokens == (QUERY_TOKENS[1],)
    with pytest.raises(ValueError, match="outside"):
        token_spans("hé", QUERY_TOKENS)


# ---------------------------------------------------------------------------- overlapping offsets
# The contract's token rule: a byte-level token that ends inside a multi-byte character shows only
# what precedes it (possibly "") and the token that completes it shows the whole character; both
# report the whole character's offsets, so offsets overlap without being identical.

OFFSET_CASES = {
    # "a 🍕 b": 🍕 is 4 UTF-8 bytes, split over two tokens (" "+2 bytes, then 2 bytes).
    "emoji over two tokens": (
        "a 🍕 b",
        [("a", 0, 1), (" ", 1, 3), ("🍕", 2, 3), (" b", 3, 5)],
        [(0, 1, (0,)), (1, 3, (1, 2)), (3, 5, (3,))],
    ),
    # ... and over three tokens (" "+1 byte, 2 bytes, 1 byte): texts " ", "", "🍕".
    "emoji over three tokens": (
        "a 🍕 b",
        [("a", 0, 1), (" ", 1, 3), ("", 2, 3), ("🍕", 2, 3), (" b", 3, 5)],
        [(0, 1, (0,)), (1, 3, (1, 2, 3)), (3, 5, (4,))],
    ),
    # e + U+0301 (2 bytes) split after its first byte; then a clean combining sequence.
    "combining sequence": (
        "e\u0301x o\u0308",
        [("e", 0, 2), ("\u0301", 1, 2), ("x", 2, 3), (" o", 3, 5), ("\u0308", 5, 6)],
        [(0, 2, (0, 1)), (2, 3, (2,)), (3, 5, (3,)), (5, 6, (4,))],
    ),
    # Truncation cut 🍕 after 2 bytes: U+FFFD in the display text, offsets cover the character.
    "truncated final character": (
        "ab🍕",
        [("a", 0, 1), ("b\ufffd", 1, 3)],
        [(0, 1, (0,)), (1, 3, (1,))],
    ),
    "every token clean": (
        "Keys rotate.",
        [("Keys", 0, 4), (" rotate", 4, 11), (".", 11, 12)],
        [(0, 4, (0,)), (4, 11, (1,)), (11, 12, (2,))],
    ),
}


def _tokens(spec):  # type: ignore[no-untyped-def]
    return [Token(id=i, text=t, segment="input", start=s, end=e) for i, (t, s, e) in enumerate(spec)]


@pytest.mark.parametrize("name", list(OFFSET_CASES))
def test_token_spans_merge_overlapping_offsets(name: str) -> None:
    text, spec, expected = OFFSET_CASES[name]
    tokens = _tokens(spec)
    spans = token_spans(text, tokens)
    assert [(s.start, s.end, s.positions) for s in spans] == expected
    # Non-overlapping and tiling the text; slices concatenate to the caller's text.
    assert all(a.end <= b.start for a, b in zip(spans, spans[1:]))
    assert spans[0].start == 0 and spans[-1].end == len(text)
    assert all(a.end == b.start for a, b in zip(spans, spans[1:]))
    assert "".join(s.text for s in spans) == text
    # The display strings concatenate to the decoded text (equal to the input unless truncated).
    display = "".join(t.text for t in tokens)
    assert display == text or (name == "truncated final character" and display == "ab\ufffd")


@pytest.mark.parametrize("name", list(OFFSET_CASES))
def test_attribution_spans_merge_overlaps_and_keep_token_weights(name: str) -> None:
    text, spec, expected = OFFSET_CASES[name]
    tokens = _tokens(spec)
    n = len(tokens)
    # 3 document slots attending to the tokens with different (row-stochastic) weights.
    rows = []
    for j in range(3):
        raw = [float((j + 1) * (t + 1) % 7 + 1) for t in range(n)]
        rows.append([v / sum(raw) for v in raw])
    document = TextEncoding(n, False, tokens, Tensor.from_list([rows], axes=["head", "slot", "token"]))
    att = token_attribution(make_explanation(), document=document, scoring=SCORING, document_text=text)
    assert att.document_tokens is not None and len(att.document_tokens) == n  # unmerged per token
    spans = att.document_spans()
    assert [(s.start, s.end, s.positions) for s in spans] == expected
    for s in spans:
        assert s.weight == pytest.approx(
            math.fsum(att.document_tokens[p].weight for p in s.positions), abs=1e-12
        )
        assert s.text == text[s.start : s.end]
    assert "".join(s.text or "" for s in spans) == text
    assert math.fsum(s.weight for s in spans) == pytest.approx(att.total, abs=1e-6)


def test_score_response_attribution_on_contract_example() -> None:
    res = ScoreResponse.from_dict(expanded("score", 200, "explained"))
    att = res.attribution(0, query_text="refund policy", document_text="Refunds accepted.")
    assert att.query_tokens is not None and len(att.query_tokens) == res.query.token_count
    # The tokenizer adds no special tokens: every token is template or input.
    assert {t.segment for t in res.query.tokens or []} <= {"template", "input"}
    assert [s.text for s in att.query_spans()] == ["refund", " policy"]
    assert [s.text for s in att.document_spans()] == ["Ref", "unds", " accepted", "."]
    exp = res.results[0].explanation
    assert exp is not None
    assert att.total == pytest.approx(exp.score(res.scoring))
    # The example's own numbers satisfy routing_score = mean(routing_contribution) to 2 decimals.
    assert exp.routing_score() == pytest.approx(res.results[0].routing_score, abs=0.01)


def test_attribution_requires_explanation() -> None:
    res = ScoreResponse.from_dict(example("score", 200, "basic"))
    with pytest.raises(ValueError, match="explain=True"):
        res.attribution(0)
    search = SearchResponse.from_dict(example("searchIndex"))
    with pytest.raises(ValueError, match="explain=True"):
        search.attribution(0)


def test_search_attribution_is_query_side_only() -> None:
    q, _ = encodings()
    body = example("searchIndex")
    search = SearchResponse.from_dict(body)
    search.query = q
    exp = make_explanation()
    search.results[0].explanation = exp
    # The response carries its own scoring parameters; no second call is needed.
    att = search.attribution(0, query_text=QUERY_TEXT)
    assert att.document_tokens is None and att.query_tokens is not None
    assert len(att.document_slot_weights) == 3
    assert att.total == pytest.approx(exp.score(search.scoring), abs=1e-6)
    assert [s.text for s in att.query_spans()] == ["h", "é", "llo", " world"]


def test_no_renormalisation_needed_for_float32_explanations() -> None:
    # value_attention arrives as float32, so its rows sum to 1 to ~1e-7 and the decomposition is
    # exact without renormalising (this test is why the old float16 workaround was dropped).
    exp = make_explanation()
    assert exp.value_attention.dtype == "float32"
    for row in exp.value_attention.to_list():  # type: ignore[union-attr]
        assert math.fsum(row) == pytest.approx(1.0, abs=1e-6)
    att = token_attribution(exp, scoring=SCORING, coupling="value")
    for i, row in enumerate(att.slot_pairs):
        assert math.fsum(row) == pytest.approx(att.query_slot_scores[i], abs=1e-7)


def test_pilot_chunks_without_offsets() -> None:
    from breadbowl import StoredDocument

    body = example("getDocument")
    body["chunks"][0]["char_start"] = None
    body["chunks"][0]["char_end"] = None
    chunk = StoredDocument.from_dict(body).chunks[0]
    assert chunk.char_start is None
    with pytest.raises(ValueError, match="no character offsets"):
        chunk.text("anything")

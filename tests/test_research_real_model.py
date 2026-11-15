"""breadbowl_embed.research against real weights; skipped unless BREADBOWL_TEST_MODEL is set.

  BREADBOWL_TEST_MODEL  a release directory (or Hub repo id already in the local cache)
  BREADBOWL_TEST_DTYPE  optional compute dtype: native (default), float32, bfloat16

About 20 short forward passes; on a CPU roughly half a minute in float32.
"""
import json
import os

import pytest
import torch

from breadbowl_embed import BreadBowl
from breadbowl_embed import research

pytestmark = pytest.mark.real_model

QUERY = "How many hearts does an octopus have?"
ANSWER = "Octopuses have three hearts."
DOCUMENT = (ANSWER + " Two branchial hearts pump blood through the gills, while a single systemic heart circulates "
            "it through the rest of the body. The Pacific is the largest and deepest of Earth's oceans.")
OTHERS = ["The Pacific is Earth's largest ocean.", "Plants use sunlight during photosynthesis."]


@pytest.fixture(scope="module")
def model():
    source = os.environ.get("BREADBOWL_TEST_MODEL")
    if not source:
        pytest.skip("set BREADBOWL_TEST_MODEL to a release directory to run the real-model checks")
    dtype = os.environ.get("BREADBOWL_TEST_DTYPE") or None
    return BreadBowl.from_pretrained(source, dtype=dtype, local_files_only=True)


def _row_tolerance(model) -> float:
    return 1e-5 if model.network.routing_head.weight.dtype == torch.float32 else 1e-2


def test_trace_equals_the_encoders_and_maps_tokens(model):
    long_document = " ".join(["The octopus is a soft-bodied, eight-limbed mollusc."] * 60)
    for kind, texts, encode in (("query", [QUERY], model.encode_queries),
                                ("document", [DOCUMENT, long_document], model.encode_documents)):
        traced = research.trace(model, texts, kind, capture="all", layers=[-1])
        encoded = encode(texts, return_attention=True, return_tokens=True)
        rebuilt = traced.encoding()
        assert torch.equal(rebuilt.routing, encoded.routing) and torch.equal(rebuilt.value, encoded.value)
        for i in range(len(texts)):
            assert torch.equal(rebuilt.attention[i], encoded.attention[i])
            assert rebuilt.offsets[i] == encoded.offsets[i] and rebuilt.truncated[i] == encoded.truncated[i]
    assert traced[1].tokens.truncated and len(traced[1].tokens) == model.config["preprocessing"]["max_document_length"]
    query = research.trace(model, [QUERY], "query")[0].tokens
    instruction = model.config["preprocessing"]["default_instruction"]
    assert query.instruction == instruction
    assert "".join(query.token_texts[t] for t, s in enumerate(query.segments) if s == "instruction") == instruction
    assert [query.text[slice(*o)] for o, s in zip(query.offsets, query.segments) if s == "text"] and \
        "".join(query.token_texts[t] for t, s in enumerate(query.segments) if s == "text").strip() == QUERY


def test_attention_rows_are_distributions(model):
    [t] = research.trace(model, [DOCUMENT], "document", capture=("logits",))
    tolerance = _row_tolerance(model)
    assert float((t.attention.sum(-1) - 1).abs().max()) < tolerance
    network_dtype = model.network.routing_head.weight.dtype
    assert torch.equal(torch.softmax(t.attention_logits.to(network_dtype), dim=-1).float(), t.attention)
    m = t.attention_map()
    assert float((m.matrix().sum(-1) - 1).abs().max()) < tolerance
    mass = m.segment_mass()
    assert float((sum(mass.values()) - 1).abs().max()) < tolerance


def test_decomposition_recombines_to_score(model):
    pair = research.inspect_pair(model, QUERY, DOCUMENT)
    scores = model.score(model.encode_queries([QUERY]), model.encode_documents([DOCUMENT]))
    d = pair.decomposition
    assert (d.final, d.routing, d.value) == (float(scores.combined[0, 0]), float(scores.routing[0, 0]),
                                             float(scores.value[0, 0]))
    assert max(abs(r) for r in d.residuals().values()) < 1e-6
    assert float(d.routing_weight.sum()) == pytest.approx(1.0, abs=1e-12)
    for side in ("query", "document"):
        a = pair.attribution(side)
        assert float(a.weights.sum()) + a.unattributed == pytest.approx(d.final, abs=1e-6)
    html = pair.report()
    assert html.count("<script") == 2 and "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert json.loads(pair.to_json())["scores"]["final"] == d.final


def test_occlusion_sanity(model):
    start = DOCUMENT.index(ANSWER)
    answer = research.occlude(model, QUERY, DOCUMENT, [(start, start + len(ANSWER))])
    other_start = DOCUMENT.index("The Pacific")
    unrelated = research.occlude(model, QUERY, DOCUMENT, [(other_start, len(DOCUMENT))])
    assert answer.occluded_text == DOCUMENT[len(ANSWER):]
    # Removing the sentence that answers the query costs far more than removing an unrelated one.
    assert answer.delta["final"] < -0.05 and answer.delta["final"] < unrelated.delta["final"] - 0.05
    span = (DOCUMENT.index("hearts"), DOCUMENT.index("hearts") + len("hearts"))
    hearts = research.occlude(model, QUERY, DOCUMENT, [span], mode="slot-mask")
    assert hearts.delta["final"] < 0
    sweep = research.occlusion_sweep(model, QUERY, DOCUMENT, unit="sentence", mode="delete")
    assert sweep.top(1)[0]["text"].strip() == ANSWER


def test_slot_ablation_and_rules(model):
    queries = model.encode_queries([QUERY])
    documents = model.encode_documents([DOCUMENT] + OTHERS)
    ablation = research.slot_ablation(queries, documents)
    assert ablation.query_slot["final"].shape == (16, 3) and len(ablation.effects()) == 32
    rules = research.compare_rules(queries, documents, base=model)
    assert rules.agreement["model"]["max_abs_delta"] == 0
    assert rules.agreement["routing-only"]["top1_agreement"] == 1.0  # the answer passage wins either way
    assert model.scoring.value_mix == 0.25


def test_collection_diagnostics(model):
    texts = [DOCUMENT] + OTHERS + ["Hearts pump blood.", "Copper carries oxygen in some molluscs."]
    documents = model.encode_documents(texts)
    queries = model.encode_queries([QUERY, "largest ocean"])
    stats = research.slot_statistics(documents, queries=queries, top_k=2)
    assert 1 < stats.per_text_effective_rank["routing_mean"] <= 16
    torch.testing.assert_close(torch.diagonal(stats.within_text_routing_cosine), torch.ones(16), atol=1e-5, rtol=0)
    assert stats.usage["pairs"] == 4
    index = model.index(documents, ids=[f"d{i}" for i in range(len(texts))])
    diagnostics = research.shortlist_diagnostics(index, queries, slot_hits=[1, 128], top_k=3)
    wide = diagnostics.rows[1]
    assert wide["candidate_recall"] == 1.0 and wide["underfilled"] == 0


def test_compute_dtype_variant(model):
    if model.network.routing_head.weight.dtype != torch.float32:
        pytest.skip("needs the float32 weights in memory")
    result = research.compare_precision(model, [QUERY], [DOCUMENT] + OTHERS, top_k=3)
    rows = {r["variant"]: r for r in result.rows}
    assert rows[result.reference]["max_abs_delta"] == 0.0
    assert 0 < rows["compute=float32 batch=1 storage=int8"]["max_abs_delta"] < 1e-2
    assert 0 < rows["compute=bfloat16 batch=1 storage=float32"]["max_abs_delta"] < 5e-2

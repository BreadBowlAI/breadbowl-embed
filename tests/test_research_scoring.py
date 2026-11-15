"""breadbowl_embed.research: score decomposition, attribution, rule overrides and interventions."""
import math

import pytest
import torch
import torch.nn.functional as F

from breadbowl_embed import Embeddings, ScoringConfig, explain, score
from breadbowl_embed.research import (ScoringRule, ablate_slots, attribute, compare_rules, decompose, edit_text,
                                      inspect_pair, occlude, occlusion_sweep, rescore, slot_ablation)


def random_pair(seed=5, q=2, d=3, slots=4, d_k=6, d_v=5, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    queries = Embeddings(F.normalize(torch.randn(q, slots, d_k, generator=g, dtype=dtype), dim=-1),
                         torch.randn(q, slots, d_v, generator=g, dtype=dtype), "test", "query")
    documents = Embeddings(F.normalize(torch.randn(d, slots + 1, d_k, generator=g, dtype=dtype), dim=-1),
                           3 * torch.randn(d, slots + 1, d_v, generator=g, dtype=dtype), "test", "document")
    return queries, documents


@pytest.mark.parametrize("rule", [None, "routing-only", "value-only", "maxsim", "maxsim+value",
                                  ScoringRule(tau_k=0.05, tau_v=0.3, value_mix=0.4)])
def test_decomposition_recombines_to_score(rule):
    queries, documents = random_pair()
    grid = decompose(queries, documents, rule=rule)
    resolved = grid[0][0].rule
    reference = score(queries, documents, config=resolved.config())
    for i in range(len(queries)):
        for j in range(len(documents)):
            d = grid[i][j]
            # The totals are the scorer's own numbers, bit for bit.
            assert d.final == float(reference.combined[i, j]) and d.routing == float(reference.routing[i, j])
            assert d.value == float(reference.value[i, j])
            # The parts recombine to them up to float32 rounding.
            for component, residual in d.residuals().items():
                assert abs(residual) < 2e-6, (component, residual)
            assert float(d.query_slot_contribution("final").sum()) == pytest.approx(d.final, abs=2e-6)
            assert float(d.document_slot_contribution("final").sum()) + d.unattributed("final") == pytest.approx(
                d.final, abs=2e-6)
            assert float(d.routing_weight.sum()) == pytest.approx(1.0, abs=1e-12)
            torch.testing.assert_close(d.value_attention.sum(-1), torch.ones(4, dtype=torch.float64))
            assert float(d.value_slot_cosine.mean()) == pytest.approx(d.value, abs=2e-6)
            assert bool((d.routing_entropy_term <= 1e-15).all())
            if resolved.qk_mode == "max":
                assert float(d.routing_entropy_term.abs().max()) == 0.0
                assert sorted(d.routing_weight.flatten().tolist())[-4:] == [0.25] * 4


def test_decomposition_matches_explain_per_slot():
    queries, documents = random_pair(q=1, d=2)
    expl = explain(queries, documents)
    grid = decompose(queries, documents)
    for j in range(2):
        e, d = expl[0][j], grid[0][j]
        slots = d.similarity.shape[0]
        torch.testing.assert_close(d.similarity.float(), e.slot_similarity)
        torch.testing.assert_close((d.query_slot_contribution("routing") * slots).float(), e.routing_contribution)
        torch.testing.assert_close(d.value_attention.float(), e.value_attention)
        torch.testing.assert_close(d.value_slot_cosine.float(), e.value_slot_cosine)
        assert d.top_pairs(3)[0]["contribution"] == pytest.approx(
            float(d.pair_contribution().abs().max()) * (1 if d.top_pairs(1)[0]["contribution"] > 0 else -1))
        assert len(d.slot_table()) == 4 and d.to_dict()["type"] == "score_decomposition"


def test_rule_overrides_do_not_touch_the_config():
    base = ScoringConfig()
    queries, documents = random_pair()
    routing_only = rescore(queries, documents, "routing-only", base=base)
    assert torch.equal(routing_only.combined, score(queries, documents).routing)
    value_only = rescore(queries, documents, "value-only")
    assert torch.equal(value_only.combined, score(queries, documents).value)
    maxsim = rescore(queries, documents, "maxsim").combined
    cos = torch.einsum("qsd,ptd->qpst", queries.routing, documents.routing)
    torch.testing.assert_close(maxsim, cos.max(-1).values.mean(-1))
    assert ScoringConfig() == base
    comparison = compare_rules(queries, documents, top_k=2)
    assert comparison.agreement["model"]["kendall_tau"] == 1.0 and comparison.agreement["model"]["max_abs_delta"] == 0
    assert {r.name for r in comparison.rules} == {"model", "routing-only", "value-only", "maxsim", "maxsim+value"}
    assert comparison.to_dict()["type"] == "rule_comparison" and "<table>" in comparison._repr_html_()
    custom = ScoringRule.preset("model").replace(tau_k=0.1)
    assert custom.name == "custom" and custom.tau_k == 0.1
    with pytest.raises(ValueError):
        ScoringRule(value_mix=2.0)
    with pytest.raises(ValueError):
        ScoringRule.preset("bm25")


def test_slot_ablation_matches_manual_rescoring():
    queries, documents = random_pair()
    ablation = slot_ablation(queries, documents, query=1)
    q = queries.select([1])
    for s in range(4):
        keep = [k for k in range(4) if k != s]
        manual = score(Embeddings(q.routing[:, keep], q.value[:, keep], "test", "query"), documents)
        assert torch.equal(ablation.query_slot["final"][s], manual.combined[0])
    keep = [1, 2, 3, 4]
    manual = score(q, Embeddings(documents.routing[:, keep], documents.value[:, keep], "test", "document"))
    assert torch.equal(ablation.document_slot["final"][0], manual.combined[0])
    assert torch.equal(ablation.baseline["final"], score(q, documents).combined[0])
    assert ablation.delta("query").shape == (4, 3) and len(ablation.effects()) == 4 + 5
    zeroed = ablate_slots(queries, documents, query_slots=[0], mode="zero")
    assert zeroed.combined.shape == (2, 3)
    assert torch.equal(ablate_slots(queries, documents).combined, score(queries, documents).combined)
    with pytest.raises(ValueError):
        ablate_slots(queries, documents, query_slots=range(4))


def test_edit_text_merges_spans():
    assert edit_text("abcdefgh", [(1, 3), (2, 5)], "_") == "a_fgh"
    assert edit_text("abcdefgh", [(0, 1), (7, 8)]) == "bcdefg"
    with pytest.raises(ValueError):
        edit_text("abc", [(2, 9)])


def test_inspect_pair_and_attribution(tiny_model):
    pair = inspect_pair(tiny_model, "fish water", "fish water cats land")
    reference = tiny_model.score(tiny_model.encode_queries(["fish water"]),
                                 tiny_model.encode_documents(["fish water cats land"]))
    d = pair.decomposition
    assert d.final == float(reference.combined[0, 0]) and d.routing == float(reference.routing[0, 0])
    for side in ("query", "document"):
        for component in ("final", "routing", "value"):
            a = pair.attribution(side, component)
            assert float(a.weights.sum()) + a.unattributed == pytest.approx(a.total, abs=2e-6)
            torch.testing.assert_close(a.by_slot.sum(-1), a.slot_contribution)
            assert sum(a.segment_totals().values()) == pytest.approx(float(a.weights.sum()))
    doc = pair.attribution("document")
    assert sum(w["weight"] for w in doc.word_weights()) == pytest.approx(float(doc.weights.sum()))
    assert doc.span_total(0, 4) == pytest.approx(float(doc.weights[0]))
    assert "not causal" in doc.note and doc.top_tokens(2)[0]["index"] in range(4)
    head = attribute(d, pair.document, side="document", head=1)
    assert float(head.weights.sum()) == pytest.approx(float(doc.weights.sum()), abs=1e-9)
    query = pair.attribution("query")
    assert set(query.segment_totals()) == {"instruction", "template", "text"}
    routing_only = inspect_pair(tiny_model, "fish water", "fish water cats land", rule="routing-only")
    assert routing_only.decomposition.final == routing_only.decomposition.routing == d.routing


def test_occlusion(tiny_model):
    query, document = "fish water", "fish water cats land"
    baseline = float(tiny_model.score(tiny_model.encode_queries([query]),
                                      tiny_model.encode_documents([document])).combined[0, 0])
    deleted = occlude(tiny_model, query, document, [(0, 5)])
    assert deleted.occluded_text == "water cats land" and deleted.baseline["final"] == baseline
    expected = float(tiny_model.score(tiny_model.encode_queries([query]),
                                      tiny_model.encode_documents(["water cats land"])).combined[0, 0])
    assert deleted.occluded["final"] == expected
    assert deleted.delta["final"] == pytest.approx(expected - baseline)
    replaced = occlude(tiny_model, query, document, [(0, 4)], mode="replace", replacement="dogs")
    assert replaced.occluded_text == "dogs water cats land"
    masked = occlude(tiny_model, query, document, [(0, 4)], mode="slot-mask")
    assert masked.masked_tokens == (0,) and masked.occluded_text is None
    instruction = occlude(tiny_model, query, document, side="query", mode="slot-mask", tokens=[0])
    assert instruction.masked_tokens == (0,)
    with pytest.raises(ValueError):
        occlude(tiny_model, query, document, [(0, 4)], tokens=[0])
    with pytest.raises(ValueError):
        occlude(tiny_model, query, document, [(0, 4)], mode="delete", tokens=None, side="title")
    with pytest.raises(ValueError):
        occlude(tiny_model, query, document, [(0, len(document))], mode="delete")


def test_occlusion_sweep(tiny_model):
    query, document = "fish water", "fish water cats land"
    sweep = occlusion_sweep(tiny_model, query, document)
    assert [u["text"] for u in sweep.units] == ["fish", "water", "cats", "land"]
    for unit, final in zip(sweep.units, sweep.scores["final"].tolist()):
        single = occlude(tiny_model, query, document, [(unit["start"], unit["end"])], mode="slot-mask")
        assert single.occluded["final"] == final
    assert sweep.token_delta().shape == (4,) and sweep.top(1)[0]["delta"] == float(sweep.delta().min())
    deleted = occlusion_sweep(tiny_model, query, document, mode="delete", limit=2)
    assert len(deleted.units) == 2
    expected = occlude(tiny_model, query, document, [(0, 4)]).occluded["final"]
    assert deleted.scores["final"][0].item() == pytest.approx(expected, abs=1e-12)
    query_side = occlusion_sweep(tiny_model, query, document, side="query", unit="token")
    assert [u["text"] for u in query_side.units] == ["fish", "water"]
    one_word = occlusion_sweep(tiny_model, query, "fish", unit="word")
    assert math.isnan(one_word.scores["final"][0].item())  # hiding the only token is undefined
    assert "type" in sweep.to_dict() and sweep.to_dict()["units"][0]["delta"]["final"] is not None

import math
import pytest
import torch
import torch.nn.functional as F
from breadbowl_embed import Embeddings, ScoringConfig, explain, score, score_routing


@pytest.mark.parametrize("batches", [(8, 128), (1, 3), (2, 1)])
def test_score_routing_is_bit_identical_to_score(batches):
    torch.manual_seed(9)
    q = Embeddings(torch.randn(5, 4, 6), torch.randn(5, 4, 3), "test", "query")
    d = Embeddings(torch.randn(7, 4, 6), torch.randn(7, 4, 3), "test", "document")
    query_batch, document_batch = batches
    full = score(q, d, query_batch_size=query_batch, document_batch_size=document_batch)
    routing = score_routing(q, d, query_batch_size=query_batch, document_batch_size=document_batch)
    assert torch.equal(routing, full.routing)
    assert torch.equal(score_routing(q, d.routing, query_batch_size=query_batch, document_batch_size=document_batch),
                       full.routing)  # documents stored without value slots
    assert score_routing(q, d.routing[:0]).shape == (5, 0)
    with pytest.raises(ValueError, match="Incompatible"):
        score_routing(q, Embeddings(d.routing, d.value, "other", "document"))
    with pytest.raises(ValueError, match="dimensions"):
        score_routing(q, torch.randn(2, 4, 5))


def test_attention_readout_matches_independent_equations_and_gradients():
    torch.manual_seed(12)
    tensors = [torch.randn(*shape, dtype=torch.float64, requires_grad=True) for shape in
               [(2, 3, 4), (5, 3, 4), (2, 3, 6), (5, 3, 6)]]
    q, d, qv, dv = tensors
    queries = Embeddings(q, qv, "test", "query")
    docs = Embeddings(d, dv, "test", "document")
    result = score(queries, docs, query_batch_size=1, document_batch_size=2)
    sim = torch.einsum("bmd,cnd->bcmn", F.normalize(q, dim=-1), F.normalize(d, dim=-1))
    routing = (0.02 * (torch.logsumexp(sim / 0.02, dim=-1) - math.log(3))).mean(-1)
    readout = torch.einsum("bcmn,cnv->bcmv", (sim / 0.1).softmax(-1), dv)
    value = (F.normalize(qv, dim=-1)[:, None] * F.normalize(readout, dim=-1)).sum(-1).mean(-1)
    expected = 0.75 * routing + 0.25 * value
    torch.testing.assert_close(result.combined, expected)
    for actual, reference in zip(torch.autograd.grad(result.combined.sum(), tensors, retain_graph=True),
                                 torch.autograd.grad(expected.sum(), tensors)):
        torch.testing.assert_close(actual, reference)


def test_raw_document_value_norms_are_not_discarded():
    q = Embeddings(torch.tensor([[[1., 0.]]]), torch.tensor([[[1., 0.]]]), "test", "query")
    d = Embeddings(torch.tensor([[[1., 0.], [1., 0.]]]), torch.tensor([[[10., 0.], [0., 1.]]]), "test", "document")
    value = score(q, d).value.item()
    assert value == pytest.approx(10 / math.sqrt(101))
    assert value > 0.99


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cancellation_and_empty_batches(dtype):
    q = Embeddings(torch.zeros(1, 2, 2, dtype=dtype), torch.ones(1, 2, 2, dtype=dtype), "test", "query")
    d = Embeddings(torch.zeros(1, 2, 2, dtype=dtype), torch.tensor([[[1., 1.], [-1., -1.]]], dtype=dtype), "test", "document")
    assert score(q, d).combined.item() == 0
    assert score(q, d.select([])).combined.shape == (1, 0)


@pytest.mark.parametrize("mode", ["logmeanexp", "max"])
def test_explain_identities_and_shapes(mode):
    torch.manual_seed(5)
    q = Embeddings(torch.randn(2, 4, 6), torch.randn(2, 4, 5), "test", "query")
    d = Embeddings(torch.randn(3, 4, 6), 3 * torch.randn(3, 4, 5), "test", "document")
    config = ScoringConfig(qk_mode=mode)
    scores = score(q, d, config=config)
    explanations = explain(q, d, config=config)
    assert len(explanations) == 2 and all(len(row) == 3 for row in explanations)
    for i in range(2):
        for j in range(3):
            e = explanations[i][j]
            assert (e.query_index, e.document_index) == (i, j)
            assert e.slot_similarity.shape == e.value_attention.shape == (4, 4)
            assert e.routing_contribution.shape == e.value_slot_cosine.shape == (4,)
            assert e.score == pytest.approx(0.75 * e.routing_score + 0.25 * e.value_score, abs=1e-6)
            assert float(e.routing_contribution.mean()) == pytest.approx(e.routing_score, abs=1e-6)
            assert float(e.value_slot_cosine.mean()) == pytest.approx(e.value_score, abs=1e-6)
            assert e.routing_score == pytest.approx(float(scores.routing[i, j]), abs=1e-6)
            assert e.value_score == pytest.approx(float(scores.value[i, j]), abs=1e-6)
            assert e.score == pytest.approx(float(scores.final[i, j]), abs=1e-6)
            torch.testing.assert_close(e.value_attention.sum(-1), torch.ones(4))
            cosine = F.normalize(q.routing[i], dim=-1) @ F.normalize(d.routing[j], dim=-1).T
            torch.testing.assert_close(e.slot_similarity, cosine)
            if mode == "max":
                torch.testing.assert_close(e.routing_contribution, cosine.max(-1).values)
            else:  # each term lies between the mean and the max cosine of its row
                assert bool((e.routing_contribution <= cosine.max(-1).values + 1e-6).all())
                assert bool((e.routing_contribution >= cosine.mean(-1) - 1e-6).all())
            matches = e.slot_matches()
            assert [m["document_slot"] for m in matches] == cosine.argmax(-1).tolist()
            assert e.to_dict()["settings"]["qk_mode"] == mode


def test_explain_reads_raw_document_values():
    q = Embeddings(torch.tensor([[[1., 0.]]]), torch.tensor([[[1., 0.]]]), "test", "query")
    d = Embeddings(torch.tensor([[[1., 0.], [1., 0.]]]), torch.tensor([[[10., 0.], [0., 1.]]]), "test", "document")
    e = explain(q, d)[0][0]
    assert e.value_score == pytest.approx(10 / math.sqrt(101))
    torch.testing.assert_close(e.value_attention, torch.tensor([[0.5, 0.5]]))


def test_incompatible_representations_fail():
    q = Embeddings(torch.ones(1, 2, 3), torch.ones(1, 2, 3), "old", "query")
    d = Embeddings(torch.ones(1, 2, 3), torch.ones(1, 2, 3), "new", "document")
    with pytest.raises(ValueError, match="Incompatible"):
        score(q, d)
    with pytest.raises(ValueError):
        ScoringConfig(tau_v=float("nan"))

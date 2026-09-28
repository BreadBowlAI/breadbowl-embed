import math
import pytest
import torch
import torch.nn.functional as F
from breadbowl_embed import Embeddings, ScoringConfig, score


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


def test_incompatible_representations_fail():
    q = Embeddings(torch.ones(1, 2, 3), torch.ones(1, 2, 3), "old", "query")
    d = Embeddings(torch.ones(1, 2, 3), torch.ones(1, 2, 3), "new", "document")
    with pytest.raises(ValueError, match="Incompatible"):
        score(q, d)
    with pytest.raises(ValueError):
        ScoringConfig(tau_v=float("nan"))

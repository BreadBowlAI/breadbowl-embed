import pytest
import torch
from breadbowl_embed import Embeddings, Index, score


def test_search_matches_exhaustive_rerank_and_survives_reload(tmp_path):
    torch.manual_seed(41)
    d = Embeddings(torch.randn(11, 3, 4), torch.randn(11, 3, 6), "test", "document")
    q = Embeddings(torch.randn(2, 3, 4), torch.randn(2, 3, 6), "test", "query")
    ids = [f"d{i:02d}" for i in range(11)]
    index = Index(d, ids)
    result = index.search(q, top_k=4, candidate_k=11, batch_size=2)
    expected = score(q, d).combined
    for i in range(2):
        expected_ids = sorted(range(11), key=lambda j: (-float(expected[i, j]), ids[j]))[:4]
        assert [r.id for r in result[i]] == [ids[j] for j in expected_ids]
    index.save(tmp_path / "index")
    loaded = Index.load(tmp_path / "index")
    assert loaded.search(q, top_k=4, candidate_k=11, batch_size=2) == result


def test_routing_candidate_selection_and_stable_ties():
    d = Embeddings(torch.ones(3, 2, 2), torch.ones(3, 2, 2), "test", "document")
    q = Embeddings(torch.ones(1, 2, 2), torch.ones(1, 2, 2), "test", "query")
    index = Index(d, ["z", "a", "b"])
    assert [r.id for r in index.search(q, top_k=2, candidate_k=2, batch_size=1)[0]] == ["a", "b"]
    with pytest.raises(ValueError):
        index.search(q, top_k=3, candidate_k=2)


def test_corrupt_artifacts_and_no_overwrite(tmp_path):
    d = Embeddings(torch.ones(1, 2, 2), torch.ones(1, 2, 2), "test", "document")
    d.save(tmp_path / "slots")
    with pytest.raises(FileExistsError):
        d.save(tmp_path / "slots")
    path = tmp_path / "slots/embeddings.safetensors"
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        Embeddings.load(tmp_path / "slots")


def test_empty_corpus():
    d = Embeddings(torch.empty(0, 2, 2), torch.empty(0, 2, 2), "test", "document")
    q = Embeddings(torch.ones(1, 2, 2), torch.ones(1, 2, 2), "test", "query")
    assert Index(d, []).search(q) == [[]]

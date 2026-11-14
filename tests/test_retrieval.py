import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from breadbowl_embed import Embeddings, Index, SearchProfile, score
from breadbowl_embed._search import coarse_candidates, search_slots_exact
from breadbowl_embed.config import canonical_json
from breadbowl_embed.retrieval import PROFILES, bytes_per_document, resolve_profile


def corpus(n=60, slots=4, d_k=8, d_v=6, queries=5, seed=3, rid="test"):
    generator = torch.Generator().manual_seed(seed)
    docs = Embeddings(F.normalize(torch.randn(n, slots, d_k, generator=generator), dim=-1),
                      2 * torch.randn(n, slots, d_v, generator=generator), rid, "document")
    q = Embeddings(F.normalize(torch.randn(queries, slots, d_k, generator=generator), dim=-1),
                   torch.randn(queries, slots, d_v, generator=generator), rid, "query")
    return docs, q, [f"d{i:03d}" for i in range(n)]


def brute_force(q, docs, ids, top_k):
    final = score(q, docs).combined
    return [[ids[j] for j in sorted(range(len(ids)), key=lambda j: (-float(final[i, j]), ids[j]))[:top_k]]
            for i in range(len(q))]


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
    # The exact profile breaks routing ties by id; slot-shortlist profiles follow the research
    # pipeline and break them by corpus position before the final (score, id) order.
    assert [r.id for r in index.search(q, top_k=2, candidate_k=2, batch_size=1, profile="exact")[0]] == ["a", "b"]
    assert [r.id for r in index.search(q, top_k=2, candidate_k=2, batch_size=1)[0]] == ["a", "z"]
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
    assert Index(d, []).search(q, profile="exact") == [[]]


def test_profiles_are_named_recorded_and_validated(tmp_path):
    assert (PROFILES["paper"].slot_hits, PROFILES["paper"].shortlist, PROFILES["paper"].rerank) == (128, 1024, 100)
    assert (PROFILES["default"].slot_hits, PROFILES["default"].shortlist, PROFILES["default"].rerank) == (512, 1024, 100)
    assert resolve_profile(None) == PROFILES["default"]
    assert resolve_profile("reference-slot-shortlist-v1") == PROFILES["paper"]
    assert resolve_profile("exact-document-routing-v1") == PROFILES["exact"]
    assert resolve_profile("paper", 50).name == "custom" and resolve_profile("paper", 100).name == "paper"
    with pytest.raises(ValueError):
        resolve_profile("exact", slot_hits=4)
    with pytest.raises(ValueError):
        resolve_profile("fastest")
    docs, q, ids = corpus(n=20)
    index = Index(docs, ids, profile="paper")
    hits = index.search(q, top_k=3)
    assert all(hit.profile == PROFILES["paper"].to_dict() for row in hits for hit in row)
    assert index.search(q, top_k=3, profile="exact")[0][0].profile["method"] == "exact-routing"
    index.save(tmp_path / "index")
    manifest = json.loads((tmp_path / "index/index.json").read_text())
    assert manifest["default_profile"] == PROFILES["paper"].to_dict()
    assert Index.load(tmp_path / "index").profile == PROFILES["paper"]


@pytest.mark.parametrize("storage", ["float32", "float16", "int8"])
@pytest.mark.parametrize("mmap", [False, True])
def test_storage_roundtrip(tmp_path, storage, mmap):
    docs, q, ids = corpus()
    index = Index(docs, ids, storage=storage, metadata=[{"n": i} for i in range(len(ids))])
    before = index.search(q, top_k=5)
    index.save(tmp_path / "index")
    loaded = Index.load(tmp_path / "index", mmap=mmap)
    assert loaded.storage == storage and loaded.ids == ids
    assert loaded.search(q, top_k=5) == before
    assert before[0][0].metadata == {"n": before[0][0].document_index}
    manifest = json.loads((tmp_path / "index/index.json").read_text())
    stored = {name: entry["dtype"] for name, entry in manifest["shards"][0]["files"].items()}
    expected = {"float32": {"routing": "float32", "value": "float32"},
                "float16": {"routing": "float16", "value": "float16"},
                "int8": {"routing": "int8", "value": "int8", "value_scale": "float32"}}[storage]
    assert stored == expected
    with pytest.raises(FileExistsError):
        index.save(tmp_path / "index")


def test_storage_sizes_match_the_paper():
    assert [bytes_per_document(s, 16, 256, 256) for s in ("float32", "float16", "int8")] == [32768, 16384, 8256]


def test_quantisation_error_bounds():
    docs, _, ids = corpus(n=40)
    for storage, routing_cosine, value_error in [("float16", 0.99999, 2e-3), ("int8", 0.9995, None)]:
        index = Index(docs, ids, storage=storage)
        everything = np.arange(len(ids))
        routing = F.normalize(index.routing_vectors(everything), dim=-1)
        assert float((routing * docs.routing).sum(-1).min()) > routing_cosine
        value = index.value_vectors(everything)
        if value_error is None:  # int8: at most half a quantisation step per number
            step = docs.value.abs().amax(-1, keepdim=True) / 127
            assert bool(((value - docs.value).abs() <= step / 2 + 1e-6).all())
        else:
            assert float(((value - docs.value).abs() / docs.value.abs().clamp_min(1)).max()) < value_error


def test_corrupt_shard_and_identity_refusal(tmp_path, tiny_model):
    docs, q, ids = corpus(n=10)
    Index(docs, ids).save(tmp_path / "index")
    with pytest.raises(ValueError, match="different model"):
        Index.load(tmp_path / "index", representation_id="another-model")
    assert len(Index.load(tmp_path / "index", representation_id="test")) == 10
    with pytest.raises(ValueError, match="different model"):
        tiny_model.load_index(tmp_path / "index")
    own = tiny_model.index(tiny_model.encode_documents(["cats land", "fish water"]), ids=["a", "b"])
    own.save(tmp_path / "own")
    assert tiny_model.load_index(tmp_path / "own").ids == ["a", "b"]
    shard = tmp_path / "index/shard-00000.value.npy"
    raw = shard.read_bytes()
    shard.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        Index.load(tmp_path / "index")
    other = Embeddings(q.routing, q.value, "another-model", "query")
    with pytest.raises(ValueError, match="Incompatible"):
        Index(docs, ids).search(other)


def test_v1_index_directory_still_loads(tmp_path):
    """The v0.1 on-disk index (safetensors slots + index.json) keeps its exact-routing behaviour."""
    docs, q, ids = corpus(n=30)
    root = tmp_path / "v1"
    root.mkdir()
    docs.save(root / "slots")
    (root / "index.json").write_text(canonical_json({
        "format": "breadbowl_index_v1", "profile": "exact-document-routing-v1", "ids": ids,
        "metadata": [{} for _ in ids], "scoring": {"qk_mode": "logmeanexp", "tau_k": 0.02, "tau_v": 0.1,
                                                   "value_mix": 0.25}}))
    loaded = Index.load(root, representation_id="test")
    assert loaded.profile == PROFILES["exact"]
    assert loaded.search(q, top_k=5, candidate_k=7) == Index(docs, ids, profile="exact").search(q, top_k=5, candidate_k=7)


def test_add_equals_single_build_and_rejects_duplicates():
    docs, q, ids = corpus(n=50)
    whole = Index(docs, ids, storage="int8")
    parts = Index(docs.select(slice(0, 20)), ids[:20], storage="int8")
    parts.add(docs.select(slice(20, 50)), ids[20:])
    assert parts.search(q, top_k=8) == whole.search(q, top_k=8)
    assert parts.search(q, top_k=8, profile="exact") == whole.search(q, top_k=8, profile="exact")
    with pytest.raises(ValueError, match="unique"):
        parts.add(docs.select([0]), [ids[0]])
    uncached = Index(docs, ids, storage="int8")
    uncached.cache_routing = False
    assert uncached.search(q, top_k=8, profile="paper") == whole.search(q, top_k=8, profile="paper")


def test_slot_shortlist_with_full_coverage_equals_brute_force():
    docs, q, ids = corpus(n=60)
    wide = SearchProfile("custom", "slot-shortlist", slot_hits=60 * 4, shortlist=60, rerank=60)
    got = [[hit.id for hit in row] for row in Index(docs, ids).search(q, top_k=10, profile=wide)]
    assert got == brute_force(q, docs, ids, 10)
    exact = [[hit.id for hit in row] for row in Index(docs, ids).search(q, top_k=10, profile="exact", candidate_k=60)]
    assert exact == brute_force(q, docs, ids, 10)


def test_narrow_slot_shortlist_follows_its_definition():
    """Shortlist = documents ranked by mean over query slots of their best hit cosine (missing = -1)."""
    docs, q, ids = corpus(n=60)
    unit = F.normalize(q.routing, dim=-1).numpy()
    blocks = [(0, docs.routing[:25].numpy()), (25, docs.routing[25:].numpy())]
    scores, hits = search_slots_exact(unit, blocks, 6)
    sims = np.einsum("qsd,ntd->qsnt", unit, docs.routing.numpy()).reshape(len(q), 4, -1)
    for i in range(len(q)):
        expected = np.sort(sims[i], axis=-1)[:, ::-1][:, :6]
        np.testing.assert_allclose(scores[i], expected, rtol=1e-6, atol=1e-6)
        best = {}
        for slot in range(4):
            for value, hit in zip(scores[i, slot], hits[i, slot]):
                doc = int(hit) // 4
                best.setdefault(doc, [-1.0] * 4)
                best[doc][slot] = max(best[doc][slot], float(value))
        means = {doc: sum(v) / 4 for doc, v in best.items()}
        expected_shortlist = sorted(means, key=lambda doc: -means[doc])[:7]
        assert coarse_candidates(scores[i], hits[i], 4, 7) == expected_shortlist
    profile = SearchProfile("custom", "slot-shortlist", slot_hits=6, shortlist=7, rerank=3)
    for i, row in enumerate(Index(docs, ids).search(q, top_k=3, profile=profile, explain=True)):
        shortlist = coarse_candidates(scores[i], hits[i], 4, 7)
        assert {hit.document_index for hit in row} <= set(shortlist)
        for hit in row:  # the slot hits that put the document on the shortlist are reported
            reported = {(h["query_slot"], h["document_slot"]) for h in hit.explanation["slot_hits"]}
            expected = {(s, int(h) % 4) for s in range(4) for h in hits[i, s] if int(h) // 4 == hit.document_index}
            assert reported == expected and reported
            assert len(hit.explanation["slot_matches"]) == 4


def test_faiss_backend_matches_torch():
    pytest.importorskip("faiss")
    docs, q, ids = corpus(n=80)
    index = Index(docs, ids)
    for profile in ("paper", "default"):
        assert index.search(q, top_k=10, profile=profile, backend="faiss") == index.search(q, top_k=10, profile=profile)

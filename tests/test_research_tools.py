"""breadbowl_embed.research: diagnostics, precision experiments, reports, JSON export, plots and the CLI."""
import json
import re

import pytest
import torch
import torch.nn.functional as F

from breadbowl_embed import BreadBowl, Embeddings, Index
from breadbowl_embed.cli import main
from breadbowl_embed.retrieval import resolve_profile
from breadbowl_embed.research import (compare_precision, compare_rules, decompose, inspect_pair, occlusion_sweep,
                                      shortlist_diagnostics, slot_ablation, slot_statistics, trace,
                                      with_compute_dtype)


def corpus(n=40, slots=4, d_k=8, d_v=6, queries=6, seed=3):
    g = torch.Generator().manual_seed(seed)
    docs = Embeddings(F.normalize(torch.randn(n, slots, d_k, generator=g), dim=-1),
                      2 * torch.randn(n, slots, d_v, generator=g), "test", "document")
    q = Embeddings(F.normalize(torch.randn(queries, slots, d_k, generator=g), dim=-1),
                   torch.randn(queries, slots, d_v, generator=g), "test", "query")
    return docs, q, [f"d{i:03d}" for i in range(n)]


def test_slot_statistics_detects_collapse_and_constant_slots():
    g = torch.Generator().manual_seed(1)
    routing = F.normalize(torch.randn(30, 4, 8, generator=g), dim=-1)
    routing[:, 1] = routing[:, 0]  # slots 0 and 1 collapsed onto one direction
    routing[:, 3] = F.normalize(torch.ones(8), dim=0)  # slot 3 is the same for every text
    docs = Embeddings(routing, torch.randn(30, 4, 5, generator=g), "test", "document")
    stats = slot_statistics(docs)
    assert stats.collapse["most_similar_pair"] == [0, 1]
    assert stats.collapse["max_within_text_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert stats.per_slot[3]["routing_resultant"] == pytest.approx(1.0, abs=1e-6)
    assert stats.per_slot[3]["routing_variance"] == pytest.approx(0.0, abs=1e-6)
    assert stats.per_slot[3]["routing_effective_rank"] < 1e-3 < stats.per_slot[0]["routing_effective_rank"]
    assert stats.per_text_effective_rank["routing_max"] <= 3.0 + 1e-6  # at most 3 distinct directions per text
    torch.testing.assert_close(torch.diagonal(stats.within_text_routing_cosine), torch.ones(4))
    assert stats.usage is None and stats.to_dict()["type"] == "slot_statistics"
    docs_corpus, queries, ids = corpus()
    with_usage = slot_statistics(Index(docs_corpus, ids, storage="int8"), queries=queries, top_k=3)
    usage = with_usage.usage
    assert usage["pairs"] == 6 * 3 and float(usage["document_slot_win_rate"].sum()) == pytest.approx(1.0)
    assert float(usage["document_slot_score_share"].sum()) + float(usage["query_slot_score_share"].sum()) > 0
    assert usage["best_match_counts"].sum() == 6 * 3 * 4
    sampled = slot_statistics(docs_corpus, sample=10)
    assert sampled.count == 10


def test_shortlist_diagnostics_agree_with_search_and_detect_underfill():
    docs, queries, ids = corpus()
    index = Index(docs, ids)
    widths = [1, 2, 8, 160]
    diagnostics = shortlist_diagnostics(index, queries, profile="paper", slot_hits=widths, top_k=5, per_query=True)
    rows = {r["slot_hits"]: r for r in diagnostics.rows}
    assert set(rows) == set(widths) and diagnostics.reference["method"] == "exact-routing"
    # Everything hit: the profile equals exact search.
    assert rows[160]["candidate_recall"] == 1.0 and rows[160]["result_recall_at_k"] == 1.0
    assert rows[160]["underfilled"] == 0 and rows[160]["candidates_mean"] == 40
    # One hit per query slot: at most 4 documents reach reranking.
    assert rows[1]["underfilled"] == 6 and rows[1]["candidates_mean"] <= 4
    assert rows[1]["underfill_rate"] == 1.0
    for width in widths:  # the profile's results are exactly Index.search's
        expected = index.search(queries, top_k=5, profile=resolve_profile("paper", slot_hits=width))
        got = [d["results"] for d in diagnostics.per_query if d["slot_hits"] == width]
        assert got == [[r.document_index for r in row] for row in expected], width
    exact = index.search(queries, top_k=5, profile="exact")
    assert [d["exact_results"] for d in diagnostics.per_query if d["slot_hits"] == 1] == \
        [[r.document_index for r in row] for row in exact]
    recall = diagnostics.per_slot_hit_recall
    assert len(recall["160"]) == 4 and all(v == 1.0 for v in recall["160"])
    with pytest.raises(ValueError):
        shortlist_diagnostics(index, queries, profile="exact")


def test_with_compute_dtype_equals_loading_in_that_dtype(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path / "release")
    native = BreadBowl.from_pretrained(tmp_path / "release")
    loaded = BreadBowl.from_pretrained(tmp_path / "release", dtype="bfloat16")
    derived = with_compute_dtype(native, "bfloat16")
    assert derived.representation_id == loaded.representation_id and derived.compute_dtype == "bfloat16"
    for (name, a), (_, b) in zip(derived.network.state_dict().items(), loaded.network.state_dict().items()):
        assert a.dtype == b.dtype and torch.equal(a, b), name
    texts = ["fish water", "cats land dogs"]
    assert torch.equal(derived.encode_documents(texts).value, loaded.encode_documents(texts).value)
    assert {p.dtype for p in native.network.parameters()} == {torch.float32}  # the source is untouched
    assert with_compute_dtype(native, "native") is native
    with pytest.raises(ValueError):
        with_compute_dtype(loaded, "float32")


def test_compare_precision(tiny_model):
    queries, documents = ["fish", "cats land"], ["fish water", "cats land", "dogs water", "land fish"]
    result = compare_precision(tiny_model, queries, documents, batch_sizes=(1, 3), top_k=2)
    names = [r["variant"] for r in result.rows]
    assert result.reference == "compute=float32 batch=1 storage=float32" and len(names) == 2 * 2 * 3
    reference = next(r for r in result.rows if r["variant"] == result.reference)
    assert reference["max_abs_delta"] == 0.0 and reference["kendall_tau"] == 1.0
    expected = tiny_model.score(tiny_model.encode_queries(queries), tiny_model.encode_documents(documents))
    assert torch.equal(result.scores[result.reference]["final"], expected.combined)
    int8 = next(r for r in result.rows if r["variant"] == "compute=float32 batch=1 storage=int8")
    assert 0 < int8["max_abs_delta"] < 0.05
    bf16 = next(r for r in result.rows if r["variant"] == "compute=bfloat16 batch=1 storage=float32")
    assert bf16["max_abs_delta"] > 0 and bf16["encode_seconds"] >= 0
    assert result.to_dict()["rows"][0]["variant"] == names[0]


def _all_objects(model):
    pair = inspect_pair(model, "fish <b>water</b>", "fish water cats land")
    encoded_q = model.encode_queries(["fish", "cats"])
    encoded_d = model.encode_documents(["fish water", "cats land", "dogs"])
    docs, queries, ids = corpus(n=10)
    traces = trace(model, ["fish water"], "document", capture="all")
    return {
        "pair": pair, "attention": pair.query, "tokens": pair.query.tokens, "decomposition": pair.decomposition,
        "attribution": pair.attribution(), "trace": traces[0], "traces": traces,
        "sweep": occlusion_sweep(model, "fish", "fish water cats"),
        "ablation": slot_ablation(encoded_q, encoded_d), "rules": compare_rules(encoded_q, encoded_d),
        "stats": slot_statistics(docs, queries=queries),
        "shortlist": shortlist_diagnostics(Index(docs, ids), queries, slot_hits=[2, 8]),
        "precision": compare_precision(model, ["fish"], ["fish water", "cats"], compute_dtypes=["float32"]),
        "decomposition_grid": decompose(encoded_q, encoded_d)[0][0],
    }


def test_every_object_exports_json_and_static_html(tiny_model, tmp_path):
    for name, obj in _all_objects(tiny_model).items():
        data = json.loads(obj.to_json())
        assert data["type"] == obj.json_type and data["schema"] == 1 and "library_version" in data, name
        assert json.loads(obj.to_json(digits=3)) is not None
        path = obj.save_json(tmp_path / f"{name}.json")
        assert json.loads(path.read_text())["type"] == obj.json_type
        html = obj._repr_html_()
        assert "<script" not in html.lower() and "<div" in html, name
    rounded = _all_objects(tiny_model)["decomposition"].to_dict(digits=2)
    assert all(len(repr(v).split(".")[-1]) <= 2 for v in rounded["slots"][0].values() if isinstance(v, float))


def test_html_report_is_self_contained_and_escapes_user_text(tiny_model, tmp_path):
    hostile = "fish </script><script>alert(1)</script> water & <img src=x onerror=alert(2)>"
    pair = inspect_pair(tiny_model, hostile, "fish water cats\u2028land")
    html = pair.report(tmp_path / "report.html", title="Report <for> you")
    assert (tmp_path / "report.html").read_text(encoding="utf-8") == html
    assert html.startswith("<!doctype html>") and "Report &lt;for&gt; you" in html
    assert "<script>alert(1)" not in html and "<img src=x" not in html
    assert html.count("<script") == 2  # the JSON data block and the inline script
    # nothing is loaded from the network
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", html)
    assert "@import" not in html and "url(" not in html
    block = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', html, re.S)[1]
    assert "<" not in block and "\u2028" not in block  # nothing can end the script element or a JS string
    data = json.loads(block)
    assert data["slots"] == {"query": 3, "document": 3} and data["heads"] == 2
    assert len(data["query"]["tokens"]) == len(pair.query.tokens) and len(data["matrices"]["contribution"]["data"]) == 3
    assert any("</script>" in t[0] or "script" in t[0] for t in data["query"]["tokens"]) or \
        "[UNK]" in "".join(t[2] for t in data["query"]["tokens"])
    assert "prefers-color-scheme: dark" in html and 'data-theme="dark"' in html
    assert 'role="radiogroup"' in html and 'aria-live="polite"' in html


def test_plots(tiny_model):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    objects = _all_objects(tiny_model)
    from breadbowl_embed.research import plot
    axes = [objects["attention"].plot(), objects["decomposition"].plot(kind="routing_weight"),
            objects["attribution"].plot(k=3), objects["sweep"].plot(), objects["stats"].plot(),
            objects["shortlist"].plot(), objects["precision"].plot(),
            plot.attention(objects["attention"], head=1, segments=("text",))]
    assert all(hasattr(ax, "figure") for ax in axes)
    with pytest.raises(ValueError):
        objects["decomposition"].plot(kind="nope")
    plt.close("all")


def run(capsys, *args) -> list[dict]:
    assert main([str(a) for a in args]) == 0
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_inspect_cli(capsys, tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path / "release")
    release = tmp_path / "release"
    [pair] = run(capsys, "inspect", "pair", "--model", release, "--query", "fish", "--document", "fish water cats",
                 "--html", tmp_path / "pair.html")
    assert pair["html"].endswith("pair.html") and (tmp_path / "pair.html").read_text().startswith("<!doctype html>")
    [full] = run(capsys, "inspect", "pair", "--model", release, "--query", "fish", "--document", "fish water",
                 "--digits", "4")
    assert full["type"] == "pair_inspection" and set(full["scores"]) == {"final", "routing", "value"}
    rows = run(capsys, "inspect", "attention", "--model", release, "--kind", "query", "--text", "fish",
               "--text", "cats land", "--per-head", "--logits")
    assert len(rows) == 2 and len(rows[0]["attention"]) == 2 and rows[0]["tokens"][0]["segment"] == "instruction"
    [traced] = run(capsys, "inspect", "trace", "--model", release, "--kind", "document", "--text", "fish water",
                   "--layers", "0", "--output", tmp_path / "trace")
    assert (tmp_path / "trace/tensors.safetensors").is_file() and "layer_0" in traced["tensors"]
    [sweep] = run(capsys, "inspect", "occlude", "--model", release, "--query", "fish", "--document",
                  "fish water cats", "--output", tmp_path / "sweep.json")
    assert json.loads((tmp_path / "sweep.json").read_text())["type"] == "occlusion_sweep" == sweep["type"]
    ablations = run(capsys, "inspect", "ablate", "--model", release, "--query", "fish", "--query", "cats",
                    "--document", "fish water", "--document", "cats land")
    assert [a["query_index"] for a in ablations] == [0, 1]
    [rules] = run(capsys, "inspect", "rules", "--model", release, "--query", "fish", "--document", "fish water",
                  "--document", "cats", "--rules", "model,maxsim")
    assert [r["name"] for r in rules["rules"]] == ["model", "maxsim"]
    docs = tmp_path / "docs.txt"
    docs.write_text("fish water\ncats land\ndogs land\nfish cats\n")
    run(capsys, "index", "--model", release, "--input", docs, "--output", tmp_path / "index")
    [stats] = run(capsys, "inspect", "slots", "--index", tmp_path / "index")
    assert stats["count"] == 4 and stats["usage"] is None
    [stats] = run(capsys, "inspect", "slots", "--model", release, "--index", tmp_path / "index", "--queries", docs)
    assert stats["usage"]["pairs"] == 4 * 4
    [short] = run(capsys, "inspect", "shortlist", "--model", release, "--index", tmp_path / "index", "--query", "fish",
                  "--slot-hits", "1,64")
    assert [r["slot_hits"] for r in short["rows"]] == [1, 64]
    [precision] = run(capsys, "inspect", "precision", "--model", release, "--query", "fish", "--document",
                      "fish water", "--document", "cats", "--storages", "int8", "--compute-dtypes", "float32")
    assert [r["storage"] for r in precision["rows"]] == ["float32", "int8"] and precision["scores"] == {}
    with pytest.raises(SystemExit):
        main(["inspect", "pair", "--query", "fish", "--document", "fish"])

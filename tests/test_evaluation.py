"""Evaluation toolkit: metric definitions, data files, statistics, resumable encoding, runs, CLI."""
import gzip
import json
import math

import numpy as np
import pytest
import torch

from breadbowl_embed import Embeddings
from breadbowl_embed.evaluation import (EvaluationResult, bootstrap_ci, build_index, compare, evaluate_rerank,
                                        evaluate_retrieval, evaluate_run, load_beir, load_qrels, load_queries,
                                        load_result, paired_test, query_metrics, rank, read_build, read_run,
                                        write_run)
from breadbowl_embed.evaluation.cli import main
from breadbowl_embed.evaluation.data import document_text, load_corpus

DOCS = {"a": "cats land", "b": "fish water", "c": "dogs land", "d": "cats water", "e": "fish land",
        "f": "dogs water", "g": "cats dogs"}
QUERIES = {"q1": "fish", "q2": "cats land", "q3": "dogs", "q4": "water"}
QRELS = {"q1": {"b": 1, "e": 2, "a": 0}, "q2": {"a": 2, "d": 1}, "q3": {"c": 0}, "q4": {"b": 1, "d": 1, "f": 1}}


# ---- metrics: hand-computed cases -------------------------------------------------------------
JUDGED = {"d1": 2, "d2": 1, "d3": 0, "d5": 1}
RANKING = ["d3", "d1", "d4", "d2", "d5"]


def test_metrics_match_hand_computed_values():
    m = query_metrics(RANKING, JUDGED, ["ndcg@3", "ndcg@5", "mrr@1", "mrr@3", "recall@3", "recall@5", "map@3",
                                        "map@5"])
    log2 = math.log2
    assert m["ndcg@3"] == pytest.approx((2 / log2(3)) / (2 + 1 / log2(3) + 1 / log2(4)), abs=1e-15)
    assert m["ndcg@5"] == pytest.approx((2 / log2(3) + 1 / log2(5) + 1 / log2(6)) / (2 + 1 / log2(3) + 1 / log2(4)),
                                        abs=1e-15)
    assert (m["mrr@1"], m["mrr@3"]) == (0.0, 0.5)
    assert (m["recall@3"], m["recall@5"]) == (pytest.approx(1 / 3), 1.0)
    # MAP@k divides by all relevant documents (3), not by min(k, 3): trec_eval's map_cut.
    assert m["map@3"] == pytest.approx((1 / 2) / 3)
    assert m["map@5"] == pytest.approx((1 / 2 + 2 / 4 + 3 / 5) / 3)
    strict = query_metrics(RANKING, JUDGED, ["ndcg@3", "recall@3", "map@5", "mrr@3"], relevance_level=2)
    assert strict["ndcg@3"] == m["ndcg@3"]  # nDCG is graded; relevance_level is for binary metrics
    assert (strict["recall@3"], strict["map@5"], strict["mrr@3"]) == (1.0, 0.5, 0.5)


def test_perfect_and_empty_rankings():
    perfect = query_metrics(["d1", "d2", "d5"], JUDGED, ["ndcg@10", "mrr@10", "recall@10", "map@10"])
    assert perfect == {"ndcg@10": 1.0, "mrr@10": 1.0, "recall@10": 1.0, "map@10": 1.0}
    assert set(query_metrics([], JUDGED, ["ndcg@10", "map@10"]).values()) == {0.0}
    # Unjudged documents count as non-relevant; negative judgments as gain 0.
    assert query_metrics(["x", "d1"], {"d1": 1, "x": -1}, ["ndcg@2"])["ndcg@2"] == pytest.approx(1 / math.log2(3))


def test_query_set_rules():
    qrels = {"q1": {"a": 1}, "q2": {"a": 0, "b": 0}, "q3": {"b": 2}, "q4": {"c": 1}}
    run = {"q1": ["a", "b"], "q2": ["a"], "q3": ["a", "b"], "q9": ["a"]}
    result = evaluate_run(qrels, run, ["ndcg@10", "recall@10"], relevance_level=1)
    assert result.skipped == ["q2"] and result.missing == ["q4"] and result.ignored == 2  # q2 (skipped) and q9
    assert result.per_query["q4"] == {"ndcg@10": 0.0, "recall@10": 0.0}
    assert result.mean["recall@10"] == pytest.approx((1 + 1 + 0) / 3)
    # A query with positives only below relevance_level stays in the mean with 0 for binary metrics.
    high = evaluate_run(qrels, run, ["recall@10", "ndcg@10"], relevance_level=2)
    assert high.per_query["q1"]["recall@10"] == 0.0 and high.per_query["q1"]["ndcg@10"] == 1.0
    with pytest.raises(ValueError):
        evaluate_run({"q": {"a": 0}}, {}, ["ndcg@10"])
    with pytest.raises(ValueError):
        evaluate_run(qrels, run, ["ndcg"])
    with pytest.raises(ValueError):
        evaluate_run(qrels, {"q1": ["a", "a"]}, ["ndcg@10"])


def test_tie_breaking():
    scores = {"b": 1.0, "a": 1.0, "c": 0.5, "z": 2.0}
    assert rank(scores) == ["z", "a", "b", "c"]
    assert rank(scores, tie_break="trec_eval") == ["z", "b", "a", "c"]
    qrels = {"q": {"a": 1}}
    assert evaluate_run(qrels, {"q": scores}, ["mrr@10"]).mean["mrr@10"] == 1 / 2
    assert evaluate_run(qrels, {"q": scores}, ["mrr@10"], tie_break="trec_eval").mean["mrr@10"] == 1 / 3
    with pytest.raises(ValueError):
        rank({"a": float("nan")})


# ---- data files ---------------------------------------------------------------------------------
def write_beir(root, *, gz=False):
    (root / "qrels").mkdir(parents=True)
    rows = [{"_id": k, "title": "", "text": v} for k, v in DOCS.items()]
    corpus = "\n".join(json.dumps(r) for r in rows) + "\n"
    if gz:
        with gzip.open(root / "corpus.jsonl.gz", "wt", encoding="utf-8") as handle:
            handle.write(corpus)
    else:
        (root / "corpus.jsonl").write_text(corpus)
    (root / "queries.jsonl").write_text("\n".join(json.dumps({"_id": k, "text": v}) for k, v in QUERIES.items()) + "\n")
    lines = ["query-id\tcorpus-id\tscore"] + [f"{q}\t{d}\t{r}" for q, docs in QRELS.items() for d, r in docs.items()]
    (root / "qrels" / "test.tsv").write_text("\n".join(lines) + "\n")
    return root


def test_beir_trec_and_run_files(tmp_path):
    for gz in (False, True):
        data = load_beir(write_beir(tmp_path / f"beir-{gz}", gz=gz), "test")
        assert data.queries == QUERIES and data.qrels == QRELS
        assert {k: v["text"] for k, v in data.corpus.items()} == DOCS
        assert data.describe()["qrels_sha256"] == load_beir(tmp_path / f"beir-{gz}").describe()["qrels_sha256"]
    (tmp_path / "trec.qrels").write_text("q1 0 b 1\nq1 0 e 2\nq2 Q0 a 0\n")
    assert load_qrels(tmp_path / "trec.qrels") == {"q1": {"b": 1, "e": 2}, "q2": {"a": 0}}
    (tmp_path / "queries.tsv").write_text("q1\tfish\nq2\tcats land\n")
    assert load_queries(tmp_path / "queries.tsv") == {"q1": "fish", "q2": "cats land"}
    (tmp_path / "corpus.tsv").write_text("id\ttitle\ttext\nx\tT\tbody\ny\t\tother\n")
    assert load_corpus(tmp_path / "corpus.tsv") == {"x": {"title": "T", "text": "body"},
                                                    "y": {"title": "", "text": "other"}}
    assert document_text({"title": "T", "text": " body "}) == "T  body"
    assert document_text({"title": "T", "text": "body"}, "text") == "body"
    run = {"q1": [("b", 0.5), ("e", 0.5), ("a", -1e-9)], "q2": [("a", 1 / 3)]}
    write_run(tmp_path / "run.trec", run)
    back = read_run(tmp_path / "run.trec")
    assert back == {"q1": {"b": 0.5, "e": 0.5, "a": -1e-9}, "q2": {"a": 1 / 3}}
    assert rank(back["q1"]) == ["b", "e", "a"]  # written order survives a round trip (ties by id)
    (tmp_path / "dup.trec").write_text("q1 Q0 a 1 1.0 t\nq1 Q0 a 2 0.5 t\n")
    with pytest.raises(ValueError, match="twice"):
        read_run(tmp_path / "dup.trec")
    (tmp_path / "bad.tsv").write_text("query-id\tcorpus-id\tscore\nq1\ta\t1.5\n")
    with pytest.raises(ValueError, match="integer"):
        load_qrels(tmp_path / "bad.tsv")


# ---- statistics ---------------------------------------------------------------------------------
def test_bootstrap_and_paired_test():
    constant = bootstrap_ci([0.5] * 20)
    assert constant["low"] == constant["high"] == constant["mean"] == 0.5
    values = {f"q{i}": float(v) for i, v in enumerate(np.random.default_rng(1).random(50))}
    first, again = bootstrap_ci(values, seed=3), bootstrap_ci(values, seed=3)
    assert first == again and first["low"] < first["mean"] < first["high"]
    same = paired_test(values, values)
    assert same["difference"] == 0 and same["p_value"] == 1.0
    better = {k: v + 0.1 for k, v in values.items()}
    result = paired_test(values, better, resamples=2000)
    assert result["difference"] == pytest.approx(0.1)
    assert result["p_value"] == 1 / 2001 and result["low"] == pytest.approx(0.1) and result["b_better"] == 50
    noisy = {k: v + (0.01 if i % 2 else -0.01) for i, (k, v) in enumerate(values.items())}
    assert paired_test(values, noisy)["p_value"] > 0.5
    with pytest.raises(ValueError, match="different queries"):
        paired_test(values, {"x": 1.0})


# ---- resumable encoding -------------------------------------------------------------------------
class Stop(Exception):
    pass


def stored(index):
    rows = list(range(len(index)))
    return index.routing_vectors(rows), index.value_vectors(rows)


def test_build_index_resumes_and_equals_a_direct_build(tiny_model, tmp_path):
    def stop_after_first_part(done, total, encoded, seconds):
        raise Stop

    with pytest.raises(Stop):
        build_index(tiny_model, DOCS, tmp_path / "build", chunk_size=3, progress=stop_after_first_part)
    assert sorted(p.name for p in (tmp_path / "build/parts").iterdir() if p.is_dir()) == ["part-00000"]
    (tmp_path / "build/parts/.part-00001.partial").mkdir()  # an interrupted write is cleaned up
    seen = []
    index = build_index(tiny_model, DOCS, tmp_path / "build", chunk_size=3,
                        progress=lambda done, total, encoded, s: seen.append((done, encoded)))
    assert seen == [(6, 3), (7, 4)]  # part 0 was not encoded again
    direct = tiny_model.index(tiny_model.encode_documents(list(DOCS.values())), ids=list(DOCS))
    assert index.ids == direct.ids
    for got, want in zip(stored(index), stored(direct), strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    int8 = build_index(tiny_model, DOCS, tmp_path / "build", chunk_size=3, storage="int8")
    direct8 = tiny_model.index(tiny_model.encode_documents(list(DOCS.values())), ids=list(DOCS), storage="int8")
    for got, want in zip(stored(int8), stored(direct8), strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    pruned = build_index(tiny_model, DOCS, tmp_path / "build", chunk_size=3, keep_parts=False,
                         progress=lambda *a: pytest.fail("a finished build is not encoded again"))
    assert pruned.ids == index.ids and not (tmp_path / "build/parts").exists()
    build_index(tiny_model, DOCS, tmp_path / "build", chunk_size=3, progress=lambda *a: pytest.fail("again"))
    info = read_build(tmp_path / "build")
    assert info["documents"] == 7 and info["batch_size"] == 1 and info["encoding_timing"]["documents"] == 7
    assert info["compute_dtype"] == tiny_model.compute_dtype
    with pytest.raises(ValueError, match="batch_size"):
        build_index(tiny_model, DOCS, tmp_path / "build", chunk_size=3, batch_size=2)
    with pytest.raises(ValueError, match="corpus_sha256"):
        build_index(tiny_model, {**DOCS, "a": "dogs"}, tmp_path / "build", chunk_size=3)
    with pytest.raises(ValueError, match="no text"):
        build_index(tiny_model, {"x": "  "}, tmp_path / "empty")
    (tmp_path / "other").mkdir()
    (tmp_path / "other/file").write_text("x")
    with pytest.raises(FileExistsError):
        build_index(tiny_model, DOCS, tmp_path / "other")


# ---- evaluation runs ----------------------------------------------------------------------------
def test_retrieval_stages_record_and_save(tiny_model, tmp_path):
    build_index(tiny_model, DOCS, tmp_path / "build", chunk_size=4)
    metrics = ["ndcg@3", "mrr@3", "recall@5", "map@5"]
    result = evaluate_retrieval(tiny_model, tmp_path / "build", QUERIES, QRELS, profile="exact", metrics=metrics,
                                dataset={"name": "tiny"})
    assert isinstance(result, EvaluationResult) and list(result.stages) == ["routing", "final"]
    index = tiny_model.load_index(tmp_path / "build/index-float32")
    evaluated = ["q1", "q2", "q4"]  # q3 has no positive judgment
    hits = index.search(tiny_model.encode_queries([QUERIES[q] for q in evaluated]), top_k=7, profile="exact")
    for qid, found in zip(evaluated, hits, strict=True):
        assert result.stages["final"].run[qid] == [(h.id, h.score) for h in found]
        by_routing = sorted(found, key=lambda h: (-h.routing_score, h.id))
        assert result.stages["routing"].run[qid] == [(h.id, h.routing_score) for h in by_routing]
    expected = evaluate_run(QRELS, {q: [h.id for h in found] for q, found in zip(evaluated, hits, strict=True)},
                            metrics)
    assert result.mean("final") == expected.mean
    settings = result.settings
    assert settings["model"]["representation_id"] == tiny_model.representation_id
    assert settings["model"]["compute_dtype"] == tiny_model.compute_dtype
    assert settings["search"]["profile"] == {"name": "exact", "method": "exact-routing", "slot_hits": None,
                                             "shortlist": None, "rerank": 100}
    assert settings["query_encoding"]["batch_size"] == 1 and settings["corpus"]["batch_size"] == 1
    assert settings["corpus"]["document_format"] == "title_text" and settings["library_version"]
    assert settings["data"]["evaluated_queries"] == 3 and settings["data"]["skipped_without_relevant"] == 1
    assert result.candidates["mean_size"] == 7 and result.candidates["queries_below_width"] == 3
    assert result.candidates["recall"] == 1.0

    paper = evaluate_retrieval(tiny_model, tmp_path / "build", QUERIES, QRELS, profile="paper", metrics=metrics,
                               shortlist_recall=True)
    assert paper.candidates["shortlist"]["width"] == 1024 and paper.settings["search"]["profile"]["slot_hits"] == 128

    out = result.save(tmp_path / "result")
    with pytest.raises(FileExistsError):
        result.save(tmp_path / "result")
    saved = load_result(out)
    assert saved["stages"]["final"]["mean"] == result.mean("final")
    assert set(saved["stages"]["final"]["confidence_intervals"]) == set(metrics)
    assert saved["per_query"]["routing"] == result.stages["routing"].metrics.per_query
    rerun = evaluate_run(QRELS, read_run(out / "runs/final.trec"), metrics)
    assert rerun.mean == result.mean("final")
    same = compare(out, result, "ndcg@3")
    assert same["difference"] == 0 and same["p_value"] == 1.0
    stages = compare(result, result, "ndcg@3", stage_a="routing", stage_b="final")
    assert stages["mean_a"] == result.mean("routing")["ndcg@3"]
    sample = evaluate_retrieval(tiny_model, index, QUERIES, QRELS, profile="exact", metrics=metrics, max_queries=2)
    assert sample.settings["data"]["evaluated_queries"] == 2
    assert sample.settings["corpus"]["source"] == "in-memory index"


def test_rerank_supplied_candidates(tiny_model, tmp_path):
    index = build_index(tiny_model, DOCS, tmp_path / "build")
    candidates = {"q1": {"a": 3.0, "b": 2.0, "e": 2.0, "c": 1.0}, "q2": ["d", "a", "g"], "q4": ["f"]}
    result = evaluate_rerank(tiny_model, index, candidates, QUERIES, QRELS, depth=3, metrics=["ndcg@3", "recall@3"])
    assert list(result.stages) == ["input", "routing", "final"]
    assert [d for d, _ in result.stages["input"].run["q1"]] == ["a", "b", "e"]  # ties by id, cut at depth
    rows = [index.ids.index(d) for d in ("a", "b", "e")]
    documents = Embeddings(index.routing_vectors(rows), index.value_vectors(rows), index.representation_id, "document")
    scores = tiny_model.score(tiny_model.encode_queries(["fish"]), documents)
    expected = sorted(zip("abe", scores.combined[0].tolist(), strict=True), key=lambda p: (-p[1], p[0]))
    assert result.stages["final"].run["q1"] == expected
    assert result.candidates["width"] == 3 and result.settings["rerank"]["depth"] == 3
    with pytest.raises(ValueError, match="not in the index"):
        evaluate_rerank(tiny_model, index, {"q1": ["a", "zz"]}, QUERIES, QRELS)
    dropped = evaluate_rerank(tiny_model, index, {"q1": ["a", "zz"]}, QUERIES, QRELS, missing_documents="drop")
    assert dropped.settings["rerank"]["dropped_candidates"] == 1
    assert dropped.settings["rerank"]["queries_without_candidates"] == 2


# ---- command line --------------------------------------------------------------------------------
def run_cli(capsys, *args):
    assert main([str(a) for a in args]) == 0
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_cli_index_retrieve_rerank_metrics_compare(capsys, tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path / "release")
    data = write_beir(tmp_path / "beir")
    model = ("--model", tmp_path / "release")
    [built] = run_cli(capsys, "index", *model, "--data", data, "--work-dir", tmp_path / "work", "--chunk-size", 3)
    assert built["documents"] == 7 and built["encoding_timing"]["documents"] == 7
    [retrieved] = run_cli(capsys, "retrieve", *model, "--data", data, "--work-dir", tmp_path / "work",
                          "--output", tmp_path / "out", "--chunk-size", 3, "--profile", "exact",
                          "--metrics", "ndcg@3,recall@5")
    assert set(retrieved["stages"]) == {"routing", "final"} and retrieved["queries"] == 3
    assert (tmp_path / "out/runs/final.trec").is_file()
    [scored] = run_cli(capsys, "metrics", "--qrels", data / "qrels/test.tsv", "--run", tmp_path / "out/runs/final.trec",
                       "--metrics", "ndcg@3,recall@5", "--ci", "--resamples", 200)
    assert scored["mean"] == retrieved["stages"]["final"]
    [compared] = run_cli(capsys, "compare", "--result-a", tmp_path / "out", "--result-b", tmp_path / "out",
                         "--stage-a", "routing", "--metric", "ndcg@3", "--resamples", 200)
    assert compared["queries"] == 3
    write_run(tmp_path / "bm25.trec", {"q1": [("a", 2.0), ("b", 1.0)], "q2": [("d", 1.0)], "q4": [("f", 1.0)]})
    [reranked] = run_cli(capsys, "rerank", *model, "--data", data, "--candidates", tmp_path / "bm25.trec",
                         "--work-dir", tmp_path / "work-rerank", "--output", tmp_path / "out-rerank")
    assert set(reranked["stages"]) == {"input", "routing", "final"}
    assert read_build(tmp_path / "work-rerank")["documents"] == 4  # only the candidates were encoded
    [via_compare] = run_cli(capsys, "compare", "--qrels", data / "qrels/test.tsv", "--run-a",
                            tmp_path / "out-rerank/runs/input.trec", "--run-b", tmp_path / "out-rerank/runs/final.trec",
                            "--resamples", 200)
    assert via_compare["queries"] == 3
    with pytest.raises(SystemExit):
        main(["retrieve", "--model", str(tmp_path / "release"), "--data", str(data), "--work-dir",
              str(tmp_path / "work"), "--output", str(tmp_path / "out")])  # output exists

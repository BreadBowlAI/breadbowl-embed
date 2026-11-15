"""Cross-check the evaluation metrics against pytrec_eval (trec_eval) and ranx when they are installed."""
import random

import pytest

from breadbowl_embed.evaluation import evaluate_run, rank


def random_case(seed: int, *, ties: bool = False, queries: int = 40):
    rng = random.Random(seed)
    qrels, run = {}, {}
    for q in range(queries):
        qid = f"q{q:03d}"
        docs = [f"d{d:04d}" for d in rng.sample(range(500), 160)]
        judged = docs[:60] if q % 3 else rng.sample(docs, 30)
        # graded 0..3; some queries with a single relevant document, one without any positive
        qrels[qid] = {d: (0 if q == 0 else rng.choice([0, 0, 1, 2, 3])) for d in judged}
        if q == 1:
            qrels[qid] = {d: 0 for d in judged}
            qrels[qid][judged[5]] = 2
        scored = docs[rng.randrange(20, 150):] if q % 5 == 0 else docs
        if ties:
            run[qid] = {d: float(rng.randrange(8)) for d in scored}
        else:
            values = rng.sample(range(10**6), len(scored))
            run[qid] = {d: v / 7.0 for d, v in zip(scored, values, strict=True)}
    return qrels, run


METRICS = ["ndcg@1", "ndcg@10", "ndcg@100", "recall@10", "recall@100", "map@10", "map@100", "mrr@10", "mrr@100"]


def pytrec_values(pytrec_eval, qrels, run, relevance_level):
    measures = {"ndcg_cut.1,10,100", "recall.10,100", "map_cut.10,100"}
    per_query = pytrec_eval.RelevanceEvaluator(qrels, measures, relevance_level=relevance_level).evaluate(run)
    out = {}
    for qid, row in per_query.items():
        out[qid] = {"ndcg@1": row["ndcg_cut_1"], "ndcg@10": row["ndcg_cut_10"], "ndcg@100": row["ndcg_cut_100"],
                    "recall@10": row["recall_10"], "recall@100": row["recall_100"], "map@10": row["map_cut_10"],
                    "map@100": row["map_cut_100"]}
    for k in (10, 100):  # pytrec_eval's recip_rank has no cutoff: evaluate the run cut at k
        cut = {qid: dict(sorted(docs.items(), key=lambda kv: (-kv[1], tuple(-ord(c) for c in kv[0])))[:k])
               for qid, docs in run.items()}
        rr = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"}, relevance_level=relevance_level).evaluate(cut)
        for qid, row in rr.items():
            out[qid][f"mrr@{k}"] = row["recip_rank"]
    return out


@pytest.mark.parametrize("relevance_level", [1, 2])
@pytest.mark.parametrize("ties", [False, True])
def test_matches_pytrec_eval(relevance_level, ties):
    pytrec_eval = pytest.importorskip("pytrec_eval")
    qrels, run = random_case(7 + relevance_level, ties=ties)
    ours = evaluate_run(qrels, run, METRICS, relevance_level=relevance_level, tie_break="trec_eval")
    theirs = pytrec_values(pytrec_eval, qrels, run, relevance_level)
    assert ours.skipped == ["q000"]  # all judgments 0: skipped here, scored 0 by pytrec_eval
    assert all(v == 0 for v in theirs["q000"].values())
    for qid, row in ours.per_query.items():
        for metric in METRICS:
            assert row[metric] == pytest.approx(theirs[qid][metric], abs=1e-12), (qid, metric)
    if ties:  # the default tie order (ascending id, as Index.search) differs from trec_eval's
        default = evaluate_run(qrels, run, METRICS, relevance_level=relevance_level)
        assert any(default.per_query[q] != ours.per_query[q] for q in ours.per_query)


def test_matches_ranx():
    ranx = pytest.importorskip("ranx")
    qrels, run = random_case(11)
    ours = evaluate_run(qrels, run, METRICS)
    evaluated = {q: qrels[q] for q in ours.per_query}
    theirs = ranx.evaluate(ranx.Qrels(evaluated), ranx.Run({q: run[q] for q in evaluated if q in run}), METRICS,
                           return_mean=False, make_comparable=True)
    order = sorted(evaluated)
    for metric in METRICS:
        values = dict(zip(order, theirs[metric].tolist(), strict=True))
        for qid in order:
            assert ours.per_query[qid][metric] == pytest.approx(values[qid], abs=1e-12), (qid, metric)


def test_trec_eval_tie_order_is_descending_id():
    pytrec_eval = pytest.importorskip("pytrec_eval")
    run = {"q": {"b": 1.0, "a": 1.0, "c": 1.0}}
    for relevant in "abc":
        rr = pytrec_eval.RelevanceEvaluator({"q": {relevant: 1}}, {"recip_rank"}).evaluate(run)["q"]["recip_rank"]
        assert rr == 1 / (rank(run["q"], tie_break="trec_eval").index(relevant) + 1)

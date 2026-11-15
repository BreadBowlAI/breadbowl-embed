"""The MTEB adapter, run end to end through mteb on hand-made tasks (skipped without mteb)."""
import pytest

mteb = pytest.importorskip("mteb")

from datasets import Dataset  # noqa: E402  (mteb depends on datasets)
from mteb.abstasks.retrieval import AbsTaskRetrieval  # noqa: E402
from mteb.abstasks.task_metadata import TaskMetadata  # noqa: E402
from mteb.models.models_protocols import CrossEncoderProtocol, EncoderProtocol, SearchProtocol  # noqa: E402

from breadbowl_embed import Embeddings  # noqa: E402
from breadbowl_embed.evaluation import BreadBowlMTEB, evaluate_retrieval, evaluate_run  # noqa: E402

DOCS = {"a": ("", "cats land"), "b": ("fish", "water"), "c": ("", "dogs land"), "d": ("", "cats water"),
        "e": ("", "fish land"), "f": ("dogs", "water"), "g": ("", "cats dogs")}
QUERIES = {"q1": "fish", "q2": "cats land", "q3": "dogs", "q4": "water"}
QRELS = {"q1": {"b": 1, "e": 2}, "q2": {"a": 2, "d": 1}, "q3": {"c": 1, "f": 1, "g": 1}, "q4": {"b": 1, "d": 1}}
TOP_RANKED = {"q1": ["a", "b", "c"], "q2": ["g", "d", "a", "b"], "q3": ["c"], "q4": ["f", "e"]}


def make_task(*, reranking: bool):
    class Tiny(AbsTaskRetrieval):
        metadata = TaskMetadata(
            name="BreadBowlTinyReranking" if reranking else "BreadBowlTinyRetrieval",
            description="A hand-made seven-document task for testing the BreadBowl MTEB adapter.",
            reference="https://github.com/BreadBowlAI/breadbowl-embed", type="Retrieval", category="t2t",
            modalities=["text"], eval_splits=["test"], eval_langs=["eng-Latn"], main_score="ndcg_at_10",
            dataset={"path": "NA", "revision": "NA"}, date=("2026-01-01", "2026-01-01"), domains=[],
            task_subtypes=[], license="apache-2.0", annotations_creators="expert-annotated", dialect=[],
            sample_creation="created", bibtex_citation="")

        def load_data(self, num_proc=None, **kwargs):
            split = {"corpus": Dataset.from_list([{"id": k, "title": t, "text": x} for k, (t, x) in DOCS.items()]),
                     "queries": Dataset.from_list([{"id": k, "text": v} for k, v in QUERIES.items()]),
                     "relevant_docs": QRELS, "top_ranked": TOP_RANKED if reranking else None}
            self.dataset = {"default": {"test": split}}
            self.data_loaded = True

    return Tiny()


def scores_of(result):
    [task_result] = result.task_results
    [scores] = task_result.scores["test"]
    return scores


def test_protocol_and_metadata(tiny_model):
    model = BreadBowlMTEB(tiny_model, profile="paper", stage="final")
    assert isinstance(model, SearchProtocol)
    assert not isinstance(model, EncoderProtocol) and not isinstance(model, CrossEncoderProtocol)
    meta = model.mteb_model_meta
    assert meta.name.startswith("local/breadbowl-embed-") and meta.revision
    assert meta.use_instructions and meta.max_tokens == 16 and meta.embed_dim == 8
    assert meta.loader_kwargs["profile"] == "paper" and meta.loader_kwargs["candidate_k"] == 100
    assert "MSMARCO" in meta.training_datasets
    fields = type(meta).model_fields  # newer mteb releases know more fields
    if "model_type" in fields:
        assert meta.model_type == ["late-interaction"]
    if "experiment_kwargs" in fields:
        assert meta.experiment_kwargs == meta.loader_kwargs
    if hasattr(getattr(mteb.models.model_meta, "ScoringFunction", None), "CUSTOM"):
        assert meta.similarity_fn_name == "custom"


def test_mteb_scores_retrieval_with_the_models_own_ranking(tiny_model, tmp_path):
    model = BreadBowlMTEB(tiny_model, profile="exact", cache_dir=tmp_path / "cache", chunk_size=3)
    result = mteb.evaluate(model, make_task(reranking=False), cache=None, show_progress_bar=False,
                           prediction_folder=tmp_path / "predictions")
    scores = scores_of(result)
    # The same documents and scores through the toolkit's own evaluation (same title + text rule).
    corpus = {k: {"title": t, "text": x} for k, (t, x) in DOCS.items()}
    from breadbowl_embed.evaluation import build_index
    index = build_index(tiny_model, corpus, tmp_path / "build")
    ours = evaluate_retrieval(tiny_model, index, QUERIES, QRELS, profile="exact",
                              metrics=["ndcg@1", "ndcg@3", "ndcg@10", "recall@3", "mrr@10"])
    mean = ours.mean("final")
    assert scores["ndcg_at_1"] == pytest.approx(mean["ndcg@1"], abs=1e-5)
    assert scores["ndcg_at_3"] == pytest.approx(mean["ndcg@3"], abs=1e-5)
    assert scores["ndcg_at_10"] == pytest.approx(mean["ndcg@10"], abs=1e-5)
    assert scores["recall_at_3"] == pytest.approx(mean["recall@3"], abs=1e-5)
    assert scores["mrr_at_10"] == pytest.approx(mean["mrr@10"], abs=1e-5)
    assert [r["call"] for r in model.records] == ["index", "search"]
    assert (tmp_path / "cache" / "BreadBowlTinyRetrieval").is_dir()  # resumable build directory

    routing = BreadBowlMTEB(tiny_model, profile="exact", stage="routing")
    routing_scores = scores_of(mteb.evaluate(routing, make_task(reranking=False), cache=None,
                                             show_progress_bar=False))
    assert routing_scores["ndcg_at_10"] == pytest.approx(ours.mean("routing")["ndcg@10"], abs=1e-5)


def test_mteb_reranking_task_scores_only_the_given_candidates(tiny_model):
    model = BreadBowlMTEB(tiny_model)
    scores = scores_of(mteb.evaluate(model, make_task(reranking=True), cache=None, show_progress_bar=False))
    index = tiny_model.index(tiny_model.encode_documents([(t + " " + x).strip() for t, x in DOCS.values()]),
                             ids=list(DOCS))
    queries = tiny_model.encode_queries(list(QUERIES.values()))
    run = {}
    for i, (qid, candidates) in enumerate(TOP_RANKED.items()):
        rows = [index.ids.index(d) for d in candidates]
        documents = Embeddings(index.routing_vectors(rows), index.value_vectors(rows), index.representation_id,
                               "document")
        run[qid] = dict(zip(candidates, tiny_model.score(queries.select(i), documents).combined[0].tolist(),
                            strict=True))
    expected = evaluate_run(QRELS, run, ["ndcg@10", "map@10"], tie_break="trec_eval")
    assert scores["ndcg_at_10"] == pytest.approx(expected.mean["ndcg@10"], abs=1e-5)
    assert scores["map_at_10"] == pytest.approx(expected.mean["map@10"], abs=1e-5)


def test_model_meta_loader_round_trip(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path / "release")
    original = BreadBowlMTEB.from_pretrained(tmp_path / "release", profile="paper", candidate_k=50, stage="routing")
    loader_kwargs = {**original.mteb_model_meta.loader_kwargs, "path": str(tmp_path / "release")}
    meta = original.mteb_model_meta.model_copy(update={"loader_kwargs": loader_kwargs})
    loaded = meta.load_model()
    assert isinstance(loaded, BreadBowlMTEB)
    assert loaded.profile == original.profile and loaded.stage == "routing"
    assert loaded.model.representation_id == original.model.representation_id
    assert loaded.mteb_model_meta.name == original.mteb_model_meta.name

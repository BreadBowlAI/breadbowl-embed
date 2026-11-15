"""MTEB adapter: let `mteb` evaluate BreadBowl retrieval with the model's own score.

MTEB scores encoder models with cosine (or MaxSim) over vectors it gets from `encode`. BreadBowl's
score is not of that form (0.75 R + 0.25 V over 16 routing/value slots), so this adapter
implements MTEB's *search* protocol instead (`mteb.models.SearchProtocol`: `index`, `search` and
`mteb_model_meta`): MTEB hands it the corpus and the queries, and it returns, per query, the
documents and scores produced by `Index.search` (or, for reranking tasks, the model's score of the
given candidates). MTEB then computes its metrics from those scores.

Checked against the protocol source of mteb 2.27.14 (the newest release when written); see
docs/evaluation.md for the versions that were run. Requires the optional extra
(`pip install "breadbowl-embed[mteb]"`); importing this module without mteb fails with a hint.

    import mteb
    from breadbowl_embed.evaluation import BreadBowlMTEB

    model = BreadBowlMTEB.from_pretrained("path/or/repo", dtype="bfloat16", profile="paper")
    results = mteb.evaluate(model, mteb.get_tasks(tasks=["SciFact"]), cache=None)
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
import time
from typing import Any
import warnings

try:
    from mteb.models import model_meta as _mteb_meta
    from mteb.models.model_meta import ModelMeta
except ImportError as exc:  # pragma: no cover - depends on the optional extra
    raise ImportError("The MTEB adapter needs the optional extra: pip install 'breadbowl-embed[mteb]'") from exc

from .._version import __version__
from ..embeddings import Embeddings
from ..retrieval import EXACT_ROUTING, SLOT_SHORTLIST, SearchProfile, resolve_profile
from .corpus import build_index

REFERENCE = "https://github.com/BreadBowlAI/breadbowl-embed"
# MTEB task names of the fine-tuning sources the v1.3 model card lists (MS MARCO, Natural
# Questions, FEVER, HotpotQA), with their MTEB variants. The weak pre-training stage's 34-subset
# collection has not been mapped to MTEB tasks, so this set is a lower bound.
TRAINING_DATASETS = frozenset({
    "MSMARCO", "MSMARCOHardNegatives", "NanoMSMARCORetrieval",
    "NQ", "NQHardNegatives", "NanoNQRetrieval",
    "FEVER", "FEVERHardNegatives", "NanoFEVERRetrieval",
    "HotpotQA", "HotpotQAHardNegatives", "NanoHotpotQARetrieval",
})


def _safe(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(part)) or "_"


def _document_text(title: str | None, text: str | None) -> str:
    """MTEB's corpus rule: (title + " " + text).strip() when there is a title, else text.strip()."""
    title, text = title or "", text or ""
    return (title + " " + text).strip() if title else text.strip()


def model_meta(model, *, name: str | None = None, revision: str | None = None, experiment: Mapping | None = None,
               training_datasets: set[str] | frozenset[str] | None = TRAINING_DATASETS, loader=None,
               loader_kwargs: Mapping | None = None) -> ModelMeta:
    """MTEB metadata for a loaded BreadBowl model.

    name defaults to the release's Hugging Face repository (when recorded) or
    "local/breadbowl-embed-<release id>"; revision to the loaded Hub commit, the recorded revision
    or the release id. training_datasets: MTEB task names the model was trained on (None:
    unknown). experiment: MTEB's experiment_kwargs; mteb passes them back to the loader, so they
    are loader arguments. Fields this mteb version does not know are left out.
    """
    info = model.info()
    hub = info["release"]["huggingface"]
    release_id = model.release_id or model.representation_id
    name = name or hub.get("repository") or f"local/breadbowl-embed-{release_id[:12]}"
    revision = revision or hub.get("loaded_revision") or hub.get("revision") or release_id[:12]
    element = 2 if model.compute_dtype in {"bfloat16", "float16"} else 4
    base = (info.get("provenance", {}).get("base_model") or {}).get("repo")
    values: dict[str, Any] = {
        "loader": loader or load_mteb_model, "loader_kwargs": dict(loader_kwargs or {}),
        "name": name, "revision": revision, "release_date": None, "languages": ["eng-Latn"],
        "n_parameters": info["parameters"], "memory_usage_mb": round(info["parameters"] * element / 2**20, 1),
        "max_tokens": info["preprocessing"]["max_document_length"],
        # 16 routing and 16 value vectors per text; MTEB records one per-vector width.
        "embed_dim": info["architecture"]["d_k"],
        "license": (info["release"].get("license") or "").lower() or None, "open_weights": True,
        "public_training_code": None, "public_training_data": None, "framework": ["PyTorch"],
        "reference": REFERENCE, "use_instructions": True,
        # Neither cosine nor MaxSim; the enum member exists in recent mteb (a string "custom" is refused).
        "similarity_fn_name": getattr(getattr(_mteb_meta, "ScoringFunction", None), "CUSTOM", None),
        "training_datasets": set(training_datasets) if training_datasets is not None else None,
        "adapted_from": base, "modalities": ["text"], "model_type": ["late-interaction"],
        "experiment_kwargs": dict(experiment) if experiment else None,
        "output_dtypes": None,  # search-only: MTEB never receives embeddings from this adapter
    }
    return ModelMeta(**{k: v for k, v in values.items() if k in ModelMeta.model_fields})


class BreadBowlMTEB:
    """MTEB search model (index + search) around a loaded `BreadBowl`.

    profile: search profile for retrieval tasks ("default" 512/1024/100, "paper" 128/1024/100,
    "exact", or a SearchProfile); slot_hits/shortlist/candidate_k override its parameters.
    candidate_k is the number of documents that reach value reranking (the profile's 100 by
    default); at most that many documents are returned per query, so MTEB's recall@k for
    k > candidate_k is computed on that list. stage: "final" (0.75 R + 0.25 V, default) or
    "routing" (R alone, over the same candidates). cache_dir: encode the corpus resumably under
    this directory (one build per task, subset, split, model and batch size); default: in memory.
    instruction: query instruction (default: the checkpoint's). Per-query instructions that a task
    supplies are appended to the query text, as MTEB does for models without instruction support.

    The settings that change results are recorded as MTEB experiment_kwargs (and so in MTEB's result
    files and cache paths), together with the library version.
    """

    def __init__(self, model, *, profile="default", slot_hits: int | None = None, shortlist: int | None = None,
                 candidate_k: int | None = None, stage: str = "final", query_batch_size: int = 1,
                 document_batch_size: int = 1, storage: str = "float32", cache_dir: str | Path | None = None,
                 chunk_size: int = 256, instruction: str | None = None, backend: str = "torch",
                 name: str | None = None, revision: str | None = None,
                 training_datasets: set[str] | frozenset[str] | None = TRAINING_DATASETS):
        if stage not in {"final", "routing"}:
            raise ValueError("stage must be 'final' or 'routing'.")
        if profile == "custom":  # as recorded in experiment_kwargs by an earlier run
            method = SLOT_SHORTLIST if slot_hits is not None else EXACT_ROUTING
            self.profile = SearchProfile("custom", method, slot_hits, shortlist, candidate_k or 100)
        else:
            self.profile = resolve_profile(profile, candidate_k, slot_hits=slot_hits, shortlist=shortlist)
        self.model = model
        self.stage = stage
        self.query_batch_size, self.document_batch_size = query_batch_size, document_batch_size
        self.storage, self.chunk_size, self.backend = storage, chunk_size, backend
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.instruction = instruction
        # Loader arguments: mteb passes experiment_kwargs back to the loader in ModelMeta.load_model.
        self.experiment = {"profile": self.profile.name, "slot_hits": self.profile.slot_hits,
                           "shortlist": self.profile.shortlist, "candidate_k": self.profile.rerank, "stage": stage,
                           "dtype": model.compute_dtype, "storage": storage, "query_batch_size": query_batch_size,
                           "document_batch_size": document_batch_size, "instruction": instruction,
                           "library_version": __version__}
        self.experiment = {k: v for k, v in self.experiment.items() if v is not None}
        # Also as loader_kwargs: mteb releases before ~2.27 pass only those to the loader.
        self.mteb_model_meta = model_meta(model, name=name, revision=revision, experiment=self.experiment,
                                          training_datasets=training_datasets, loader_kwargs=self.experiment)
        self._index = None
        self._index_key = None
        self.records: list[dict] = []  # one entry per index()/search() call: settings and timings

    @classmethod
    def from_pretrained(cls, model_id_or_path: str | Path, *, revision: str | None = None, device: str | None = None,
                        dtype=None, local_files_only: bool = False, **options) -> BreadBowlMTEB:
        """Load a release (directory or Hub id) and wrap it; options go to the constructor."""
        from ..model import BreadBowl
        model = BreadBowl.from_pretrained(model_id_or_path, revision=revision, device=device, dtype=dtype,
                                          local_files_only=local_files_only)
        return cls(model, **options)

    # ---- mteb.models.SearchProtocol -----------------------------------------------------------
    def index(self, corpus, *, task_metadata, hf_split: str, hf_subset: str, encode_kwargs=None,
              num_proc: int | None = None, **kwargs) -> None:
        start = time.monotonic()
        columns = corpus.column_names
        if "text" not in columns:
            raise ValueError("Only text corpora are supported.")
        ids = [str(i) for i in corpus["id"]]
        titles = corpus["title"] if "title" in columns else [None] * len(ids)
        items, empty = [], 0
        for doc_id, title, text in zip(ids, titles, corpus["text"], strict=True):
            content = _document_text(title, text)
            if content:
                items.append((doc_id, content))
            else:
                empty += 1  # nothing to encode; such documents are never retrieved
        if empty:
            warnings.warn(f"{empty} empty documents in {task_metadata.name} were not indexed.", stacklevel=2)
        if self.cache_dir is not None:
            directory = (self.cache_dir / _safe(task_metadata.name) / _safe(hf_subset) / _safe(hf_split)
                         / f"{self.model.representation_id[:16]}-b{self.document_batch_size}")
            self._index = build_index(self.model, items, directory, document_format="text", storage=self.storage,
                                      profile=self.profile, chunk_size=self.chunk_size,
                                      batch_size=self.document_batch_size)
        else:
            encoded = self.model.encode_documents([text for _, text in items], batch_size=self.document_batch_size)
            self._index = self.model.index(encoded, ids=[doc_id for doc_id, _ in items], storage=self.storage,
                                           profile=self.profile)
        self._index_key = (task_metadata.name, hf_subset, hf_split)
        self.records.append({"call": "index", "task": task_metadata.name, "subset": hf_subset, "split": hf_split,
                             "documents": len(items), "empty_documents": empty,
                             "seconds": time.monotonic() - start})

    def search(self, queries, *, task_metadata, hf_split: str, hf_subset: str, top_k: int, encode_kwargs=None,
               top_ranked: Mapping[str, list[str]] | None = None, num_proc: int | None = None,
               **kwargs) -> dict[str, dict[str, float]]:
        if self._index is None or self._index_key != (task_metadata.name, hf_subset, hf_split):
            raise RuntimeError("search() was called before index() for this task, subset and split.")
        start = time.monotonic()
        qids = [str(i) for i in queries["id"]]
        texts = list(queries["text"])
        if any(not isinstance(t, str) for t in texts):
            raise NotImplementedError("Conversational (multi-turn) queries are not supported.")
        if "instruction" in queries.column_names:
            texts = [t + " " + i if i else t for t, i in zip(texts, queries["instruction"], strict=True)]
        encoded = self.model.encode_queries(texts, batch_size=self.query_batch_size, instruction=self.instruction)
        if top_ranked is not None:
            results = self._rerank(qids, encoded, top_ranked)
        else:
            width = min(top_k, self.profile.rerank)
            results = {}
            for first in range(0, len(qids), 64):
                hits = self._index.search(encoded.select(slice(first, first + 64)), top_k=self.profile.rerank,
                                          profile=self.profile, backend=self.backend)
                for qid, found in zip(qids[first:first + 64], hits, strict=True):
                    if self.stage == "routing":
                        found = sorted(found, key=lambda h: (-h.routing_score, h.id))
                    results[qid] = {h.id: (h.routing_score if self.stage == "routing" else h.score)
                                    for h in found[:width]}
        self.records.append({"call": "search", "task": task_metadata.name, "subset": hf_subset, "split": hf_split,
                             "queries": len(qids), "top_k": top_k, "reranking_task": top_ranked is not None,
                             "seconds": time.monotonic() - start})
        return results

    def _rerank(self, qids, encoded: Embeddings, top_ranked) -> dict[str, dict[str, float]]:
        position = {doc_id: i for i, doc_id in enumerate(self._index.ids)}
        results = {}
        for i, qid in enumerate(qids):
            docs = [d for d in dict.fromkeys(top_ranked.get(qid, [])) if d in position]
            if not docs:
                results[qid] = {}
                continue
            rows = [position[d] for d in docs]
            documents = Embeddings(self._index.routing_vectors(rows), self._index.value_vectors(rows),
                                   self._index.representation_id, "document")
            scores = self.model.score(encoded.select(i), documents)
            values = scores.routing[0] if self.stage == "routing" else scores.combined[0]
            results[qid] = dict(zip(docs, values.tolist(), strict=True))
        return results


def load_mteb_model(model_name: str, revision: str | None = None, *, path: str | Path | None = None,
                    device: str | None = None, dtype=None, local_files_only: bool = False,
                    library_version: str | None = None, **options) -> BreadBowlMTEB:
    """The ModelMeta loader: `ModelMeta.load_model()` calls it with the metadata's name, revision,
    loader_kwargs and experiment_kwargs. `path` loads a local release directory instead of the Hub
    repository `model_name` (the metadata keeps name and revision)."""
    from ..model import BreadBowl
    if library_version is not None and library_version != __version__:
        warnings.warn(f"These settings were recorded with breadbowl-embed {library_version}; "
                      f"{__version__} is installed.", stacklevel=2)
    model = BreadBowl.from_pretrained(path or model_name, revision=None if path else revision, device=device,
                                      dtype=dtype, local_files_only=local_files_only)
    return BreadBowlMTEB(model, name=model_name, revision=revision if path else None, **options)

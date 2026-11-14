"""Stored document slots, the paper's retrieval profile and exact document-routing search."""
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ._scoring import score_qk
from ._search import coarse_candidates, search_slots_exact, search_slots_faiss
from ._version import __version__
from .config import ScoringConfig, canonical_json, sha256_file
from .embeddings import Embeddings
from .scoring import explain as explain_pairs
from .scoring import score

INDEX_FORMAT_V1 = "breadbowl_index_v1"
INDEX_FORMAT = "breadbowl_index_v2"
STORAGE_TYPES = ("float32", "float16", "int8")
_INT8_LEVELS = 127


@dataclass(frozen=True)
class SearchResult:
    id: str
    score: float
    routing_score: float
    value_score: float
    document_index: int
    metadata: dict = field(default_factory=dict)
    explanation: dict | None = None
    profile: dict | None = None  # name and parameters of the search profile that produced it

    def to_dict(self):
        return asdict(self)


SLOT_SHORTLIST = "slot-shortlist"
EXACT_ROUTING = "exact-routing"


@dataclass(frozen=True)
class SearchProfile:
    """How candidates reach value reranking.

    method "slot-shortlist" (the paper's pipeline): every query slot searches all document routing
        slots and keeps its top `slot_hits`; documents are shortlisted by the mean over query slots
        of their best hit cosine (a query slot that missed the document counts -1), keeping
        `shortlist`; exact routing rescoring keeps `rerank`; values rerank those.
    method "exact-routing" (v0.1's index): exact routing score for every document, keep `rerank`.

    Named profiles (PROFILES): "default" slot-shortlist 512/1024/100; "paper" slot-shortlist
    128/1024/100, the setting of the reported results; "exact" exact-routing, rerank 100. Changing
    a named profile's parameters renames it "custom".
    """
    name: str = "default"
    method: str = SLOT_SHORTLIST
    slot_hits: int | None = 512
    shortlist: int | None = 1024
    rerank: int = 100

    def __post_init__(self):
        if self.method not in {SLOT_SHORTLIST, EXACT_ROUTING}:
            raise ValueError(f"Unknown search method {self.method!r}.")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Profiles need a name.")
        names = ("rerank", "slot_hits", "shortlist") if self.method == SLOT_SHORTLIST else ("rerank",)
        for name in names:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"Profile {name} must be a positive integer.")
        if self.method == EXACT_ROUTING and (self.slot_hits is not None or self.shortlist is not None):
            object.__setattr__(self, "slot_hits", None)
            object.__setattr__(self, "shortlist", None)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "SearchProfile":
        return cls(**{k: value[k] for k in ("name", "method", "slot_hits", "shortlist", "rerank")})

    def replace(self, **changes) -> "SearchProfile":
        changes = {k: v for k, v in changes.items() if v is not None}
        values = {**self.to_dict(), **changes}
        if any(values[k] != getattr(self, k) for k in changes):
            values["name"] = "custom"
        return SearchProfile(**values)


PROFILES = {
    "default": SearchProfile("default", SLOT_SHORTLIST, 512, 1024, 100),
    "paper": SearchProfile("paper", SLOT_SHORTLIST, 128, 1024, 100),
    "exact": SearchProfile("exact", EXACT_ROUTING, None, None, 100),
}
# Names used by v0.1 manifests and earlier drafts.
PROFILE_ALIASES = {"exact-document-routing-v1": "exact", "reference": "paper", "reference-slot-shortlist-v1": "paper"}
DEFAULT = PROFILES["default"]
PAPER = REFERENCE = PROFILES["paper"]
EXACT = PROFILES["exact"]


def resolve_profile(profile=None, candidate_k: int | None = None, *, slot_hits: int | None = None,
                    shortlist: int | None = None) -> SearchProfile:
    """A SearchProfile from None (default), a name or a SearchProfile, with optional overrides."""
    if profile is None:
        profile = DEFAULT
    if isinstance(profile, str):
        key = PROFILE_ALIASES.get(profile, profile)
        if key not in PROFILES:
            raise ValueError(f"Unknown profile {profile!r}; expected one of {sorted(PROFILES)}.")
        profile = PROFILES[key]
    if not isinstance(profile, SearchProfile):
        raise TypeError("profile must be a SearchProfile, a profile name, or None.")
    if profile.method == EXACT_ROUTING and (slot_hits is not None or shortlist is not None):
        raise ValueError("slot_hits and shortlist apply to slot-shortlist profiles only.")
    return profile.replace(rerank=candidate_k, slot_hits=slot_hits, shortlist=shortlist)


def _fixed_point(x: np.ndarray):
    """Symmetric int8 per vector, scale = largest magnitude / 127 (as the paper's storage study)."""
    scale = np.abs(x).max(-1, keepdims=True) / _INT8_LEVELS
    scale[scale == 0] = 1
    q = np.rint(x / scale)
    return q.astype(np.int8), scale[..., 0].astype(np.float32)


def quantize(embeddings: Embeddings, storage: str) -> dict:
    """Arrays stored for a document batch. Routing scales are dropped: routing is compared by cosine."""
    if storage not in STORAGE_TYPES:
        raise ValueError(f"storage must be one of {STORAGE_TYPES}.")
    routing = embeddings.routing.detach().float().cpu().numpy()
    value = embeddings.value.detach().float().cpu().numpy()
    if storage == "float32":
        return {"routing": routing.astype(np.float32), "value": value.astype(np.float32)}
    if storage == "float16":
        return {"routing": routing.astype(np.float16), "value": value.astype(np.float16)}
    routing_q, _ = _fixed_point(routing)
    value_q, value_scale = _fixed_point(value)
    return {"routing": routing_q, "value": value_q, "value_scale": value_scale}


def bytes_per_document(storage: str, slots: int, d_k: int, d_v: int) -> int:
    width = {"float32": 4, "float16": 2, "int8": 1}[storage]
    return slots * (d_k + d_v) * width + (slots * 4 if storage == "int8" else 0)


class Index:
    """Document routing/value slots with ids, persistent, searchable without a server.

    Storage is float32 (default), float16 or int8 (per-vector symmetric fixed point; value slots
    keep a float32 scale). `profile` is the search profile used when search() is not given one
    (see SearchProfile; "default" unless set). Results are deterministic: ties order by id.
    """

    def __init__(self, embeddings: Embeddings, ids: list[str], *,
                 scoring: ScoringConfig | None = None, metadata: list[dict] | None = None,
                 storage: str = "float32", profile: SearchProfile | str | None = None):
        if embeddings.kind != "document":
            raise ValueError("Indexes require document embeddings.")
        if storage not in STORAGE_TYPES:
            raise ValueError(f"storage must be one of {STORAGE_TYPES}.")
        self.representation_id = embeddings.representation_id
        self.storage = storage
        self.profile = resolve_profile(profile)
        self.scoring = scoring or ScoringConfig()
        self.slots, self.d_k = embeddings.routing.shape[1:]
        self.d_v = embeddings.value.shape[2]
        self.ids: list[str] = []
        self.metadata: list[dict] = []
        self._id_set: set[str] = set()
        self._shards: list[dict] = []
        self._starts: list[int] = []
        self._faiss = None
        self._unit = None
        self.cache_routing = True
        self.add(embeddings, ids, metadata=metadata)

    @classmethod
    def from_embeddings(cls, embeddings, *, ids, scoring=None, metadata=None, storage="float32", profile=None):
        return cls(embeddings, ids, scoring=scoring, metadata=metadata, storage=storage, profile=profile)

    def __len__(self):
        return len(self.ids)

    # ---- building ---------------------------------------------------------------------------
    def add(self, embeddings: Embeddings, ids: list[str], *, metadata: list[dict] | None = None) -> None:
        """Append documents as a new shard, stored with this index's storage type."""
        if embeddings.kind != "document" or embeddings.representation_id != self.representation_id:
            raise ValueError("Add document embeddings from the model this index was built with.")
        if tuple(embeddings.routing.shape[1:]) != (self.slots, self.d_k) or embeddings.value.shape[2] != self.d_v:
            raise ValueError("Slot count or dimensions differ from the index.")
        if len(ids) != len(embeddings) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("Provide one non-empty string ID per document.")
        if len(set(ids)) != len(ids) or self._id_set.intersection(ids):
            raise ValueError("Document IDs must be unique.")
        if metadata is not None and (len(metadata) != len(ids) or not all(isinstance(x, dict) for x in metadata)):
            raise ValueError("Provide one metadata object per document.")
        metadata = json.loads(canonical_json(metadata if metadata is not None else [{} for _ in ids]))
        if len(ids):
            self._starts.append(len(self.ids))
            self._shards.append(quantize(embeddings, self.storage))
        self.ids.extend(ids)
        self._id_set.update(ids)
        self.metadata.extend(metadata)
        self._faiss = None
        self._unit = None

    # ---- stored vectors ---------------------------------------------------------------------
    def _gather(self, name: str, indices) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        width = self.d_k if name == "routing" else self.d_v
        out = np.empty((len(indices), self.slots, width), dtype=np.float32)
        if not len(indices):
            return out
        if indices.min() < 0 or indices.max() >= len(self):
            raise IndexError("Document index out of range.")
        starts = np.asarray(self._starts)
        shard_of = np.searchsorted(starts, indices, side="right") - 1
        local = indices - starts[shard_of]
        for shard in np.unique(shard_of):
            rows = np.nonzero(shard_of == shard)[0]
            arrays = self._shards[shard]
            block = np.asarray(arrays[name][local[rows]], dtype=np.float32)
            if name == "value" and "value_scale" in arrays:
                block *= np.asarray(arrays["value_scale"][local[rows]], dtype=np.float32)[..., None]
            out[rows] = block
        return out

    def routing_vectors(self, indices) -> torch.Tensor:
        """Routing slots as read back from storage (direction only for int8)."""
        return torch.from_numpy(self._gather("routing", indices))

    def value_vectors(self, indices) -> torch.Tensor:
        """Raw value slots as read back from storage."""
        return torch.from_numpy(self._gather("value", indices))

    def _unit_routing_blocks(self, documents_per_block: int):
        for start, arrays in zip(self._starts, self._shards):
            routing = arrays["routing"]
            for offset in range(0, len(routing), documents_per_block):
                block = np.array(routing[offset:offset + documents_per_block], dtype=np.float32)
                norms = np.linalg.norm(block, axis=-1, keepdims=True)
                np.divide(block, norms, out=block, where=norms > 0)
                yield start + offset, block

    @property
    def embeddings(self) -> Embeddings:
        """All stored documents as Embeddings (read back from storage; routing re-normalised)."""
        everything = np.arange(len(self))
        routing = F.normalize(self.routing_vectors(everything), dim=-1)
        return Embeddings(routing, self.value_vectors(everything), self.representation_id, "document")

    def _cached_unit_routing(self) -> torch.Tensor:
        """Unit float32 routing [documents, slots, d_k], built on first slot search.

        Costs documents * slots * d_k * 4 bytes of memory (as much again as float32 storage);
        set `cache_routing = False` to re-read storage block by block on every search instead.
        """
        if self._unit is None:
            blocks = [torch.from_numpy(block) for _, block in self._unit_routing_blocks(65536)]
            self._unit = torch.cat(blocks) if blocks else torch.empty(0, self.slots, self.d_k)
        return self._unit

    def _faiss_index(self):
        if self._faiss is None:
            try:
                import faiss
            except ImportError as exc:  # pragma: no cover - depends on the optional extra
                raise ImportError("backend='faiss' needs the optional extra: "
                                  "pip install 'breadbowl-embed[faiss]'") from exc
            index = faiss.IndexFlatIP(self.d_k)
            for _, block in self._unit_routing_blocks(65536):
                index.add(np.ascontiguousarray(block.reshape(-1, self.d_k)))
            self._faiss = index
        return self._faiss

    # ---- search -----------------------------------------------------------------------------
    @torch.inference_mode()
    def search(self, queries: Embeddings, *, top_k: int = 10, candidate_k: int | None = None,
               batch_size: int = 128, profile: SearchProfile | str | None = None,
               explain: bool = False, backend: str = "torch",
               query_batch_size: int = 32) -> list[list[SearchResult]]:
        """Top documents per query, ranked by (1 - value_mix) * routing + value_mix * value.

        profile: a SearchProfile or a name ("default", "paper", "exact"); None uses the index's
        profile. `candidate_k` overrides the profile's `rerank` width (documents that reach value
        reranking). `backend` applies to the slot search of slot-shortlist profiles: "torch"
        (exact, default) or "faiss" (flat inner-product index, optional extra). Every result
        records the profile it was produced with. With explain=True each result also carries the
        query-slot -> document-slot matches and, for slot-shortlist profiles, the slot hits that
        put the document on the shortlist.
        """
        profile = resolve_profile(profile if profile is not None else self.profile, candidate_k)
        if type(top_k) is not int or top_k < 1 or profile.rerank < top_k:
            raise ValueError("Require candidate_k (rerank width) >= top_k >= 1.")
        if type(batch_size) is not int or batch_size < 1 or type(query_batch_size) is not int or query_batch_size < 1:
            raise ValueError("batch sizes must be positive integers.")
        if backend not in {"torch", "faiss"}:
            raise ValueError("backend must be 'torch' or 'faiss'.")
        if queries.kind != "query":
            raise ValueError("Search with query embeddings.")
        if queries.representation_id != self.representation_id:
            raise ValueError("Incompatible representations: re-encode with the same model and preprocessing.")
        if tuple(queries.routing.shape[2:]) != (self.d_k,) or queries.value.shape[2] != self.d_v:
            raise ValueError("Query/document dimensions differ.")
        queries = queries.to("cpu").without_tokens()
        if not len(self):
            return [[] for _ in range(len(queries))]
        results = []
        for start in range(0, len(queries), query_batch_size):
            batch = queries.select(slice(start, start + query_batch_size))
            if profile.method == SLOT_SHORTLIST:
                candidates = self._reference_candidates(batch, profile, backend, batch_size)
            else:
                candidates = [(self._exact_candidates(batch.select(i), profile.rerank, batch_size), None)
                              for i in range(len(batch))]
            for i, (selected, hits) in enumerate(candidates):
                results.append(self._rerank(batch.select(i), selected, top_k, batch_size, explain, hits,
                                            profile.to_dict()))
        return results

    def _exact_candidates(self, query: Embeddings, width: int, batch_size: int) -> list[int]:
        parts = []
        for first in range(0, len(self), batch_size):
            rows = np.arange(first, min(first + batch_size, len(self)))
            parts.append(score_qk(query.routing, self.routing_vectors(rows), tau_k=self.scoring.tau_k,
                                  mode=self.scoring.qk_mode)[0])
        values = torch.cat(parts).numpy()
        if len(values) <= width:
            chosen = np.arange(len(values))
        else:
            threshold = -np.partition(-values, width - 1)[width - 1]
            above = np.nonzero(values > threshold)[0]
            tied = sorted(np.nonzero(values == threshold)[0].tolist(), key=lambda j: self.ids[j])
            chosen = np.concatenate([above, np.asarray(tied[:width - len(above)], dtype=np.int64)])
        return sorted(chosen.tolist(), key=lambda j: (-float(values[j]), self.ids[j]))

    def _reference_candidates(self, batch: Embeddings, profile: SearchProfile, backend: str, batch_size: int):
        unit = F.normalize(batch.routing.float(), dim=-1).numpy()
        k = min(profile.slot_hits, len(self) * self.slots)
        if backend == "faiss":
            scores, hits = search_slots_faiss(unit, self._faiss_index(), k)
        else:
            width = max(1, 65536 // self.slots)
            if self.cache_routing:
                cached = self._cached_unit_routing()
                blocks = ((first, cached[first:first + width].numpy()) for first in range(0, len(cached), width))
            else:
                blocks = self._unit_routing_blocks(width)
            scores, hits = search_slots_exact(unit, blocks, k)
        out = []
        for j in range(len(batch)):
            shortlist = coarse_candidates(scores[j], hits[j], self.slots, profile.shortlist)
            if not shortlist:
                out.append(([], None))
                continue
            routing = torch.cat([
                score_qk(batch.routing[j:j + 1], self.routing_vectors(shortlist[s:s + batch_size]),
                         tau_k=self.scoring.tau_k, mode=self.scoring.qk_mode)[0]
                for s in range(0, len(shortlist), batch_size)])
            order = sorted(range(len(shortlist)), key=lambda i: (-float(routing[i]), shortlist[i]))
            out.append(([shortlist[i] for i in order[:profile.rerank]], (scores[j], hits[j])))
        return out

    def _rerank(self, query: Embeddings, selected: list[int], top_k: int, batch_size: int,
                explain: bool, hits, profile: dict) -> list[SearchResult]:
        if not selected:
            return []
        documents = Embeddings(self.routing_vectors(selected), self.value_vectors(selected),
                               self.representation_id, "document")
        scores = score(query, documents, config=self.scoring, document_batch_size=batch_size)
        ranked = sorted(range(len(selected)), key=lambda j: (-float(scores.combined[0, j]), self.ids[selected[j]]))
        results = []
        for j in ranked[:top_k]:
            di = selected[j]
            explanation = None
            if explain:
                pair = explain_pairs(query, documents.select(j), config=self.scoring)[0][0]
                explanation = {"slot_matches": pair.slot_matches()}
                if hits is not None:
                    hit_scores, hit_ids = hits
                    explanation["slot_hits"] = [
                        {"query_slot": int(qs), "rank": int(rank),
                         "document_slot": int(hit_ids[qs, rank] % self.slots),
                         "cosine": float(hit_scores[qs, rank])}
                        for qs, rank in zip(*np.nonzero(hit_ids // self.slots == di))]
            results.append(SearchResult(self.ids[di], float(scores.combined[0, j]), float(scores.routing[0, j]),
                                        float(scores.value[0, j]), di, self.metadata[di], explanation,
                                        dict(profile)))
        return results

    # ---- persistence ------------------------------------------------------------------------
    def describe(self) -> dict:
        return {"format": INDEX_FORMAT, "representation_id": self.representation_id, "documents": len(self),
                "storage": self.storage, "slots": self.slots, "routing_dim": self.d_k, "value_dim": self.d_v,
                "bytes_per_document": bytes_per_document(self.storage, self.slots, self.d_k, self.d_v),
                "shards": len(self._shards), "scoring": self.scoring.to_dict(),
                "default_profile": self.profile.to_dict()}

    def save(self, directory: str | Path) -> None:
        """Write .npy shards and index.json (last). Refuses to overwrite an existing directory."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        shards = []
        for number, (start, arrays) in enumerate(zip(self._starts, self._shards)):
            files = {}
            for name, array in arrays.items():
                relative = f"shard-{number:05d}.{name}.npy"
                np.save(root / relative, np.ascontiguousarray(array), allow_pickle=False)
                files[name] = {"path": relative, "sha256": sha256_file(root / relative),
                               "dtype": str(array.dtype), "shape": list(array.shape)}
            shards.append({"start": start, "count": len(arrays["routing"]), "files": files})
        manifest = {"format": INDEX_FORMAT, "library_version": __version__, "kind": "document",
                    "representation_id": self.representation_id, "storage": self.storage,
                    "representation": {"routing": "L2-normalised; int8 stores direction only",
                                       "value": "raw; int8 stores a float32 scale per slot",
                                       "quantization": "per-vector symmetric fixed point, scale=max|x|/127"},
                    "slots": self.slots, "routing_dim": self.d_k, "value_dim": self.d_v,
                    "scoring": self.scoring.to_dict(), "default_profile": self.profile.to_dict(),
                    "count": len(self), "ids": self.ids, "metadata": self.metadata, "shards": shards}
        (root / "index.json.tmp").write_text(canonical_json(manifest) + "\n")
        os.replace(root / "index.json.tmp", root / "index.json")

    @classmethod
    def load(cls, directory: str | Path, *, representation_id: str | None = None, model=None,
             mmap: bool = False, verify: bool = True) -> "Index":
        """Load an index. With `model` or `representation_id`, refuse one built by another model.

        mmap=True memory-maps the shards instead of reading them (v2 format only); verify=False
        skips the shard checksums (faster for large indexes).
        """
        expected = model.representation_id if model is not None else representation_id
        root = Path(directory)
        manifest = json.loads((root / "index.json").read_text())
        if manifest.get("format") == INDEX_FORMAT_V1:
            # v0.1 indexes recorded their only profile; keep searching them the same way.
            if manifest.get("profile") != "exact-document-routing-v1":
                raise ValueError("Unsupported index format or retrieval profile.")
            embeddings = Embeddings.load(root / "slots")
            _check_identity(embeddings.representation_id, expected)
            return cls(embeddings, manifest["ids"], scoring=ScoringConfig(**manifest["scoring"]),
                       metadata=manifest["metadata"], profile=EXACT)
        if manifest.get("format") != INDEX_FORMAT or manifest.get("kind") != "document":
            raise ValueError("Unsupported index format or retrieval profile.")
        _check_identity(manifest["representation_id"], expected)
        if manifest["storage"] not in STORAGE_TYPES:
            raise ValueError("Unsupported index storage type.")
        index = cls.__new__(cls)
        index.representation_id = manifest["representation_id"]
        index.storage = manifest["storage"]
        index.profile = SearchProfile.from_dict(manifest["default_profile"])
        index.scoring = ScoringConfig(**manifest["scoring"])
        index.slots, index.d_k, index.d_v = manifest["slots"], manifest["routing_dim"], manifest["value_dim"]
        index.ids, index.metadata = list(manifest["ids"]), list(manifest["metadata"])
        index._id_set = set(index.ids)
        if (len(index._id_set) != len(index.ids) or len(index.ids) != manifest["count"]
                or len(index.metadata) != len(index.ids)):
            raise ValueError("Index ids/metadata do not match the manifest count.")
        index._shards, index._starts, index._faiss, index._unit = [], [], None, None
        index.cache_routing = True
        expected_start = 0
        for shard in manifest["shards"]:
            arrays = {}
            for name, entry in shard["files"].items():
                relative = Path(entry["path"])
                if relative.is_absolute() or ".." in relative.parts or name not in {"routing", "value", "value_scale"}:
                    raise ValueError("Invalid index shard member.")
                path = root / relative
                if verify and sha256_file(path) != entry["sha256"]:
                    raise ValueError(f"Index shard checksum mismatch: {relative}.")
                array = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
                if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
                    raise ValueError(f"Index shard shape/dtype differs from the manifest: {relative}.")
                arrays[name] = array
            if shard["start"] != expected_start or len(arrays["routing"]) != shard["count"]:
                raise ValueError("Index shards are not contiguous.")
            if (arrays["routing"].shape[1:] != (index.slots, index.d_k)
                    or arrays["value"].shape[1:] != (index.slots, index.d_v)):
                raise ValueError("Index shard dimensions differ from the manifest.")
            if (index.storage == "int8") != ("value_scale" in arrays):
                raise ValueError("Index shard files do not match the storage type.")
            index._starts.append(shard["start"])
            index._shards.append(arrays)
            expected_start += shard["count"]
        if expected_start != len(index.ids):
            raise ValueError("Index shards do not cover every id.")
        return index


def _check_identity(actual: str, expected: str | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError("This index was built from a different model representation "
                         f"({actual[:12]}...) than the one requested ({expected[:12]}...). Re-encode the corpus.")

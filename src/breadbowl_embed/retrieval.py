"""Small-corpus exact routing retrieval followed by value reranking."""
from dataclasses import asdict, dataclass, field
import heapq
import json
import os
from pathlib import Path

import torch

from ._scoring import score_qk
from .config import ScoringConfig, canonical_json
from .embeddings import Embeddings
from .scoring import score, validate_pair


@dataclass(frozen=True)
class SearchResult:
    id: str
    score: float
    routing_score: float
    value_score: float
    document_index: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class Index:
    """In-memory corpus with persistent slots; no server or database required.

    Profile: exact-document-routing-v1. This scans all routing slots, retains
    candidate_k documents by exact routing score, then reranks their values.
    It is not the per-slot shortlist pipeline used for the paper's BEIR run.
    """
    profile = "exact-document-routing-v1"

    def __init__(self, embeddings: Embeddings, ids: list[str], *,
                 scoring: ScoringConfig | None = None, metadata: list[dict] | None = None):
        if embeddings.kind != "document":
            raise ValueError("Indexes require document embeddings.")
        if len(ids) != len(embeddings) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("Provide one non-empty string ID per document.")
        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique.")
        if metadata is not None and (len(metadata) != len(ids) or not all(isinstance(x, dict) for x in metadata)):
            raise ValueError("Provide one metadata object per document.")
        self.embeddings = embeddings.to("cpu")
        self.ids = list(ids)
        self.metadata = json.loads(canonical_json(metadata if metadata is not None else [{} for _ in ids]))
        self.scoring = scoring or ScoringConfig()

    @classmethod
    def from_embeddings(cls, embeddings, *, ids, scoring=None, metadata=None):
        return cls(embeddings, ids, scoring=scoring, metadata=metadata)

    @torch.inference_mode()
    def search(self, queries: Embeddings, *, top_k: int = 10, candidate_k: int = 100,
               batch_size: int = 128) -> list[list[SearchResult]]:
        if type(top_k) is not int or type(candidate_k) is not int or top_k < 1 or candidate_k < top_k:
            raise ValueError("Require candidate_k >= top_k >= 1.")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        queries = queries.to("cpu")
        validate_pair(queries, self.embeddings)
        results = []
        for qi in range(len(queries)):
            q = queries.select(qi)
            best = []
            for start in range(0, len(self.ids), batch_size):
                d = self.embeddings.select(slice(start, start + batch_size))
                routing = score_qk(q.routing, d.routing, tau_k=self.scoring.tau_k, mode=self.scoring.qk_mode)[0]
                candidates = [(-float(s), self.ids[start + j], start + j) for j, s in enumerate(routing)]
                best = heapq.nsmallest(candidate_k, best + candidates)
            selected = [item[2] for item in best]
            if not selected:
                results.append([])
                continue
            scores = score(q, self.embeddings.select(selected), config=self.scoring, document_batch_size=batch_size)
            ranked = [SearchResult(self.ids[di], float(scores.combined[0, j]),
                                   float(scores.routing[0, j]), float(scores.value[0, j]), di,
                                   self.metadata[di]) for j, di in enumerate(selected)]
            results.append(sorted(ranked, key=lambda row: (-row.score, row.id))[:top_k])
        return results

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        self.embeddings.save(root / "slots")
        manifest = {"format": "breadbowl_index_v1", "profile": self.profile, "ids": self.ids,
                    "metadata": self.metadata, "scoring": self.scoring.to_dict()}
        (root / "index.json.tmp").write_text(canonical_json(manifest) + "\n")
        os.replace(root / "index.json.tmp", root / "index.json")

    @classmethod
    def load(cls, directory: str | Path) -> "Index":
        root = Path(directory)
        manifest = json.loads((root / "index.json").read_text())
        if manifest.get("format") != "breadbowl_index_v1" or manifest.get("profile") != cls.profile:
            raise ValueError("Unsupported index format or retrieval profile.")
        return cls(Embeddings.load(root / "slots"), manifest["ids"],
                   scoring=ScoringConfig(**manifest["scoring"]), metadata=manifest["metadata"])

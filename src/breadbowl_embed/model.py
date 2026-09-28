"""User-facing model loader and query/document encoders."""
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from .architecture import QKVRetriever
from .config import FORMAT, ScoringConfig, canonical_json, fingerprint, sha256_file, validate_config
from .embeddings import Embeddings
from .retrieval import Index, SearchResult
from .scoring import Scores, score


def validate_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported model manifest format.")
    required = {"config.json", "model.safetensors", "backbone/config.json", "tokenizer/tokenizer_config.json"}
    files = manifest.get("files", {})
    if not required.issubset(files):
        raise ValueError("Model manifest is missing required release files.")
    for relative, digest in files.items():
        member = PurePosixPath(relative)
        # Hub snapshots use symlinks into a shared blob cache. Validate the
        # manifest name lexically and verify bytes, without rejecting those links.
        path = root / relative
        if member.is_absolute() or ".." in member.parts or not path.is_file():
            raise ValueError("Invalid model manifest member.")
        if sha256_file(path) != digest:
            raise ValueError(f"Checksum mismatch for {relative}.")
    # Tokenizer/config loaders may read ancillary files: reject unverified members.
    for name in ("backbone", "tokenizer"):
        for member in (root / name).rglob("*"):
            if member.is_file() and member.relative_to(root).as_posix() not in files:
                raise ValueError("Unverified tokenizer/backbone file.")
    return manifest


class BreadBowl:
    """One model, paired routing/value slots, no service dependency.

    `network` exposes the underlying torch.nn.Module for research use.
    Encoders return CPU tensors and retain raw document-value magnitudes.
    """
    def __init__(self, network: QKVRetriever, tokenizer, config: dict, *,
                 representation_id: str, device: str = "cpu"):
        validate_config(config)
        self.config = deepcopy(config)
        self.scoring = ScoringConfig(**config["scoring"])
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.network = network.to(self.device).eval()
        self.representation_id = representation_id

    @classmethod
    def from_pretrained(cls, model_id_or_path: str | Path, *, revision: str | None = None,
                        device: str = "cpu", dtype: str = "native",
                        local_files_only: bool = False) -> "BreadBowl":
        """Load a complete verified release. Native preserves mixed parameter dtypes.

        CPU is the conservative default. CUDA and MPS must be selected explicitly.
        A HF repo containing only a paper is not yet a loadable model release.
        """
        path = Path(model_id_or_path)
        if not path.is_dir():
            path = Path(snapshot_download(str(model_id_or_path), revision=revision,
                                          local_files_only=local_files_only,
                                          allow_patterns=["config.json", "manifest.json", "model.safetensors",
                                                          "backbone/*", "tokenizer/*"]))
        if not (path / "manifest.json").exists():
            raise ValueError("This repository has no BreadBowl weight release yet (manifest.json missing).")
        manifest = validate_manifest(path)
        config = json.loads((path / "config.json").read_text())
        validate_config(config)
        network, tokenizer = QKVRetriever.from_config(
            config["architecture"], train_config=config.get("inference_setup", {}),
            backbone_init="config", backbone_source_override=str(path / "backbone"),
            tokenizer_source_override=str(path / "tokenizer"), local_files_only_override=True,
            enable_gradient_checkpointing=False,
        )
        network.load_state_dict(load_file(str(path / "model.safetensors")), strict=True, assign=True)
        if dtype != "native":
            allowed = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
            if dtype not in allowed:
                raise ValueError(f"Unsupported dtype {dtype!r}.")
            network.to(dtype=allowed[dtype])
        identity = fingerprint({"model": manifest["representation_id"], "compute_dtype": dtype})
        return cls(network, tokenizer, config, representation_id=identity, device=device)

    def save_pretrained(self, directory: str | Path) -> None:
        """Create an inference-only release with a checksum manifest."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        backbone_config = deepcopy(self.network.backbone.config)
        backbone_config.use_cache = False
        backbone_config.save_pretrained(root / "backbone")
        tokenizer = deepcopy(self.tokenizer)
        if hasattr(tokenizer, "backend_tokenizer"):
            tokenizer.backend_tokenizer.no_padding()
            tokenizer.backend_tokenizer.no_truncation()
        tokenizer.save_pretrained(root / "tokenizer")
        tokenizer_config_path = root / "tokenizer/tokenizer_config.json"
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        # Loader location metadata must not change cache identity on a resave.
        for key in ("is_local", "name_or_path", "_name_or_path"):
            tokenizer_config.pop(key, None)
        tokenizer_config_path.write_text(canonical_json(tokenizer_config) + "\n")
        config = deepcopy(self.config)
        config["architecture"]["backbone"].update({"pretrained": "backbone", "local_files_only": True,
                                                    "gradient_checkpointing": False})
        (root / "config.json").write_text(canonical_json(config) + "\n")
        # Clone aliases deliberately so strict state-dict loading preserves every key.
        tensors = {k: v.detach().cpu().contiguous().clone() for k, v in self.network.state_dict().items()}
        save_file(tensors, str(root / "model.safetensors"))
        del tensors
        files = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}
        encoding_config = {k: v for k, v in config.items() if k not in {"scoring", "provenance"}}
        identity = fingerprint({"weights": files["model.safetensors"], "encoding_config": encoding_config,
                                "assets": {k: v for k, v in files.items() if k.startswith(("backbone/", "tokenizer/"))}})
        manifest = {"format": FORMAT, "representation_id": identity, "files": files}
        (root / "manifest.json").write_text(canonical_json(manifest) + "\n")

    @torch.inference_mode()
    def _encode(self, texts: list[str], kind: str, *, batch_size: int,
                instructions: list[str] | None = None) -> Embeddings:
        if isinstance(texts, str) or not isinstance(texts, (list, tuple)) or not all(isinstance(t, str) for t in texts):
            raise TypeError("Pass a list of strings, not one string.")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if instructions is not None and (len(instructions) != len(texts) or not all(isinstance(s, str) for s in instructions)):
            raise ValueError("Provide one instruction per query.")
        prep = self.config["preprocessing"]
        if kind == "query":
            formatted = [prep["query_format"].format(query=t, instruction=(instructions[i] if instructions is not None
                         else prep["default_instruction"])) for i, t in enumerate(texts)]
            limit = prep["max_query_length"]
        else:
            formatted = [prep["document_format"].format(text=t) for t in texts]
            limit = prep["max_document_length"]
        routes, values = [], []
        for start in range(0, len(formatted), batch_size):
            batch = self.tokenizer(formatted[start:start + batch_size], padding=True,
                                   truncation=True, max_length=limit, return_tensors="pt")
            # The architecture consumes only these two fields; token_type_ids are not used.
            output = self.network(input_ids=batch["input_ids"].to(self.device),
                                  attention_mask=batch["attention_mask"].to(self.device))
            routes.append(torch.nn.functional.normalize(output.routing.float(), dim=-1).cpu())
            values.append(output.value.float().cpu())
        architecture = self.config["architecture"]
        slots = architecture["slot_extractor"]["num_slots"]
        if routes:
            routing, value = torch.cat(routes), torch.cat(values)
        else:
            routing = torch.empty(0, slots, architecture["heads"]["d_k"])
            value = torch.empty(0, slots, architecture["heads"]["d_v"])
        return Embeddings(routing, value, self.representation_id, kind)

    def encode_queries(self, texts: list[str], *, batch_size: int = 8,
                       instructions: list[str] | None = None) -> Embeddings:
        return self._encode(texts, "query", batch_size=batch_size, instructions=instructions)

    def encode_documents(self, texts: list[str], *, batch_size: int = 8) -> Embeddings:
        """Encode one passage per item. Long items are truncated, not silently chunked."""
        return self._encode(texts, "document", batch_size=batch_size)

    def score(self, queries: Embeddings, documents: Embeddings, **kwargs) -> Scores:
        if queries.representation_id != self.representation_id:
            raise ValueError("Representations were produced by a different model configuration.")
        return score(queries, documents, config=self.scoring, **kwargs)

    def index(self, documents: Embeddings, *, ids: list[str], metadata=None) -> Index:
        if documents.representation_id != self.representation_id:
            raise ValueError("Representations were produced by a different model configuration.")
        return Index(documents, ids, scoring=self.scoring, metadata=metadata)

    def rerank(self, query: str, documents: list[str] | Embeddings, *, ids: list[str] | None = None,
               top_k: int | None = None, batch_size: int = 8) -> list[SearchResult]:
        """Rerank supplied candidates; pass cached document embeddings to encode once."""
        encoded = documents if isinstance(documents, Embeddings) else self.encode_documents(documents, batch_size=batch_size)
        ids = [str(i) for i in range(len(encoded))] if ids is None else ids
        if len(ids) != len(encoded) or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("Provide unique non-empty IDs matching the documents.")
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError("top_k must be positive.")
        scores = self.score(self.encode_queries([query]), encoded)
        ranked = [SearchResult(ids[i], float(scores.combined[0, i]), float(scores.routing[0, i]),
                               float(scores.value[0, i]), i) for i in range(len(ids))]
        return sorted(ranked, key=lambda r: (-r.score, r.id))[:top_k]

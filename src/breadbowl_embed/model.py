"""User-facing model loader, tokenizer view and query/document encoders."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from ._weights import read_safetensors
from .architecture import QKVRetriever
from .config import (ARCHITECTURE_NAME, FORMAT, LOADER_ONLY_TOKENIZER_KEYS, ScoringConfig, canonical_json,
                     compute_dtype_name, encoding_config, fingerprint, format_with_span, sha256_file,
                     validate_config)
from .embeddings import Embeddings
from .retrieval import Index, SearchResult
from .scoring import Explanation, Scores, explain, score, score_routing
from .tokenization import TokenizedText, display_strings, map_offsets, segments_for

_TORCH_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
# One text per forward pass: on CPU it is the fastest setting for full-length documents, and it is
# the only one whose results do not depend on which other texts share the batch (docs/validation.md).
DEFAULT_BATCH_SIZE = 1


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


def release_identity(config: dict, files: dict[str, str]) -> str:
    """Representation identity of a release: weights, encoding configuration and tokenizer/backbone files."""
    return fingerprint({"weights": files["model.safetensors"], "encoding_config": encoding_config(config),
                        "assets": {k: v for k, v in files.items() if k.startswith(("backbone/", "tokenizer/"))}})


def write_manifest(root: Path, config: dict) -> dict:
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()
             and p.name != "manifest.json"}
    manifest = {"format": FORMAT, "representation_id": release_identity(config, files), "files": files}
    (root / "manifest.json").write_text(canonical_json(manifest) + "\n")
    return manifest


def set_release_metadata(directory: str | Path, *, repository: str | None = None, revision: str | None = None,
                         license: str | None = None) -> dict:
    """Record publish-time facts in a local schema-2 release (before upload).

    Updates config.json's release section and the manifest checksums. The representation
    identity does not change (release metadata is outside it); this is checked.
    """
    root = Path(directory)
    before = validate_manifest(root)
    config = json.loads((root / "config.json").read_text())
    validate_config(config)
    if config["schema_version"] < 2:
        raise ValueError("Release metadata needs a schema-2 release; re-export it with this version.")
    hub = config["release"]["huggingface"]
    if repository is not None:
        hub["repository"] = repository
    if revision is not None:
        hub["revision"] = revision
    if license is not None:
        config["release"]["license"] = license
    validate_config(config)
    (root / "config.json").write_text(canonical_json(config) + "\n")
    manifest = write_manifest(root, config)
    if manifest["representation_id"] != before["representation_id"]:  # pragma: no cover - defensive
        raise RuntimeError("Release metadata changed the representation identity.")
    return config["release"]


def _construction_context():
    """Build the network without initialising weights that are about to be replaced."""
    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # pragma: no cover - other transformers layouts
        return nullcontext()
    return no_init_weights()


class BreadBowl:
    """One model, paired routing/value slots, no service dependency.

    `network` exposes the underlying torch.nn.Module for research use.
    Encoders return CPU tensors and retain raw document-value magnitudes.
    """
    def __init__(self, network: QKVRetriever, tokenizer, config: dict, *,
                 representation_id: str, device: str = "cpu", compute_dtype: str = "native",
                 release_id: str | None = None, weights_sha256: str | None = None,
                 loaded_revision: str | None = None):
        validate_config(config)
        self.config = deepcopy(config)
        self.scoring = ScoringConfig(**config["scoring"])
        self.tokenizer = tokenizer
        # The model was trained and evaluated with right padding and right truncation; a causal
        # backbone gives different states for left-padded text.
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.special_ids = frozenset(int(i) for i in getattr(tokenizer, "all_special_ids", []))
        self.device = torch.device(device)
        self.network = network.to(self.device).eval()
        self.representation_id = representation_id
        self.compute_dtype = compute_dtype
        self.release_id = release_id
        self.weights_sha256 = weights_sha256
        self.loaded_revision = loaded_revision

    @classmethod
    def from_pretrained(cls, model_id_or_path: str | Path, *, revision: str | None = None,
                        device: str | None = None, dtype=None,
                        local_files_only: bool = False) -> "BreadBowl":
        """Load a complete verified release (local directory or Hugging Face repo id).

        dtype: None/"native" keeps the stored parameter dtypes (float32 for v1.3; mixed for the
        v1.2 preview); "float32", "bfloat16" or "float16" (or a torch.dtype) casts each tensor
        while it is read, so peak memory stays near the cast model's size. The compute dtype is
        part of the representation identity. device defaults to "cpu"; CUDA and MPS must be
        selected explicitly.
        """
        dtype_name = compute_dtype_name(dtype)
        device = device or "cpu"
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
        with _construction_context():
            network, tokenizer = QKVRetriever.from_config(
                config["architecture"], train_config=config.get("inference_setup", {}),
                backbone_init="config", backbone_source_override=str(path / "backbone"),
                tokenizer_source_override=str(path / "tokenizer"), local_files_only_override=True,
                enable_gradient_checkpointing=False,
            )
        target = _TORCH_DTYPES.get(dtype_name)
        # Native: Safetensors maps the file, so weights are shared, reclaimable page cache rather
        # than private memory. A cast needs new storage: read and cast tensor by tensor instead.
        state = (load_file(str(path / "model.safetensors")) if target is None
                 else read_safetensors(path / "model.safetensors", target))
        network.load_state_dict(state, strict=True, assign=True)
        del state
        if target is not None:  # as Module.to(dtype): floating buffers (e.g. rotary frequencies) too
            for module in network.modules():
                for name, buffer in module._buffers.items():
                    if buffer is not None and buffer.is_floating_point():
                        module._buffers[name] = buffer.to(target)
        identity = fingerprint({"model": manifest["representation_id"], "compute_dtype": dtype_name})
        # A Hub snapshot directory is named after its commit.
        loaded_revision = path.name if path.parent.name == "snapshots" else None
        return cls(network, tokenizer, config, representation_id=identity, device=device,
                   compute_dtype=dtype_name, release_id=manifest["representation_id"],
                   weights_sha256=manifest["files"]["model.safetensors"], loaded_revision=loaded_revision)

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
        for key in LOADER_ONLY_TOKENIZER_KEYS:
            tokenizer_config.pop(key, None)
        tokenizer_config_path.write_text(canonical_json(tokenizer_config) + "\n")
        # Clone aliases deliberately so strict state-dict loading preserves every key.
        tensors = {k: v.detach().cpu().contiguous().clone() for k, v in self.network.state_dict().items()}
        save_file(tensors, str(root / "model.safetensors"))
        del tensors
        config = deepcopy(self.config)
        config["architecture"]["backbone"].update({"pretrained": "backbone", "local_files_only": True,
                                                    "gradient_checkpointing": False})
        if config["schema_version"] >= 2:
            config["release"]["weights_sha256"] = sha256_file(root / "model.safetensors")
        (root / "config.json").write_text(canonical_json(config) + "\n")
        write_manifest(root, config)

    # ---- tokenisation -----------------------------------------------------------------------
    def format_texts(self, texts: list[str], kind: str, *,
                     instructions: list[str] | None = None) -> list[tuple[str, tuple[int, int]]]:
        """The exact model inputs and where each caller text sits inside them."""
        prep = self.config["preprocessing"]
        if kind == "query":
            return [format_with_span(prep["query_format"], "query", query=t,
                                     instruction=prep["default_instruction"] if instructions is None else instructions[i])
                    for i, t in enumerate(texts)]
        if kind != "document":
            raise ValueError("kind must be 'query' or 'document'.")
        return [format_with_span(prep["document_format"], "text", text=t) for t in texts]

    def _limit(self, kind: str) -> int:
        prep = self.config["preprocessing"]
        return prep["max_query_length"] if kind == "query" else prep["max_document_length"]

    @staticmethod
    def _check_texts(texts, instructions) -> None:
        if isinstance(texts, str) or not isinstance(texts, (list, tuple)) or not all(isinstance(t, str) for t in texts):
            raise TypeError("Pass a list of strings, not one string.")
        if instructions is not None and (len(instructions) != len(texts) or not all(isinstance(s, str) for s in instructions)):
            raise ValueError("Provide one instruction per query.")

    @staticmethod
    def _resolve_instructions(texts, instruction, instructions):
        if instruction is not None and instructions is not None:
            raise ValueError("Pass either instruction or instructions, not both.")
        if instruction is not None:
            if not isinstance(instruction, str):
                raise ValueError("instruction must be a string.")
            return [instruction] * len(texts) if isinstance(texts, (list, tuple)) else None
        return instructions

    def _tokenize_batch(self, texts: list[str], formatted: list[tuple[str, tuple[int, int]]], limit: int,
                        details: bool):
        """The one tokenizer call behind both `tokenize` and the encoders.

        Returns the padded model inputs and, with details, one TokenizedText per text.
        """
        batch = self.tokenizer([text for text, _ in formatted], padding=True, truncation=True, max_length=limit,
                               return_tensors="pt", return_offsets_mapping=details)
        if not details:
            return batch, None
        untruncated = [len(ids) for ids in self.tokenizer([text for text, _ in formatted], truncation=False)["input_ids"]]
        mask = batch["attention_mask"].bool()
        items = []
        for b, (text, (model_input, span)) in enumerate(zip(texts, formatted)):
            ids = batch["input_ids"][b][mask[b]]
            id_list = ids.tolist()
            offsets = map_offsets(batch["offset_mapping"][b][mask[b]].tolist(), span)
            items.append(TokenizedText(
                text=text, formatted=model_input, span=span, input_ids=ids,
                tokens=tuple(self.tokenizer.convert_ids_to_tokens(id_list)),
                token_texts=display_strings(self.tokenizer, id_list), offsets=offsets,
                segments=segments_for(id_list, offsets, self.special_ids),
                truncated=untruncated[b] > len(id_list), untruncated_token_count=untruncated[b]))
        return batch, items

    def tokenize(self, texts: list[str], kind: str, *, instruction: str | None = None,
                 instructions: list[str] | None = None) -> list[TokenizedText]:
        """How each text will be encoded, without running the model.

        kind is "query" or "document"; queries get the instruction template (default: the
        checkpoint's instruction). Uses the same tokenizer call as the encoders, so input ids,
        token counts, truncation and offsets equal what `encode_queries`/`encode_documents`
        report with return_tokens=True.
        """
        instructions = self._resolve_instructions(texts, instruction, instructions)
        self._check_texts(texts, instructions)
        if kind == "document" and instructions is not None:
            raise ValueError("Instructions apply to queries only.")
        formatted = self.format_texts(list(texts), kind, instructions=instructions)
        items = []
        for start in range(0, len(formatted), 64):
            _, part = self._tokenize_batch(list(texts[start:start + 64]), formatted[start:start + 64],
                                           self._limit(kind), True)
            items.extend(part)
        return items

    # ---- encoding ---------------------------------------------------------------------------
    @torch.inference_mode()
    def _encode(self, texts: list[str], kind: str, *, batch_size: int,
                instructions: list[str] | None = None, return_attention: bool = False,
                return_tokens: bool = False) -> Embeddings:
        self._check_texts(texts, instructions)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        formatted = self.format_texts(list(texts), kind, instructions=instructions)
        routes, values, attentions, tokenized = [], [], [], []
        for start in range(0, len(formatted), batch_size):
            chunk = formatted[start:start + batch_size]
            batch, items = self._tokenize_batch(list(texts[start:start + batch_size]), chunk, self._limit(kind),
                                                return_tokens)
            mask = batch["attention_mask"]
            if not bool(mask.sum(dim=1).all()):
                raise ValueError("A text produced no tokens (empty document?); it cannot be encoded.")
            # The architecture consumes only these two fields; token_type_ids are not used.
            output = self.network(input_ids=batch["input_ids"].to(self.device),
                                  attention_mask=mask.to(self.device))
            routes.append(F.normalize(output.routing.float(), dim=-1).cpu())
            values.append(output.value.float().cpu())
            if return_attention:
                attention = output.attention.float().cpu()
                if attention.ndim == 3:  # single-head extractors return [batch, slots, tokens]
                    attention = attention.unsqueeze(1)
                attentions.extend(attention[b][..., mask[b].bool()] for b in range(len(chunk)))
            if return_tokens:
                tokenized.extend(items)
        architecture = self.config["architecture"]
        slots = architecture["slot_extractor"]["num_slots"]
        if routes:
            routing, value = torch.cat(routes), torch.cat(values)
        else:
            routing = torch.empty(0, slots, architecture["heads"]["d_k"])
            value = torch.empty(0, slots, architecture["heads"]["d_v"])
        extras = {}
        if return_attention:
            extras["attention"] = tuple(attentions)
        if return_tokens:
            extras.update(input_ids=tuple(t.input_ids for t in tokenized), tokens=tuple(t.tokens for t in tokenized),
                          token_texts=tuple(t.token_texts for t in tokenized),
                          offsets=tuple(t.offsets for t in tokenized), truncated=tuple(t.truncated for t in tokenized))
        return Embeddings(routing, value, self.representation_id, kind, **extras)

    def encode_queries(self, texts: list[str], *, batch_size: int = DEFAULT_BATCH_SIZE, instruction: str | None = None,
                       instructions: list[str] | None = None, return_attention: bool = False,
                       return_tokens: bool = False) -> Embeddings:
        """Encode queries as "{instruction}\\nQuery: {query}" (the checkpoint's format).

        instruction: one instruction for every query (default: the checkpoint's). instructions:
        one per query. return_attention adds per-query [heads, slots, tokens] slot attention;
        return_tokens adds input_ids, token strings, display strings, offsets into each query and
        truncation flags. batch_size defaults to 1: each text's slots then do not depend on the
        other texts encoded with it; larger batches pad and change results slightly (most in
        bfloat16), and on CPU they are not faster.
        """
        instructions = self._resolve_instructions(texts, instruction, instructions)
        return self._encode(texts, "query", batch_size=batch_size, instructions=instructions,
                            return_attention=return_attention, return_tokens=return_tokens)

    def encode_documents(self, texts: list[str], *, batch_size: int = DEFAULT_BATCH_SIZE,
                         return_attention: bool = False, return_tokens: bool = False) -> Embeddings:
        """Encode one passage per item. Long items are truncated, not silently chunked.

        batch_size defaults to 1 (see encode_queries).
        """
        return self._encode(texts, "document", batch_size=batch_size,
                            return_attention=return_attention, return_tokens=return_tokens)

    # ---- scoring and retrieval --------------------------------------------------------------
    def _check(self, embeddings: Embeddings) -> None:
        if embeddings.representation_id != self.representation_id:
            raise ValueError("Representations were produced by a different model configuration.")

    def score(self, queries: Embeddings, documents: Embeddings, **kwargs) -> Scores:
        self._check(queries)
        return score(queries, documents, config=self.scoring, **kwargs)

    def score_routing(self, queries: Embeddings, documents: Embeddings | torch.Tensor, **kwargs) -> torch.Tensor:
        """Routing scores R only, for documents without value slots; equals score(...).routing."""
        self._check(queries)
        return score_routing(queries, documents, config=self.scoring, **kwargs)

    def explain(self, queries: Embeddings, documents: Embeddings) -> list[list[Explanation]]:
        """Slot-level decomposition of every (query, document) score: result[i][j]."""
        self._check(queries)
        return explain(queries, documents, config=self.scoring)

    def index(self, documents: Embeddings, *, ids: list[str], metadata=None, storage: str = "float32",
              profile=None) -> Index:
        """An Index over encoded documents; `profile` is its default search profile ("default")."""
        self._check(documents)
        return Index(documents.without_tokens(), ids, scoring=self.scoring, metadata=metadata, storage=storage,
                     profile=profile)

    def load_index(self, directory: str | Path, *, mmap: bool = False, verify: bool = True) -> Index:
        """Load an index, refusing one built from a different model representation."""
        return Index.load(directory, model=self, mmap=mmap, verify=verify)

    def rerank(self, query: str, documents: list[str] | Embeddings, *, ids: list[str] | None = None,
               top_k: int | None = None, batch_size: int = DEFAULT_BATCH_SIZE,
               instruction: str | None = None) -> list[SearchResult]:
        """Rerank supplied candidates; pass cached document embeddings to encode once."""
        encoded = documents if isinstance(documents, Embeddings) else self.encode_documents(documents, batch_size=batch_size)
        ids = [str(i) for i in range(len(encoded))] if ids is None else ids
        if len(ids) != len(encoded) or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("Provide unique non-empty IDs matching the documents.")
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError("top_k must be positive.")
        scores = self.score(self.encode_queries([query], instruction=instruction), encoded)
        ranked = [SearchResult(ids[i], float(scores.combined[0, i]), float(scores.routing[0, i]),
                               float(scores.value[0, i]), i) for i in range(len(ids))]
        return sorted(ranked, key=lambda r: (-r.score, r.id))[:top_k]

    def info(self) -> dict:
        """Identity, release facts, provenance and settings of the loaded model."""
        parameters = list(self.network.parameters())
        release = self.config.get("release") or {}
        hub = release.get("huggingface") or {}
        return {"representation_id": self.representation_id, "release_id": self.release_id,
                "schema_version": self.config["schema_version"], "compute_dtype": self.compute_dtype,
                "device": str(self.device),
                "parameter_dtypes": sorted({str(p.dtype).removeprefix("torch.") for p in parameters}),
                "parameters": sum(p.numel() for p in parameters),
                "backbone": type(self.network.backbone).__name__,
                "release": {"architecture": release.get("architecture", ARCHITECTURE_NAME),
                            "license": release.get("license"),
                            "weights_sha256": self.weights_sha256 or release.get("weights_sha256"),
                            "huggingface": {"repository": hub.get("repository"), "revision": hub.get("revision"),
                                            "loaded_revision": self.loaded_revision}},
                "architecture": {"slots": self.config["architecture"]["slot_extractor"]["num_slots"],
                                 "slot_heads": self.config["architecture"]["slot_extractor"].get("num_heads", 1),
                                 "d_attn": self.config["architecture"]["slot_extractor"]["d_attn"],
                                 "d_k": self.config["architecture"]["heads"]["d_k"],
                                 "d_v": self.config["architecture"]["heads"]["d_v"]},
                "preprocessing": self.config["preprocessing"], "scoring": self.scoring.to_dict(),
                "default_batch_size": DEFAULT_BATCH_SIZE,
                "provenance": self.config.get("provenance", {})}

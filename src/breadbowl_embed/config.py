"""Versioned inference configuration; no training-machine paths are required."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import string
from typing import Any

FORMAT = "breadbowl_embed_v1"
# Schema 1: v1.2-preview and the first v1.3 export. Schema 2 adds the `release` section
# (architecture name, licence, weights sha256, Hugging Face repository and revision). Both load.
SCHEMA_VERSION = 2
SCHEMA_VERSIONS = (1, 2)
ARCHITECTURE_NAME = "qkv-slot-encoder"
DEFAULT_LICENSE = "Apache-2.0"
# The configuration that decides how text becomes slots. The release identity covers exactly
# these keys (plus weights and tokenizer/backbone files); release metadata, scoring and
# provenance do not change it. "schema_version" below is the encoding schema, still 1.
ENCODING_KEYS = ("format", "architecture", "preprocessing", "inference_setup", "routing_normalized",
                 "value_normalized")
# Loader bookkeeping that tokenizers write into tokenizer_config.json. It describes how a copy
# was loaded, not the tokenizer, so it must not change a release's representation identity.
LOADER_ONLY_TOKENIZER_KEYS = ("is_local", "name_or_path", "_name_or_path", "local_files_only",
                              "cache_dir", "revision", "token", "trust_remote_code", "_commit_hash")
COMPUTE_DTYPES = ("native", "float32", "bfloat16", "float16")


def encoding_config(config: dict) -> dict:
    """The part of a release config that its representation identity covers.

    Schema 1 releases fingerprinted every key except scoring and provenance; for their keys that
    is the same set as ENCODING_KEYS plus schema_version 1, so identities are unchanged.
    """
    if config.get("schema_version") == 1:
        return {k: v for k, v in config.items() if k not in {"scoring", "provenance"}}
    return {**{k: config[k] for k in ENCODING_KEYS if k in config}, "schema_version": 1}


def release_section(*, license: str | None = DEFAULT_LICENSE, repository: str | None = None,
                    revision: str | None = None, weights_sha256: str | None = None) -> dict:
    return {"architecture": ARCHITECTURE_NAME, "license": license, "weights_sha256": weights_sha256,
            "huggingface": {"repository": repository, "revision": revision}}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ScoringConfig:
    qk_mode: str = "logmeanexp"
    tau_k: float = 0.02
    tau_v: float = 0.1
    value_mix: float = 0.25

    def __post_init__(self):
        if self.qk_mode not in {"logmeanexp", "max"}:
            raise ValueError("Public releases support normalized logmeanexp or max routing.")
        if not all(math.isfinite(x) and x > 0 for x in (self.tau_k, self.tau_v)):
            raise ValueError("Scoring temperatures must be finite and positive.")
        if not math.isfinite(self.value_mix) or not 0 <= self.value_mix <= 1:
            raise ValueError("value_mix must be between zero and one.")

    def to_dict(self):
        return asdict(self)


def validate_config(config: dict) -> None:
    if config.get("format") != FORMAT or config.get("schema_version") not in SCHEMA_VERSIONS:
        raise ValueError("Unsupported BreadBowl model format/schema version.")
    if config["schema_version"] >= 2:
        release = config.get("release")
        if not isinstance(release, dict) or not isinstance(release.get("architecture"), str):
            raise ValueError("Schema 2 configs need a release section with an architecture name.")
        hub = release.get("huggingface")
        if not isinstance(hub, dict) or not all(hub.get(k) is None or isinstance(hub[k], str)
                                                for k in ("repository", "revision")):
            raise ValueError("release.huggingface needs repository/revision strings or nulls.")
        for key in ("license", "weights_sha256"):
            if release.get(key) is not None and not isinstance(release[key], str):
                raise ValueError(f"release.{key} must be a string or null.")
    ScoringConfig(**config["scoring"])
    prep = config["preprocessing"]
    for field in ("max_query_length", "max_document_length"):
        if type(prep[field]) is not int or prep[field] <= 0:
            raise ValueError(f"Invalid {field}.")
    for field in ("query_format", "document_format", "default_instruction"):
        if not isinstance(prep[field], str):
            raise ValueError(f"Invalid {field}.")
    prep["query_format"].format(query="test", instruction="test")
    prep["document_format"].format(text="test")
    if config.get("routing_normalized") is not True or config.get("value_normalized") is not False:
        raise ValueError("Expected normalized routing and raw value representations.")


def compute_dtype_name(dtype) -> str:
    """Normalise a user dtype (None, string or torch.dtype) to one of COMPUTE_DTYPES."""
    if dtype is None:
        return "native"
    name = str(dtype).removeprefix("torch.")
    name = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16", "half": "float16",
            "float": "float32"}.get(name, name)
    if name not in COMPUTE_DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {COMPUTE_DTYPES}.")
    return name


def format_with_span(template: str, field: str, **values: str) -> tuple[str, tuple[int, int]]:
    """Render `template` like str.format and return where `field`'s value landed.

    The span is the first occurrence of the field, as character offsets in the result; it lets
    token offsets in the formatted text be mapped back into the caller's original text.
    """
    formatter = string.Formatter()
    pieces, span = [], None
    position = 0
    for literal, name, spec, conversion in formatter.parse(template):
        pieces.append(literal)
        position += len(literal)
        if name is None:
            continue
        value = formatter.convert_field(values[name], conversion)
        rendered = formatter.format_field(value, spec or "")
        if name == field and span is None and conversion is None and not spec:
            span = (position, position + len(rendered))
        pieces.append(rendered)
        position += len(rendered)
    text = "".join(pieces)
    if text != template.format(**values):  # pragma: no cover - defensive
        raise ValueError("Template rendering mismatch.")
    if span is None:
        raise ValueError(f"Template {template!r} does not contain a plain {{{field}}} field.")
    return text, span

"""Versioned inference configuration; no training-machine paths are required."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

FORMAT = "breadbowl_embed_v1"


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
    if config.get("format") != FORMAT or config.get("schema_version") != 1:
        raise ValueError("Unsupported BreadBowl model format/schema version.")
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

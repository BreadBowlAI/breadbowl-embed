"""Response models: plain dataclasses parsed from the contract's JSON.

Why dataclasses rather than pydantic: the SDK has to install cleanly next to ML stacks that pin
pydantic 1 or 2, and the response shapes are fixed by the OpenAPI contract, which
``tests/test_contract.py`` checks these classes against. Parsing is lenient (unknown fields are
kept in ``.raw``, never rejected) so a newer server does not break an older client.

Field names are the wire names. Timestamps stay ISO-8601 strings and counters stay decimal
strings, exactly as the contract sends them.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from ._tensor import Tensor

__all__ = [
    "ResponseMeta",
    "BaseModel",
    "ModelRef",
    "HealthStatus",
    "ReadinessStatus",
    "HuggingFaceRef",
    "TokenizerRef",
    "ModelIdentity",
    "Representation",
    "ModelDimensions",
    "TokenLimits",
    "TextFormats",
    "ScoringParameters",
    "SearchProfile",
    "ModelCapabilities",
    "ServiceLimits",
    "Model",
    "Usage",
    "Token",
    "TextEncoding",
    "EncodedText",
    "Explanation",
    "SlotVectors",
    "EncodeResponse",
    "ScoreResult",
    "ScoreResponse",
    "ResolvedChunkingConfig",
    "Index",
    "DeleteIndexResponse",
    "DocumentError",
    "DocumentOutcome",
    "UpsertDocumentsResponse",
    "StoredChunk",
    "StoredDocument",
    "DeleteDocumentResponse",
    "SearchResult",
    "SearchResponse",
    "UsageCounters",
    "TenantLimits",
    "RemainingAllowance",
    "KeyRateLimit",
    "CallingApiKey",
    "UsageSummary",
    "MetadataValue",
    "Metadata",
]

MetadataValue = Union[str, float, bool]
Metadata = Dict[str, MetadataValue]

M = TypeVar("M", bound="BaseModel")


@dataclass(frozen=True)
class ResponseMeta:
    """HTTP facts about the response a model was parsed from."""

    request_id: Optional[str]
    status_code: int
    headers: Mapping[str, str]
    idempotency_key: Optional[str] = None

    @property
    def idempotent_replayed(self) -> bool:
        """``True`` when the server replayed a stored response for this ``Idempotency-Key``."""
        return str(self.headers.get("idempotent-replayed", "")).lower() == "true"


_HINTS: Dict[type, Dict[str, Any]] = {}


def _hints(cls: type) -> Dict[str, Any]:
    hints = _HINTS.get(cls)
    if hints is None:
        module = sys.modules[cls.__module__]
        hints = get_type_hints(cls, vars(module))
        _HINTS[cls] = hints
    return hints


def _convert(tp: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(tp)
    if origin is Union:
        members = [a for a in get_args(tp) if a is not type(None)]
        if len(members) == 1:
            return _convert(members[0], value)
        return value  # scalar unions (metadata values) pass through
    if origin in (list, List):
        (item,) = get_args(tp) or (Any,)
        return [_convert(item, v) for v in value] if isinstance(value, list) else value
    if origin in (dict, Dict):
        args = get_args(tp)
        item = args[1] if len(args) == 2 else Any
        return {k: _convert(item, v) for k, v in value.items()} if isinstance(value, dict) else value
    if isinstance(tp, type):
        if issubclass(tp, BaseModel):
            return tp.from_dict(value) if isinstance(value, Mapping) else value
        if tp is Tensor:
            return Tensor.from_dict(value) if isinstance(value, Mapping) else value
        if tp is float and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
    return value


def _unconvert(value: Any) -> Any:
    if isinstance(value, (BaseModel, Tensor)):
        return value.to_dict()
    if isinstance(value, list):
        return [_unconvert(v) for v in value]
    if isinstance(value, dict):
        return {k: _unconvert(v) for k, v in value.items()}
    return value


class BaseModel:
    """Shared parsing, serialization and response metadata for every response model."""

    _raw: Dict[str, Any]
    _meta: Optional[ResponseMeta]

    @classmethod
    def from_dict(cls: Type[M], data: Mapping[str, Any]) -> M:
        hints = _hints(cls)
        kwargs: Dict[str, Any] = {}
        for f in dataclasses.fields(cls):  # type: ignore[arg-type]
            if f.name in data:
                kwargs[f.name] = _convert(hints[f.name], data[f.name])
            elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                kwargs[f.name] = None  # lenient: a server that omits a required field still parses
        obj = cls(**kwargs)
        object.__setattr__(obj, "_raw", dict(data))
        object.__setattr__(obj, "_meta", None)
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """The wire JSON for this object (optional fields that are ``None`` are omitted)."""
        out: Dict[str, Any] = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            optional = f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
            if value is None and optional:
                continue
            out[f.name] = _unconvert(value)
        return out

    @property
    def raw(self) -> Dict[str, Any]:
        """The JSON object exactly as received, including fields this SDK version does not know."""
        return getattr(self, "_raw", None) or self.to_dict()

    @property
    def meta(self) -> Optional[ResponseMeta]:
        """Request id, status and headers of the HTTP response (top-level objects only)."""
        return getattr(self, "_meta", None)

    @property
    def request_id(self) -> Optional[str]:
        """The ``x-request-id`` of the response this object came from. Quote it in bug reports."""
        meta = self.meta
        return meta.request_id if meta else None

    def _with_meta(self: M, meta: ResponseMeta) -> M:
        object.__setattr__(self, "_meta", meta)
        return self


# ---------------------------------------------------------------------------- health & models


@dataclass
class ModelRef(BaseModel):
    id: str
    revision: str


@dataclass
class HealthStatus(BaseModel):
    ok: bool


@dataclass
class ReadinessStatus(BaseModel):
    """``/readyz``: ``ok`` and the served model when ready; ``ok=False`` (and no model) on 503."""

    ok: bool
    model: Optional[ModelRef] = None


@dataclass
class HuggingFaceRef(BaseModel):
    repo: str
    revision: str
    url: str


@dataclass
class TokenizerRef(BaseModel):
    repo: str
    revision: str


@dataclass
class ModelIdentity(BaseModel):
    architecture: str
    weights_sha256: str
    huggingface: HuggingFaceRef
    tokenizer: TokenizerRef
    license: str


@dataclass
class Representation(BaseModel):
    """Everything that must match for stored vectors to be comparable. ``compute_dtype`` is
    ``bfloat16`` on the hosted service and ``unknown`` for indexes created by the pilot."""

    id: str
    compute_dtype: str
    slot_schema_version: int
    stored_dtype: str
    routing_normalization: str
    value_normalization: str


@dataclass
class ModelDimensions(BaseModel):
    slots: int
    routing_dim: int
    value_dim: int
    attention_heads: int


@dataclass
class TokenLimits(BaseModel):
    query_max_tokens: int
    document_max_tokens: int


@dataclass
class TextFormats(BaseModel):
    query_format: str
    document_format: str
    default_instruction: str
    truncation_side: str


@dataclass
class ScoringParameters(BaseModel):
    routing_aggregation: str
    tau_k: float
    tau_v: float
    routing_weight: float
    value_weight: float


@dataclass
class SearchProfile(BaseModel):
    """A named search pipeline an index pins for life. ``slot_hits``/``shortlist`` are
    ``None`` for the legacy ``pilot_maxsim`` profile."""

    name: str
    description: str
    candidate_generation: str
    slot_hits: Optional[int]
    shortlist: Optional[int]
    default_candidate_k: int
    max_candidate_k: int
    scoring: ScoringParameters
    tie_policy: str
    filter_operations: List[str]


@dataclass
class ModelCapabilities(BaseModel):
    input_types: List[str]
    encode_returns: List[str]
    encoding_formats: List[str]
    dtypes: List[str]
    attention_heads: List[str]
    explain: bool
    return_vectors: bool
    offset_unit: str


@dataclass
class ServiceLimits(BaseModel):
    max_request_bytes: int
    max_inputs_per_encode: int
    max_documents_per_score: int
    max_documents_per_upsert: int
    max_query_bytes: int
    max_document_bytes: int
    max_instruction_bytes: int
    max_doc_id_length: int
    max_metadata_keys: int
    max_metadata_bytes: int
    max_top_k: int
    max_candidate_k: int
    max_attention_values_per_request: int
    max_chunks_per_upsert: int
    max_chunks_per_index: Optional[int]
    """Chunks one index may hold, counted after replacements; ``None``: no limit."""
    max_page_size: int


@dataclass
class Model(BaseModel):
    id: str
    name: str
    revision: str
    aliases: List[str]
    status: str
    default: bool
    identity: ModelIdentity
    representation: Representation
    dimensions: ModelDimensions
    token_limits: TokenLimits
    text_formats: TextFormats
    scoring_defaults: ScoringParameters
    default_search_profile: str
    search_profiles: List[SearchProfile]
    capabilities: ModelCapabilities
    limits: ServiceLimits
    created_at: str

    def search_profile(self, name: Optional[str] = None) -> SearchProfile:
        """The named profile (default: ``default_search_profile``)."""
        wanted = name or self.default_search_profile
        for profile in self.search_profiles:
            if profile.name == wanted:
                return profile
        raise KeyError(f"model {self.id} has no search profile {wanted!r}")


# ---------------------------------------------------------------------------- shared


@dataclass
class Usage(BaseModel):
    input_tokens: int
    document_count: int
    chunk_count: int
    candidate_count: int
    latency_ms: int


@dataclass
class Token(BaseModel):
    id: int
    text: str
    segment: str
    start: Optional[int]
    end: Optional[int]

    @property
    def span(self) -> Optional[Tuple[int, int]]:
        """``(start, end)`` in code points of the caller's text, or ``None`` for template/special."""
        if self.start is None or self.end is None:
            return None
        return (self.start, self.end)


@dataclass
class TextEncoding(BaseModel):
    """How one text was encoded; ``tokens``/``attention`` only when requested."""

    token_count: int
    truncated: bool
    tokens: Optional[List[Token]] = None
    attention: Optional[Tensor] = None


@dataclass
class EncodedText(BaseModel):
    """One ``encode`` output: a ``TextEncoding`` plus ``index`` and the requested slots."""

    index: int
    token_count: int
    truncated: bool
    tokens: Optional[List[Token]] = None
    attention: Optional[Tensor] = None
    routing: Optional[Tensor] = None
    value: Optional[Tensor] = None


def _mean(values: List[float]) -> float:
    return math.fsum(values) / len(values) if values else float("nan")


@dataclass
class Explanation(BaseModel):
    """The four tensors a pair score is computed from (see the contract's ``Explanation``)."""

    slot_similarity: Tensor
    routing_contribution: Tensor
    value_attention: Tensor
    value_slot_cosine: Tensor

    def routing_score(self) -> float:
        """``mean(routing_contribution)``; equals the result's ``routing_score`` up to rounding."""
        return _mean(self.routing_contribution.flat())

    def value_score(self) -> float:
        """``mean(value_slot_cosine)``; equals the result's ``value_score`` up to rounding."""
        return _mean(self.value_slot_cosine.flat())

    def score(self, scoring: ScoringParameters) -> float:
        return scoring.routing_weight * self.routing_score() + scoring.value_weight * self.value_score()


@dataclass
class SlotVectors(BaseModel):
    routing: Tensor
    value: Tensor


# ---------------------------------------------------------------------------- encode & score


@dataclass
class EncodeResponse(BaseModel):
    model: ModelRef
    input_type: str
    data: List[EncodedText]
    usage: Usage
    instruction: Optional[str] = None


@dataclass
class ScoreResult(BaseModel):
    index: int
    doc_id: str
    score: float
    routing_score: float
    value_score: float
    metadata: Metadata
    document: TextEncoding
    explanation: Optional[Explanation] = None


@dataclass
class ScoreResponse(BaseModel):
    model: ModelRef
    scoring: ScoringParameters
    query: TextEncoding
    results: List[ScoreResult]
    usage: Usage

    def attribution(
        self,
        result: Union[int, ScoreResult],
        *,
        query_text: Optional[str] = None,
        document_text: Optional[str] = None,
        coupling: str = "score",
    ) -> Any:
        """Per-token attribution of one result's score; see :func:`breadbowl.token_attribution`.

        ``result`` is a result object or its position in ``results``. Requires a request made
        with ``explain=True`` and ``return_attention=True``.
        """
        from ._helpers import token_attribution

        res = self.results[result] if isinstance(result, int) else result
        if res.explanation is None:
            raise ValueError("result has no explanation; call score(..., explain=True)")
        return token_attribution(
            res.explanation,
            query=self.query,
            document=res.document,
            scoring=self.scoring,
            query_text=query_text,
            document_text=document_text,
            coupling=coupling,
        )


# ---------------------------------------------------------------------------- indexes & documents


@dataclass
class ResolvedChunkingConfig(BaseModel):
    strategy: str
    max_tokens: int
    overlap_tokens: int
    minimum_chunk_tokens: int


@dataclass
class Index(BaseModel):
    id: str
    name: str
    model: ModelRef
    representation: Representation
    search_profile: SearchProfile
    text_retention: str
    chunking: ResolvedChunkingConfig
    metadata_schema: Dict[str, str]
    status: str
    document_count: int
    chunk_count: int
    created_at: str


@dataclass
class DeleteIndexResponse(BaseModel):
    index_id: str
    deleted: bool
    documents_deleted: int


@dataclass
class DocumentError(BaseModel):
    code: str
    message: str
    retryable: bool


@dataclass
class DocumentOutcome(BaseModel):
    doc_id: str
    status: str
    content_sha256: Optional[str] = None
    token_count: Optional[int] = None
    chunk_count: Optional[int] = None
    truncated: Optional[bool] = None
    error: Optional[DocumentError] = None


@dataclass
class UpsertDocumentsResponse(BaseModel):
    index_id: str
    documents: List[DocumentOutcome]
    upserted_documents: int
    upserted_chunks: int
    failed_documents: int
    usage: Usage

    @property
    def failed(self) -> List[DocumentOutcome]:
        """Outcomes with ``status == "failed"``. A 200 response can still contain failures."""
        return [d for d in self.documents if d.status == "failed"]

    @property
    def retryable(self) -> List[DocumentOutcome]:
        """Failed outcomes whose error says resending the same document may succeed."""
        return [d for d in self.failed if d.error is not None and d.error.retryable]


@dataclass
class StoredChunk(BaseModel):
    chunk_id: str
    chunk_index: int
    token_start: int
    token_end: int
    token_count: int
    char_start: Optional[int]
    char_end: Optional[int]
    vectors: Optional[SlotVectors] = None

    def text(self, document_text: str) -> str:
        """This chunk's slice of your own copy of the document text (offsets are code points).

        Raises ``ValueError`` for chunks stored by the pilot service, which have no offsets."""
        if self.char_start is None or self.char_end is None:
            raise ValueError(f"chunk {self.chunk_id} has no character offsets (stored before v1.3)")
        return document_text[self.char_start : self.char_end]


@dataclass
class StoredDocument(BaseModel):
    index_id: str
    doc_id: str
    status: str
    content_sha256: str
    metadata: Metadata
    token_count: int
    chunk_count: int
    chunks: List[StoredChunk]
    created_at: str
    updated_at: str
    error: Optional[str] = None


@dataclass
class DeleteDocumentResponse(BaseModel):
    index_id: str
    doc_id: str
    deleted: bool


# ---------------------------------------------------------------------------- search


@dataclass
class SearchResult(BaseModel):
    doc_id: str
    chunk_id: str
    chunk_index: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    score: float
    routing_score: float
    value_score: float
    metadata: Metadata
    explanation: Optional[Explanation] = None
    vectors: Optional[SlotVectors] = None

    def text(self, document_text: str) -> str:
        """The hit's slice of your own copy of the document (offsets are code points)."""
        return document_text[self.char_start : self.char_end]


@dataclass
class SearchResponse(BaseModel):
    index_id: str
    model: ModelRef
    search_profile: SearchProfile
    scoring: ScoringParameters
    top_k: int
    candidate_k: int
    query: TextEncoding
    results: List[SearchResult]
    usage: Usage

    def attribution(
        self,
        result: Union[int, SearchResult],
        *,
        query_text: Optional[str] = None,
        coupling: str = "score",
    ) -> Any:
        """Query-side token attribution of one hit, using this response's ``scoring``. Document
        text is not stored, so there is no document side (document-slot weights only).
        Requires ``explain=True``, and ``return_attention=True`` for query tokens."""
        from ._helpers import token_attribution

        res = self.results[result] if isinstance(result, int) else result
        if res.explanation is None:
            raise ValueError("result has no explanation; search with explain=True")
        return token_attribution(
            res.explanation,
            query=self.query,
            document=None,
            scoring=self.scoring,
            query_text=query_text,
            coupling=coupling,
        )


# ---------------------------------------------------------------------------- usage


@dataclass
class UsageCounters(BaseModel):
    """Current-month counters. Totals are decimal strings (``int(x)`` is exact in Python)."""

    input_tokens: str
    requests: str
    encodes: str
    scores: str
    searches: str
    document_upserts: str
    documents: int

    def as_ints(self) -> Dict[str, int]:
        return {f.name: int(getattr(self, f.name)) for f in dataclasses.fields(self)}


@dataclass
class TenantLimits(BaseModel):
    input_tokens: str
    documents: int
    requests_per_minute: int
    max_concurrent_requests: int


@dataclass
class RemainingAllowance(BaseModel):
    input_tokens: str
    documents: int


@dataclass
class KeyRateLimit(BaseModel):
    requests_per_minute: Optional[int] = None
    max_concurrent_requests: Optional[int] = None


@dataclass
class CallingApiKey(BaseModel):
    id: str
    key_prefix: str
    scopes: List[str]
    expires_at: Optional[str]
    rate_limit: Optional[KeyRateLimit]
    name: Optional[str] = None


@dataclass
class UsageSummary(BaseModel):
    period_start: str
    period_end: str
    usage: UsageCounters
    limits: TenantLimits
    remaining: RemainingAllowance
    api_key: CallingApiKey

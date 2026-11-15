"""The SDK agrees with the OpenAPI contract.

* The fixtures and the generated module were produced from the current contract.
* Every response example validates against its schema, parses into the SDK model and
  serializes back unchanged; every model field is a schema property and vice versa, with the
  same required/optional split.
* Every request example in the contract is what the SDK sends for the equivalent call, and
  SDK-built requests with every option set validate against the request schemas (which use
  ``additionalProperties: false``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Type

import pytest

yaml = pytest.importorskip("yaml")
jsonschema = pytest.importorskip("jsonschema")
referencing = pytest.importorskip("referencing")

from referencing.jsonschema import DRAFT202012  # noqa: E402

import breadbowl  # noqa: E402
from breadbowl import _generated, _models  # noqa: E402
from breadbowl._models import BaseModel  # noqa: E402
from tests.conftest import FIXTURES, SIBLING_FIXTURE, SPEC_PATH, MockAPI  # noqa: E402

if not SPEC_PATH.exists():  # pragma: no cover - e.g. running from an sdist
    pytest.skip("contract not available outside the repository", allow_module_level=True)

SPEC_BYTES = SPEC_PATH.read_bytes()
SPEC: Dict[str, Any] = yaml.safe_load(SPEC_BYTES)
SPEC_URI = "urn:bb-embed:v1"
REGISTRY = referencing.Registry().with_resource(SPEC_URI, DRAFT202012.create_resource(SPEC))
INDEX_ID = "0b8a3d0e-5c1f-4f7e-9a51-3e2d6c4b1a90"


def validator(pointer: str) -> Any:
    return jsonschema.Draft202012Validator({"$ref": f"{SPEC_URI}#{pointer}"}, registry=REGISTRY)


def schema_validator(name: str) -> Any:
    return validator(f"/components/schemas/{name}")


def drop_nones(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: drop_nones(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [drop_nones(v) for v in value]
    return value


def test_validation_is_not_vacuous() -> None:
    encode = schema_validator("EncodeRequest")
    assert encode.is_valid({"input": "x", "input_type": "query"})
    assert not encode.is_valid({"input": "x", "input_type": "query", "return_components": True})
    assert not encode.is_valid({"input": "x", "input_type": "passage"})
    body = json.loads(
        json.dumps(FIXTURES["operations"]["getUsage"]["responses"]["200"]["examples"]["default"])
    )
    body["usage"]["input_tokens"] = 1834221  # counters are strings
    assert not schema_validator("UsageSummary").is_valid(body)


# ---------------------------------------------------------------------------- freshness


def test_fixtures_were_generated_from_this_contract() -> None:
    assert FIXTURES["spec"]["sha256"] == hashlib.sha256(SPEC_BYTES).hexdigest(), (
        "fixtures are stale: run `node scripts/generate-fixtures.mjs` in sdks/typescript"
    )


def test_generated_module_is_from_this_contract() -> None:
    assert hashlib.sha256(SPEC_BYTES).hexdigest() == _generated.SPEC_SHA256, (
        "src/breadbowl/_generated.py is stale: run `python scripts/generate_types.py`"
    )
    assert SPEC["info"]["version"] == breadbowl.SPEC_VERSION


def test_both_sdks_share_identical_fixtures() -> None:
    if not SIBLING_FIXTURE.exists():
        pytest.skip("TypeScript package not present")
    here = (SPEC_PATH.parents[2] / "sdks/python/tests/fixtures/contract.json").read_bytes()
    assert SIBLING_FIXTURE.read_bytes() == here


def test_operation_table_matches_fixture_generator() -> None:
    ops = FIXTURES["operations"]
    assert set(ops) == set(_generated.OPERATIONS)
    for op_id, op in _generated.OPERATIONS.items():
        f = ops[op_id]
        assert (op.method, op.path, op.idempotency_key, op.authenticated) == (
            f["method"],
            f["path"],
            f["idempotency_key"],
            f["authenticated"],
        )


# ---------------------------------------------------------------------------- errors


def test_operation_error_codes_are_the_specs_x_error_codes() -> None:
    for op_id, op in _generated.OPERATIONS.items():
        spec_op = SPEC["paths"][op.path][op.method.lower()]
        expected = {str(k): tuple(v) for k, v in (spec_op.get("x-error-codes") or {}).items()}
        assert op.error_codes == expected, op_id
        assert {k: list(v) for k, v in op.error_codes.items()} == FIXTURES["operations"][op_id]["error_codes"]
        for status, codes in op.error_codes.items():
            assert status in spec_op["responses"], (op_id, status)
            assert set(codes) <= set(_generated.ERROR_CODES[int(status)])


def _sample(schema: Dict[str, Any]) -> Any:
    s = resolve(schema["$ref"].lstrip("#")) if "$ref" in schema else schema
    if "oneOf" in s:
        return _sample(s["oneOf"][0])
    if "enum" in s:
        return s["enum"][0]
    if s.get("examples"):
        return s["examples"][0]
    if s.get("pattern", "").startswith("^sha256:"):
        return "sha256:" + "0" * 64
    if s.get("type") == "string":
        return "x"
    if s.get("type") in ("integer", "number"):
        return 3
    if s.get("type") == "array":
        return [_sample(s["items"])]
    if s.get("type") == "object":
        return {k: _sample(v) for k, v in (s.get("properties") or {}).items()}
    raise AssertionError(s)


def test_x_error_details_have_typed_accessors() -> None:
    from breadbowl._errors import error_class_for

    details_map = SPEC["components"]["schemas"]["Error"]["x-error-details"]
    assert set(details_map) == set(_generated.ERROR_DETAILS) == set(FIXTURES["error_details"])
    status_of = {code: int(st) for st, codes in _generated.ERROR_CODES.items() for code in codes}
    for code, schema_name in details_map.items():
        typed = _generated.ERROR_DETAILS[code]
        assert typed.__name__ == schema_name
        props, required = flatten(resolve(f"/components/schemas/{schema_name}"))
        assert set(typed.__annotations__) == set(props), schema_name
        assert set(typed.__required_keys__) == required, schema_name
        details = _sample({"$ref": f"#/components/schemas/{schema_name}"})
        schema_validator(schema_name).validate(details)
        cls = error_class_for(status_of[code], code)
        err = cls("m", status_code=status_of[code], code=code, request_id=None, details=details)
        assert err.details_for(code) == details
        for field, value in details.items():
            name = "index_status" if field == "status" else field
            assert getattr(err, name) == value, (cls.__name__, name)
    assert (
        breadbowl.APIStatusError("m", status_code=422, code="validation_error", request_id=None).details_for(
            "validation_error"
        )
        is None
    )


# ---------------------------------------------------------------------------- response models

# (model class, JSON pointer of its schema). Inline objects are addressed by their path.
MODEL_SCHEMAS: List[Tuple[Type[BaseModel], str]] = [
    (_models.HealthStatus, "/components/schemas/HealthStatus"),
    (_models.ModelRef, "/components/schemas/ModelRef"),
    (_models.Model, "/components/schemas/Model"),
    (_models.ModelIdentity, "/components/schemas/ModelIdentity"),
    (_models.HuggingFaceRef, "/components/schemas/ModelIdentity/properties/huggingface"),
    (_models.TokenizerRef, "/components/schemas/ModelIdentity/properties/tokenizer"),
    (_models.Representation, "/components/schemas/Representation"),
    (_models.ModelDimensions, "/components/schemas/ModelDimensions"),
    (_models.TokenLimits, "/components/schemas/TokenLimits"),
    (_models.TextFormats, "/components/schemas/TextFormats"),
    (_models.ScoringParameters, "/components/schemas/ScoringParameters"),
    (_models.SearchProfile, "/components/schemas/SearchProfile"),
    (_models.ModelCapabilities, "/components/schemas/ModelCapabilities"),
    (_models.ServiceLimits, "/components/schemas/ServiceLimits"),
    (_models.Usage, "/components/schemas/Usage"),
    (_models.Token, "/components/schemas/Token"),
    (_models.TextEncoding, "/components/schemas/TextEncoding"),
    (_models.Explanation, "/components/schemas/Explanation"),
    (_models.SlotVectors, "/components/schemas/SlotVectors"),
    (_models.EncodeResponse, "/components/schemas/EncodeResponse"),
    (_models.ScoreResult, "/components/schemas/ScoreResult"),
    (_models.ScoreResponse, "/components/schemas/ScoreResponse"),
    (_models.ResolvedChunkingConfig, "/components/schemas/ResolvedChunkingConfig"),
    (_models.Index, "/components/schemas/Index"),
    (_models.DeleteIndexResponse, "/components/schemas/DeleteIndexResponse"),
    (_models.DocumentError, "/components/schemas/DocumentError"),
    (_models.DocumentOutcome, "/components/schemas/DocumentOutcome"),
    (_models.UpsertDocumentsResponse, "/components/schemas/UpsertDocumentsResponse"),
    (_models.StoredChunk, "/components/schemas/StoredChunk"),
    (_models.StoredDocument, "/components/schemas/StoredDocument"),
    (_models.DeleteDocumentResponse, "/components/schemas/DeleteDocumentResponse"),
    (_models.SearchResult, "/components/schemas/SearchResult"),
    (_models.SearchResponse, "/components/schemas/SearchResponse"),
    (_models.UsageSummary, "/components/schemas/UsageSummary"),
    (_models.UsageCounters, "/components/schemas/UsageSummary/properties/usage"),
    (_models.TenantLimits, "/components/schemas/UsageSummary/properties/limits"),
    (_models.RemainingAllowance, "/components/schemas/UsageSummary/properties/remaining"),
    (_models.CallingApiKey, "/components/schemas/CallingApiKey"),
    (_models.KeyRateLimit, "/components/schemas/CallingApiKey/properties/rate_limit"),
]


def resolve(pointer: str) -> Dict[str, Any]:
    node: Any = SPEC
    for part in pointer.strip("/").split("/"):
        node = node[part]
    while "$ref" in node:
        node = resolve(node["$ref"].lstrip("#"))
    return dict(node)


def flatten(schema: Dict[str, Any]) -> Tuple[Dict[str, Any], set]:
    """Properties and required names, merging allOf parts."""
    props: Dict[str, Any] = dict(schema.get("properties") or {})
    required = set(schema.get("required") or [])
    for part in schema.get("allOf") or []:
        sub = resolve(part["$ref"].lstrip("#")) if "$ref" in part else part
        p, r = flatten(sub)
        props.update(p)
        required |= r
    return props, required


def test_every_response_schema_has_a_model() -> None:
    covered = {p.split("/")[3] for _, p in MODEL_SCHEMAS}
    response_schemas = {
        r["schema"].split("/")[-1]
        for op in FIXTURES["operations"].values()
        for status, r in op["responses"].items()
        if r["schema"] and int(status) < 300
    }
    # Lists are SyncPage/AsyncPage; ReadinessStatus is checked separately (it merges NotReadyStatus).
    assert response_schemas - covered == {"ModelList", "IndexList", "ReadinessStatus"}


@pytest.mark.parametrize("cls, pointer", MODEL_SCHEMAS, ids=[c.__name__ for c, _ in MODEL_SCHEMAS])
def test_model_fields_match_schema(cls: Type[BaseModel], pointer: str) -> None:
    props, required = flatten(resolve(pointer))
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    assert set(fields) == set(props), f"{cls.__name__}: fields differ from {pointer}"
    for name, f in fields.items():
        has_default = f.default is not dataclasses.MISSING
        assert has_default == (name not in required), f"{cls.__name__}.{name}: required-ness differs"


def test_readiness_model_covers_both_bodies() -> None:
    names = {f.name for f in dataclasses.fields(_models.ReadinessStatus)}
    props, _ = flatten(resolve("/components/schemas/ReadinessStatus"))
    not_ready, _ = flatten(resolve("/components/schemas/NotReadyStatus"))
    assert names == set(props) | set(not_ready)


def response_examples() -> List[Tuple[str, str, str, Any]]:
    out = []
    for op_id, op in FIXTURES["operations"].items():
        for status, r in op["responses"].items():
            for name, body in r["examples"].items():
                out.append((op_id, status, name, body))
    return out


@pytest.mark.parametrize("op_id, status, name, body", response_examples(), ids=lambda v: str(v)[:24])
def test_response_examples_validate_against_their_schema(
    op_id: str, status: str, name: str, body: Any
) -> None:
    schema = FIXTURES["operations"][op_id]["responses"][status]["schema"]
    validator(schema.lstrip("#")).validate(body)


def test_expanded_examples_still_validate() -> None:
    for key, body in FIXTURES["expanded"].items():
        op_id, status, _ = key.split("/")
        schema = FIXTURES["operations"][op_id]["responses"][status]["schema"]
        validator(schema.lstrip("#")).validate(body)


PARSERS: Dict[str, Type[BaseModel]] = {
    "getHealth": _models.HealthStatus,
    "getReadiness": _models.ReadinessStatus,
    "getModel": _models.Model,
    "encode": _models.EncodeResponse,
    "score": _models.ScoreResponse,
    "createIndex": _models.Index,
    "getIndex": _models.Index,
    "deleteIndex": _models.DeleteIndexResponse,
    "upsertDocuments": _models.UpsertDocumentsResponse,
    "getDocument": _models.StoredDocument,
    "deleteDocument": _models.DeleteDocumentResponse,
    "searchIndex": _models.SearchResponse,
    "getUsage": _models.UsageSummary,
}
LIST_ITEMS = {"listModels": _models.Model, "listIndexes": _models.Index}


@pytest.mark.parametrize("op_id, status, name, body", response_examples(), ids=lambda v: str(v)[:24])
def test_response_examples_round_trip_through_models(op_id: str, status: str, name: str, body: Any) -> None:
    if int(status) >= 400 and op_id != "getReadiness":
        return  # error envelopes are covered by test_errors
    if op_id in LIST_ITEMS:
        items = [LIST_ITEMS[op_id].from_dict(x).to_dict() for x in body["data"]]
        assert drop_nones(items) == drop_nones(body["data"])
        return
    parsed = PARSERS[op_id].from_dict(body)
    assert drop_nones(parsed.to_dict()) == drop_nones(body)
    assert parsed.raw == body


# ---------------------------------------------------------------------------- requests


def _client(api: MockAPI) -> breadbowl.BreadBowl:
    import httpx

    return breadbowl.BreadBowl(
        "k", base_url="https://api.test", http_client=httpx.Client(transport=httpx.MockTransport(api.handler))
    )


def _sdk_call(client: breadbowl.BreadBowl, op_id: str, ex: Dict[str, Any]) -> None:
    ex = dict(ex)
    if op_id == "encode":
        text = ex.pop("input")
        if "return" in ex:
            ex["return_"] = ex.pop("return")
        client.encode(text, **ex)
    elif op_id == "score":
        client.score(ex.pop("query"), ex.pop("documents"), **ex)
    elif op_id == "createIndex":
        client.indexes.create(ex.pop("name"), **ex)
    elif op_id == "upsertDocuments":
        client.indexes.documents.upsert(INDEX_ID, ex["documents"])
    elif op_id == "searchIndex":
        client.indexes.search(INDEX_ID, ex.pop("query"), **ex)
    else:  # pragma: no cover
        raise AssertionError(op_id)


def request_examples() -> List[Tuple[str, str, Any]]:
    return [
        (op_id, name, ex)
        for op_id, op in FIXTURES["operations"].items()
        for name, ex in op["request_examples"].items()
    ]


@pytest.mark.parametrize("op_id, name, ex", request_examples(), ids=lambda v: str(v)[:24])
def test_contract_request_examples_are_what_the_sdk_sends(op_id: str, name: str, ex: Dict[str, Any]) -> None:
    api = MockAPI()
    status = "201" if op_id == "createIndex" else "200"
    api.add(next(iter(FIXTURES["operations"][op_id]["responses"][status]["examples"].values())), int(status))
    _sdk_call(_client(api), op_id, ex)
    assert api.last.body == ex
    validator(FIXTURES["operations"][op_id]["request_schema"].lstrip("#")).validate(api.last.body)


def _all_options_calls(client: breadbowl.BreadBowl) -> None:
    client.encode(
        ["a", "b"],
        input_type="document",
        model="m",
        instruction="Find it.",
        return_=["routing", "value", "attention", "tokens"],
        encoding_format="float",
        dtype="float32",
        attention_heads="all",
    )
    client.score(
        "q",
        [breadbowl.Document("a", "t", {"k": 1, "flag": True, "s": "x"}), {"doc_id": "b", "text": "u"}],
        model="m",
        instruction="Find it.",
        explain=True,
        return_attention=True,
        attention_heads="mean",
        encoding_format="base64",
        dtype="float16",
    )
    client.indexes.create(
        "n",
        model="m",
        search_profile="p",
        text_retention="none",
        chunking={
            "strategy": "token_window",
            "max_tokens": 128,
            "overlap_tokens": 16,
            "minimum_chunk_tokens": 8,
        },
        metadata_schema={"a": "string", "b": "number", "c": "boolean"},
    )
    client.indexes.documents.upsert(INDEX_ID, [breadbowl.Document("d", "t", {"year": 2026})])
    client.indexes.search(
        INDEX_ID,
        "q",
        top_k=3,
        candidate_k=50,
        filter={"year": 2026},
        instruction="i",
        explain=True,
        return_attention=True,
        return_vectors=True,
        attention_heads="mean",
        encoding_format="float",
        dtype="float32",
    )


def test_sdk_requests_with_every_option_validate() -> None:
    api = MockAPI()
    for op_id, status in [
        ("encode", 200),
        ("score", 200),
        ("createIndex", 201),
        ("upsertDocuments", 200),
        ("searchIndex", 200),
    ]:
        api.add(
            next(iter(FIXTURES["operations"][op_id]["responses"][str(status)]["examples"].values())), status
        )
    _all_options_calls(_client(api))
    by_path = {
        "/v1/encode": "EncodeRequest",
        "/v1/score": "ScoreRequest",
        "/v1/indexes": "CreateIndexRequest",
        f"/v1/indexes/{INDEX_ID}/documents": "UpsertDocumentsRequest",
        f"/v1/indexes/{INDEX_ID}/search": "SearchRequest",
    }
    for req in api.requests:
        schema_validator(by_path[req.path]).validate(req.body)
    # Every optional property of every request schema was exercised at least once.
    for path, name in by_path.items():
        props, _ = flatten(resolve(f"/components/schemas/{name}"))
        sent = set().union(*(set(r.body) for r in api.requests if r.path == path))
        assert sent == set(props), (name, set(props) - sent)


def test_query_parameters_validate() -> None:
    api = MockAPI()
    api.add(FIXTURES["operations"]["getDocument"]["responses"]["200"]["examples"]["default"])
    api.add(FIXTURES["operations"]["listIndexes"]["responses"]["200"]["examples"]["default"])
    client = _client(api)
    client.indexes.documents.get(
        INDEX_ID, "doc-1", return_vectors=False, encoding_format="float", dtype="float16"
    )
    client.indexes.list(name="prod", limit=100, cursor="abc")
    params = {
        p["name"]: p["schema"]
        for path in ("/v1/indexes/{index_id}/documents/{doc_id}", "/v1/indexes")
        for method in ("get",)
        for p in [
            resolve(x["$ref"].lstrip("#")) if "$ref" in x else x
            for x in SPEC["paths"][path][method].get("parameters", [])
        ]
        if p["in"] == "query"
    }
    for req in api.requests:
        for name, value in req.params.items():
            schema = params[name]
            typed: Optional[Any] = value
            if (resolve(schema["$ref"].lstrip("#")) if "$ref" in schema else schema).get("type") == "boolean":
                typed = {"true": True, "false": False}[value]
            elif schema.get("type") == "integer":
                typed = int(value)
            jsonschema.Draft202012Validator(schema, registry=REGISTRY).validate(
                typed
            ) if "$ref" not in schema else validator(schema["$ref"].lstrip("#")).validate(typed)


def test_error_envelope_schema_matches_sdk_fields() -> None:
    props, required = flatten(resolve("/components/schemas/Error"))
    err = breadbowl.APIStatusError("m", status_code=400, code="bad_request", request_id="r")
    for name in props:
        assert hasattr(err, name), name
    assert required == {"code", "message"}
    json.dumps(FIXTURES["errors"])  # fixtures are plain JSON


def test_types_module_exports_every_generated_detail_and_request_type() -> None:
    import breadbowl.types as types

    for code, typed in _generated.ERROR_DETAILS.items():
        assert getattr(types, typed.__name__) is typed, code
    for name in (
        "EncodeRequest",
        "ScoreRequest",
        "CreateIndexRequest",
        "SearchRequest",
        "UpsertDocumentsRequest",
    ):
        assert getattr(types, name) is getattr(_generated, name)

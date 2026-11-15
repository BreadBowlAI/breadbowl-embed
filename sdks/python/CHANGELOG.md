# Changelog

## 0.1.0

First release, built against the bb-embed v1.3 public contract (`sdks/openapi/bb-embed.v1.yaml`,
OpenAPI `info.version` 1.3.0). 0.1.0 targets the v1.3 API only.

- Sync `BreadBowl` and async `AsyncBreadBowl` clients for all 15 public operations:
  - `health.live` and `health.ready`
  - `models.list` and `models.get`
  - `encode` and `score`
  - `indexes.create`, `indexes.list`, `indexes.get`, `indexes.delete` and `indexes.search`
  - `indexes.documents.upsert`, `indexes.documents.get` and `indexes.documents.delete`
  - `usage.get`
  - `client.index(id)`, a handle bound to one index
- Dataclass response models with `.request_id`, `.meta` and `.raw`.
- Request `TypedDict`s and enum `Literal`s generated from the contract.
- Exception classes per status and per error code.
- Bounded retries with jittered backoff that honour `Retry-After`, following the contract's
  guidance:
  - Retried: 429 except `usage_limit_exceeded`, 500/502/503/504, timeouts and connection
    failures, `idempotency_request_in_progress`, and `index_not_ready` while
    `details.status == "creating"`.
  - Only 2xx responses are billed, so failed attempts are never billed twice.
  - `Idempotency-Key` on every mutation, reused across retries.
- Typed error details (`x-error-details`): `details_for(code)`, typed properties on the
  specific exception classes, and `TypedDict`s in `breadbowl.types`.
- Cursor pagination: iterate a page for every item, or use `iter_pages()`. The async paginator is
  awaitable and async-iterable.
- `Tensor`: pure-Python decoding (float16 through `struct`), `to_numpy()` with the `numpy` extra,
  and shape and length checks.
- `token_attribution`, `token_spans` and `attention_matrix`.
- `ScoreResponse.attribution()` and `SearchResponse.attribution()`.
- Depends on httpx only. numpy is optional. Python 3.9+.
- Contract features used by the SDK:
  - `SearchResponse` carries the full `search_profile`, `scoring`, `top_k` and `candidate_k`.
  - Explanation tensors are always float32.
  - `next_cursor` is a string exactly when `has_more` is true; a violation raises.
  - Stored chunks may have null character offsets (pilot data).
- Shared fixtures are generated from the contract's examples. Examples marked `x-abbreviated`
  are expanded; any other example with inconsistent tensors fails generation.
- Apache-2.0 `LICENSE` file.
- `documents.upsert_batched` (and `IndexHandle.upsert_batched`), sync and async:
  - any number of documents, sent in batches of up to 50
  - a batch is split and retried on `chunk_limit_exceeded`
  - outcomes merged in input order
  - `BatchUpsertError` with the partial result and the pending `doc_id`s
- Upserts use `upsert_timeout` (default 180 s) because the hosted service encodes about 3 chunks/s.
- `RepresentationUnavailableError` (409, never retried) and `ChunkLimitExceededError` (422), with
  typed details.
- `Representation.compute_dtype` and `ServiceLimits.max_chunks_per_upsert`.
- Spans follow the contract's token rule: offsets of tokens that split a character overlap, so
  `token_spans`, `query_spans()` and `document_spans()` merge overlapping offsets (union range,
  summed weight, all positions). The spans never overlap and tile the text. `TokenSpan` now has
  `positions` and `tokens`.

# Changelog

## 0.1.0

First release, built against the bb-embed v1.3 public contract (`sdks/openapi/bb-embed.v1.yaml`,
OpenAPI `info.version` 1.3.0). 0.1.0 targets the v1.3 API only.

- `BreadBowl` client for all 15 public operations:
  - `health.live` and `health.ready`
  - `models.list` and `models.get`
  - `encode` and `score`
  - `indexes.create`, `indexes.list`, `indexes.get`, `indexes.delete` and `indexes.search`
  - `indexes.documents.upsert`, `indexes.documents.get` and `indexes.documents.delete`
  - `usage.get`
  - `index(id)`, a handle bound to one index
- Types generated with openapi-typescript; the client's parameters and results are those types.
- Zero runtime dependencies. Uses `fetch`, with an injectable `fetch`. ESM and CJS builds with
  declarations. Node 18+, Bun, Deno, browsers.
- Error classes per status and per error code, with `requestId`, `details`, `issues` and
  `retryAfter`.
- Bounded retries with jittered backoff that honour `Retry-After`, following the contract's
  guidance:
  - Retried: 429 except `usage_limit_exceeded`, 500/502/503/504, timeouts and network errors,
    `idempotency_request_in_progress`, and `index_not_ready` while `details.status` is
    `creating`.
  - Only 2xx responses are billed, so failed attempts are never billed twice.
  - `Idempotency-Key` on every mutation, reused across retries.
  - Per-attempt timeouts, and caller aborts.
- Typed error details (`x-error-details`, generated as `ErrorDetailsByCode`): `detailsFor(code)`
  and typed getters on the specific error classes.
- Cursor pagination: `await` for a page, `for await` for every item.
- `decodeTensor`: float16 decoded by hand, float32, the nested-float encoding, shape checks and
  indexing helpers. `encodeTensor` builds tensors.
- `CodePointIndex`, `tokenSpans` and `sliceCodePoints` for code-point offsets.
- `parseCounter` and `parseCounters` for decimal-string counters.
- `tokenAttribution`, `explanationScores` and `attentionMatrix` for explaining scores.
- The client refuses an API key in a browser page unless `dangerouslyAllowBrowser` is set.
- Contract features used by the SDK:
  - `SearchResponse` carries the full `search_profile`, `scoring`, `top_k` and `candidate_k`.
  - Explanation tensors are always float32.
  - `next_cursor` is a string exactly when `has_more` is true; a violation raises.
  - Stored chunks may have null character offsets (pilot data).
- Shared fixtures are generated from the contract's examples. Examples marked `x-abbreviated`
  are expanded; any other example with inconsistent tensors fails generation.
- Apache-2.0 `LICENSE` file.
- `documents.upsertBatched` (and `IndexHandle.upsertBatched`):
  - any number of documents, sent in batches of up to 50
  - a batch is split and retried on `chunk_limit_exceeded`
  - outcomes merged in input order
  - `BatchUpsertError` with the partial result and the pending `doc_id`s
- Upserts use `upsertTimeoutMs` (default 180 s) because the hosted service encodes about 3 chunks/s.
- `RepresentationUnavailableError` (409, never retried) and `ChunkLimitExceededError` (422), with
  typed details. `Representation.compute_dtype` and `ServiceLimits.max_chunks_per_upsert` come
  from the generated types.
- `IndexChunkLimitExceededError` (422 `index_chunk_limit_exceeded`, an index would hold more than
  `limits.max_chunks_per_index` chunks) with `limit`, `current`, `requested` and `replaced`;
  `upsertBatched` does not split or retry it. `ServiceLimits.max_chunks_per_index` (`null`: no
  limit) comes from the generated types.
- Spans follow the contract's token rule: offsets of tokens that split a character overlap, so
  `tokenSpans`, `querySpans()` and `documentSpans()` merge overlapping offsets (union range,
  summed weight, all positions). The spans never overlap and tile the text. `tokenSpans` returns
  merged spans (`positions`, `tokens`), and `mergeOffsetRanges` is exported.

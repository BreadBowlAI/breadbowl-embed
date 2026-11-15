# @breadbowl/embed

TypeScript client for the BreadBowl Embed API (`bb-embed` v1.3): encoding, scoring, managed
indexes and search, including the model's raw data:

- routing and value slots
- slot→token attention
- tokens with offsets
- the tensors every score is computed from

The client has no runtime dependencies and uses `fetch`. It runs on Node 18+, Bun, Deno and
modern browsers, and ships as ESM and CJS with type declarations. Its types are generated from the
OpenAPI contract.

This package is the HTTP client for the API. The model itself (to run it on your own machine) is
the Python library `breadbowl-embed` at the root of the same repository. The source of this SDK
is [`sdks/typescript`](https://github.com/BreadBowlAI/breadbowl-embed/tree/main/sdks/typescript)
in [BreadBowlAI/breadbowl-embed](https://github.com/BreadBowlAI/breadbowl-embed).

```bash
npm install @breadbowl/embed
```

Until the package is published on npm, build it from the repository and install the tarball
(needs Node.js and pnpm):

```bash
git clone https://github.com/BreadBowlAI/breadbowl-embed.git
cd breadbowl-embed/sdks/typescript
pnpm install && pnpm run build && npm pack     # writes breadbowl-embed-0.1.0.tgz
# then, in your project:
npm install /path/to/breadbowl-embed/sdks/typescript/breadbowl-embed-0.1.0.tgz
```

**Version 0.1.0 targets the v1.3 API**
([`sdks/openapi/bb-embed.v1.yaml`](https://github.com/BreadBowlAI/breadbowl-embed/blob/main/sdks/openapi/bb-embed.v1.yaml),
`info.version` 1.3.0).
It does not work with the older pilot API that some deployments still serve, so set `baseURL` to a
v1.3 deployment.

> [!WARNING]
> **Never put an account API key in a browser bundle.** Anyone who opens the page can read it and
> spend your quota. Call the API from your server, or from a server-side proxy that adds the key.
> The client refuses an API key when it detects a browser page. It never reads one from the
> environment there, and there is no default key. `dangerouslyAllowBrowser: true` overrides this;
> use it only for a key that is safe to expose.

## Quickstart

```ts
import { BreadBowl } from '@breadbowl/embed';

const client = new BreadBowl(); // BREADBOWL_API_KEY (Node, Bun, Deno); or { apiKey: '...' }

const res = await client.score({
  query: 'refund policy',
  documents: [
    { doc_id: 'a', text: 'Refunds are available within 30 days.' },
    { doc_id: 'b', text: 'Keys can be rotated from the dashboard.' },
  ],
});
for (const r of res.results) console.log(r.doc_id, r.score, r.routing_score, r.value_score);
console.log(res.requestId); // quote this when you report a problem
```

A plain string document gets its position as `doc_id`: `documents: ['text a', 'text b']`.

Deno: `import { BreadBowl } from 'npm:@breadbowl/embed';`

Self-hosted: `new BreadBowl({ apiKey, baseURL: 'http://localhost:8080' })` or set
`BREADBOWL_BASE_URL`. The default is the contract's server URL (`DEFAULT_BASE_URL`).

In a browser, send requests to your own proxy: `new BreadBowl({ apiKey: null, baseURL: '/api/bb' })`.
The proxy adds the key on the server.

Request and response fields use the wire names (`input_type`, `top_k`, ...), and the types are
the generated contract types (`import type { SearchResponse, components } from '@breadbowl/embed'`).

## Encode and tensors

```ts
import { decodeTensor } from '@breadbowl/embed';

const out = await client.encode({
  input: ['Keys can be rotated from the dashboard.'],
  input_type: 'document',
  return: ['routing', 'value', 'attention', 'tokens'],
  attention_heads: 'mean',
});
const routing = decodeTensor(out.data[0]!.routing!); // Float32Array + shape + axes
routing.shape;          // [16, 256]
routing.get(0, 3);      // one value
routing.slice(0);       // slot 0 as a Float32Array view
routing.toNested();     // number[][]
```

`decodeTensor` decodes little-endian float16 by hand; it does not rely on `Float16Array`. It
handles float32 and the `float` (nested JSON) encoding too. Values from `float` are rounded to
`dtype`, so both encodings decode to the same numbers.

Decoding checks the byte length against `shape × itemsize` and the nesting against `shape`; a
mismatch throws `TensorDecodeError`.

## Indexes and search

```ts
const index = await client.indexes.create({ name: 'support-articles', metadata_schema: { source: 'string' } });

const up = await client.indexes.documents.upsert(index.id, [
  { doc_id: 'doc-1', text: 'Keys can be rotated from the dashboard.', metadata: { source: 'docs' } },
  { doc_id: 'doc-2', text: 'Refunds are available within 30 days.' },
]);
// A 200 can still contain per-document failures:
for (const d of up.documents) if (d.status === 'failed') console.log(d.doc_id, d.error?.code, d.error?.retryable);

const hits = await client.indexes.search(index.id, { query: 'How do I rotate a key?', top_k: 5, filter: { source: 'docs' } });

const idx = client.index(index.id); // bound handle
await idx.search('rotate key');
await idx.getDocument('doc-1', { return_vectors: true });
```

### Many documents

One request takes at most 50 documents and `limits.max_chunks_per_upsert` (128) chunks.
`upsertBatched` takes any number of documents:

- It sends them in batches of `batchSize` (default 50), each with its own `Idempotency-Key`.
- A batch rejected with `chunk_limit_exceeded` is split into `ceil(requested / limit)` parts and
  retried, because chunk counts are only known after the server tokenizes.
- A single document over the limit becomes a failed outcome with code `chunk_limit_exceeded`.
- The outcomes come back merged, in input order.

```ts
import { BatchUpsertError } from '@breadbowl/embed';

try {
  const res = await client.indexes.documents.upsertBatched(index.id, myDocuments); // or idx.upsertBatched(...)
  console.log(res.upsertedDocuments, res.upsertedChunks, res.requestIds, res.failed);
} catch (err) {
  if (err instanceof BatchUpsertError) console.log(err.error, err.result.upsertedDocuments, err.pending);
  else throw err;
}
```

Upserts are slow: about 2.4 to 3.5 chunks per second on the hosted CPU, so a request at the chunk
limit takes 40 to 60 s, and longer while other upserts are queued. Upserts use `upsertTimeoutMs`
(default 180 000 ms; the contract recommends at least 120 s). All other calls use `timeoutMs`
(60 000 ms). For interactive use, send a few documents at a time.

## Pagination

```ts
const page = await client.indexes.list({ limit: 50 }); // first page
page.data; page.hasMore; page.nextCursor; page.requestId;
await page.nextPage();

for await (const index of client.indexes.list()) console.log(index.name); // every page
```

The contract guarantees that `next_cursor` is a string exactly when `has_more` is true. A server
that breaks this makes `nextPage()` throw `APIResponseError` instead of stopping silently.

## Offsets are code points

Token `start`/`end` and chunk `char_start`/`char_end` count Unicode code points. JavaScript
strings index UTF-16 code units, so `text.slice(start, end)` is wrong once the text contains an
emoji or another character outside the BMP. Use the helpers:

```ts
import { CodePointIndex, tokenSpans, sliceCodePoints } from '@breadbowl/embed';

const idx = new CodePointIndex(text);       // build once per text
idx.slice(hit.char_start, hit.char_end);    // the hit's chunk of your own copy of the document
tokenSpans(text, tokens);                   // merged: [{ start, end, utf16Start, utf16End, text, positions, tokens }]
```

### Tokens, offsets and spans

A character can be split across byte-level tokens; an emoji is 4 UTF-8 bytes, for example.

- The token that ends inside the character shows only what precedes it (possibly `''`).
- The token that completes the character shows the whole character.
- Both report the whole character's offsets, so the offsets of neighbouring tokens can overlap
  without being identical (`[4, 6)` and `[5, 6)`).

A token's `text` is a display string: the tokens' texts concatenate to the decoded text. Its
offsets are not, because they may overlap a neighbour's.

`tokenSpans`, `querySpans()` and `documentSpans()` merge overlapping offsets into one span: the
union range, the summed weight, and every token position. The spans never overlap, and for a
complete tokenization they tile the text; a truncated text is tiled up to the cut. **Highlight
with these merged spans**, never with per-token offsets. Per-token weights stay unmerged in
`queryTokens` and `documentTokens`.

`mergeOffsetRanges` applies the same rule to your own ranges. Stored chunks are different: with
`overlap_tokens` they overlap by design, and the SDK never merges them.

A stored chunk (`getDocument`) written by the pilot service has `char_start`/`char_end` set to
`null`. Search hits always carry offsets.

## Explaining a score

```ts
import { tokenAttribution, explanationScores } from '@breadbowl/embed';

const query = 'refund policy';
const doc = 'Refunds accepted.';
const res = await client.score({
  query, documents: [{ doc_id: 'a', text: doc }], explain: true, return_attention: true, attention_heads: 'mean',
});
const r = res.results[0]!;
const att = tokenAttribution(r.explanation!, {
  query: res.query, document: r.document, scoring: res.scoring, queryText: query, documentText: doc,
});
att.total;                 // = r.score (explanation tensors are always float32)
att.querySpans();          // merged, non-overlapping: [{ start, end, text, weight, positions }]
att.documentSpans();
att.slotPairs;             // [16][16] share of the score per (query slot, document slot)
explanationScores(r.explanation!, res.scoring); // { routingScore, valueScore, score } recomputed
```

How it is computed:

1. Each query slot's share of the score is
   `s[i] = (routing_weight · routing_contribution[i] + value_weight · value_slot_cosine[i]) / slots`.
2. `s[i]` is spread over document slots by `value_attention` (`coupling: 'value'`), by
   `softmax(slot_similarity / tau_k)` (`'routing'`), or by both (`'score'`, the default).
3. Slot→token attention carries the shares to tokens. Each side sums to the score.

Search returns only the query side, because document text is not stored. A search response
carries its own `scoring`, `search_profile`, `top_k` and `candidate_k`, so you can pass
`scoring: res.scoring` without fetching the index.

## Counters

Counters that grow without bound (`usage.input_tokens`, `requests`, ...) are decimal strings, so
no precision is lost. Convert them with `parseCounter('50000000')` (a `bigint`), or with
`parseCounters(summary.usage)` for every counter at once.

## Errors

Non-2xx responses throw `APIStatusError` subclasses. Each one carries `status`, `code`,
`errorMessage`, `requestId`, `details`, `issues`, `retryAfter` and `headers`:

- 400 `BadRequestError`
- 401 `AuthenticationError` (`ApiKeyExpiredError`)
- 403 `PermissionDeniedError` (`TenantSuspendedError`, `InsufficientScopeError`)
- 404 `NotFoundError`
- 409 `ConflictError` (`IndexNameConflictError`, `IndexNotReadyError`, `ModelRevisionUnavailableError`, `RepresentationUnavailableError`, `IdempotencyRequestInProgressError`)
- 413 `PayloadTooLargeError`
- 415 `UnsupportedMediaTypeError`
- 422 `UnprocessableEntityError` (`RequestValidationError`, `InvalidCursorError`, `UnsupportedModelError`, `UnsupportedSearchProfileError`, `UnsupportedFilterError`, `AttentionLimitExceededError`, `DocumentLimitExceededError`, `ChunkLimitExceededError`, `IdempotencyKeyReusedError`)
- 429 `RateLimitError` (`RateLimitExceededError`, `ConcurrencyLimitExceededError`, `UsageLimitExceededError`)
- 503 `ServiceUnavailableError`
- other 5xx `InternalServerError`

An unknown `code` maps to its status class. Failures without a response throw
`APIConnectionError` or `APITimeoutError`. Aborting through your `signal` throws
`APIUserAbortError`.

`details` has a fixed shape per code: the contract's `x-error-details`, generated as
`ErrorDetailsByCode`.

- `err.detailsFor('index_not_ready')?.status` is typed.
- The specific classes have getters, for example:
  - `IndexNotReadyError.indexStatus`
  - `IndexNameConflictError.indexId`
  - `InsufficientScopeError.requiredScope`
  - `UnsupportedModelError.served`
  - `AttentionLimitExceededError.requestedValues` and `.limit`
  - `DocumentLimitExceededError.limit`, `.current` and `.newDocuments`
  - `ChunkLimitExceededError.limit` and `.requested`
  - `RepresentationUnavailableError.representation` and `.served`

`RepresentationUnavailableError` means the pinned model revision is served with another
representation, for example another compute dtype. It is never retried; rebuild the index if it
persists.

## Retries, idempotency, timeouts

Defaults:

- `maxRetries: 2`
- exponential backoff with jitter, from 0.5 s up to 8 s
- `Retry-After` honoured up to `maxRetryWaitMs` (60 s)
- `timeoutMs: 60000` per attempt
- `upsertTimeoutMs: 180000` per attempt for upserts

Every operation except the health probes is retried on:

- 429, except `usage_limit_exceeded`
- 500, 502, 503 and 504
- 409 `idempotency_request_in_progress`
- 409 `index_not_ready` while `details.status` is `creating`
- timeouts and network errors

These follow the contract's guidance. Mutations reuse one `Idempotency-Key` (a UUID) for every
attempt.

Only 2xx responses are billed, so a failed `encode`, `score` or `search` attempt is never billed
twice. A request that succeeded on the server but timed out on the client is billed again when it
is retried; pass `maxRetries: 0` if that matters to you.

`usage_limit_exceeded` (the allowance resets next month) and an `index_not_ready` in `error`
state are never retried.

Per call: `client.usage.get({ signal, timeoutMs, maxRetries, headers })`. On mutations you can
also pass your own key: `{ idempotencyKey }`.

Every result has non-enumerable properties, so `JSON.stringify` leaves them out:

- `requestId`
- `meta`: `status`, `headers`, `idempotencyKey` and `idempotentReplayed`

Pass `fetch` to use a custom implementation (proxies, tests, instrumentation).

## Development

The package has its own lockfile (`pnpm-lock.yaml`); use pnpm (the version is pinned in
`packageManager`) and Node.js 22.12 or newer for the development tools (the published package
itself runs on Node 18+). The OpenAPI contract is
`../openapi/bb-embed.v1.yaml` (`sdks/openapi/` in the repository), shared with the Python SDK in
`../python`. From this directory (`sdks/typescript`):

```bash
pnpm install --frozen-lockfile
pnpm run generate       # types (openapi-typescript) + shared fixtures for both SDKs, from the contract
pnpm run generate:check
pnpm run build && pnpm run typecheck && pnpm run lint && pnpm run test
npm pack --dry-run      # what would be published: dist/, README.md, CHANGELOG.md, LICENSE, package.json
```

`typecheck` checks `src/` against browser-only libraries, without Node types, so the library
cannot use Node APIs by accident.

`test/contract.test.ts` validates every contract example and every SDK-built request against the
spec with ajv, and asserts that the generated types and fixtures are current.

The tests use a scripted mock `fetch`; the global `fetch` is replaced so that any real network
request fails the test.

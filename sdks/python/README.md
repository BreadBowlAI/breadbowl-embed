# breadbowl

Python client for the BreadBowl Embed API (`bb-embed` v1.3): encoding, scoring, managed indexes
and search. The model is open, so the API returns its raw data too (routing and value slots,
slot→token attention, tokens with character offsets, and the tensors every score is computed
from). This SDK decodes that data and helps you use it.

- Sync `BreadBowl` and async `AsyncBreadBowl` clients
- Typed response models, typed errors and request ids on both
- Bounded retries that honour `Retry-After`, and `Idempotency-Key` on every mutation
- Cursor pagination iterators
- Tensor decoding without numpy (float16 too), `.to_numpy()` when numpy is installed
- `token_attribution`: a score broken down over slot pairs and the tokens of both texts

The only dependency is `httpx`. The SDK does not need torch or the model library.

`breadbowl` (this package, import `breadbowl`) is the HTTP client for the API. To run the model
on your own machine instead, use the `breadbowl-embed` library (import `breadbowl_embed`) from the
root of the same repository. The source of this SDK is
[`sdks/python`](https://github.com/BreadBowlAI/breadbowl-embed/tree/main/sdks/python) in
[BreadBowlAI/breadbowl-embed](https://github.com/BreadBowlAI/breadbowl-embed).

Install from the repository:

```bash
pip install "breadbowl @ git+https://github.com/BreadBowlAI/breadbowl-embed.git#subdirectory=sdks/python"
pip install "breadbowl[numpy] @ git+https://github.com/BreadBowlAI/breadbowl-embed.git#subdirectory=sdks/python"   # adds Tensor.to_numpy()
```

Once the package is published on PyPI: `pip install breadbowl` (or `"breadbowl[numpy]"`).

Python 3.9 or newer.

**Version 0.1.0 targets the v1.3 API**
([`sdks/openapi/bb-embed.v1.yaml`](https://github.com/BreadBowlAI/breadbowl-embed/blob/main/sdks/openapi/bb-embed.v1.yaml),
`info.version` 1.3.0).
It does not work with the older pilot API that some deployments still serve, so set `base_url`
to a v1.3 deployment.

## Quickstart

```python
from breadbowl import BreadBowl

client = BreadBowl()  # reads BREADBOWL_API_KEY; or BreadBowl(api_key="...")

res = client.score(
    "refund policy",
    [
        {"doc_id": "a", "text": "Refunds are available within 30 days."},
        {"doc_id": "b", "text": "Keys can be rotated from the dashboard."},
    ],
)
for r in res.results:
    print(r.doc_id, r.score, r.routing_score, r.value_score)
print(res.request_id)  # quote this when you report a problem
```

A plain string document gets its position as `doc_id`: `client.score("q", ["text a", "text b"])`.

Keep the API key on a server. Do not put it in a URL, a browser bundle or a log.

### Self-hosting

```python
client = BreadBowl(api_key="...", base_url="http://localhost:8080")  # or BREADBOWL_BASE_URL
```

The default base URL comes from the contract's `servers` entry
(`breadbowl.DEFAULT_BASE_URL`).

## Encode

```python
out = client.encode(
    ["Keys can be rotated from the dashboard."],
    input_type="document",
    return_=["routing", "value", "attention", "tokens"],  # wire field `return`
    attention_heads="mean",
)
item = out.data[0]
item.routing.shape  # (16, 256), L2-normalised
item.value.to_list()  # nested Python floats, no numpy needed
item.routing.to_numpy()  # float16 ndarray (pip install "breadbowl[numpy]")
item.attention.axes  # ('head', 'slot', 'token'); rows sum to 1
[t.text for t in item.tokens]
```

## Indexes and search

```python
index = client.indexes.create("support-articles", metadata_schema={"source": "string"})

up = client.indexes.documents.upsert(
    index.id,
    [
        {
            "doc_id": "doc-1",
            "text": "Keys can be rotated from the dashboard.",
            "metadata": {"source": "docs"},
        },
        {"doc_id": "doc-2", "text": "Refunds are available within 30 days."},
    ],
)
for failed in up.failed:  # a 200 can still contain per-document failures
    print(failed.doc_id, failed.error.code, failed.error.retryable)

hits = client.indexes.search(index.id, "How do I rotate a key?", top_k=5, filter={"source": "docs"})
for hit in hits.results:
    print(hit.doc_id, hit.chunk_index, hit.score, hit.text(my_copy_of[hit.doc_id]))

idx = client.index(index.id)  # or a handle bound to one index
idx.search("rotate key")
idx.get_document("doc-1", return_vectors=True)
idx.delete_document("doc-1")
```

Text is never stored. `hit.text(document_text)` and `chunk.text(document_text)` slice your own copy
of the document by the chunk's code-point offsets.

### Many documents

One request takes at most 50 documents and `limits.max_chunks_per_upsert` (128) chunks.
`upsert_batched` takes any number of documents:

- It sends them in batches of `batch_size` (default 50), each with its own `Idempotency-Key`.
- Chunk counts are only known after the server tokenizes. A batch rejected with
  `chunk_limit_exceeded` is split into `ceil(requested / limit)` parts and retried.
- A single document that alone exceeds the limit becomes a failed outcome with code
  `chunk_limit_exceeded`.
- The per-document outcomes come back merged, in input order.

```python
from breadbowl import BatchUpsertError

try:
    result = client.indexes.documents.upsert_batched(index.id, my_documents)  # or idx.upsert_batched(...)
    print(result.upserted_documents, result.upserted_chunks, result.request_ids)
    for failed in result.failed:
        print(failed.doc_id, failed.error.code, failed.error.retryable)
except BatchUpsertError as e:  # a whole request failed, e.g. 409 representation_unavailable
    print(e.error, e.result.upserted_documents, e.pending)  # earlier batches are durable
```

Upserts are slow: about 2.4 to 3.5 chunks per second on the hosted CPU, so a request at the chunk
limit takes 40 to 60 s, and longer while other upserts are queued. Upserts use their own timeout,
`upsert_timeout`, which defaults to 180 s; the contract recommends at least 120 s. All other calls
use `timeout` (60 s). For interactive use, send a few documents at a time.

## Pagination

`list()` returns the first page. Iterating it walks every page:

```python
page = client.indexes.list(limit=50)
page.data, page.has_more, page.next_cursor, page.request_id
for index in page:  # all indexes, page after page
    print(index.name)
for p in page.iter_pages():
    ...
```

The async client returns a paginator: `await client.indexes.list()` gives the first page, and
`async for index in client.indexes.list()` gives every item.

The contract guarantees that `next_cursor` is a string exactly when `has_more` is true. A server
that sends `has_more: true` without a cursor raises `APIResponseError` instead of silently
stopping.

## Explaining a score

Ask for `explain=True` and `return_attention=True`, then use `attribution`:

```python
query = "refund policy"
doc = "Refunds accepted."
res = client.score(
    query, [{"doc_id": "a", "text": doc}], explain=True, return_attention=True, attention_heads="mean"
)
att = res.attribution(0, query_text=query, document_text=doc)
att.total  # = the result's score (explanation tensors are float32)
[(s.text, round(s.weight, 4)) for s in att.query_spans()]  # per query word piece
[(s.text, round(s.weight, 4)) for s in att.document_spans()]  # per document word piece
att.slot_pairs  # [16][16]: share of the score per (query slot, document slot)
```

How it is computed:

1. Each query slot's share of the score is
   `s[i] = (routing_weight · routing_contribution[i] + value_weight · value_slot_cosine[i]) / 16`,
   so `Σ s[i] = score`.
2. `s[i]` is spread over document slots by `coupling`:
   - `"value"` spreads it by `value_attention`.
   - `"routing"` spreads it by `softmax(slot_similarity / tau_k)`, the weights of the log-mean-exp
     (its gradient).
   - `"score"` (default) spreads the routing part by the routing weights and the value part by the
     value weights.
3. Slot→token attention carries the shares to tokens on each side. Query and document token weights
   each sum to the score.

Weights can be negative. Template and special tokens get weight too, but `*_spans()` keep only
your text, merged so they never overlap (see "Tokens, offsets and spans"). Pass `head=` to use
one attention head instead of the mean.

Search returns only the query's attention, because document text is not stored. A search
response carries its own `scoring`, `search_profile`, `top_k` and `candidate_k`, so
`hits.attribution(0, query_text=q)` needs no second call. Search with `explain=True` and
`return_attention=True`.

`Explanation.routing_score()` and `.value_score()` recompute the means from the tensors, so you
can check them against `routing_score` and `value_score`.

## Tensors

`Tensor` holds `{encoding, dtype, shape, axes, data}` as sent:

- `to_list()`, `flat()`, `t[i, j]` and `axis("token")` are pure Python.
- `to_numpy()` and `np.asarray(t)` need numpy.
- Values sent with `encoding_format="float"` are rounded to `dtype`, so both encodings decode to
  the same numbers.
- The request's `dtype` (float16 by default) applies to slots and attention. The four explanation
  tensors are always float32.

Decoding checks the data length against `shape × itemsize`, and the nesting against `shape`.
A mismatch raises `TensorDecodeError`, which is also a `ValueError`.

## Offsets

Token `start`/`end` and chunk `char_start`/`char_end` count Unicode code points. A Python string
also indexes by code point, so `text[start:end]` is correct.
`breadbowl.token_spans(text, tokens)` returns the merged spans of the input tokens and checks
that the offsets fit the text.

### Tokens, offsets and spans

A character can be split across byte-level tokens; an emoji is 4 UTF-8 bytes, for example.

- The token that ends inside the character shows only what precedes it (possibly `""`).
- The token that completes the character shows the whole character.
- Both report the whole character's offsets, so the offsets of neighbouring tokens can overlap
  without being identical (`[4, 6)` and `[5, 6)`).

A token's `text` is a display string: the tokens' texts concatenate to the decoded text. Its
offsets are not, because they may overlap a neighbour's.

Every helper that returns spans, `token_spans(text, tokens)`, `query_spans()` and
`document_spans()`, merges overlapping offsets into one span: the union range, the summed weight,
and every token position. The spans never overlap, and for a complete tokenization they tile the
text; a truncated text is tiled up to the cut. **Highlight with these merged spans**, never with
per-token offsets. Per-token weights stay unmerged in `query_tokens` and `document_tokens`.

```python
for span in breadbowl.token_spans(text, item.tokens):
    print(span.start, span.end, repr(span.text), span.positions)  # "".join(s.text) == text
```

Stored chunks are different: with `overlap_tokens` they overlap by design, and the SDK never
merges them.

A stored chunk written by the pilot service has `char_start`/`char_end` set to `None`;
`chunk.text(...)` raises `ValueError` for it.

## Errors

Every non-2xx response raises an `APIStatusError` subclass. Each one has `status_code`, `code`,
`message`, `request_id`, `details`, `issues`, `retry_after` and `headers`:

```python
from breadbowl import RateLimitError, RequestValidationError, NotFoundError, APIConnectionError

try:
    client.indexes.get("missing")
except NotFoundError as e:
    print(e.code, e.request_id)
except RequestValidationError as e:
    print(e.issues)
```

| Status | Class (subclasses by `code`) |
| --- | --- |
| 400 | `BadRequestError` |
| 401 | `AuthenticationError` (`ApiKeyExpiredError`) |
| 403 | `PermissionDeniedError` (`TenantSuspendedError`, `InsufficientScopeError`) |
| 404 | `NotFoundError` |
| 409 | `ConflictError` (`IndexNameConflictError`, `IndexNotReadyError`, `ModelRevisionUnavailableError`, `RepresentationUnavailableError`, `IdempotencyRequestInProgressError`) |
| 413 / 415 | `PayloadTooLargeError` / `UnsupportedMediaTypeError` |
| 422 | `UnprocessableEntityError` (`RequestValidationError`, `InvalidCursorError`, `UnsupportedModelError`, `UnsupportedSearchProfileError`, `UnsupportedFilterError`, `AttentionLimitExceededError`, `DocumentLimitExceededError`, `ChunkLimitExceededError`, `IdempotencyKeyReusedError`) |
| 429 | `RateLimitError` (`RateLimitExceededError`, `ConcurrencyLimitExceededError`, `UsageLimitExceededError`) |
| 503 | `ServiceUnavailableError` |
| other 5xx | `InternalServerError` |

An unknown `code` maps to its status class. When no HTTP response arrives, the SDK raises
`APIConnectionError`, or `APITimeoutError` for a timeout.

`details` has a fixed shape per code (the contract's `x-error-details`):

- `err.details_for("index_not_ready")` returns the details when the code matches.
- `breadbowl.types` has a `TypedDict` for each shape.
- The specific classes have typed properties, for example:
  - `IndexNotReadyError.index_status`
  - `IndexNameConflictError.index_id`
  - `InsufficientScopeError.required_scope`
  - `UnsupportedModelError.served`
  - `AttentionLimitExceededError.requested_values` and `.limit`
  - `DocumentLimitExceededError.limit`, `.current` and `.new_documents`
  - `ChunkLimitExceededError.limit` and `.requested`
  - `RepresentationUnavailableError.representation` and `.served`

`RepresentationUnavailableError` means the index's pinned model revision is served with another
representation, for example another compute dtype. It is not retried: it resolves when the
deployment serves the pinned representation again; otherwise rebuild the index.

## Retries, idempotency and timeouts

`max_retries=2` (3 attempts), exponential backoff with jitter (0.5 s doubling, at most 8 s).
`Retry-After` is honoured. A `Retry-After` longer than `max_retry_wait` (60 s) is raised instead
of waited for.

Every operation except the health probes is retried on:

- 429, except `usage_limit_exceeded`
- 500, 502, 503 and 504
- 409 `idempotency_request_in_progress`
- 409 `index_not_ready` while `details.status` is `creating`
- timeouts and connection failures

These follow the contract's guidance. Mutations send one `Idempotency-Key` for all attempts.

Only 2xx responses are billed, so a failed `encode`, `score` or `search` attempt is never billed
twice. A request that succeeded on the server but timed out on the client is billed again when it
is retried; set `max_retries=0` if that matters to you.

`usage_limit_exceeded` is never retried: the allowance resets next month. An `index_not_ready`
whose `details.status` is `error` is not retried either; delete and recreate the index.

Each mutation call generates a UUID `Idempotency-Key`, or uses the one you pass:
`client.indexes.create("n", idempotency_key="...")`.

```python
client = BreadBowl(timeout=30.0, max_retries=4, upsert_timeout=300.0)
fast = client.with_options(timeout=5.0, max_retries=0)  # shares the connection pool
```

Pass `http_client=httpx.Client(...)` for proxies, TLS settings or a custom transport.

## Response metadata

Every top-level result has:

- `.request_id`
- `.meta`: status, headers, `idempotency_key` and `idempotent_replayed`
- `.raw`: the JSON exactly as received, including fields this SDK version does not know

`to_dict()` serialises a model back to wire JSON.

## Models

Responses are plain dataclasses whose field names are the wire names. Timestamps stay ISO-8601
strings. Counters stay decimal strings: `int(x)` is exact, and `usage.usage.as_ints()` converts
them all.

Dataclasses rather than pydantic keep the install light. They also avoid version conflicts with
ML environments that pin pydantic 1 or 2.

Parsing is lenient: an unknown field never breaks an older client. The test suite checks the
models against the OpenAPI contract instead.

## Async

```python
import asyncio
from breadbowl import AsyncBreadBowl


async def main():
    async with AsyncBreadBowl() as client:
        res = await client.indexes.search("idx_123", "rotate key")
        async for index in client.indexes.list():
            print(index.name)


asyncio.run(main())
```

Works on asyncio and trio (through anyio).

## Development

The HTTP surface is defined by the OpenAPI contract in `../openapi/bb-embed.v1.yaml`
(`sdks/openapi/` in the repository), shared with the TypeScript SDK in `../typescript`. Two
scripts generate code from it. From this directory (`sdks/python`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/generate_types.py              # src/breadbowl/_generated.py (enums, request TypedDicts, operation table)
python scripts/generate_types.py --check
(cd ../typescript && pnpm install && node scripts/generate-fixtures.mjs)   # shared fixtures for both SDKs (Node.js and pnpm)
pytest
ruff check . && ruff format --check . && mypy
```

`tests/test_contract.py` fails when the generated code or fixtures are stale. It also checks the
models against the contract:

- Every response example validates against its schema, parses, and serialises back unchanged.
- Model fields and required-ness match the schemas.
- The contract's request examples are exactly what the SDK sends.
- SDK requests with every option set validate against the strict request schemas.

The tests use scripted mock transports and never contact a server: a request to any host other
than loopback fails the test.

// Bounded retries per operation kind, Retry-After, idempotency keys, timeouts and aborts.
import { describe, expect, it } from 'vitest';
import {
  APIConnectionError,
  APITimeoutError,
  APIUserAbortError,
  IndexNotReadyError,
  InternalServerError,
  RateLimitExceededError,
  RequestValidationError,
  ServiceUnavailableError,
  UsageLimitExceededError,
  parseRetryAfter,
} from '../src/index.js';
import { MockServer, connectionRefused, errorBody, example, makeClient } from './helpers.js';

const INDEX_ID = '0b8a3d0e-5c1f-4f7e-9a51-3e2d6c4b1a90';
const RATE = errorBody('rate_limit_exceeded');
const INTERNAL = errorBody('internal_error');
const UNAVAILABLE = errorBody('service_unavailable');

describe('retries', () => {
  it('429 honours Retry-After then succeeds', async () => {
    const server = new MockServer().add(RATE, 429, { 'Retry-After': '2' }).add(example('encode'));
    const res = await makeClient(server).encode({ input: 'q', input_type: 'query' });
    expect(server.sleeps).toEqual([2000]);
    expect(res.requestId).toBe('req-test-1');
  });

  it('parses HTTP-date Retry-After', () => {
    expect(parseRetryAfter('Wed, 21 Oct 2015 07:28:00 GMT')).toBe(0);
    expect(parseRetryAfter('7')).toBe(7);
    expect(parseRetryAfter('soon')).toBeNull();
  });

  it('usage_limit_exceeded is not retried', async () => {
    const server = new MockServer().add(errorBody('usage_limit_exceeded'), 429, { 'Retry-After': '86400' });
    const err = await makeClient(server).encode({ input: 'q', input_type: 'query' }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(UsageLimitExceededError);
    expect((err as UsageLimitExceededError).retryAfter).toBe(86400);
    expect(server.requests).toHaveLength(1);
  });

  it('a Retry-After beyond maxRetryWaitMs is thrown', async () => {
    const server = new MockServer().add(RATE, 429, { 'Retry-After': '600' });
    await expect(makeClient(server).usage.get()).rejects.toBeInstanceOf(RateLimitExceededError);
    expect(server.sleeps).toEqual([]);
  });

  it('gives up after maxRetries with jittered exponential backoff', async () => {
    const server = new MockServer().add(UNAVAILABLE, 503).add(UNAVAILABLE, 503).add(UNAVAILABLE, 503);
    await expect(makeClient(server).models.list()).rejects.toBeInstanceOf(ServiceUnavailableError);
    expect(server.requests).toHaveLength(3);
    expect(server.sleeps[0]).toBeGreaterThanOrEqual(250);
    expect(server.sleeps[0]).toBeLessThanOrEqual(500);
    expect(server.sleeps[1]).toBeGreaterThanOrEqual(500);
    expect(server.sleeps[1]).toBeLessThanOrEqual(1000);
  });

  it('maxRetries 0 per request', async () => {
    const server = new MockServer().add(UNAVAILABLE, 503);
    await expect(makeClient(server).usage.get({ maxRetries: 0 })).rejects.toBeInstanceOf(ServiceUnavailableError);
    expect(server.requests).toHaveLength(1);
  });

  it.each([500, 502, 504])('%i is retried for reads', async (status) => {
    const server = new MockServer().add(INTERNAL, status).add(example('getIndex'));
    await makeClient(server).indexes.get(INDEX_ID);
    expect(server.requests).toHaveLength(2);
  });

  // Only 2xx responses are billed, so failed encode/score/search attempts are retried too.
  it.each([500, 502, 504])('%i is retried for encode, score and search', async (status) => {
    const server = new MockServer()
      .add(INTERNAL, status)
      .add(example('searchIndex'))
      .add(INTERNAL, status)
      .add(example('encode'))
      .add(INTERNAL, status)
      .add(example('score', 200, 'basic'));
    const c = makeClient(server);
    await c.indexes.search(INDEX_ID, 'q');
    await c.encode({ input: 'q', input_type: 'query' });
    await c.score({ query: 'q', documents: ['a'] });
    expect(server.requests).toHaveLength(6);
  });

  it('a 500 is thrown after the retries are spent', async () => {
    const server = new MockServer().add(INTERNAL, 500).add(INTERNAL, 500).add(INTERNAL, 500);
    await expect(makeClient(server).indexes.search(INDEX_ID, 'q')).rejects.toBeInstanceOf(InternalServerError);
    expect(server.requests).toHaveLength(3);
  });

  it('503 is retried for score', async () => {
    const server = new MockServer().add(UNAVAILABLE, 503, { 'Retry-After': '1' }).add(example('score', 200, 'basic'));
    await makeClient(server).score({ query: 'q', documents: ['a', 'b'] });
    expect(server.sleeps).toEqual([1000]);
  });

  it('mutations reuse one Idempotency-Key across retries', async () => {
    const server = new MockServer().add(INTERNAL, 500).add(UNAVAILABLE, 503).add(example('createIndex', 201), 201);
    const index = await makeClient(server).indexes.create({ name: 'production' });
    const keys = new Set(server.requests.map((r) => r.headers.get('idempotency-key')));
    expect(server.requests).toHaveLength(3);
    expect(keys.size).toBe(1);
    expect(keys.has(index.meta.idempotencyKey)).toBe(true);
  });

  it('each mutation call gets a fresh key', async () => {
    const server = new MockServer().add(example('deleteDocument')).add(example('deleteDocument'));
    const c = makeClient(server);
    await c.indexes.documents.delete(INDEX_ID, 'd');
    await c.indexes.documents.delete(INDEX_ID, 'd');
    expect(server.requests[0]!.headers.get('idempotency-key')).not.toBe(server.requests[1]!.headers.get('idempotency-key'));
  });

  it('idempotency_request_in_progress is retried with the same key (backoff without Retry-After)', async () => {
    const server = new MockServer().add(errorBody('idempotency_request_in_progress'), 409).add(example('deleteIndex'));
    await makeClient(server).indexes.delete(INDEX_ID, { idempotencyKey: 'k1' });
    expect(server.requests.map((r) => r.headers.get('idempotency-key'))).toEqual(['k1', 'k1']);
    expect(server.sleeps).toHaveLength(1);
  });

  it('idempotency_request_in_progress with Retry-After is retried with the same key', async () => {
    const server = new MockServer()
      .add(errorBody('idempotency_request_in_progress'), 409, { 'Retry-After': '3' })
      .add(example('upsertDocuments'));
    await makeClient(server).indexes.documents.upsert(INDEX_ID, [{ doc_id: 'd', text: 't' }], { idempotencyKey: 'fixed' });
    expect(server.sleeps).toEqual([3000]);
    expect(server.requests.map((r) => r.headers.get('idempotency-key'))).toEqual(['fixed', 'fixed']);
  });

  it('index_not_ready is retried only while details.status is creating', async () => {
    const creating = { error: { code: 'index_not_ready', message: 'creating', details: { status: 'creating' } } };
    const failed = { error: { code: 'index_not_ready', message: 'error', details: { status: 'error' } } };
    const server = new MockServer()
      .add(creating, 409, { 'Retry-After': '5' })
      .add(example('searchIndex'))
      .add(failed, 409, { 'Retry-After': '5' });
    const c = makeClient(server);
    await c.indexes.search(INDEX_ID, 'q');
    expect(server.sleeps).toEqual([5000]);
    const err = await c.indexes.search(INDEX_ID, 'q').catch((e: unknown) => e);
    expect(err).toBeInstanceOf(IndexNotReadyError);
    expect((err as IndexNotReadyError).indexStatus).toBe('error');
    expect(server.requests).toHaveLength(3);
  });

  it('422 is never retried', async () => {
    const server = new MockServer().add(errorBody('validation_error'), 422);
    const err = await makeClient(server).indexes.documents.upsert(INDEX_ID, [{ doc_id: 'd', text: 't' }]).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(RequestValidationError);
    expect((err as RequestValidationError).issues[0]!.path).toEqual(['documents', 3, 'doc_id']);
    expect(server.requests).toHaveLength(1);
  });

  it('connection failures are retried for encode', async () => {
    const server = new MockServer().fail(connectionRefused()).fail(new TypeError('fetch failed')).add(example('encode'));
    await makeClient(server).encode({ input: 'q', input_type: 'query' });
    expect(server.requests).toHaveLength(3);
  });

  it('a connection failure is thrown after the retries are spent', async () => {
    const server = new MockServer().fail(new TypeError('a')).fail(new TypeError('b')).fail(new TypeError('c'));
    await expect(makeClient(server).encode({ input: 'q', input_type: 'query' })).rejects.toBeInstanceOf(APIConnectionError);
    expect(server.requests).toHaveLength(3);
  });

  it('network failures are retried for reads and mutations', async () => {
    const server = new MockServer()
      .fail(new TypeError('fetch failed'))
      .add(example('getUsage'))
      .fail(new TypeError('socket hang up'))
      .add(example('deleteIndex'));
    const c = makeClient(server);
    await c.usage.get();
    await c.indexes.delete(INDEX_ID);
    expect(server.requests).toHaveLength(4);
  });

  it('times out per attempt', async () => {
    const hanging = (_url: string, init: RequestInit): Promise<Response> =>
      new Promise((_resolve, reject) => {
        init.signal?.addEventListener('abort', () => {
          reject(new DOMException('aborted', 'AbortError'));
        });
      });
    let calls = 0;
    const counting = (url: string, init: RequestInit): Promise<Response> => {
      calls += 1;
      return hanging(url, init);
    };
    const c = makeClient(new MockServer(), { fetch: counting, timeoutMs: 20, maxRetries: 1 });
    const err = await c.encode({ input: 'q', input_type: 'query' }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(APITimeoutError);
    expect(calls).toBe(2); // timeouts are retried (the contract's guidance)
  });

  it('a caller abort is never retried', async () => {
    const controller = new AbortController();
    const server = new MockServer();
    const fetchThatAborts = (url: string, init: RequestInit): Promise<Response> => {
      controller.abort();
      return server.fetch(url, init);
    };
    server.add(UNAVAILABLE, 503);
    const c = makeClient(server, { fetch: fetchThatAborts });
    await expect(c.usage.get({ signal: controller.signal })).rejects.toBeInstanceOf(APIUserAbortError);
    await expect(c.usage.get({ signal: controller.signal })).rejects.toBeInstanceOf(APIUserAbortError);
  });
});

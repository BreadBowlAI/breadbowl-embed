import { describe, expect, it } from 'vitest';
import * as sdk from '../src/index.js';
import {
  APIResponseError,
  APIStatusError,
  ERROR_DETAILS,
  InsufficientScopeError,
  InternalServerError,
  errorClassFor,
} from '../src/index.js';
import { FIXTURES, MockServer, makeClient } from './helpers.js';

const EXPECTED: Record<string, string> = {
  bad_request: 'BadRequestError',
  authentication_failed: 'AuthenticationError',
  api_key_expired: 'ApiKeyExpiredError',
  tenant_suspended: 'TenantSuspendedError',
  insufficient_scope: 'InsufficientScopeError',
  not_found: 'NotFoundError',
  index_name_conflict: 'IndexNameConflictError',
  index_not_ready: 'IndexNotReadyError',
  model_revision_unavailable: 'ModelRevisionUnavailableError',
  representation_unavailable: 'RepresentationUnavailableError',
  idempotency_request_in_progress: 'IdempotencyRequestInProgressError',
  payload_too_large: 'PayloadTooLargeError',
  unsupported_media_type: 'UnsupportedMediaTypeError',
  validation_error: 'RequestValidationError',
  invalid_cursor: 'InvalidCursorError',
  unsupported_model: 'UnsupportedModelError',
  unsupported_search_profile: 'UnsupportedSearchProfileError',
  unsupported_filter: 'UnsupportedFilterError',
  attention_limit_exceeded: 'AttentionLimitExceededError',
  document_limit_exceeded: 'DocumentLimitExceededError',
  chunk_limit_exceeded: 'ChunkLimitExceededError',
  idempotency_key_reused: 'IdempotencyKeyReusedError',
  rate_limit_exceeded: 'RateLimitExceededError',
  concurrency_limit_exceeded: 'ConcurrencyLimitExceededError',
  usage_limit_exceeded: 'UsageLimitExceededError',
  internal_error: 'InternalServerError',
  service_unavailable: 'ServiceUnavailableError',
};

describe('contract error examples', () => {
  it.each(FIXTURES.errors.map((e) => [`${e.status} ${e.code}`, e] as const))('%s', async (_n, e) => {
    const headers: Record<string, string> = { 'x-request-id': 'req-err-1' };
    if (e.status === 429) headers['Retry-After'] = '12';
    const server = new MockServer().add(e.body, e.status, headers);
    const err = await makeClient(server, { maxRetries: 0 }).usage.get().catch((x: unknown) => x);
    expect(err).toBeInstanceOf(APIStatusError);
    const s = err as APIStatusError;
    expect(s.constructor.name).toBe(EXPECTED[e.code]);
    expect(s.name).toBe(EXPECTED[e.code]);
    expect(s.status).toBe(e.status);
    expect(s.code).toBe(e.code);
    expect(s.errorMessage).toBe(e.body.error.message);
    expect(s.requestId).toBe('req-err-1');
    expect(s.message).toContain('req-err-1');
    expect(s.details).toEqual(e.body.error.details ?? {});
    if (e.code in ERROR_DETAILS && e.body.error.details) {
      expect(s.detailsFor(e.code as keyof typeof ERROR_DETAILS)).toEqual(e.body.error.details);
    }
    expect(s.issues).toEqual(e.body.error.issues ?? []);
    if (e.status === 429) expect(s.retryAfter).toBe(12);
  });

  it('every documented code has its own exported class', () => {
    for (const [status, codes] of Object.entries(FIXTURES.error_codes)) {
      for (const code of codes) {
        const cls = errorClassFor(Number(status), code);
        expect(cls.name).toBe(EXPECTED[code]);
        expect((sdk as Record<string, unknown>)[cls.name]).toBe(cls);
      }
    }
  });
});

describe('fallbacks', () => {
  it('maps unknown codes by status', () => {
    expect(errorClassFor(422, 'brand_new')).toBe(sdk.UnprocessableEntityError);
    expect(errorClassFor(429, 'brand_new')).toBe(sdk.RateLimitError);
    expect(errorClassFor(502, null)).toBe(InternalServerError);
    expect(errorClassFor(418, 'teapot')).toBe(APIStatusError);
    expect(errorClassFor(400, 'not_found')).toBe(sdk.BadRequestError);
  });

  it('typed details', () => {
    const e = new InsufficientScopeError('x', { status: 403, code: 'insufficient_scope', requestId: null, details: { required_scope: 'indexes:write' } });
    expect(e.requiredScope).toBe('indexes:write');
    const c = new sdk.IndexNameConflictError('x', { status: 409, code: 'index_name_conflict', requestId: null, details: { index_id: 'i' } });
    expect(c.indexId).toBe('i');
  });

  it('non-JSON error bodies', async () => {
    const server = new MockServer().addText('<html>Bad gateway</html>', 502);
    const err = await makeClient(server, { maxRetries: 0 }).models.get('m').catch((x: unknown) => x);
    expect(err).toBeInstanceOf(InternalServerError);
    expect((err as InternalServerError).code).toBeNull();
    expect((err as InternalServerError).errorMessage).toContain('Bad gateway');
  });

  it('401 exposes WWW-Authenticate and is not retried', async () => {
    const server = new MockServer().add({ error: { code: 'authentication_failed', message: 'nope' } }, 401, {
      'WWW-Authenticate': 'Bearer realm="bb-embed", error="invalid_token"',
    });
    const err = await makeClient(server).usage.get().catch((x: unknown) => x);
    expect(err).toBeInstanceOf(sdk.AuthenticationError);
    expect((err as APIStatusError).headers.get('www-authenticate')).toMatch(/^Bearer/);
    expect(server.requests).toHaveLength(1);
  });

  it('2xx without JSON is an APIResponseError', async () => {
    const server = new MockServer().addText('not json', 200);
    await expect(makeClient(server).usage.get()).rejects.toBeInstanceOf(APIResponseError);
  });

  it('hierarchy', () => {
    expect(new sdk.APITimeoutError('t')).toBeInstanceOf(sdk.APIConnectionError);
    for (const name of Object.values(EXPECTED)) {
      const Cls = (sdk as unknown as Record<string, typeof APIStatusError>)[name]!;
      expect(new Cls('m', { status: 400, code: null, requestId: null })).toBeInstanceOf(sdk.BreadBowlError);
    }
  });
});

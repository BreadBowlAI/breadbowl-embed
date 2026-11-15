// Transport: fetch, per-attempt timeouts, bounded retries, idempotency keys, response metadata.
import {
  APIConnectionError,
  APIResponseError,
  APIStatusError,
  APITimeoutError,
  APIUserAbortError,
  BreadBowlError,
  errorClassFor,
} from './errors.js';
import { DEFAULT_BASE_URL, OPERATIONS, type OperationId } from './generated/meta.js';
import type { ValidationIssue } from './types.js';
import { VERSION } from './version.js';

export type FetchLike = (input: string, init: RequestInit) => Promise<Response>;

export interface ClientOptions {
  /**
   * Account API key. Defaults to `BREADBOWL_API_KEY` in server runtimes (Node, Bun, Deno).
   * Pass `null` to send no `Authorization` header, e.g. to a same-origin proxy that adds it.
   * Never ship an account key to a browser; see the README.
   */
  apiKey?: string | null | undefined;
  /** Defaults to `BREADBOWL_BASE_URL` or the hosted service. Set it for self-hosted deployments. */
  baseURL?: string | undefined;
  /** A `fetch` implementation (default: the global one). */
  fetch?: FetchLike | undefined;
  /** Per-attempt timeout in milliseconds (default 60 000). */
  timeoutMs?: number | undefined;
  /**
   * Per-attempt timeout for `upsertDocuments` (default 180 000). Upserts are encoded at about
   * 3 chunks/s on the hosted CPU, so a full request can take over a minute.
   */
  upsertTimeoutMs?: number | undefined;
  /** Retries after the first attempt (default 2). */
  maxRetries?: number | undefined;
  /** A `Retry-After` longer than this is not waited for; the error is thrown (default 60 000). */
  maxRetryWaitMs?: number | undefined;
  defaultHeaders?: Record<string, string> | undefined;
  /** Allow an API key in a browser page. Off by default; only for keys that are safe to expose. */
  dangerouslyAllowBrowser?: boolean | undefined;
}

export interface RequestOptions {
  signal?: AbortSignal | undefined;
  timeoutMs?: number | undefined;
  maxRetries?: number | undefined;
  headers?: Record<string, string> | undefined;
}

export interface MutationOptions extends RequestOptions {
  /** Reused on every retry of this call. Default: a fresh UUID per call. */
  idempotencyKey?: string | undefined;
}

export interface ResponseMeta {
  requestId: string | null;
  status: number;
  headers: Headers;
  idempotencyKey: string | null;
  /** `true` when the server replayed a stored response for this `Idempotency-Key`. */
  idempotentReplayed: boolean;
}

/** A response body plus non-enumerable `requestId` and `meta` (not serialized by JSON.stringify). */
export type WithMeta<T> = T & { readonly requestId: string | null; readonly meta: ResponseMeta };

export function attachMeta<T extends object>(body: T, meta: ResponseMeta): WithMeta<T> {
  Object.defineProperty(body, 'meta', { value: meta, enumerable: false, configurable: true });
  Object.defineProperty(body, 'requestId', { value: meta.requestId, enumerable: false, configurable: true });
  return body as WithMeta<T>;
}

export interface CallSpec {
  operation: OperationId;
  pathParams?: Record<string, string>;
  query?: Record<string, string | number | boolean | undefined>;
  body?: unknown;
  idempotencyKey?: string;
  /** Non-2xx statuses whose body is a result rather than an error (readyz's 503). */
  acceptStatuses?: readonly number[];
}

/**
 * probe: health checks report the current state and are never retried. Everything else follows
 * the contract's retry guidance. Mutations carry one Idempotency-Key for all attempts; encode,
 * score and search store nothing and only 2xx responses are billed, so a failed attempt can be
 * retried without double billing.
 */
type RetryKind = 'probe' | 'retry';

function retryKind(operation: OperationId): RetryKind {
  return OPERATIONS[operation].authenticated ? 'retry' : 'probe';
}

const INITIAL_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 8000;

export function parseRetryAfter(value: string | null): number | null {
  if (value === null) return null;
  const trimmed = value.trim();
  if (/^\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed);
  const when = Date.parse(trimmed);
  if (Number.isNaN(when)) return null;
  return Math.max(0, (when - Date.now()) / 1000);
}

export function randomUUID(): string {
  const c = (globalThis as { crypto?: Crypto }).crypto;
  if (c && typeof c.randomUUID === 'function') return c.randomUUID();
  const bytes = new Uint8Array(16);
  if (c && typeof c.getRandomValues === 'function') c.getRandomValues(bytes);
  else for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256);
  bytes[6] = ((bytes[6] ?? 0) & 0x0f) | 0x40;
  bytes[8] = ((bytes[8] ?? 0) & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function readEnv(name: string): string | undefined {
  const g = globalThis as {
    process?: { env?: Record<string, string | undefined> };
    Deno?: { env?: { get(key: string): string | undefined } };
  };
  try {
    const fromProcess = g.process?.env?.[name];
    if (fromProcess) return fromProcess;
    return g.Deno?.env?.get(name);
  } catch {
    return undefined; // Deno without --allow-env
  }
}

export function isBrowser(): boolean {
  const g = globalThis as { document?: unknown; window?: unknown };
  return typeof g.window !== 'undefined' && typeof g.document !== 'undefined';
}

function jsonBody(body: unknown): string {
  return JSON.stringify(body, (_key, value: unknown) => {
    if (typeof value === 'number' && !Number.isFinite(value)) {
      throw new BreadBowlError(`request contains a non-finite number (${String(value)}); the API rejects it`);
    }
    return value;
  });
}

export class APIClientCore {
  readonly apiKey: string | null;
  readonly baseURL: string;
  readonly timeoutMs: number;
  readonly upsertTimeoutMs: number;
  readonly maxRetries: number;
  readonly maxRetryWaitMs: number;
  readonly defaultHeaders: Readonly<Record<string, string>>;
  protected readonly fetchImpl: FetchLike;
  /** @internal Sleep used between retries; tests replace it. */
  _sleep: (ms: number, signal?: AbortSignal) => Promise<void> = sleep;

  constructor(options: ClientOptions = {}) {
    const browser = isBrowser();
    let apiKey: string | null;
    if (options.apiKey === null) apiKey = null;
    else apiKey = options.apiKey ?? (browser ? undefined : readEnv('BREADBOWL_API_KEY')) ?? null;
    if (apiKey === null && options.apiKey !== null) {
      throw new BreadBowlError(
        'No API key. Pass { apiKey } or set BREADBOWL_API_KEY (server runtimes). ' +
          'Pass { apiKey: null } only when a proxy adds the key for you.',
      );
    }
    if (apiKey !== null && browser && !options.dangerouslyAllowBrowser) {
      throw new BreadBowlError(
        'Refusing to use an API key in a browser: anyone can read it from the page. Call the API ' +
          'from your server, or use { apiKey: null } with a proxy. Set dangerouslyAllowBrowser: true ' +
          'only for a key that is safe to expose.',
      );
    }
    if (!options.fetch && typeof (globalThis as { fetch?: unknown }).fetch !== 'function') {
      throw new BreadBowlError('No fetch implementation in this runtime; pass { fetch }.');
    }
    this.apiKey = apiKey;
    this.baseURL = (options.baseURL ?? readEnv('BREADBOWL_BASE_URL') ?? DEFAULT_BASE_URL).replace(/\/+$/, '');
    this.timeoutMs = options.timeoutMs ?? 60_000;
    this.upsertTimeoutMs = options.upsertTimeoutMs ?? 180_000;
    this.maxRetries = options.maxRetries ?? 2;
    this.maxRetryWaitMs = options.maxRetryWaitMs ?? 60_000;
    this.defaultHeaders = { ...(options.defaultHeaders ?? {}) };
    // Call through globalThis so a browser's fetch keeps its `this`.
    this.fetchImpl = options.fetch ?? ((input, init) => globalThis.fetch(input, init));
    if (this.maxRetries < 0) throw new RangeError('maxRetries must be >= 0');
  }

  newIdempotencyKey(): string {
    return randomUUID();
  }

  protected buildURL(spec: CallSpec): string {
    const op = OPERATIONS[spec.operation];
    let path: string = op.path;
    for (const [name, value] of Object.entries(spec.pathParams ?? {})) {
      // One path segment each: "/" in a doc_id is sent as %2F.
      path = path.replace(`{${name}}`, encodeURIComponent(value));
    }
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(spec.query ?? {})) {
      if (value !== undefined) params.append(key, String(value));
    }
    const qs = params.toString();
    return `${this.baseURL}${path}${qs ? `?${qs}` : ''}`;
  }

  protected buildHeaders(spec: CallSpec, options: RequestOptions): Headers {
    const op = OPERATIONS[spec.operation];
    const headers = new Headers({ Accept: 'application/json' });
    if (!isBrowser()) headers.set('User-Agent', `breadbowl-typescript/${VERSION}`);
    for (const [k, v] of Object.entries(this.defaultHeaders)) headers.set(k, v);
    if (op.authenticated && this.apiKey !== null) headers.set('Authorization', `Bearer ${this.apiKey}`);
    if (spec.body !== undefined) headers.set('Content-Type', 'application/json');
    if (spec.idempotencyKey !== undefined) headers.set('Idempotency-Key', spec.idempotencyKey);
    for (const [k, v] of Object.entries(options.headers ?? {})) headers.set(k, v);
    return headers;
  }

  private backoff(attempt: number): number {
    const base = Math.min(MAX_BACKOFF_MS, INITIAL_BACKOFF_MS * 2 ** attempt);
    return base * (0.5 + Math.random() / 2);
  }

  /** Milliseconds to wait before the next attempt, or null to give up. */
  private retryDelay(
    kind: RetryKind,
    attempt: number,
    maxRetries: number,
    failure: { error: APIStatusError } | { cause: unknown },
  ): number | null {
    if (kind === 'probe' || attempt >= maxRetries) return null;
    // Timeouts and connection failures.
    if (!('error' in failure)) return this.backoff(attempt);
    const { status, code, retryAfter } = failure.error;
    let retryable: boolean;
    if (status === 429) retryable = code !== 'usage_limit_exceeded'; // that one resets next month
    else if (status === 409) {
      retryable =
        code === 'idempotency_request_in_progress' ||
        (code === 'index_not_ready' && failure.error.detailsFor('index_not_ready')?.status === 'creating');
    } else retryable = status === 500 || status === 502 || status === 503 || status === 504;
    if (!retryable) return null;
    if (retryAfter !== null) {
      const ms = retryAfter * 1000;
      return ms <= this.maxRetryWaitMs ? ms : null;
    }
    return this.backoff(attempt);
  }

  private makeError(response: Response, text: string): APIStatusError {
    let body: unknown = text;
    try {
      body = JSON.parse(text);
    } catch {
      // not JSON (e.g. a proxy's HTML error page)
    }
    let code: string | null = null;
    let message = response.statusText || `HTTP ${response.status}`;
    let details: Record<string, unknown> | undefined;
    let issues: ValidationIssue[] | undefined;
    const envelope = (body as { error?: unknown } | null)?.error;
    if (envelope && typeof envelope === 'object') {
      const e = envelope as { code?: unknown; message?: unknown; details?: unknown; issues?: unknown };
      code = typeof e.code === 'string' ? e.code : null;
      if (typeof e.message === 'string' && e.message) message = e.message;
      if (e.details && typeof e.details === 'object') details = e.details as Record<string, unknown>;
      if (Array.isArray(e.issues)) issues = e.issues as ValidationIssue[];
    } else if (typeof body === 'string' && body.trim()) {
      message = body.trim().slice(0, 500);
    }
    const Cls = errorClassFor(response.status, code);
    return new Cls(message, {
      status: response.status,
      code,
      requestId: response.headers.get('x-request-id'),
      details,
      issues,
      retryAfter: parseRetryAfter(response.headers.get('retry-after')),
      headers: response.headers,
      body,
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters -- the caller names the contract type of the body
  async request<T>(spec: CallSpec, options: RequestOptions = {}): Promise<{ body: T; meta: ResponseMeta }> {
    const op = OPERATIONS[spec.operation];
    const url = this.buildURL(spec);
    const headers = this.buildHeaders(spec, options);
    const body = spec.body === undefined ? undefined : jsonBody(spec.body);
    const kind = retryKind(spec.operation);
    const maxRetries = options.maxRetries ?? this.maxRetries;
    const timeoutMs = options.timeoutMs ?? this.timeoutMs;
    const userSignal = options.signal;

    for (let attempt = 0; ; attempt++) {
      if (isAborted(userSignal)) throw new APIUserAbortError('request aborted', { cause: userSignal?.reason });
      const controller = new AbortController();
      // An object, not a `let`: the timer callback mutates it and narrowing must not assume false.
      const attemptState = { timedOut: false };
      const timer = setTimeout(() => {
        attemptState.timedOut = true;
        controller.abort();
      }, timeoutMs);
      const onAbort = (): void => {
        controller.abort(userSignal?.reason);
      };
      userSignal?.addEventListener('abort', onAbort, { once: true });
      let response: Response;
      let text: string;
      try {
        const init: RequestInit = { method: op.method, headers, signal: controller.signal };
        if (body !== undefined) init.body = body;
        response = await this.fetchImpl(url, init);
        text = await response.text();
      } catch (cause) {
        if (isAborted(userSignal)) throw new APIUserAbortError('request aborted', { cause });
        const { timedOut } = attemptState;
        const delay = this.retryDelay(kind, attempt, maxRetries, { cause });
        if (delay === null) {
          throw timedOut
            ? new APITimeoutError(`request timed out after ${timeoutMs} ms`, { cause })
            : new APIConnectionError(`connection failed: ${String((cause as Error | null)?.message ?? cause)}`, {
                cause,
              });
        }
        await this._sleep(delay, userSignal);
        continue;
      } finally {
        clearTimeout(timer);
        userSignal?.removeEventListener('abort', onAbort);
      }

      const meta: ResponseMeta = {
        requestId: response.headers.get('x-request-id'),
        status: response.status,
        headers: response.headers,
        idempotencyKey: spec.idempotencyKey ?? null,
        idempotentReplayed: response.headers.get('idempotent-replayed')?.toLowerCase() === 'true',
      };
      if (response.ok || spec.acceptStatuses?.includes(response.status)) {
        let parsed: unknown;
        try {
          parsed = JSON.parse(text);
        } catch {
          throw new APIResponseError(`expected a JSON body from ${spec.operation}, got ${JSON.stringify(text.slice(0, 200))}`, {
            status: response.status,
            requestId: meta.requestId,
            body: text,
          });
        }
        if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
          throw new APIResponseError(`expected a JSON object from ${spec.operation}`, {
            status: response.status,
            requestId: meta.requestId,
            body: parsed,
          });
        }
        return { body: parsed as T, meta };
      }
      const error = this.makeError(response, text);
      const delay = this.retryDelay(kind, attempt, maxRetries, { error });
      if (delay === null) throw error;
      await this._sleep(delay, userSignal);
    }
  }
}

// A function, so control-flow narrowing does not assume `aborted` is unchanged across awaits.
function isAborted(signal: AbortSignal | undefined): boolean {
  return signal?.aborted === true;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new APIUserAbortError('request aborted', { cause: signal.reason }));
      return;
    }
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(new APIUserAbortError('request aborted', { cause: signal?.reason }));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

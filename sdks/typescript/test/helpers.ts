import { readFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BreadBowl, type ClientOptions, type FetchLike } from '../src/index.js';

const here = dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = resolve(here, '../../..');
export const SPEC_PATH = resolve(REPO_ROOT, 'sdks/openapi/bb-embed.v1.yaml');
export const FIXTURE_PATH = resolve(here, 'fixtures/contract.json');
export const SIBLING_FIXTURE = resolve(REPO_ROOT, 'sdks/python/tests/fixtures/contract.json');
export const HAS_SPEC = existsSync(SPEC_PATH);

export interface Fixtures {
  spec: { sha256: string; version: string; server_url: string };
  error_codes: Record<string, string[]>;
  error_details: Record<string, string>;
  operations: Record<
    string,
    {
      method: string;
      path: string;
      authenticated: boolean;
      idempotency_key: boolean;
      request_schema: string | null;
      request_examples: Record<string, any>;
      responses: Record<string, { schema: string | null; headers: string[]; examples: Record<string, any>; abbreviated: string[] }>;
      error_codes: Record<string, string[]>;
    }
  >;
  errors: { response: string; example: string; code: string; status: number; body: any }[];
  expanded: Record<string, any>;
}

export const FIXTURES = JSON.parse(readFileSync(FIXTURE_PATH, 'utf8')) as Fixtures;

const clone = <T>(v: T): T => structuredClone(v);

/** A response example from the contract (verbatim). */
export function example(operationId: string, status: number | string = 200, name?: string): any {
  const examples = FIXTURES.operations[operationId]!.responses[String(status)]!.examples;
  const key = name ?? Object.keys(examples)[0]!;
  return clone(examples[key]);
}

/** A response example with abbreviated tensors tiled to their full shape. */
export function expanded(operationId: string, status: number | string, name: string): any {
  return clone(FIXTURES.expanded[`${operationId}/${status}/${name}`]);
}

export function errorBody(code: string): any {
  const e = FIXTURES.errors.find((x) => x.code === code);
  if (!e) throw new Error(code);
  return clone(e.body);
}

export interface Recorded {
  method: string;
  url: URL;
  headers: Headers;
  body: any;
}

type Scripted =
  | { kind: 'response'; status: number; body?: unknown; text?: string; headers: Record<string, string> }
  | { kind: 'throw'; error: unknown };

/** A scripted server behind a mock fetch: queue responses, then inspect recorded requests. */
export class MockServer {
  readonly queue: Scripted[] = [];
  readonly requests: Recorded[] = [];
  readonly sleeps: number[] = [];

  add(body: unknown, status = 200, headers: Record<string, string> = {}, requestId: string | null = 'req-test-1'): this {
    const h = { ...headers };
    if (requestId !== null && !('x-request-id' in h)) h['x-request-id'] = requestId;
    this.queue.push({ kind: 'response', status, body, headers: h });
    return this;
  }

  addText(text: string, status: number, headers: Record<string, string> = {}): this {
    this.queue.push({ kind: 'response', status, text, headers });
    return this;
  }

  fail(error: unknown): this {
    this.queue.push({ kind: 'throw', error });
    return this;
  }

  readonly fetch: FetchLike = (input, init) => {
    const headers = new Headers(init.headers);
    this.requests.push({
      method: init.method ?? 'GET',
      url: new URL(input),
      headers,
      body: typeof init.body === 'string' ? JSON.parse(init.body) : undefined,
    });
    const next = this.queue.shift();
    if (!next) return Promise.reject(new Error(`unexpected request ${init.method ?? 'GET'} ${input}`));
    if (next.kind === 'throw') return Promise.reject(next.error instanceof Error ? next.error : new Error(String(next.error)));
    const body = next.text ?? (next.body === undefined ? null : JSON.stringify(next.body));
    const respHeaders = new Headers(next.headers);
    if (next.body !== undefined && !respHeaders.has('content-type')) respHeaders.set('content-type', 'application/json');
    return Promise.resolve(new Response(body, { status: next.status, headers: respHeaders }));
  };

  get last(): Recorded {
    return this.requests[this.requests.length - 1]!;
  }
}

export const API_KEY = 'test-key-not-real';
export const BASE_URL = 'https://api.test';

export function makeClient(server: MockServer, options: ClientOptions = {}): BreadBowl {
  const client = new BreadBowl({ apiKey: API_KEY, baseURL: BASE_URL, fetch: server.fetch, ...options });
  client._sleep = (ms: number) => {
    server.sleeps.push(ms);
    return Promise.resolve();
  };
  return client;
}

/** A fetch failure the way undici reports a refused connection. */
export function connectionRefused(): TypeError {
  return new TypeError('fetch failed', { cause: Object.assign(new Error('connect ECONNREFUSED'), { code: 'ECONNREFUSED' }) });
}

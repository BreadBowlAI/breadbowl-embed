// Every operation: method, path, query, headers, body and parsed result.
import { describe, expect, it } from 'vitest';
import { BreadBowl, OPERATIONS, Page, VERSION, BreadBowlError, DEFAULT_BASE_URL } from '../src/index.js';
import { API_KEY, BASE_URL, MockServer, example, makeClient } from './helpers.js';

const INDEX_ID = '0b8a3d0e-5c1f-4f7e-9a51-3e2d6c4b1a90';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

interface Case {
  operation: keyof typeof OPERATIONS;
  run: (c: BreadBowl) => PromiseLike<any>;
  status?: number;
  response: any;
  method: string;
  path: string;
  query?: Record<string, string>;
  body?: any;
  idempotent?: boolean;
  authenticated?: boolean;
  check: (r: any) => void;
}

const CASES: Case[] = [
  {
    operation: 'getHealth',
    run: (c) => c.health.live(),
    response: example('getHealth'),
    method: 'GET',
    path: '/healthz',
    authenticated: false,
    check: (r) => expect(r.ok).toBe(true),
  },
  {
    operation: 'getReadiness',
    run: (c) => c.health.ready(),
    response: example('getReadiness'),
    method: 'GET',
    path: '/readyz',
    authenticated: false,
    check: (r) => expect(r.model.revision).toBeTruthy(),
  },
  {
    operation: 'listModels',
    run: (c) => c.models.list({ limit: 5 }),
    response: example('listModels'),
    method: 'GET',
    path: '/v1/models',
    query: { limit: '5' },
    check: (p: Page<any>) => {
      expect(p).toBeInstanceOf(Page);
      expect(p.data[0].dimensions.slots).toBe(16);
      expect(p.hasMore).toBe(false);
    },
  },
  {
    operation: 'getModel',
    run: (c) => c.models.get('breadbowl-embed'),
    response: example('getModel'),
    method: 'GET',
    path: '/v1/models/breadbowl-embed',
    check: (m) => expect(m.search_profiles.map((p: any) => p.name)).toContain(m.default_search_profile),
  },
  {
    operation: 'encode',
    run: (c) => c.encode({ input: 'rotate key', input_type: 'query', return: ['routing', 'attention', 'tokens'], attention_heads: 'mean' }),
    response: example('encode'),
    method: 'POST',
    path: '/v1/encode',
    body: { input: 'rotate key', input_type: 'query', return: ['routing', 'attention', 'tokens'], attention_heads: 'mean' },
    check: (r) => expect(r.data[0].tokens[7]).toMatchObject({ start: 0, end: 6 }),
  },
  {
    operation: 'score',
    run: (c) =>
      c.score({ query: 'refund policy', documents: [{ doc_id: 'a', text: 'Refunds.', metadata: { source: 'help-center' } }, 'Keys.'] }),
    response: example('score', 200, 'basic'),
    method: 'POST',
    path: '/v1/score',
    body: {
      query: 'refund policy',
      documents: [
        { doc_id: 'a', text: 'Refunds.', metadata: { source: 'help-center' } },
        { doc_id: '1', text: 'Keys.' },
      ],
    },
    check: (r) => expect(r.results.map((x: any) => x.doc_id)).toEqual(['a', 'b']),
  },
  {
    operation: 'createIndex',
    run: (c) => c.indexes.create({ name: 'production', chunking: { max_tokens: 256 }, metadata_schema: { year: 'number' } }),
    status: 201,
    response: example('createIndex', 201),
    method: 'POST',
    path: '/v1/indexes',
    body: { name: 'production', chunking: { max_tokens: 256 }, metadata_schema: { year: 'number' } },
    idempotent: true,
    check: (r) => expect(r.id).toBe(INDEX_ID),
  },
  {
    operation: 'listIndexes',
    run: (c) => c.indexes.list({ name: 'production', limit: 1 }),
    response: example('listIndexes'),
    method: 'GET',
    path: '/v1/indexes',
    query: { name: 'production', limit: '1' },
    check: (p: Page<any>) => {
      expect(p.hasMore).toBe(true);
      expect(p.nextCursor).toBeTruthy();
    },
  },
  {
    operation: 'getIndex',
    run: (c) => c.indexes.get(INDEX_ID),
    response: example('getIndex'),
    method: 'GET',
    path: `/v1/indexes/${INDEX_ID}`,
    check: (r) => expect(r.status).toBe('ready'),
  },
  {
    operation: 'deleteIndex',
    run: (c) => c.indexes.delete(INDEX_ID),
    response: example('deleteIndex'),
    method: 'DELETE',
    path: `/v1/indexes/${INDEX_ID}`,
    idempotent: true,
    check: (r) => expect(r.documents_deleted).toBe(1204),
  },
  {
    operation: 'upsertDocuments',
    run: (c) =>
      c.indexes.documents.upsert(INDEX_ID, [
        { doc_id: 'doc-1', text: 'Keys.', metadata: { year: 2026 } },
        { doc_id: 'doc-2', text: 'Refunds.' },
      ]),
    response: example('upsertDocuments'),
    method: 'POST',
    path: `/v1/indexes/${INDEX_ID}/documents`,
    body: { documents: [{ doc_id: 'doc-1', text: 'Keys.', metadata: { year: 2026 } }, { doc_id: 'doc-2', text: 'Refunds.' }] },
    idempotent: true,
    check: (r) => expect(r.failed_documents).toBe(1),
  },
  {
    operation: 'getDocument',
    run: (c) => c.indexes.documents.get(INDEX_ID, 'a/b c', { return_vectors: true, dtype: 'float32' }),
    response: example('getDocument'),
    method: 'GET',
    path: `/v1/indexes/${INDEX_ID}/documents/a%2Fb%20c`,
    query: { return_vectors: 'true', dtype: 'float32' },
    check: (r) => expect(r.chunks[0].char_end).toBe(69),
  },
  {
    operation: 'deleteDocument',
    run: (c) => c.indexes.documents.delete(INDEX_ID, 'doc-1'),
    response: example('deleteDocument'),
    method: 'DELETE',
    path: `/v1/indexes/${INDEX_ID}/documents/doc-1`,
    idempotent: true,
    check: (r) => expect(r.deleted).toBe(true),
  },
  {
    operation: 'searchIndex',
    run: (c) => c.indexes.search(INDEX_ID, { query: 'How do I rotate a key?', top_k: 10, candidate_k: 100, filter: { source: 'docs' } }),
    response: example('searchIndex'),
    method: 'POST',
    path: `/v1/indexes/${INDEX_ID}/search`,
    body: { query: 'How do I rotate a key?', top_k: 10, candidate_k: 100, filter: { source: 'docs' } },
    check: (r) => expect(r.results).toHaveLength(2),
  },
  {
    operation: 'getUsage',
    run: (c) => c.usage.get(),
    response: example('getUsage'),
    method: 'GET',
    path: '/v1/usage',
    check: (r) => expect(r.usage.input_tokens).toBe('1834221'),
  },
];

describe('operations', () => {
  it('cover every operation in the contract', () => {
    expect(CASES.map((c) => c.operation).sort()).toEqual(Object.keys(OPERATIONS).sort());
  });

  it.each(CASES.map((c) => [c.operation, c] as const))('%s', async (_name, c) => {
    const server = new MockServer().add(c.response, c.status ?? 200);
    const result = await c.run(makeClient(server));
    const req = server.last;
    expect(req.method).toBe(c.method);
    expect(req.url.origin).toBe(BASE_URL);
    expect(req.url.pathname).toBe(c.path);
    expect(Object.fromEntries(req.url.searchParams)).toEqual(c.query ?? {});
    expect(req.body).toEqual(c.body);
    expect(req.headers.get('user-agent')).toBe(`breadbowl-typescript/${VERSION}`);
    expect(req.headers.get('accept')).toBe('application/json');
    expect(req.headers.get('authorization')).toBe(c.authenticated === false ? null : `Bearer ${API_KEY}`);
    expect(req.headers.get('content-type')).toBe(c.body === undefined ? null : 'application/json');
    if (c.idempotent) expect(req.headers.get('idempotency-key')).toMatch(UUID);
    else expect(req.headers.get('idempotency-key')).toBeNull();
    expect(result.requestId).toBe('req-test-1');
    c.check(result);
  });
});

describe('response metadata', () => {
  it('is non-enumerable and carries replay information', async () => {
    const server = new MockServer().add(example('deleteIndex'), 200, { 'Idempotent-Replayed': 'true' });
    const res = await makeClient(server).indexes.delete(INDEX_ID, { idempotencyKey: 'my-key-1' });
    expect(server.last.headers.get('idempotency-key')).toBe('my-key-1');
    expect(res.meta).toMatchObject({ status: 200, idempotencyKey: 'my-key-1', idempotentReplayed: true, requestId: 'req-test-1' });
    expect(JSON.parse(JSON.stringify(res))).toEqual(example('deleteIndex'));
    expect(Object.keys(res)).not.toContain('meta');
  });

  it('readiness 503 resolves ok:false and is not retried', async () => {
    const server = new MockServer().add({ ok: false }, 503);
    const res = await makeClient(server).health.ready();
    expect(res.ok).toBe(false);
    expect(server.requests).toHaveLength(1);
  });
});

describe('index handle', () => {
  it('binds every index operation', async () => {
    const server = new MockServer()
      .add(example('searchIndex'))
      .add(example('upsertDocuments'))
      .add(example('getDocument'))
      .add(example('deleteDocument'))
      .add(example('getIndex'))
      .add(example('deleteIndex'));
    const idx = makeClient(server).index(INDEX_ID);
    await idx.search('q');
    expect(server.last.body).toEqual({ query: 'q' });
    await idx.upsert([{ doc_id: 'd', text: 't' }]);
    await idx.getDocument('d');
    await idx.deleteDocument('d');
    await idx.get();
    await idx.delete();
    expect(server.requests.map((r) => r.method)).toEqual(['POST', 'POST', 'GET', 'DELETE', 'GET', 'DELETE']);
  });
});

describe('client options', () => {
  it('reads the key and base URL from the environment', async () => {
    const prev = { key: process.env.BREADBOWL_API_KEY, url: process.env.BREADBOWL_BASE_URL };
    process.env.BREADBOWL_API_KEY = 'from-env';
    process.env.BREADBOWL_BASE_URL = 'https://self-hosted.example:8080/';
    try {
      const server = new MockServer().add(example('getUsage'));
      const c = new BreadBowl({ fetch: server.fetch });
      expect(c.baseURL).toBe('https://self-hosted.example:8080');
      await c.usage.get();
      expect(server.last.headers.get('authorization')).toBe('Bearer from-env');
      expect(server.last.url.host).toBe('self-hosted.example:8080');
    } finally {
      process.env.BREADBOWL_API_KEY = prev.key ?? '';
      process.env.BREADBOWL_BASE_URL = prev.url ?? '';
      if (prev.key === undefined) delete process.env.BREADBOWL_API_KEY;
      if (prev.url === undefined) delete process.env.BREADBOWL_BASE_URL;
    }
  });

  it('requires a key unless apiKey is explicitly null', () => {
    const prev = process.env.BREADBOWL_API_KEY;
    delete process.env.BREADBOWL_API_KEY;
    try {
      expect(() => new BreadBowl({ fetch: new MockServer().fetch })).toThrow(/No API key/);
      expect(new BreadBowl({ apiKey: null, fetch: new MockServer().fetch }).apiKey).toBeNull();
    } finally {
      if (prev !== undefined) process.env.BREADBOWL_API_KEY = prev;
    }
  });

  it('defaults to the contract server URL', () => {
    const prev = process.env.BREADBOWL_BASE_URL;
    delete process.env.BREADBOWL_BASE_URL;
    try {
      expect(new BreadBowl({ apiKey: 'k' }).baseURL).toBe(DEFAULT_BASE_URL.replace(/\/+$/, ''));
    } finally {
      if (prev !== undefined) process.env.BREADBOWL_BASE_URL = prev;
    }
  });

  it('sends default and per-request headers', async () => {
    const server = new MockServer().add(example('getUsage'));
    await makeClient(server, { defaultHeaders: { 'X-Trace': '1' } }).usage.get({ headers: { 'X-Extra': '2' } });
    expect(server.last.headers.get('x-trace')).toBe('1');
    expect(server.last.headers.get('x-extra')).toBe('2');
  });

  it('checks arguments before sending', async () => {
    const c = makeClient(new MockServer());
    expect(() => c.indexes.get('')).toThrow(TypeError);
    expect(() => c.index('')).toThrow(TypeError);
    await expect(
      c.indexes.documents.upsert(INDEX_ID, [{ doc_id: 'a', text: 'b', metadata: { x: Number.NaN } }]),
    ).rejects.toThrow(BreadBowlError);
  });
});

// documents.upsertBatched and the upsert timeout, against a simulated server that chunks
// documents (metadata.chunks chunks each) and enforces a chunk limit per request.
import { describe, expect, it } from 'vitest';
import { BatchUpsertError, BreadBowl, RepresentationUnavailableError, type DocumentInput, type FetchLike } from '../src/index.js';

const INDEX_ID = 'idx-1';

class ChunkingServer {
  readonly batches: string[][] = [];
  readonly keys: (string | null)[] = [];
  constructor(
    readonly limit = 6,
    readonly failDoc: string | null = null,
    readonly breakAfter: number | null = null,
  ) {}

  readonly fetch: FetchLike = (_url, init) => {
    const docs = (JSON.parse(init.body as string) as { documents: DocumentInput[] }).documents;
    this.batches.push(docs.map((d) => d.doc_id));
    this.keys.push(new Headers(init.headers).get('idempotency-key'));
    const json = (status: number, body: unknown, rid: string): Promise<Response> =>
      Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'x-request-id': rid } }));
    if (this.breakAfter !== null && this.batches.length > this.breakAfter) {
      return json(409, { error: { code: 'representation_unavailable', message: 'dtype', details: { representation: { compute_dtype: 'bfloat16' }, served: null } } }, 'r-409');
    }
    const chunks = (d: DocumentInput): number => Number(d.metadata?.chunks ?? 1);
    const requested = docs.reduce((a, d) => a + chunks(d), 0);
    if (requested > this.limit) {
      return json(422, { error: { code: 'chunk_limit_exceeded', message: 'too many', details: { limit: this.limit, requested } } }, 'r-422');
    }
    const outcomes = [...docs].reverse().map((d) =>
      d.doc_id === this.failDoc
        ? { doc_id: d.doc_id, status: 'failed', error: { code: 'encoding_failed', message: 'worker', retryable: true } }
        : { doc_id: d.doc_id, status: 'upserted', content_sha256: '0'.repeat(64), token_count: 10, chunk_count: chunks(d), truncated: false },
    );
    const ok = outcomes.filter((o) => o.status === 'upserted');
    return json(200, {
      index_id: INDEX_ID,
      documents: outcomes,
      upserted_documents: ok.length,
      upserted_chunks: ok.reduce((a, o) => a + (o.chunk_count ?? 0), 0),
      failed_documents: outcomes.length - ok.length,
      usage: { input_tokens: 10 * ok.length, document_count: ok.length, chunk_count: 0, candidate_count: 0, latency_ms: 5 },
    }, `r-${this.batches.length}`);
  };
}

const client = (server: ChunkingServer): BreadBowl => {
  const c = new BreadBowl({ apiKey: 'k', baseURL: 'https://api.test', fetch: server.fetch });
  c._sleep = () => Promise.resolve();
  return c;
};
const docs = (n: number, chunks = 1): DocumentInput[] =>
  Array.from({ length: n }, (_, i) => ({ doc_id: `d${i}`, text: `text ${i}`, metadata: { chunks } }));

describe('upsertBatched', () => {
  it('splits by document count and merges outcomes in input order', async () => {
    const server = new ChunkingServer(1000);
    const res = await client(server).indexes.documents.upsertBatched(INDEX_ID, docs(120));
    expect(server.batches.map((b) => b.length)).toEqual([50, 50, 20]);
    expect(res.documents.map((d) => d.doc_id)).toEqual(docs(120).map((d) => d.doc_id));
    expect(res).toMatchObject({ upsertedDocuments: 120, failedDocuments: 0, upsertedChunks: 120, inputTokens: 1200 });
    expect(res.requestIds).toEqual(['r-1', 'r-2', 'r-3']);
    expect(new Set(server.keys).size).toBe(3);
  });

  it('splits a batch rejected with chunk_limit_exceeded and retries', async () => {
    const server = new ChunkingServer(6);
    const res = await client(server).indexes.documents.upsertBatched(INDEX_ID, docs(10, 2));
    expect(server.batches[0]).toHaveLength(10);
    for (const b of server.batches.slice(1)) expect(b.length * 2).toBeLessThanOrEqual(6);
    expect(res.upsertedDocuments).toBe(10);
    expect(res.documents.map((d) => d.doc_id)).toEqual(docs(10).map((d) => d.doc_id));
    expect(new Set(server.keys).size).toBe(server.keys.length);
  });

  it('reports a single document over the limit as a failed outcome', async () => {
    const server = new ChunkingServer(6);
    const res = await client(server).indexes.documents.upsertBatched(INDEX_ID, [
      { doc_id: 'small', text: 'a', metadata: { chunks: 1 } },
      { doc_id: 'huge', text: 'b', metadata: { chunks: 9 } },
      { doc_id: 'tail', text: 'c', metadata: { chunks: 2 } },
    ]);
    expect(res.documents.map((d) => d.doc_id)).toEqual(['small', 'huge', 'tail']);
    expect(res.documents[1]).toMatchObject({ status: 'failed', error: { code: 'chunk_limit_exceeded', retryable: false } });
    expect(res.documents[1]!.error!.message).toContain('9 chunks');
    expect(res.failed.map((d) => d.doc_id)).toEqual(['huge']);
  });

  it('merges per-document failures', async () => {
    const server = new ChunkingServer(1000, 'd3');
    const res = await client(server).index(INDEX_ID).upsertBatched(docs(5), { batchSize: 2 });
    expect(server.batches.map((b) => b.length)).toEqual([2, 2, 1]);
    expect(res.retryable.map((d) => d.doc_id)).toEqual(['d3']);
  });

  it('stops on a request failure with partial results and pending doc_ids', async () => {
    const server = new ChunkingServer(1000, null, 1);
    const err = await client(server).indexes.documents.upsertBatched(INDEX_ID, docs(7), { batchSize: 3 }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(BatchUpsertError);
    const b = err as BatchUpsertError;
    expect(b.error).toBeInstanceOf(RepresentationUnavailableError);
    expect((b.error as RepresentationUnavailableError).served).toBeNull();
    expect(b.result.documents.map((d) => d.doc_id)).toEqual(['d0', 'd1', 'd2']);
    expect(b.pending).toEqual(['d3', 'd4', 'd5', 'd6']);
    expect(server.batches).toHaveLength(2); // not retried
  });

  it('rejects duplicate doc_ids and bad batch sizes before sending', async () => {
    const server = new ChunkingServer();
    const c = client(server);
    await expect(c.indexes.documents.upsertBatched(INDEX_ID, [docs(1)[0]!, docs(1)[0]!])).rejects.toThrow(/d0/);
    await expect(c.indexes.documents.upsertBatched(INDEX_ID, docs(2), { batchSize: 0 })).rejects.toThrow(RangeError);
    expect(server.batches).toEqual([]);
  });
});

describe('upsert timeout', () => {
  it('defaults to 180 s for upserts only and is configurable', async () => {
    const seen: number[] = [];
    const c = new BreadBowl({ apiKey: 'k', baseURL: 'https://api.test', fetch: new ChunkingServer().fetch });
    expect(c.upsertTimeoutMs).toBe(180_000);
    expect(c.timeoutMs).toBe(60_000);
    expect(new BreadBowl({ apiKey: 'k', fetch: new ChunkingServer().fetch, upsertTimeoutMs: 300_000 }).upsertTimeoutMs).toBe(300_000);
    // The per-attempt timer uses upsertTimeoutMs: a 100 ms default timeout does not cut a 150 ms upsert.
    const slow: FetchLike = async (url, init) => {
      await new Promise((r) => setTimeout(r, 150));
      seen.push(1);
      return new ChunkingServer().fetch(url, init);
    };
    const quick = new BreadBowl({ apiKey: 'k', baseURL: 'https://api.test', fetch: slow, timeoutMs: 100, maxRetries: 0 });
    await expect(quick.indexes.documents.upsert(INDEX_ID, docs(1))).resolves.toMatchObject({ upserted_documents: 1 });
    expect(seen).toEqual([1]);
  });
});

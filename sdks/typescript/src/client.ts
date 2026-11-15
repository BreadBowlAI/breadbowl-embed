// The BreadBowl client: a thin, typed layer over the generated contract types.
import { APIClientCore, attachMeta, type CallSpec, type ClientOptions, type MutationOptions, type RequestOptions, type WithMeta } from './core.js';
import type { OperationId } from './generated/meta.js';
import { ChunkLimitExceededError } from './errors.js';
import { BatchUpsertError, splitBatch, tooLarge, type UpsertBatchResult, makeBatchResult } from './batching.js';
import { Page, PagePromise, type PageBody } from './pagination.js';
import type {
  CreateIndexRequest,
  DeleteDocumentResponse,
  DeleteIndexResponse,
  DocumentInput,
  EncodeRequest,
  EncodeResponse,
  EncodingFormat,
  HealthStatus,
  Index,
  Model,
  NotReadyStatus,
  ReadinessStatus,
  ScoreDocument,
  ScoreRequest,
  ScoreResponse,
  SearchRequest,
  SearchResponse,
  StoredDocument,
  TensorDtype,
  UpsertDocumentsResponse,
  UsageSummary,
} from './types.js';

export interface ListParams {
  limit?: number | undefined;
  cursor?: string | undefined;
}

export interface ListIndexesParams extends ListParams {
  /** Only the index with exactly this name (at most one). */
  name?: string | undefined;
}

export interface GetDocumentParams {
  return_vectors?: boolean | undefined;
  encoding_format?: EncodingFormat | undefined;
  dtype?: TensorDtype | undefined;
}

/** `score` parameters; a plain string document gets its position as `doc_id`. */
export type ScoreParams = Omit<ScoreRequest, 'documents'> & { documents: (ScoreDocument | string)[] };

function requireId(name: string, value: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new TypeError(`${name} must be a non-empty string`);
  }
  return value;
}

function stripUndefined<T extends object>(value: T): T {
  return Object.fromEntries(Object.entries(value).filter(([, v]) => v !== undefined)) as T;
}

abstract class Resource {
  constructor(protected readonly client: BreadBowl) {}

  protected async call<T extends object>(spec: CallSpec, options?: RequestOptions): Promise<WithMeta<T>> {
    const { body, meta } = await this.client.request<T>(spec, options);
    return attachMeta(body, meta);
  }

  protected async mutate<T extends object>(
    spec: Omit<CallSpec, 'idempotencyKey'>,
    options?: MutationOptions,
  ): Promise<WithMeta<T>> {
    const key = options?.idempotencyKey ?? this.client.newIdempotencyKey();
    return this.call<T>({ ...spec, idempotencyKey: key }, options);
  }

  protected paginate<T>(
    operation: OperationId,
    query: Record<string, string | number | undefined>,
    cursor: string | undefined,
    options?: RequestOptions,
  ): PagePromise<T> {
    const fetchPage = async (cur: string | undefined): Promise<Page<T>> => {
      const { body, meta } = await this.client.request<PageBody<T>>(
        { operation, query: { ...query, cursor: cur } },
        options,
      );
      return new Page<T>(body, meta, cur, (next) => fetchPage(next));
    };
    return new PagePromise(() => fetchPage(cursor));
  }
}

export class Models extends Resource {
  /** Models served by this deployment, default first. */
  list(params: ListParams = {}, options?: RequestOptions): PagePromise<Model> {
    return this.paginate<Model>('listModels', { limit: params.limit }, params.cursor, options);
  }

  /** A model by immutable id or alias. */
  get(model: string, options?: RequestOptions): Promise<WithMeta<Model>> {
    return this.call<Model>({ operation: 'getModel', pathParams: { model: requireId('model', model) } }, options);
  }
}

export class Documents extends Resource {
  /**
   * Upsert 1–50 documents (at most `limits.max_chunks_per_upsert` chunks). A 200 can still contain
   * per-document failures: check `failed_documents`. Uses `upsertTimeoutMs` (180 s by default).
   */
  upsert(
    indexId: string,
    documents: DocumentInput[],
    options?: MutationOptions,
  ): Promise<WithMeta<UpsertDocumentsResponse>> {
    if (!Array.isArray(documents)) throw new TypeError('documents must be an array');
    return this.mutate<UpsertDocumentsResponse>(
      {
        operation: 'upsertDocuments',
        pathParams: { index_id: requireId('indexId', indexId) },
        body: { documents: documents.map((d) => stripUndefined(d)) },
      },
      { ...options, timeoutMs: options?.timeoutMs ?? this.client.upsertTimeoutMs },
    );
  }

  /**
   * Upsert any number of documents in requests of up to `batchSize` (≤ 50) documents. A batch
   * rejected with `chunk_limit_exceeded` is split and retried; a single document over the chunk
   * limit becomes a failed outcome. Outcomes are merged in input order. Any other failure throws
   * `BatchUpsertError` with the outcomes so far and the pending `doc_id`s. Every request gets its
   * own Idempotency-Key.
   */
  async upsertBatched(
    indexId: string,
    documents: DocumentInput[],
    options: Omit<MutationOptions, 'idempotencyKey'> & { batchSize?: number | undefined } = {},
  ): Promise<UpsertBatchResult> {
    const { batchSize = 50, ...requestOptions } = options;
    if (!Number.isInteger(batchSize) || batchSize < 1) throw new RangeError('batchSize must be a positive integer');
    const seen = new Map<string, number>();
    documents.forEach((d, i) => {
      const prev = seen.get(d.doc_id);
      if (prev !== undefined) throw new TypeError(`doc_id ${JSON.stringify(d.doc_id)} appears at positions ${prev} and ${i}`);
      seen.set(d.doc_id, i);
    });
    type Batch = [number, DocumentInput][];
    const indexed: Batch = documents.map((d, i) => [i, d]);
    const queue: Batch[] = [];
    for (let i = 0; i < indexed.length; i += batchSize) queue.push(indexed.slice(i, i + batchSize));
    const outcomes = new Map<number, UpsertDocumentsResponse['documents'][number]>();
    const responses: WithMeta<UpsertDocumentsResponse>[] = [];
    for (let batch = queue.shift(); batch; batch = queue.shift()) {
      let response: WithMeta<UpsertDocumentsResponse>;
      try {
        response = await this.upsert(indexId, batch.map(([, d]) => d), requestOptions);
      } catch (err) {
        if (err instanceof ChunkLimitExceededError) {
          const [first] = batch;
          if (batch.length === 1 && first) outcomes.set(first[0], tooLarge(first[1].doc_id, err));
          else queue.unshift(...splitBatch(batch, err));
          continue;
        }
        const partial = makeBatchResult(indexId, documents.length, outcomes, responses);
        const pending = [batch, ...queue].flatMap((b) => b.map(([, d]) => d.doc_id));
        throw new BatchUpsertError(err, partial, pending);
      }
      responses.push(response);
      const byId = new Map(response.documents.map((o) => [o.doc_id, o]));
      for (const [position, doc] of batch) {
        outcomes.set(
          position,
          byId.get(doc.doc_id) ?? {
            doc_id: doc.doc_id,
            status: 'failed',
            error: { code: 'missing_outcome', message: 'no outcome in the response', retryable: true },
          },
        );
      }
    }
    return makeBatchResult(indexId, documents.length, outcomes, responses);
  }

  get(
    indexId: string,
    docId: string,
    params: GetDocumentParams = {},
    options?: RequestOptions,
  ): Promise<WithMeta<StoredDocument>> {
    return this.call<StoredDocument>(
      {
        operation: 'getDocument',
        pathParams: { index_id: requireId('indexId', indexId), doc_id: requireId('docId', docId) },
        query: { return_vectors: params.return_vectors, encoding_format: params.encoding_format, dtype: params.dtype },
      },
      options,
    );
  }

  /** `deleted` is false when the document did not exist. */
  delete(indexId: string, docId: string, options?: MutationOptions): Promise<WithMeta<DeleteDocumentResponse>> {
    return this.mutate<DeleteDocumentResponse>(
      {
        operation: 'deleteDocument',
        pathParams: { index_id: requireId('indexId', indexId), doc_id: requireId('docId', docId) },
      },
      options,
    );
  }
}

export class Indexes extends Resource {
  readonly documents: Documents;

  constructor(client: BreadBowl) {
    super(client);
    this.documents = new Documents(client);
  }

  create(params: CreateIndexRequest, options?: MutationOptions): Promise<WithMeta<Index>> {
    return this.mutate<Index>({ operation: 'createIndex', body: stripUndefined(params) }, options);
  }

  /** Indexes, newest first. */
  list(params: ListIndexesParams = {}, options?: RequestOptions): PagePromise<Index> {
    return this.paginate<Index>('listIndexes', { name: params.name, limit: params.limit }, params.cursor, options);
  }

  get(indexId: string, options?: RequestOptions): Promise<WithMeta<Index>> {
    return this.call<Index>({ operation: 'getIndex', pathParams: { index_id: requireId('indexId', indexId) } }, options);
  }

  delete(indexId: string, options?: MutationOptions): Promise<WithMeta<DeleteIndexResponse>> {
    return this.mutate<DeleteIndexResponse>(
      { operation: 'deleteIndex', pathParams: { index_id: requireId('indexId', indexId) } },
      options,
    );
  }

  /** Search an index. A string is shorthand for `{ query }`. Every attempt is billed. */
  search(indexId: string, params: SearchRequest | string, options?: RequestOptions): Promise<WithMeta<SearchResponse>> {
    const body = typeof params === 'string' ? { query: params } : stripUndefined(params);
    return this.call<SearchResponse>(
      { operation: 'searchIndex', pathParams: { index_id: requireId('indexId', indexId) }, body },
      options,
    );
  }
}

/** Operations bound to one index: `client.index('idx_...')`. */
export class IndexHandle {
  constructor(
    private readonly client: BreadBowl,
    readonly id: string,
  ) {
    requireId('indexId', id);
  }

  get(options?: RequestOptions): Promise<WithMeta<Index>> {
    return this.client.indexes.get(this.id, options);
  }

  delete(options?: MutationOptions): Promise<WithMeta<DeleteIndexResponse>> {
    return this.client.indexes.delete(this.id, options);
  }

  search(params: SearchRequest | string, options?: RequestOptions): Promise<WithMeta<SearchResponse>> {
    return this.client.indexes.search(this.id, params, options);
  }

  upsert(documents: DocumentInput[], options?: MutationOptions): Promise<WithMeta<UpsertDocumentsResponse>> {
    return this.client.indexes.documents.upsert(this.id, documents, options);
  }

  upsertBatched(
    documents: DocumentInput[],
    options?: Omit<MutationOptions, 'idempotencyKey'> & { batchSize?: number | undefined },
  ): Promise<UpsertBatchResult> {
    return this.client.indexes.documents.upsertBatched(this.id, documents, options);
  }

  getDocument(docId: string, params?: GetDocumentParams, options?: RequestOptions): Promise<WithMeta<StoredDocument>> {
    return this.client.indexes.documents.get(this.id, docId, params, options);
  }

  deleteDocument(docId: string, options?: MutationOptions): Promise<WithMeta<DeleteDocumentResponse>> {
    return this.client.indexes.documents.delete(this.id, docId, options);
  }
}

export class Usage extends Resource {
  /** Current-month usage, limits, remaining allowance and the calling key's settings. */
  get(options?: RequestOptions): Promise<WithMeta<UsageSummary>> {
    return this.call<UsageSummary>({ operation: 'getUsage' }, options);
  }
}

export class Health extends Resource {
  /** `GET /healthz` (no key needed). */
  live(options?: RequestOptions): Promise<WithMeta<HealthStatus>> {
    return this.call<HealthStatus>({ operation: 'getHealth' }, options);
  }

  /** `GET /readyz`. Resolves `{ ok: false }` on 503 instead of throwing. Never retried. */
  ready(options?: RequestOptions): Promise<WithMeta<ReadinessStatus | NotReadyStatus>> {
    return this.call<ReadinessStatus | NotReadyStatus>({ operation: 'getReadiness', acceptStatuses: [503] }, options);
  }
}

/**
 * ```ts
 * const client = new BreadBowl(); // BREADBOWL_API_KEY
 * const res = await client.score({ query: 'refund policy', documents: ['Refunds within 30 days.'] });
 * ```
 */
export class BreadBowl extends APIClientCore {
  readonly models: Models;
  readonly indexes: Indexes;
  readonly usage: Usage;
  readonly health: Health;

  constructor(options: ClientOptions = {}) {
    super(options);
    this.models = new Models(this);
    this.indexes = new Indexes(this);
    this.usage = new Usage(this);
    this.health = new Health(this);
  }

  /** Encode 1–50 queries or documents (`return` defaults to `['routing', 'value']`). */
  async encode(params: EncodeRequest, options?: RequestOptions): Promise<WithMeta<EncodeResponse>> {
    const { body, meta } = await this.request<EncodeResponse>(
      { operation: 'encode', body: stripUndefined(params) },
      options,
    );
    return attachMeta(body, meta);
  }

  /** Rank 1–50 transient documents against one query. */
  async score(params: ScoreParams, options?: RequestOptions): Promise<WithMeta<ScoreResponse>> {
    const documents = params.documents.map((d, i) => (typeof d === 'string' ? { doc_id: String(i), text: d } : stripUndefined(d)));
    const { body, meta } = await this.request<ScoreResponse>(
      { operation: 'score', body: { ...stripUndefined(params), documents } satisfies ScoreRequest },
      options,
    );
    return attachMeta(body, meta);
  }

  /** A handle bound to one index. */
  index(indexId: string): IndexHandle {
    return new IndexHandle(this, indexId);
  }
}

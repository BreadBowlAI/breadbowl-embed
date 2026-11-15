// Batched upserts (see Documents.upsertBatched). The API takes at most 50 documents and
// `limits.max_chunks_per_upsert` chunks per request; chunk counts are only known after the
// server tokenizes, so a batch rejected with chunk_limit_exceeded is split into
// ceil(requested / limit) parts (at least two) and retried.
import type { WithMeta } from './core.js';
import { BreadBowlError, type ChunkLimitExceededError } from './errors.js';
import type { DocumentInput, DocumentOutcome, UpsertDocumentsResponse } from './types.js';

export interface UpsertBatchResult {
  indexId: string;
  /** One outcome per input document, in input order. */
  documents: DocumentOutcome[];
  /** Every successful request's response, in the order sent (each with its requestId). */
  responses: WithMeta<UpsertDocumentsResponse>[];
  upsertedDocuments: number;
  failedDocuments: number;
  upsertedChunks: number;
  inputTokens: number;
  failed: DocumentOutcome[];
  /** Failed outcomes whose error says resending the same document may succeed. */
  retryable: DocumentOutcome[];
  requestIds: (string | null)[];
}

/**
 * A request of a batched upsert failed as a whole (after the client's retries). Earlier batches
 * are durable: `result` holds their outcomes and `pending` the doc_ids that were not written.
 */
export class BatchUpsertError extends BreadBowlError {
  readonly error: unknown;
  readonly result: UpsertBatchResult;
  readonly pending: string[];

  constructor(error: unknown, result: UpsertBatchResult, pending: string[]) {
    super(
      `batched upsert stopped with ${pending.length} documents pending (${result.upsertedDocuments} upserted): ${String(
        error instanceof Error ? error.message : error,
      )}`,
      { cause: error },
    );
    this.error = error;
    this.result = result;
    this.pending = pending;
  }
}

type Batch = [number, DocumentInput][];

export function splitBatch(batch: Batch, error: ChunkLimitExceededError): Batch[] {
  let parts = 2;
  if (error.limit && error.requested) parts = Math.max(2, Math.ceil(error.requested / error.limit));
  parts = Math.min(parts, batch.length);
  const size = Math.ceil(batch.length / parts);
  const out: Batch[] = [];
  for (let i = 0; i < batch.length; i += size) out.push(batch.slice(i, i + size));
  return out;
}

export function tooLarge(docId: string, error: ChunkLimitExceededError): DocumentOutcome {
  return {
    doc_id: docId,
    status: 'failed',
    error: {
      code: 'chunk_limit_exceeded',
      message: `this document alone produces ${String(error.requested)} chunks; the limit per request is ${String(
        error.limit,
      )}. Shorten it or use larger chunks.`,
      retryable: false,
    },
  };
}

export function makeBatchResult(
  indexId: string,
  total: number,
  outcomes: Map<number, DocumentOutcome>,
  responses: WithMeta<UpsertDocumentsResponse>[],
): UpsertBatchResult {
  const documents: DocumentOutcome[] = [];
  for (let i = 0; i < total; i++) {
    const outcome = outcomes.get(i);
    if (outcome) documents.push(outcome);
  }
  const failed = documents.filter((d) => d.status !== 'upserted');
  return {
    indexId,
    documents,
    responses,
    upsertedDocuments: documents.length - failed.length,
    failedDocuments: failed.length,
    upsertedChunks: responses.reduce((a, r) => a + r.upserted_chunks, 0),
    inputTokens: responses.reduce((a, r) => a + r.usage.input_tokens, 0),
    failed,
    retryable: failed.filter((d) => d.error?.retryable === true),
    requestIds: responses.map((r) => r.requestId),
  };
}

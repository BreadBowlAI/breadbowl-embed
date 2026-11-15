// Typed errors mapped from the contract's {error: {code, message}} envelope. The shape of
// `details` per code comes from the contract's x-error-details (generated ErrorDetailsByCode).
import type { ErrorDetailsByCode } from './generated/meta.js';
import type { ModelRef, Representation, ValidationIssue } from './types.js';

export class BreadBowlError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** A tensor's data does not match its dtype, shape or encoding. */
export class TensorDecodeError extends BreadBowlError {}

/** The request never produced an HTTP response. */
export class APIConnectionError extends BreadBowlError {}

/** The request timed out (per attempt, see `timeoutMs`). */
export class APITimeoutError extends APIConnectionError {}

/** The caller aborted the request through its `signal`. Never retried. */
export class APIUserAbortError extends BreadBowlError {}

/** A 2xx response whose body is not the JSON the contract promises. */
export class APIResponseError extends BreadBowlError {
  readonly status: number;
  readonly requestId: string | null;
  readonly body: unknown;
  constructor(message: string, init: { status: number; requestId: string | null; body?: unknown }) {
    super(message);
    this.status = init.status;
    this.requestId = init.requestId;
    this.body = init.body;
  }
}

export interface APIStatusErrorInit {
  status: number;
  code: string | null;
  requestId: string | null;
  details?: Record<string, unknown> | undefined;
  issues?: ValidationIssue[] | undefined;
  /** Seconds, from `Retry-After`. */
  retryAfter?: number | null | undefined;
  headers?: Headers | undefined;
  body?: unknown;
}

/** A non-2xx response. `code` is the stable machine-readable error code. */
export class APIStatusError extends BreadBowlError {
  readonly status: number;
  readonly code: string | null;
  /** The server's human-readable message (not stable; do not parse). */
  readonly errorMessage: string;
  readonly requestId: string | null;
  readonly details: Record<string, unknown>;
  readonly issues: ValidationIssue[];
  readonly retryAfter: number | null;
  readonly headers: Headers;
  readonly body: unknown;

  constructor(message: string, init: APIStatusErrorInit) {
    const rid = init.requestId ? ` (request id ${init.requestId})` : '';
    super(`${init.status} ${init.code ?? 'error'}: ${message}${rid}`);
    this.status = init.status;
    this.code = init.code;
    this.errorMessage = message;
    this.requestId = init.requestId;
    this.details = init.details ?? {};
    this.issues = init.issues ?? [];
    this.retryAfter = init.retryAfter ?? null;
    this.headers = init.headers ?? new Headers();
    this.body = init.body;
  }

  /**
   * `details` typed by the contract for `code`, or null when this error has another code or
   * no details. `err.detailsFor('index_not_ready')?.status`.
   */
  detailsFor<C extends keyof ErrorDetailsByCode>(code: C): ErrorDetailsByCode[C] | null {
    if (this.code !== code || Object.keys(this.details).length === 0) return null;
    return this.details as unknown as ErrorDetailsByCode[C];
  }
}

const str = (v: unknown): string | null => (typeof v === 'string' ? v : null);
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const isStrings = (v: unknown): v is string[] => Array.isArray(v) && v.every((x) => typeof x === 'string');
const strs = (v: unknown): string[] | null => (isStrings(v) ? v : null);

export class BadRequestError extends APIStatusError {}
export class AuthenticationError extends APIStatusError {}
export class ApiKeyExpiredError extends AuthenticationError {}
export class PermissionDeniedError extends APIStatusError {}
export class TenantSuspendedError extends PermissionDeniedError {}
export class InsufficientScopeError extends PermissionDeniedError {
  get requiredScope(): string | null {
    return str(this.details.required_scope);
  }
}
export class NotFoundError extends APIStatusError {}
export class ConflictError extends APIStatusError {}
export class IndexNameConflictError extends ConflictError {
  /** The existing index with that name. */
  get indexId(): string | null {
    return str(this.details.index_id);
  }
}
export class IndexNotReadyError extends ConflictError {
  /** `creating` (retried automatically after `Retry-After`) or `error` (delete and recreate). */
  get indexStatus(): 'creating' | 'error' | null {
    const v = this.details.status;
    return v === 'creating' || v === 'error' ? v : null;
  }
}
export class ModelRevisionUnavailableError extends ConflictError {
  /** The model revision the index is pinned to. */
  get model(): ModelRef | null {
    const v = this.details.model as Partial<ModelRef> | undefined;
    return v && typeof v.id === 'string' && typeof v.revision === 'string' ? { id: v.id, revision: v.revision } : null;
  }
}
/**
 * The pinned revision is served with another representation (e.g. another compute dtype). Not
 * retried: it resolves when the deployment serves the pinned representation again; otherwise
 * create a new index and re-ingest.
 */
export class RepresentationUnavailableError extends ConflictError {
  /** The representation the index is pinned to. */
  get representation(): Representation | null {
    const v = this.details.representation;
    return v && typeof v === 'object' ? (v as Representation) : null;
  }
  /** What the deployment serves now (null while it cannot tell). */
  get served(): Representation | null {
    const v = this.details.served;
    return v && typeof v === 'object' ? (v as Representation) : null;
  }
}
export class IdempotencyRequestInProgressError extends ConflictError {}
export class PayloadTooLargeError extends APIStatusError {}
export class UnsupportedMediaTypeError extends APIStatusError {}
export class UnprocessableEntityError extends APIStatusError {}
/** `validation_error`: see `issues`. */
export class RequestValidationError extends UnprocessableEntityError {}
export class InvalidCursorError extends UnprocessableEntityError {}
export class UnsupportedModelError extends UnprocessableEntityError {
  /** Model ids and aliases this deployment serves. */
  get served(): string[] | null {
    return strs(this.details.served);
  }
}
export class UnsupportedSearchProfileError extends UnprocessableEntityError {
  /** Profile names the model offers. */
  get searchProfiles(): string[] | null {
    return strs(this.details.search_profiles);
  }
}
export class UnsupportedFilterError extends UnprocessableEntityError {
  get field(): string | null {
    return str(this.details.field);
  }
  get supportedOperations(): string[] | null {
    return strs(this.details.supported_operations);
  }
  get declaredType(): string | null {
    return str(this.details.declared_type);
  }
}
export class AttentionLimitExceededError extends UnprocessableEntityError {
  get requestedValues(): number | null {
    return num(this.details.requested_values);
  }
  get limit(): number | null {
    return num(this.details.limit);
  }
}
export class DocumentLimitExceededError extends UnprocessableEntityError {
  get limit(): number | null {
    return num(this.details.limit);
  }
  get current(): number | null {
    return num(this.details.current);
  }
  get newDocuments(): number | null {
    return num(this.details.new_documents);
  }
}
/** The documents of one upsert chunk into more than `limits.max_chunks_per_upsert` chunks. */
export class ChunkLimitExceededError extends UnprocessableEntityError {
  get limit(): number | null {
    return num(this.details.limit);
  }
  get requested(): number | null {
    return num(this.details.requested);
  }
}
export class IdempotencyKeyReusedError extends UnprocessableEntityError {}
export class RateLimitError extends APIStatusError {}
export class RateLimitExceededError extends RateLimitError {}
export class ConcurrencyLimitExceededError extends RateLimitError {}
/** The monthly allowance is exhausted; `retryAfter` points at the next month. */
export class UsageLimitExceededError extends RateLimitError {}
/** 500 and any other 5xx except 503. */
export class InternalServerError extends APIStatusError {}
export class ServiceUnavailableError extends APIStatusError {}

type StatusErrorClass = new (message: string, init: APIStatusErrorInit) => APIStatusError;

const BY_CODE: Record<string, StatusErrorClass> = {
  bad_request: BadRequestError,
  authentication_failed: AuthenticationError,
  api_key_expired: ApiKeyExpiredError,
  tenant_suspended: TenantSuspendedError,
  insufficient_scope: InsufficientScopeError,
  not_found: NotFoundError,
  index_name_conflict: IndexNameConflictError,
  index_not_ready: IndexNotReadyError,
  model_revision_unavailable: ModelRevisionUnavailableError,
  representation_unavailable: RepresentationUnavailableError,
  idempotency_request_in_progress: IdempotencyRequestInProgressError,
  payload_too_large: PayloadTooLargeError,
  unsupported_media_type: UnsupportedMediaTypeError,
  validation_error: RequestValidationError,
  invalid_cursor: InvalidCursorError,
  unsupported_model: UnsupportedModelError,
  unsupported_search_profile: UnsupportedSearchProfileError,
  unsupported_filter: UnsupportedFilterError,
  attention_limit_exceeded: AttentionLimitExceededError,
  document_limit_exceeded: DocumentLimitExceededError,
  chunk_limit_exceeded: ChunkLimitExceededError,
  idempotency_key_reused: IdempotencyKeyReusedError,
  rate_limit_exceeded: RateLimitExceededError,
  concurrency_limit_exceeded: ConcurrencyLimitExceededError,
  usage_limit_exceeded: UsageLimitExceededError,
  internal_error: InternalServerError,
  service_unavailable: ServiceUnavailableError,
};

const BY_STATUS: Record<number, StatusErrorClass> = {
  400: BadRequestError,
  401: AuthenticationError,
  403: PermissionDeniedError,
  404: NotFoundError,
  409: ConflictError,
  413: PayloadTooLargeError,
  415: UnsupportedMediaTypeError,
  422: UnprocessableEntityError,
  429: RateLimitError,
  503: ServiceUnavailableError,
};

function statusClass(status: number): StatusErrorClass {
  const cls = BY_STATUS[status];
  if (cls) return cls;
  return status >= 500 ? InternalServerError : APIStatusError;
}

/**
 * The error class for a status and code. A known code wins only when its status matches;
 * an unknown code is handled by its status, as the contract requires.
 */
export function errorClassFor(status: number, code: string | null | undefined): StatusErrorClass {
  const base = statusClass(status);
  const specific = code ? BY_CODE[code] : undefined;
  if (specific && (specific === base || specific.prototype instanceof base)) return specific;
  return base;
}

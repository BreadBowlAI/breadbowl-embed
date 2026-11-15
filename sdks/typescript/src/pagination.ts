// Cursor pagination: {data, has_more, next_cursor}.
import { APIResponseError } from './errors.js';
import type { ResponseMeta } from './core.js';

export interface PageBody<T> {
  data: T[];
  has_more: boolean;
  next_cursor: string | null;
}

/** One page. `for await (const item of page)` yields this page's items and every later page's. */
export class Page<T> implements AsyncIterable<T> {
  readonly data: T[];
  readonly hasMore: boolean;
  readonly nextCursor: string | null;
  readonly meta: ResponseMeta;

  constructor(
    body: PageBody<T>,
    meta: ResponseMeta,
    private readonly cursor: string | undefined,
    private readonly fetchPage: (cursor: string) => Promise<Page<T>>,
  ) {
    this.data = body.data;
    this.hasMore = body.has_more;
    this.nextCursor = body.next_cursor;
    this.meta = meta;
  }

  get requestId(): string | null {
    return this.meta.requestId;
  }

  hasNextPage(): boolean {
    return this.hasMore;
  }

  /** The next page, or null after the last one. */
  async nextPage(): Promise<Page<T> | null> {
    if (!this.hasMore) return null;
    // The contract guarantees next_cursor is a string exactly when has_more is true.
    if (this.nextCursor === null) {
      throw new APIResponseError('server returned has_more: true without a next_cursor', {
        status: this.meta.status,
        requestId: this.meta.requestId,
      });
    }
    if (this.nextCursor === this.cursor) {
      throw new APIResponseError('server returned the same cursor twice; stopping to avoid an endless loop', {
        status: this.meta.status,
        requestId: this.meta.requestId,
      });
    }
    return this.fetchPage(this.nextCursor);
  }

  async *iterPages(): AsyncGenerator<Page<T>, void, undefined> {
    yield this;
    let page = await this.nextPage();
    while (page) {
      yield page;
      page = await page.nextPage();
    }
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<T, void, undefined> {
    for await (const page of this.iterPages()) {
      yield* page.data;
    }
  }
}

/**
 * What `list()` returns: `await` it for the first page, or `for await` over it for every item
 * across pages.
 */
export class PagePromise<T> implements PromiseLike<Page<T>>, AsyncIterable<T> {
  private promise: Promise<Page<T>> | null = null;

  constructor(private readonly first: () => Promise<Page<T>>) {}

  private start(): Promise<Page<T>> {
    this.promise ??= this.first();
    return this.promise;
  }

  then<R1 = Page<T>, R2 = never>(
    onfulfilled?: ((value: Page<T>) => R1 | PromiseLike<R1>) | null,
    onrejected?: ((reason: unknown) => R2 | PromiseLike<R2>) | null,
  ): Promise<R1 | R2> {
    return this.start().then(onfulfilled, onrejected);
  }

  catch<R = never>(onrejected?: ((reason: unknown) => R | PromiseLike<R>) | null): Promise<Page<T> | R> {
    return this.start().catch(onrejected);
  }

  finally(onfinally?: (() => void) | null): Promise<Page<T>> {
    return this.start().finally(onfinally);
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<T, void, undefined> {
    const page = await this.start();
    yield* page;
  }
}

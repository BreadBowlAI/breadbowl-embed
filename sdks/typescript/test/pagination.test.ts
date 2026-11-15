import { describe, expect, it } from 'vitest';
import { APIResponseError, Page } from '../src/index.js';
import { MockServer, example, makeClient } from './helpers.js';

function page(names: string[], cursor: string | null): unknown {
  const template = example('listIndexes').data[0];
  return { data: names.map((n) => ({ ...template, id: `id-${n}`, name: n })), has_more: cursor !== null, next_cursor: cursor };
}

function threePages(server: MockServer): void {
  server.add(page(['a', 'b'], 'c1'), 200, {}, 'r1');
  server.add(page(['c', 'd'], 'c2'), 200, {}, 'r2');
  server.add(page(['e'], null), 200, {}, 'r3');
}

describe('pagination', () => {
  it('for await walks every page and keeps filters', async () => {
    const server = new MockServer();
    threePages(server);
    const names: string[] = [];
    for await (const index of makeClient(server).indexes.list({ limit: 2, name: 'x' })) names.push(index.name);
    expect(names).toEqual(['a', 'b', 'c', 'd', 'e']);
    expect(server.requests.map((r) => Object.fromEntries(r.url.searchParams))).toEqual([
      { name: 'x', limit: '2' },
      { name: 'x', limit: '2', cursor: 'c1' },
      { name: 'x', limit: '2', cursor: 'c2' },
    ]);
  });

  it('await gives the first page; nextPage and iterPages', async () => {
    const server = new MockServer();
    threePages(server);
    const first = await makeClient(server).indexes.list({ limit: 2 });
    expect(first).toBeInstanceOf(Page);
    expect(first.requestId).toBe('r1');
    expect(first.data.map((i) => i.name)).toEqual(['a', 'b']);
    const second = await first.nextPage();
    const ids: (string | null)[] = [];
    for await (const p of second!.iterPages()) ids.push(p.requestId);
    expect(ids).toEqual(['r2', 'r3']);
  });

  it('starting cursor and last page', async () => {
    const server = new MockServer().add(page(['z'], null));
    const p = await makeClient(server).models.list({ cursor: 'abc' });
    expect(Object.fromEntries(server.last.url.searchParams)).toEqual({ cursor: 'abc' });
    expect(p.hasNextPage()).toBe(false);
    expect(await p.nextPage()).toBeNull();
  });

  it('has_more without next_cursor violates the contract and is an error', async () => {
    const server = new MockServer().add({ data: [], has_more: true, next_cursor: null });
    const items: unknown[] = [];
    const consume = async (): Promise<void> => {
      for await (const x of makeClient(server).indexes.list()) items.push(x);
    };
    await expect(consume()).rejects.toThrow(/without a next_cursor/);
  });

  it('a repeated cursor stops with an error', async () => {
    const server = new MockServer().add(page(['a'], 'same')).add(page(['b'], 'same'));
    const seen: unknown[] = [];
    const consume = async (): Promise<void> => {
      for await (const item of makeClient(server).indexes.list()) seen.push(item);
    };
    await expect(consume()).rejects.toBeInstanceOf(APIResponseError);
  });

  it('the first request is made once even when awaited twice', async () => {
    const server = new MockServer().add(page(['a'], null));
    const pending = makeClient(server).indexes.list();
    await pending;
    await pending;
    expect(server.requests).toHaveLength(1);
  });
});

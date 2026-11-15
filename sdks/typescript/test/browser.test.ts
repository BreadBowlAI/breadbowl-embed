// Browser safety: an account key is refused in a page unless explicitly allowed.
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { BreadBowl, BreadBowlError } from '../src/index.js';
import { MockServer, example } from './helpers.js';

const g = globalThis as { window?: unknown; document?: unknown };

describe('in a browser', () => {
  beforeEach(() => {
    g.window = globalThis;
    g.document = {};
  });
  afterEach(() => {
    delete g.window;
    delete g.document;
  });

  it('refuses an API key by default', () => {
    expect(() => new BreadBowl({ apiKey: 'k', fetch: new MockServer().fetch })).toThrow(BreadBowlError);
    expect(() => new BreadBowl({ apiKey: 'k', fetch: new MockServer().fetch })).toThrow(/browser/);
  });

  it('never reads the key from the environment', () => {
    process.env.BREADBOWL_API_KEY = 'leaked';
    try {
      expect(() => new BreadBowl({ fetch: new MockServer().fetch })).toThrow(/No API key/);
    } finally {
      delete process.env.BREADBOWL_API_KEY;
    }
  });

  it('works through a proxy with apiKey: null and sends no Authorization or User-Agent', async () => {
    const server = new MockServer().add(example('getUsage'));
    const c = new BreadBowl({ apiKey: null, baseURL: 'https://my-app.example/bb', fetch: server.fetch });
    await c.usage.get();
    expect(server.last.headers.get('authorization')).toBeNull();
    expect(server.last.headers.get('user-agent')).toBeNull();
    expect(server.last.url.pathname).toBe('/bb/v1/usage');
  });

  it('allows a key only with dangerouslyAllowBrowser', () => {
    expect(new BreadBowl({ apiKey: 'k', dangerouslyAllowBrowser: true, fetch: new MockServer().fetch }).apiKey).toBe('k');
  });
});

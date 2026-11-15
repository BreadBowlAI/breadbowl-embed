// The SDK agrees with the OpenAPI contract, at runtime (ajv, JSON Schema 2020-12) and at
// compile time (the client's parameter and result types are the generated schema types).
import { createHash } from 'node:crypto';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { Ajv2020 } from 'ajv/dist/2020.js';
import { describe, expect, expectTypeOf, it } from 'vitest';
import { parse } from 'yaml';
import {
  APIStatusError,
  BreadBowl,
  ERROR_DETAILS,
  OPERATIONS,
  SPEC_SHA256,
  SPEC_VERSION,
  errorClassFor,
  type components,
} from '../src/index.js';
import { FIXTURES, FIXTURE_PATH, HAS_SPEC, MockServer, SIBLING_FIXTURE, SPEC_PATH, makeClient } from './helpers.js';

type S = components['schemas'];
const INDEX_ID = '0b8a3d0e-5c1f-4f7e-9a51-3e2d6c4b1a90';

const suite = HAS_SPEC ? describe : describe.skip;

suite('contract agreement', () => {
  const raw = readFileSync(SPEC_PATH);
  const spec = parse(raw.toString('utf8')) as Record<string, any>;
  const sha = createHash('sha256').update(raw).digest('hex');
  const ajv = new Ajv2020({ strict: false, validateFormats: false, validateSchema: false, allErrors: true });
  ajv.addSchema(spec, 'urn:bb-embed:v1');
  const validator = (pointer: string) => ajv.compile({ $ref: `urn:bb-embed:v1#${pointer}` });
  const check = (pointer: string, value: unknown): void => {
    const validate = validator(pointer);
    const ok = validate(value);
    expect(ok, `${pointer}: ${JSON.stringify(validate.errors)}`).toBe(true);
  };
  const resolvePointer = (pointer: string): any => {
    let node: any = spec;
    for (const part of pointer.replace(/^#?\//, '').split('/')) node = node[part];
    return node.$ref ? resolvePointer(node.$ref) : node;
  };

  it('validation is not vacuous', () => {
    const encode = validator('/components/schemas/EncodeRequest');
    expect(encode({ input: 'x', input_type: 'query' })).toBe(true);
    expect(encode({ input: 'x', input_type: 'query', return_components: true })).toBe(false);
    expect(encode({ input: 'x', input_type: 'passage' })).toBe(false);
    const usage = validator('/components/schemas/UsageSummary');
    const body = structuredClone(FIXTURES.operations.getUsage!.responses['200']!.examples.default);
    body.usage.input_tokens = 1834221;
    expect(usage(body)).toBe(false);
  });

  describe('freshness', () => {
    it('fixtures were generated from this contract', () => {
      expect(FIXTURES.spec.sha256, 'run `pnpm run generate:fixtures`').toBe(sha);
    });
    it('generated types were generated from this contract', () => {
      expect(SPEC_SHA256, 'run `pnpm run generate:types`').toBe(sha);
      expect(readFileSync(resolve(SPEC_PATH, '../../../sdks/typescript/src/generated/openapi.ts'), 'utf8')).toContain(sha);
      expect(SPEC_VERSION).toBe(spec.info.version);
    });
    it('both SDKs share byte-identical fixtures', () => {
      if (!existsSync(SIBLING_FIXTURE)) return;
      expect(readFileSync(SIBLING_FIXTURE)).toEqual(readFileSync(FIXTURE_PATH));
    });
    it('operation table matches the fixture generator', () => {
      expect(Object.keys(OPERATIONS).sort()).toEqual(Object.keys(FIXTURES.operations).sort());
      for (const [id, op] of Object.entries(OPERATIONS)) {
        const f = FIXTURES.operations[id]!;
        expect([op.method, op.path, op.idempotencyKey, op.authenticated]).toEqual([f.method, f.path, f.idempotency_key, f.authenticated]);
      }
    });
  });

  describe('errors', () => {
    // errors.test.ts maps every code of the union to its class; here the per-operation tables.
    it('per-operation x-error-codes match the spec and the union the classes are tested against', () => {
      for (const [id, op] of Object.entries(OPERATIONS)) {
        expect(op.errorCodes, id).toEqual(spec.paths[op.path][op.method.toLowerCase()]['x-error-codes'] ?? {});
        expect(op.errorCodes, id).toEqual(FIXTURES.operations[id]!.error_codes);
        for (const [status, codes] of Object.entries(op.errorCodes)) {
          for (const code of codes) expect(FIXTURES.error_codes[status], `${id} ${status}`).toContain(code);
          expect(Object.keys(spec.paths[op.path][op.method.toLowerCase()].responses), id).toContain(status);
        }
      }
    });

    // A plausible value for a schema, to drive the typed accessors.
    const sample = (schema: any): unknown => {
      const s = schema.$ref ? resolvePointer(schema.$ref) : schema;
      if (s.oneOf) return sample(s.oneOf[0]);
      if (s.enum) return s.enum[0];
      if (s.examples?.length) return s.examples[0];
      if (typeof s.pattern === 'string' && s.pattern.startsWith('^sha256:')) return `sha256:${'0'.repeat(64)}`;
      if (s.const !== undefined) return s.const;
      if (s.type === 'string') return 'x';
      if (s.type === 'integer' || s.type === 'number') return 3;
      if (s.type === 'array') return [sample(s.items)];
      if (s.type === 'object') return Object.fromEntries(Object.entries(s.properties ?? {}).map(([k, v]) => [k, sample(v)]));
      throw new Error(JSON.stringify(s));
    };
    const camel = (name: string): string => (name === 'status' ? 'indexStatus' : name.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase()));
    const statusOf = (code: string): number =>
      Number(Object.entries(FIXTURES.error_codes).find(([, codes]) => codes.includes(code))![0]);

    it('x-error-details: every field has a typed accessor and detailsFor returns the details', () => {
      expect(ERROR_DETAILS).toEqual(spec.components.schemas.Error['x-error-details']);
      expect(FIXTURES.error_details).toEqual(ERROR_DETAILS);
      for (const [code, schemaName] of Object.entries(ERROR_DETAILS)) {
        const details = sample({ $ref: `#/components/schemas/${schemaName}` }) as Record<string, unknown>;
        const status = statusOf(code);
        const Cls = errorClassFor(status, code);
        const err = new Cls('m', { status, code, requestId: null, details });
        expect(err.detailsFor(code as keyof typeof ERROR_DETAILS), code).toEqual(details);
        for (const [field, value] of Object.entries(details)) {
          const getter = camel(field);
          expect(getter in err, `${Cls.name}.${getter}`).toBe(true);
          expect((err as unknown as Record<string, unknown>)[getter], `${Cls.name}.${getter}`).toEqual(value);
        }
        check(`/components/schemas/${schemaName}`, details);
      }
      const other = new APIStatusError('m', { status: 409, code: 'index_not_ready', requestId: null });
      expect(other.detailsFor('index_name_conflict')).toBeNull();
    });
  });

  describe('responses', () => {
    const examples = Object.entries(FIXTURES.operations).flatMap(([id, op]) =>
      Object.entries(op.responses).flatMap(([status, r]) =>
        Object.entries(r.examples).map(([name, body]) => [`${id} ${status} ${name}`, r.schema!, body] as const),
      ),
    );
    it.each(examples)('%s validates against its schema', (_name, schema, body) => {
      check(schema.replace(/^#/, ''), body);
    });
    it('expanded examples still validate', () => {
      for (const [key, body] of Object.entries(FIXTURES.expanded)) {
        const [id, status] = key.split('/') as [string, string];
        check(FIXTURES.operations[id]!.responses[status]!.schema!.replace(/^#/, ''), body);
      }
    });
    it('the client returns response bodies unchanged', async () => {
      for (const [id, status] of [['getModel', 200], ['getUsage', 200], ['getDocument', 200]] as const) {
        const body = FIXTURES.operations[id]!.responses[String(status)]!.examples.default;
        const server = new MockServer().add(body, status);
        const c = makeClient(server);
        const result = id === 'getModel' ? await c.models.get('m') : id === 'getUsage' ? await c.usage.get() : await c.indexes.documents.get(INDEX_ID, 'd');
        expect(JSON.parse(JSON.stringify(result))).toEqual(body);
      }
    });
  });

  describe('requests', () => {
    const call = async (c: BreadBowl, id: string, ex: any): Promise<void> => {
      if (id === 'encode') await c.encode(ex);
      else if (id === 'score') await c.score(ex);
      else if (id === 'createIndex') await c.indexes.create(ex);
      else if (id === 'upsertDocuments') await c.indexes.documents.upsert(INDEX_ID, ex.documents);
      else if (id === 'searchIndex') await c.indexes.search(INDEX_ID, ex);
      else throw new Error(id);
    };
    const reqExamples = Object.entries(FIXTURES.operations).flatMap(([id, op]) =>
      Object.entries(op.request_examples).map(([name, ex]) => [`${id} ${name}`, id, ex] as const),
    );
    it.each(reqExamples)('contract example %s is exactly what the SDK sends', async (_n, id, ex) => {
      const op = FIXTURES.operations[id]!;
      const status = id === 'createIndex' ? '201' : '200';
      const server = new MockServer().add(Object.values(op.responses[status]!.examples)[0], Number(status));
      await call(makeClient(server), id, structuredClone(ex));
      expect(server.last.body).toEqual(ex);
      check(op.request_schema!.replace(/^#/, ''), server.last.body);
    });

    it('requests with every option validate and exercise every property', async () => {
      const server = new MockServer();
      for (const [id, status] of [['encode', 200], ['score', 200], ['createIndex', 201], ['upsertDocuments', 200], ['searchIndex', 200]] as const) {
        server.add(Object.values(FIXTURES.operations[id]!.responses[String(status)]!.examples)[0], status);
      }
      const c = makeClient(server);
      await c.encode({
        model: 'm', input: ['a', 'b'], input_type: 'document', instruction: 'Find it.',
        return: ['routing', 'value', 'attention', 'tokens'], encoding_format: 'float', dtype: 'float32', attention_heads: 'all',
      });
      await c.score({
        model: 'm', query: 'q', instruction: 'Find it.', documents: [{ doc_id: 'a', text: 't', metadata: { k: 1, flag: true, s: 'x' } }, 'u'],
        explain: true, return_attention: true, attention_heads: 'mean', encoding_format: 'base64', dtype: 'float16',
      });
      await c.indexes.create({
        name: 'n', model: 'm', search_profile: 'p', text_retention: 'none',
        chunking: { strategy: 'token_window', max_tokens: 128, overlap_tokens: 16, minimum_chunk_tokens: 8 },
        metadata_schema: { a: 'string', b: 'number', c: 'boolean' },
      });
      await c.indexes.documents.upsert(INDEX_ID, [{ doc_id: 'd', text: 't', metadata: { year: 2026 } }]);
      await c.indexes.search(INDEX_ID, {
        query: 'q', instruction: 'i', top_k: 3, candidate_k: 50, filter: { year: 2026 }, explain: true,
        return_attention: true, return_vectors: true, attention_heads: 'mean', encoding_format: 'float', dtype: 'float32',
      });
      const byPath: Record<string, string> = {
        '/v1/encode': 'EncodeRequest',
        '/v1/score': 'ScoreRequest',
        '/v1/indexes': 'CreateIndexRequest',
        [`/v1/indexes/${INDEX_ID}/documents`]: 'UpsertDocumentsRequest',
        [`/v1/indexes/${INDEX_ID}/search`]: 'SearchRequest',
      };
      for (const req of server.requests) check(`/components/schemas/${byPath[req.url.pathname]!}`, req.body);
      for (const [path, name] of Object.entries(byPath)) {
        const props = Object.keys(resolvePointer(`/components/schemas/${name}`).properties as object).sort();
        const sent = [...new Set(server.requests.filter((r) => r.url.pathname === path).flatMap((r) => Object.keys(r.body)))].sort();
        expect(sent, name).toEqual(props);
      }
    });

    it('query parameters validate against their schemas', async () => {
      const server = new MockServer()
        .add(FIXTURES.operations.getDocument!.responses['200']!.examples.default)
        .add(FIXTURES.operations.listIndexes!.responses['200']!.examples.default);
      const c = makeClient(server);
      await c.indexes.documents.get(INDEX_ID, 'doc-1', { return_vectors: false, encoding_format: 'float', dtype: 'float16' });
      await c.indexes.list({ name: 'prod', limit: 100, cursor: 'abc' });
      const params: Record<string, any> = {};
      for (const path of ['/v1/indexes/{index_id}/documents/{doc_id}', '/v1/indexes']) {
        for (const p of spec.paths[path].get.parameters as any[]) {
          const param = p.$ref ? resolvePointer(p.$ref) : p;
          if (param.in === 'query') params[param.name] = param.schema;
        }
      }
      for (const req of server.requests) {
        for (const [name, value] of req.url.searchParams) {
          const schema = params[name].$ref ? resolvePointer(params[name].$ref) : params[name];
          const typed = schema.type === 'boolean' ? value === 'true' : schema.type === 'integer' ? Number(value) : value;
          if (schema.type === 'boolean') expect(['true', 'false']).toContain(value);
          const validate = ajv.compile({ ...schema, $id: undefined });
          expect(validate(typed), `${name}=${value}`).toBe(true);
        }
      }
    });
  });

  it('client parameter and result types are the generated schema types (compile time)', () => {
    expectTypeOf<Parameters<BreadBowl['encode']>[0]>().toEqualTypeOf<S['EncodeRequest']>();
    expectTypeOf<Parameters<BreadBowl['indexes']['create']>[0]>().toEqualTypeOf<S['CreateIndexRequest']>();
    expectTypeOf<Parameters<BreadBowl['indexes']['documents']['upsert']>[1]>().toEqualTypeOf<S['DocumentInput'][]>();
    expectTypeOf<Parameters<BreadBowl['indexes']['search']>[1]>().toEqualTypeOf<S['SearchRequest'] | string>();
    expectTypeOf<Awaited<ReturnType<BreadBowl['encode']>>>().toExtend<S['EncodeResponse']>();
    expectTypeOf<Awaited<ReturnType<BreadBowl['score']>>>().toExtend<S['ScoreResponse']>();
    expectTypeOf<Awaited<ReturnType<BreadBowl['indexes']['search']>>>().toExtend<S['SearchResponse']>();
    expectTypeOf<Awaited<ReturnType<BreadBowl['indexes']['get']>>>().toExtend<S['Index']>();
    expectTypeOf<Awaited<ReturnType<BreadBowl['indexes']['documents']['get']>>>().toExtend<S['StoredDocument']>();
    expectTypeOf<Awaited<ReturnType<BreadBowl['usage']['get']>>>().toExtend<S['UsageSummary']>();
    expectTypeOf<Awaited<ReturnType<BreadBowl['indexes']['list']>>['data'][number]>().toEqualTypeOf<S['Index']>();
    expectTypeOf<Awaited<ReturnType<BreadBowl['models']['list']>>['data'][number]>().toEqualTypeOf<S['Model']>();
  });
});

#!/usr/bin/env node
// Generates the shared contract fixtures for BOTH SDKs from the OpenAPI examples.
//
//   node scripts/generate-fixtures.mjs          # write both copies
//   node scripts/generate-fixtures.mjs --check  # exit 1 if either copy is stale
//
// Outputs (byte-identical):
//   sdks/typescript/test/fixtures/contract.json
//   sdks/python/tests/fixtures/contract.json
//
// The fixtures hold every request and response example of the public contract, verbatim, plus
// an "expanded" copy of the examples the contract marks `x-abbreviated: true` (their tensor
// bytes are tiled to `product(shape) × itemsize` so decoders can be tested on contract-shaped
// data). Generation fails if an example that is NOT marked abbreviated is inconsistent.
// Both SDK test suites check `spec.sha256` against the current contract, so the fixtures
// cannot silently drift from it.
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'yaml';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../../..');
const specPath = resolve(repoRoot, 'sdks/openapi/bb-embed.v1.yaml');
const outputs = [
  resolve(repoRoot, 'sdks/typescript/test/fixtures/contract.json'),
  resolve(repoRoot, 'sdks/python/tests/fixtures/contract.json'),
];

const raw = readFileSync(specPath);
const spec = parse(raw.toString('utf8'));

function deref(node) {
  let current = node;
  const seen = new Set();
  while (current && typeof current === 'object' && typeof current.$ref === 'string') {
    if (seen.has(current.$ref)) throw new Error(`circular $ref ${current.$ref}`);
    seen.add(current.$ref);
    const parts = current.$ref.replace(/^#\//, '').split('/');
    current = parts.reduce((acc, key) => acc?.[key.replace(/~1/g, '/').replace(/~0/g, '~')], spec);
    if (current === undefined) throw new Error(`unresolved ${[...seen].pop()}`);
  }
  return current;
}

function mediaExamples(media) {
  if (!media) return {};
  if (media.example !== undefined) return { default: media.example };
  const out = {};
  for (const [name, ex] of Object.entries(media.examples ?? {})) {
    out[name] = deref(ex).value;
  }
  return out;
}

function abbreviatedNames(media) {
  return Object.entries(media?.examples ?? {})
    .filter(([, ex]) => deref(ex)['x-abbreviated'] === true)
    .map(([name]) => name);
}

function serverUrl() {
  const server = spec.servers?.[0];
  if (!server) return null;
  return server.url.replace(/\{(\w+)\}/g, (_, name) => server.variables?.[name]?.default ?? '');
}


// ------------------------------------------------------------------ operations
const METHODS = ['get', 'post', 'put', 'patch', 'delete'];
const operations = {};
for (const [path, item] of Object.entries(spec.paths)) {
  const shared = (item.parameters ?? []).map(deref);
  for (const method of METHODS) {
    const op = item[method];
    if (!op) continue;
    const params = [...shared, ...(op.parameters ?? []).map(deref)];
    const security = op.security ?? spec.security ?? [];
    const scopes = security.flatMap((req) => Object.values(req).flat());
    const responses = {};
    const errorStatuses = [];
    for (const [status, response] of Object.entries(op.responses)) {
      const resolved = deref(response);
      const media = resolved.content?.['application/json'];
      const schemaRef = media?.schema?.$ref ?? null;
      responses[status] = {
        schema: schemaRef,
        headers: Object.keys(resolved.headers ?? {}),
        examples: mediaExamples(media),
        abbreviated: abbreviatedNames(media),
      };
      if (Number(status) >= 400) errorStatuses.push(Number(status));
    }
    const body = op.requestBody?.content?.['application/json'];
    operations[op.operationId] = {
      method: method.toUpperCase(),
      path,
      authenticated: security.length > 0,
      scopes,
      idempotency_key: params.some((p) => p.in === 'header' && p.name === 'Idempotency-Key'),
      path_params: params.filter((p) => p.in === 'path').map((p) => p.name),
      query_params: params.filter((p) => p.in === 'query').map((p) => p.name),
      request_schema: body?.schema?.$ref ?? null,
      request_examples: mediaExamples(body),
      responses,
      error_statuses: errorStatuses,
      // x-error-codes: the codes this operation can return, per status.
      error_codes: op['x-error-codes'] ?? {},
    };
  }
}

// ------------------------------------------------------------------ error examples
const errors = [];
for (const [name, response] of Object.entries(spec.components.responses)) {
  const media = response.content?.['application/json'];
  for (const [exampleName, body] of Object.entries(mediaExamples(media))) {
    errors.push({ response: name, example: exampleName, code: body.error.code, body });
  }
}
// Attach the HTTP status each shared error response is used with.
const statusOfResponse = {};
for (const op of Object.values(spec.paths).flatMap((item) => METHODS.map((m) => item[m]).filter(Boolean))) {
  for (const [status, response] of Object.entries(op.responses)) {
    if (response.$ref) statusOfResponse[response.$ref.split('/').pop()] = Number(status);
  }
}
for (const e of errors) e.status = statusOfResponse[e.response] ?? null;

// ------------------------------------------------------------------ expanded tensors
const ITEMSIZE = { float16: 2, float32: 4 };
function isTensor(v) {
  return v && typeof v === 'object' && typeof v.encoding === 'string' && Array.isArray(v.shape) && 'data' in v;
}
function expand(value) {
  let changed = false;
  const walk = (v) => {
    if (Array.isArray(v)) return v.map(walk);
    if (!v || typeof v !== 'object') return v;
    if (isTensor(v)) {
      const size = v.shape.reduce((a, b) => a * b, 1);
      if (v.encoding === 'base64') {
        const bytes = Buffer.from(v.data, 'base64');
        const need = size * ITEMSIZE[v.dtype];
        if (bytes.length === need) return v;
        changed = true;
        const out = Buffer.alloc(need);
        for (let i = 0; i < need; i++) out[i] = bytes[i % bytes.length];
        return { ...v, data: out.toString('base64') };
      }
      const flat = [v.data].flat(Infinity);
      if (flat.length === size) return v;
      changed = true;
      const tiled = Array.from({ length: size }, (_, i) => flat[i % flat.length]);
      const nest = (values, dims) => {
        if (dims.length === 1) return values;
        const step = values.length / dims[0];
        return Array.from({ length: dims[0] }, (_, i) => nest(values.slice(i * step, (i + 1) * step), dims.slice(1)));
      };
      return { ...v, data: nest(tiled, v.shape) };
    }
    return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, walk(x)]));
  };
  const result = walk(value);
  return changed ? result : null;
}
const expanded = {};
for (const [opId, op] of Object.entries(operations)) {
  for (const [status, response] of Object.entries(op.responses)) {
    for (const [name, example] of Object.entries(response.examples)) {
      const full = expand(example);
      const marked = response.abbreviated.includes(name);
      if (full && !marked) {
        throw new Error(`${opId} ${status} ${name}: tensor data is inconsistent but the example is not marked x-abbreviated`);
      }
      if (marked) expanded[`${opId}/${status}/${name}`] = full ?? example;
    }
  }
}

// Union of every operation's x-error-codes, per status.
const errorCodes = {};
for (const op of Object.values(operations)) {
  for (const [status, codes] of Object.entries(op.error_codes)) {
    errorCodes[status] = [...new Set([...(errorCodes[status] ?? []), ...codes])];
  }
}
const sortedErrorCodes = Object.fromEntries(Object.entries(errorCodes).sort(([a], [b]) => Number(a) - Number(b)));

const fixture = {
  $comment:
    'GENERATED by sdks/typescript/scripts/generate-fixtures.mjs from the OpenAPI contract. Do not edit; regenerate.',
  spec: {
    path: relative(repoRoot, specPath),
    sha256: createHash('sha256').update(raw).digest('hex'),
    version: spec.info.version,
    title: spec.info.title,
    license: spec.info.license ?? null,
    server_url: serverUrl(),
  },
  error_codes: sortedErrorCodes,
  // x-error-details on the Error schema: code -> schema name of its `details`.
  error_details: spec.components.schemas.Error['x-error-details'] ?? {},
  operations,
  errors,
  expanded,
};

const text = `${JSON.stringify(fixture, null, 2)}\n`;
const check = process.argv.includes('--check');
let stale = false;
for (const out of outputs) {
  let current = null;
  try {
    current = readFileSync(out, 'utf8');
  } catch {
    current = null;
  }
  if (current === text) continue;
  if (check) {
    console.error(`stale: ${relative(repoRoot, out)}`);
    stale = true;
  } else {
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(out, text);
    console.log(`wrote ${relative(repoRoot, out)}`);
  }
}
if (stale) {
  console.error('Run `node scripts/generate-fixtures.mjs` in sdks/typescript.');
  process.exit(1);
}
if (check) console.log('fixtures up to date');

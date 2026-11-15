// tokenAttribution, code-point offsets, counters.
import { describe, expect, it } from 'vitest';
import {
  BreadBowlError,
  CodePointIndex,
  attentionMatrix,
  encodeTensor,
  explanationScores,
  parseCounter,
  parseCounters,
  sliceCodePoints,
  tokenAttribution,
  tokenSpans,
  type Explanation,
  type Token,
  type UsageSummary,
} from '../src/index.js';
import { example, expanded } from './helpers.js';

const SCORING = { tau_k: 0.02, tau_v: 0.1, routing_weight: 0.75, value_weight: 0.25 };
const sum = (xs: number[]): number => xs.reduce((a, b) => a + b, 0);
const softmax = (row: number[], tau: number): number[] => {
  const e = row.map((v) => Math.exp(v / tau));
  return e.map((x) => x / sum(e));
};

function makeExplanation(): Explanation {
  const sim = [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1]];
  const va = sim.map((r) => softmax(r, SCORING.tau_v));
  const rc = sim.map((r) => SCORING.tau_k * (Math.log(sum(r.map((v) => Math.exp(v / SCORING.tau_k)))) - Math.log(3)));
  return {
    slot_similarity: encodeTensor(sim.flat(), [2, 3], ['query_slot', 'document_slot']),
    routing_contribution: encodeTensor(rc, [2], ['query_slot'], { encoding: 'float' }),
    value_attention: encodeTensor(va.flat(), [2, 3], ['query_slot', 'document_slot']),
    value_slot_cosine: encodeTensor([0.5, -0.2], [2], ['query_slot'], { encoding: 'float' }),
  };
}

const tok = (spec: [string, Token['segment'], number | null, number | null][]): Token[] =>
  spec.map(([text, segment, start, end], id) => ({ id, text, segment, start, end }));

// Non-BMP and accented characters: code points differ from UTF-16 units.
const QUERY_TEXT = 'héllo \u{1F44B} world';
const QUERY_TOKENS = tok([
  ['Query:', 'template', null, null],
  ['h', 'input', 0, 1],
  ['é', 'input', 1, 2],
  ['llo', 'input', 2, 5],
  [' 👋', 'input', 5, 7],
  [' world', 'input', 7, 13],
]);
const Q_ATT = [
  [[0.1, 0.4, 0.1, 0.1, 0.2, 0.1], [0, 0, 0, 0, 1, 0]],
  [[0.1, 0.2, 0.1, 0.3, 0.2, 0.1], [0.2, 0, 0, 0, 0.8, 0]],
].flat(2);
const DOC_TOKENS = tok([
  ['a1', 'input', 0, 1],
  ['a2', 'input', 0, 1],
  ['b', 'input', 1, 2],
]);
const D_ATT = [0.5, 0.25, 0.25, 0, 0, 1, 0.2, 0.2, 0.6];

const query = { tokens: QUERY_TOKENS, attention: encodeTensor(Q_ATT, [2, 2, 6], ['head', 'slot', 'token']) };
const document = { tokens: DOC_TOKENS, attention: encodeTensor(D_ATT, [1, 3, 3], ['head', 'slot', 'token'], { encoding: 'float' }) };

describe('tokenAttribution', () => {
  it.each(['score', 'value', 'routing'] as const)('is an exact decomposition (%s)', (coupling) => {
    const exp = makeExplanation();
    const { score } = explanationScores(exp, SCORING);
    const att = tokenAttribution(exp, { query, document, scoring: SCORING, coupling });
    expect(att.total).toBeCloseTo(score, 6);
    att.slotPairs.forEach((row, i) => {
      expect(sum(row)).toBeCloseTo(att.querySlotScores[i]!, 6);
    });
    expect(sum(att.documentSlotWeights)).toBeCloseTo(score, 6);
    expect(sum(att.queryTokens!.map((w) => w.weight))).toBeCloseTo(score, 5);
    expect(sum(att.documentTokens!.map((w) => w.weight))).toBeCloseTo(score, 5);
  });

  it('maps tokens to text by code points and merges shared spans', () => {
    const att = tokenAttribution(makeExplanation(), { query, document, scoring: SCORING, queryText: QUERY_TEXT, documentText: 'ab' });
    expect(att.queryTokens!.map((w) => w.text)).toEqual([null, 'h', 'é', 'llo', ' 👋', ' world']);
    expect(att.querySpans().map((s) => s.text)).toEqual(['h', 'é', 'llo', ' 👋', ' world']);
    const doc = att.documentSpans();
    expect(doc.map((s) => [s.start, s.end, s.positions])).toEqual([[0, 1, [0, 1]], [1, 2, [2]]]);
    expect(doc[0]!.weight).toBeCloseTo(att.documentTokens![0]!.weight + att.documentTokens![1]!.weight, 9);
  });

  it('head selection and attentionMatrix', () => {
    const one = tokenAttribution(makeExplanation(), { query, scoring: SCORING, head: 0 });
    const mean = tokenAttribution(makeExplanation(), { query, scoring: SCORING });
    expect(one.queryTokens![0]!.weight).not.toBeCloseTo(mean.queryTokens![0]!.weight, 6);
    expect(mean.documentTokens).toBeNull();
    expect(attentionMatrix(query.attention)[1]![4]).toBeCloseTo(0.9, 6);
    expect(() => attentionMatrix(query.attention, 2)).toThrow(RangeError);
  });

  it('reports shape mismatches', () => {
    expect(() => tokenAttribution(makeExplanation(), { query: { ...query, tokens: QUERY_TOKENS.slice(0, 5) } })).toThrow(/6 tokens but 5/);
  });

  it('works on the contract explained example', () => {
    const res = expanded('score', 200, 'explained');
    const r = res.results[0];
    const att = tokenAttribution(r.explanation, {
      query: res.query,
      document: r.document,
      scoring: res.scoring,
      queryText: 'refund policy',
      documentText: 'Refunds accepted.',
    });
    expect(att.querySpans().map((s) => s.text)).toEqual(['refund', ' policy']);
    expect(att.documentSpans().map((s) => s.text)).toEqual(['Ref', 'unds', ' accepted', '.']);
    const scores = explanationScores(r.explanation, res.scoring);
    expect(att.total).toBeCloseTo(scores.score, 6);
    expect(scores.routingScore).toBeCloseTo(r.routing_score, 2);
    expect(scores.valueScore).toBeCloseTo(r.value_score, 2);
  });
});

describe('code-point offsets', () => {
  it('converts to UTF-16 and slices', () => {
    const idx = new CodePointIndex(QUERY_TEXT);
    expect(idx.length).toBe(13);
    expect(QUERY_TEXT.length).toBe(14);
    expect(idx.toUtf16(7)).toBe(8);
    expect(idx.slice(5, 7)).toBe(' 👋');
    expect(sliceCodePoints(QUERY_TEXT, 7, 13)).toBe(' world');
    expect(() => idx.toUtf16(14)).toThrow(RangeError);
  });

  it('tokenSpans skips template tokens and gives both offset units', () => {
    const spans = tokenSpans(QUERY_TEXT, QUERY_TOKENS);
    expect(spans.map((s) => s.positions)).toEqual([[1], [2], [3], [4], [5]]);
    expect(spans[0]!.tokens).toEqual([QUERY_TOKENS[1]]);
    expect(spans[4]).toMatchObject({ start: 7, end: 13, utf16Start: 8, utf16End: 14, text: ' world' });
  });
});

// The contract's token rule: a byte-level token that ends inside a multi-byte character shows only
// what precedes it (possibly '') and the token that completes it shows the whole character; both
// report the whole character's offsets, so offsets overlap without being identical.
type Case = [string, string, [string, number, number][], [number, number, number[]][]];
const OFFSET_CASES: Case[] = [
  // 🍕 is 4 UTF-8 bytes (2 UTF-16 units, 1 code point), split over two tokens.
  ['emoji over two tokens', 'a 🍕 b', [['a', 0, 1], [' ', 1, 3], ['🍕', 2, 3], [' b', 3, 5]], [[0, 1, [0]], [1, 3, [1, 2]], [3, 5, [3]]]],
  // ... and over three tokens: texts ' ', '', '🍕'.
  ['emoji over three tokens', 'a 🍕 b', [['a', 0, 1], [' ', 1, 3], ['', 2, 3], ['🍕', 2, 3], [' b', 3, 5]], [[0, 1, [0]], [1, 3, [1, 2, 3]], [3, 5, [4]]]],
  // e + U+0301 split after its first byte; then a clean combining sequence o + U+0308.
  ['combining sequence', 'e\u0301x o\u0308', [['e', 0, 2], ['\u0301', 1, 2], ['x', 2, 3], [' o', 3, 5], ['\u0308', 5, 6]], [[0, 2, [0, 1]], [2, 3, [2]], [3, 5, [3]], [5, 6, [4]]]],
  // Truncation cut 🍕 after 2 bytes: U+FFFD in the display text, offsets cover the character.
  ['truncated final character', 'ab🍕', [['a', 0, 1], ['b\ufffd', 1, 3]], [[0, 1, [0]], [1, 3, [1]]]],
  ['every token clean', 'Keys rotate.', [['Keys', 0, 4], [' rotate', 4, 11], ['.', 11, 12]], [[0, 4, [0]], [4, 11, [1]], [11, 12, [2]]]],
];
const inputTokens = (spec: Case[2]): Token[] => spec.map(([text, start, end], id) => ({ id, text, segment: 'input', start, end }));

describe('overlapping token offsets', () => {
  it.each(OFFSET_CASES)('tokenSpans merges and tiles: %s', (name, text, spec, expected) => {
    const tokens = inputTokens(spec);
    const spans = tokenSpans(text, tokens);
    expect(spans.map((s) => [s.start, s.end, s.positions])).toEqual(expected);
    expect(spans[0]!.start).toBe(0);
    expect(spans[spans.length - 1]!.end).toBe(new CodePointIndex(text).length);
    spans.slice(1).forEach((s, i) => {
      expect(s.start).toBe(spans[i]!.end);
    });
    expect(spans.map((s) => s.text).join('')).toBe(text);
    expect(spans.map((s) => s.utf16End - s.utf16Start).reduce((a, b) => a + b, 0)).toBe(text.length);
    const display = tokens.map((t) => t.text).join('');
    expect(display === text || (name === 'truncated final character' && display === 'ab\ufffd')).toBe(true);
  });

  it.each(OFFSET_CASES)('attribution spans merge overlaps and keep per-token weights: %s', (_name, text, spec, expected) => {
    const tokens = inputTokens(spec);
    const n = tokens.length;
    const rows = Array.from({ length: 3 }, (_, j) => {
      const raw = Array.from({ length: n }, (_, t) => ((j + 1) * (t + 1)) % 7 + 1);
      return raw.map((v) => v / sum(raw));
    });
    const document = { tokens, attention: encodeTensor(rows.flat(), [1, 3, n], ['head', 'slot', 'token']) };
    const att = tokenAttribution(makeExplanation(), { document, scoring: SCORING, documentText: text });
    expect(att.documentTokens).toHaveLength(n); // unmerged per token
    const spans = att.documentSpans();
    expect(spans.map((s) => [s.start, s.end, s.positions])).toEqual(expected);
    for (const s of spans) {
      expect(s.weight).toBeCloseTo(sum(s.positions.map((p) => att.documentTokens![p]!.weight)), 9);
    }
    expect(spans.map((s) => s.text).join('')).toBe(text);
    expect(sum(spans.map((s) => s.weight))).toBeCloseTo(att.total, 5);
  });
});

describe('counters', () => {
  it('parse to bigint without precision loss', () => {
    expect(parseCounter('9007199254740993')).toBe(9007199254740993n);
    expect(() => parseCounter('-1')).toThrow(BreadBowlError);
    expect(() => parseCounter('01')).toThrow(BreadBowlError);
    const usage = example('getUsage') as UsageSummary;
    const c = parseCounters(usage.usage);
    expect(c.input_tokens).toBe(1834221n);
    expect('documents' in c).toBe(false);
  });
});

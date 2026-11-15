// Exact decomposition of a pair score over slot pairs and tokens. See the README section
// "Explaining a score" for the formulas; they mirror the Python SDK's `token_attribution`.
import { CodePointIndex, mergeOffsetRanges, type OffsetRange } from './offsets.js';
import { decodeTensor, DecodedTensor } from './tensor.js';
import type { Explanation, ScoringParameters, Tensor, TextEncoding, Token } from './types.js';

const DEFAULT_SCORING = { tau_k: 0.02, routing_weight: 0.75, value_weight: 0.25 };

export type Coupling = 'score' | 'value' | 'routing';

/** One token's weight, unmerged. */
export interface TokenWeight {
  position: number;
  token: Token | null;
  weight: number;
  /**
   * The slice of the caller's text covered by this token's offsets (when the text was given).
   * A character split across byte-level tokens appears in each of their slices, so do not
   * concatenate these: use the merged spans. `token.text` is the token's display string.
   */
  text: string | null;
}

/**
 * Weight of one merged span: tokens whose offsets overlap are combined (union range, summed
 * weight, every position), so spans never overlap and tile the text.
 */
export interface SpanWeight {
  /** Code-point offsets. */
  start: number;
  end: number;
  weight: number;
  positions: number[];
  text: string | null;
}

export interface TokenAttribution {
  /** Σ_i s[i]; equals the result's score up to float16 rounding. */
  total: number;
  /** s[i]: each query slot's share of the score. */
  querySlotScores: number[];
  /** P[i][j]: share flowing from query slot i to document slot j. */
  slotPairs: number[][];
  /** Σ_i P[i][j]. */
  documentSlotWeights: number[];
  queryTokens: TokenWeight[] | null;
  documentTokens: TokenWeight[] | null;
  /**
   * Merged, non-overlapping query weights per span of the text (input tokens only), in text
   * order. Highlight with these; per-token weights stay unmerged in `queryTokens`.
   */
  querySpans(): SpanWeight[];
  documentSpans(): SpanWeight[];
}

export interface AttributionOptions {
  query?: Pick<TextEncoding, 'attention' | 'tokens'> | null | undefined;
  document?: Pick<TextEncoding, 'attention' | 'tokens'> | null | undefined;
  /** Weights and tau_k (`ScoreResponse.scoring` or the index's `search_profile.scoring`). */
  scoring?: Pick<ScoringParameters, 'tau_k' | 'routing_weight' | 'value_weight'> | null | undefined;
  coupling?: Coupling | undefined;
  queryText?: string | undefined;
  documentText?: string | undefined;
  /** One attention head instead of the mean over heads. */
  head?: number | undefined;
}

function matrix(t: DecodedTensor, rows: number, cols: number, name: string): number[][] {
  if (t.shape.length !== 2 || t.shape[0] !== rows || t.shape[1] !== cols) {
    throw new RangeError(`${name} has shape ${JSON.stringify(t.shape)}, expected [${rows}, ${cols}]`);
  }
  return Array.from({ length: rows }, (_, r) => Array.from(t.slice(r)));
}

function sum(values: Iterable<number>): number {
  let s = 0;
  for (const v of values) s += v;
  return s;
}

/** Slot→token attention as [slots][tokens]: one head, or the mean over heads. */
export function attentionMatrix(attention: Tensor | DecodedTensor, head?: number): number[][] {
  const t = attention instanceof DecodedTensor ? attention : decodeTensor(attention);
  if (t.shape.length !== 3) throw new RangeError(`attention must be [heads, slots, tokens], got ${JSON.stringify(t.shape)}`);
  const [heads = 0, slots = 0, tokens = 0] = t.shape;
  if (head !== undefined) {
    if (!Number.isInteger(head) || head < -heads || head >= heads) throw new RangeError(`head ${head} out of range for ${heads} heads`);
    const h = (head + heads) % heads;
    return Array.from({ length: slots }, (_, s) => Array.from(t.slice(h, s)));
  }
  const out = Array.from({ length: slots }, () => new Array<number>(tokens).fill(0));
  for (let h = 0; h < heads; h++) {
    for (let s = 0; s < slots; s++) {
      const row = t.slice(h, s);
      const acc = out[s] as number[];
      for (let k = 0; k < tokens; k++) acc[k] = (acc[k] ?? 0) + (row[k] ?? 0);
    }
  }
  return out.map((row) => row.map((v) => v / heads));
}

function tokenWeights(
  slotWeights: number[],
  side: Pick<TextEncoding, 'attention' | 'tokens'> | null | undefined,
  text: string | undefined,
  head: number | undefined,
  name: string,
): TokenWeight[] | null {
  if (!side?.attention) return null;
  const a = attentionMatrix(side.attention, head);
  if (a.length !== slotWeights.length) {
    throw new RangeError(`${name} attention has ${a.length} slots, the explanation has ${slotWeights.length}`);
  }
  const nTokens = a[0]?.length ?? 0;
  const tokens = side.tokens ?? null;
  if (tokens && tokens.length !== nTokens) {
    throw new RangeError(`${name} attention covers ${nTokens} tokens but ${tokens.length} tokens were returned`);
  }
  const index = text !== undefined ? new CodePointIndex(text) : null;
  return Array.from({ length: nTokens }, (_, t) => {
    let w = 0;
    for (let s = 0; s < a.length; s++) w += (slotWeights[s] ?? 0) * (a[s]?.[t] ?? 0);
    const token = tokens?.[t] ?? null;
    const piece = index && token && token.start !== null && token.end !== null ? index.slice(token.start, token.end) : null;
    return { position: t, token, weight: w, text: piece };
  });
}

function spans(weights: TokenWeight[] | null, text: string | undefined): SpanWeight[] {
  if (!weights) return [];
  const byPosition = new Map(weights.map((w) => [w.position, w]));
  const ranges: OffsetRange[] = [];
  for (const w of weights) {
    const start = w.token?.start ?? null;
    const end = w.token?.end ?? null;
    if (start !== null && end !== null) ranges.push({ start, end, position: w.position });
  }
  const index = text !== undefined ? new CodePointIndex(text) : null;
  return mergeOffsetRanges(ranges).map(({ start, end, positions }) => ({
    start,
    end,
    weight: sum(positions.map((p) => byPosition.get(p)?.weight ?? 0)),
    positions,
    text: index ? index.slice(start, end) : null,
  }));
}

/**
 * Decompose a pair score over query slots, slot pairs and tokens.
 *
 * s[i] = (routing_weight·routing_contribution[i] + value_weight·value_slot_cosine[i]) / n_query_slots,
 * spread over document slots by the coupling (`value`: value_attention; `routing`:
 * softmax_j(slot_similarity / tau_k); `score`: routing part by routing, value part by value).
 * Token weights follow through each side's slot→token attention; each side sums to the total.
 */
export function tokenAttribution(explanation: Explanation, options: AttributionOptions = {}): TokenAttribution {
  const coupling = options.coupling ?? 'score';
  const params = { ...DEFAULT_SCORING, ...(options.scoring ?? {}) };
  const simT = decodeTensor(explanation.slot_similarity);
  if (simT.shape.length !== 2) throw new RangeError('slot_similarity must be 2-D');
  const [nQ = 0, nD = 0] = simT.shape;
  const sim = matrix(simT, nQ, nD, 'slot_similarity');
  // Explanation tensors are float32, so value_attention rows sum to 1 to float32 precision.
  const valueAtt = matrix(decodeTensor(explanation.value_attention), nQ, nD, 'value_attention');
  const r = Array.from(decodeTensor(explanation.routing_contribution).data);
  const v = Array.from(decodeTensor(explanation.value_slot_cosine).data);
  if (r.length !== nQ || v.length !== nQ) {
    throw new RangeError('routing_contribution and value_slot_cosine need one value per query slot');
  }
  const routingPart = r.map((x) => (params.routing_weight * x) / nQ);
  const valuePart = v.map((x) => (params.value_weight * x) / nQ);
  const slotScores = routingPart.map((x, i) => x + (valuePart[i] ?? 0));
  const routingCpl = sim.map((row) => {
    const top = Math.max(...row);
    const e = row.map((x) => Math.exp((x - top) / params.tau_k));
    const z = sum(e);
    return e.map((x) => x / z);
  });
  const pairs = sim.map((_, i) =>
    Array.from({ length: nD }, (_, j) => {
      const rc = routingCpl[i]?.[j] ?? 0;
      const vc = valueAtt[i]?.[j] ?? 0;
      if (coupling === 'score') return (routingPart[i] ?? 0) * rc + (valuePart[i] ?? 0) * vc;
      return (slotScores[i] ?? 0) * (coupling === 'value' ? vc : rc);
    }),
  );
  const docSlots = Array.from({ length: nD }, (_, j) => sum(pairs.map((row) => row[j] ?? 0)));
  const queryTokens = tokenWeights(slotScores, options.query, options.queryText, options.head, 'query');
  const documentTokens = tokenWeights(docSlots, options.document, options.documentText, options.head, 'document');
  return {
    total: sum(slotScores),
    querySlotScores: slotScores,
    slotPairs: pairs,
    documentSlotWeights: docSlots,
    queryTokens,
    documentTokens,
    querySpans: () => spans(queryTokens, options.queryText),
    documentSpans: () => spans(documentTokens, options.documentText),
  };
}

/** routing_score = mean(routing_contribution), value_score = mean(value_slot_cosine). */
export function explanationScores(
  explanation: Explanation,
  scoring?: Pick<ScoringParameters, 'routing_weight' | 'value_weight'>,
): { routingScore: number; valueScore: number; score: number } {
  const mean = (t: Tensor): number => {
    const d = decodeTensor(t).data;
    return sum(d) / d.length;
  };
  const routingScore = mean(explanation.routing_contribution);
  const valueScore = mean(explanation.value_slot_cosine);
  const w = { ...DEFAULT_SCORING, ...(scoring ?? {}) };
  return { routingScore, valueScore, score: w.routing_weight * routingScore + w.value_weight * valueScore };
}

// The API's offsets are Unicode code points; JavaScript strings index UTF-16 code units.
import type { Token } from './types.js';

/**
 * Maps code-point offsets of one string to UTF-16 indices. Build it once per text and reuse it.
 *
 * ```ts
 * const idx = new CodePointIndex('héllo 👋 world');
 * idx.slice(token.start, token.end);
 * ```
 */
export class CodePointIndex {
  readonly text: string;
  /** utf16[cp] = UTF-16 index of code point `cp`; utf16[length] = text.length. */
  private readonly utf16: Uint32Array;

  constructor(text: string) {
    this.text = text;
    const offsets: number[] = [];
    let i = 0;
    while (i < text.length) {
      offsets.push(i);
      const cp = text.codePointAt(i) ?? 0;
      i += cp > 0xffff ? 2 : 1;
    }
    offsets.push(text.length);
    this.utf16 = Uint32Array.from(offsets);
  }

  /** Length in code points. */
  get length(): number {
    return this.utf16.length - 1;
  }

  /** UTF-16 index of a code-point offset (0..length). */
  toUtf16(codePointOffset: number): number {
    if (!Number.isInteger(codePointOffset) || codePointOffset < 0 || codePointOffset > this.length) {
      throw new RangeError(
        `code-point offset ${codePointOffset} is outside a text of ${this.length} code points; is this the exact string that was sent?`,
      );
    }
    return this.utf16[codePointOffset] ?? this.text.length;
  }

  /** The substring between two code-point offsets (end-exclusive). */
  slice(start: number, end: number): string {
    return this.text.slice(this.toUtf16(start), this.toUtf16(end));
  }
}

/** Slice a string by code-point offsets (convenience for one-off use). */
export function sliceCodePoints(text: string, start: number, end: number): string {
  return new CodePointIndex(text).slice(start, end);
}

export interface OffsetRange {
  /** Code-point offsets. */
  start: number;
  end: number;
  /** Position of the token in the `tokens` array (and on the attention token axis). */
  position: number;
}

export interface MergedRange {
  start: number;
  end: number;
  /** Every merged token position, ascending. */
  positions: number[];
}

/**
 * Merge code-point ranges that overlap into sorted, non-overlapping ranges, keeping every
 * position. A byte-level token that ends inside a multi-byte character reports the whole
 * character's offsets, as does the token that completes it, so token offsets can overlap
 * without being identical (`[4, 6)` and `[5, 6)`). Identical and nested ranges merge; touching
 * ranges (`[0, 4)`, `[4, 6)`) stay separate.
 */
export function mergeOffsetRanges(ranges: readonly OffsetRange[]): MergedRange[] {
  const sorted = [...ranges].sort((a, b) => a.start - b.start || a.end - b.end || a.position - b.position);
  const out: MergedRange[] = [];
  for (const r of sorted) {
    const last = out[out.length - 1];
    if (last && (r.start < last.end || (r.start === r.end && last.start <= r.start && r.start <= last.end))) {
      last.end = Math.max(last.end, r.end);
      last.positions.push(r.position);
      last.positions.sort((a, b) => a - b);
    } else {
      out.push({ start: r.start, end: r.end, positions: [r.position] });
    }
  }
  return out;
}

/** A run of the caller's text covered by input tokens, merged so spans never overlap. */
export interface TokenSpan {
  /** Code-point offsets. */
  start: number;
  end: number;
  /** UTF-16 offsets for slicing a JavaScript string. */
  utf16Start: number;
  utf16End: number;
  /** The caller's text over [start, end). Spans' texts concatenate to the text. */
  text: string;
  /** Positions of the merged tokens in the `tokens` array (and on the attention token axis). */
  positions: number[];
  tokens: Token[];
}

/**
 * The input tokens' coverage of `text` as merged, non-overlapping spans in text order (template
 * and special tokens are skipped). Tokens whose offsets overlap (a character split across
 * byte-level tokens) form one span, so for a complete tokenization the spans tile the text.
 * Highlight with these, not with per-token offsets. A token's own `text` is a display string:
 * the tokens' texts concatenate to the decoded text, while their offsets may overlap.
 */
export function tokenSpans(text: string | CodePointIndex, tokens: readonly Token[]): TokenSpan[] {
  const index = typeof text === 'string' ? new CodePointIndex(text) : text;
  const ranges: OffsetRange[] = [];
  tokens.forEach((token, position) => {
    if (token.start === null || token.end === null) return;
    index.toUtf16(token.start); // validates both offsets against the text
    index.toUtf16(token.end);
    ranges.push({ start: token.start, end: token.end, position });
  });
  return mergeOffsetRanges(ranges).map(({ start, end, positions }) => {
    const utf16Start = index.toUtf16(start);
    const utf16End = index.toUtf16(end);
    return {
      start,
      end,
      utf16Start,
      utf16End,
      text: index.text.slice(utf16Start, utf16End),
      positions,
      tokens: positions.map((p) => tokens[p] as Token),
    };
  });
}

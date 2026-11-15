// Counters that grow without bound are decimal strings on the wire; keep them as strings and
// convert with BigInt when you need arithmetic.
import { BreadBowlError } from './errors.js';
import type { CounterString } from './types.js';

const COUNTER = /^(0|[1-9][0-9]*)$/;

/** Parse a contract counter (`"50000000"`) to a bigint. */
export function parseCounter(value: CounterString): bigint {
  if (typeof value !== 'string' || !COUNTER.test(value)) {
    throw new BreadBowlError(`not a counter string: ${JSON.stringify(value)}`);
  }
  return BigInt(value);
}

/** The string-typed (counter) fields of `T`, as bigints. */
export type CounterFields<T> = { [K in keyof T as T[K] extends string ? K : never]: bigint };

/** Every counter-string field of an object as a bigint (other fields are left out). */
export function parseCounters<T extends object>(value: T): CounterFields<T> {
  const out: Record<string, bigint> = {};
  for (const [k, v] of Object.entries(value)) {
    if (typeof v === 'string' && COUNTER.test(v)) out[k] = BigInt(v);
  }
  return out as unknown as CounterFields<T>;
}

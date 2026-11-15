import { describe, expect, it } from 'vitest';
import {
  TensorDecodeError,
  decodeTensor,
  encodeTensor,
  float16BitsToNumber,
  numberToFloat16Bits,
  type Tensor,
} from '../src/index.js';
import { example, expanded } from './helpers.js';

function f16(bits: number[]): string {
  const bytes = new Uint8Array(bits.length * 2);
  const view = new DataView(bytes.buffer);
  bits.forEach((b, i) => {
    view.setUint16(i * 2, b, true);
  });
  return Buffer.from(bytes).toString('base64');
}

function f32(values: number[]): string {
  return Buffer.from(new Float32Array(values).buffer).toString('base64');
}

describe('float16', () => {
  it('decodes known bit patterns', () => {
    const t = decodeTensor({
      encoding: 'base64', dtype: 'float16', shape: [7], axes: ['dim'], byte_order: 'little',
      data: f16([0x3c00, 0xc000, 0x3800, 0x7bff, 0x0001, 0x0000, 0x8000]),
    });
    expect(Array.from(t.data.subarray(0, 5))).toEqual([1, -2, 0.5, 65504, 2 ** -24]);
    expect(Object.is(t.data[6], -0)).toBe(true);
  });

  it('infinity and NaN', () => {
    expect(float16BitsToNumber(0x7c00)).toBe(Infinity);
    expect(float16BitsToNumber(0xfc00)).toBe(-Infinity);
    expect(float16BitsToNumber(0x7e00)).toBeNaN();
  });

  it('encoding round-trips all 65 536 bit patterns (except NaN payloads)', () => {
    for (let bits = 0; bits < 0x10000; bits++) {
      const v = float16BitsToNumber(bits);
      if (Number.isNaN(v)) continue;
      expect(numberToFloat16Bits(v)).toBe(bits);
    }
  });

  it('rounds to nearest even', () => {
    expect(float16BitsToNumber(numberToFloat16Bits(0.71))).toBe(0.7099609375);
    expect(numberToFloat16Bits(1 + 2 ** -11)).toBe(0x3c00); // tie → even
    expect(numberToFloat16Bits(1 + 3 * 2 ** -11)).toBe(0x3c02);
    expect(numberToFloat16Bits(70000)).toBe(0x7c00);
    expect(numberToFloat16Bits(2 ** -26)).toBe(0);
  });

  it('float encoding is rounded to dtype, so both encodings agree', () => {
    const asFloat = decodeTensor({ encoding: 'float', dtype: 'float16', shape: [1], axes: ['query_slot'], data: [0.71] });
    const asB64 = decodeTensor({ encoding: 'base64', dtype: 'float16', shape: [1], axes: ['query_slot'], byte_order: 'little', data: f16([numberToFloat16Bits(0.71)]) });
    expect(asFloat.data[0]).toBe(asB64.data[0]);
  });
});

describe('float32 and layout', () => {
  it('base64 float32, row-major, indexing helpers', () => {
    const t = decodeTensor({ encoding: 'base64', dtype: 'float32', shape: [2, 3], axes: ['slot', 'dim'], byte_order: 'little', data: f32([1, 2, 3, 4, 5, 0.1]) });
    expect(t.toNested()).toEqual([[1, 2, 3], [4, 5, Math.fround(0.1)]]);
    expect(t.get(1, 2)).toBe(Math.fround(0.1));
    expect(t.get(-1, 0)).toBe(4);
    expect(Array.from(t.slice(1))).toEqual([4, 5, Math.fround(0.1)]);
    expect(t.axis('dim')).toBe(1);
    expect(t.size).toBe(6);
    expect(() => t.get(2, 0)).toThrow(RangeError);
    expect(() => t.get(0)).toThrow(RangeError);
  });

  it('3-D float tensors', () => {
    const data = [[[0.5, 0.25], [0.125, 0.125]]];
    const t = decodeTensor({ encoding: 'float', dtype: 'float32', shape: [1, 2, 2], axes: ['head', 'slot', 'token'], data });
    expect(t.toNested()).toEqual(data);
    expect(Array.from(t.slice(0, 1))).toEqual([0.125, 0.125]);
  });

  it('encodeTensor round-trips both encodings and dtypes', () => {
    const values = [1, 2.5, -3, 0];
    for (const dtype of ['float16', 'float32'] as const) {
      for (const encoding of ['base64', 'float'] as const) {
        const t = encodeTensor(values, [2, 2], ['slot', 'dim'], { dtype, encoding });
        expect(Array.from(decodeTensor(t).data)).toEqual(values);
      }
    }
  });

  it('zero-sized axes', () => {
    const t = decodeTensor({ encoding: 'base64', dtype: 'float16', shape: [16, 0], axes: ['slot', 'token'], byte_order: 'little', data: '' });
    expect(t.toNested()).toEqual(Array.from({ length: 16 }, () => []));
  });
});

describe('validation', () => {
  const cases: [string, unknown, RegExp][] = [
    ['short data', { encoding: 'base64', dtype: 'float16', shape: [3], axes: ['dim'], byte_order: 'little', data: f16([1, 2]) }, /needs 6/],
    ['float32 length', { encoding: 'base64', dtype: 'float32', shape: [2], axes: ['dim'], byte_order: 'little', data: f16([1, 2]) }, /needs 8/],
    ['bad base64', { encoding: 'base64', dtype: 'float16', shape: [1], axes: ['dim'], byte_order: 'little', data: '!!!!' }, /base64/],
    ['unpadded', { encoding: 'base64', dtype: 'float16', shape: [1], axes: ['dim'], byte_order: 'little', data: 'AAA' }, /base64/],
    ['dtype', { encoding: 'base64', dtype: 'float64', shape: [1], axes: ['dim'], data: '' }, /dtype/],
    ['encoding', { encoding: 'hex', dtype: 'float16', shape: [1], axes: ['dim'], data: '' }, /encoding/],
    ['axes', { encoding: 'base64', dtype: 'float16', shape: [1, 1], axes: ['dim'], data: '' }, /axes/],
    ['shape', { encoding: 'base64', dtype: 'float16', shape: [-1], axes: ['dim'], data: '' }, /shape/],
    ['byte order', { encoding: 'base64', dtype: 'float16', shape: [1], axes: ['dim'], byte_order: 'big', data: f16([1]) }, /byte_order/],
    ['ragged', { encoding: 'float', dtype: 'float16', shape: [2, 2], axes: ['slot', 'dim'], data: [[1, 2], [3]] }, /length 1/],
    ['too deep', { encoding: 'float', dtype: 'float16', shape: [2], axes: ['dim'], data: [[1], [2]] }, /finite number/],
    ['boolean', { encoding: 'float', dtype: 'float16', shape: [2], axes: ['dim'], data: [1, true] }, /finite number/],
    ['scalar data', { encoding: 'float', dtype: 'float32', shape: [1], axes: ['dim'], data: 3 }, /array/],
  ];
  it.each(cases)('%s', (_name, tensor, message) => {
    expect(() => decodeTensor(tensor as Tensor)).toThrow(TensorDecodeError);
    expect(() => decodeTensor(tensor as Tensor)).toThrow(message);
  });
});

describe('contract tensors', () => {
  it('expanded encode example decodes to the declared shapes', () => {
    const item = expanded('encode', 200, 'queryWithAttention').data[0];
    const routing = decodeTensor(item.routing);
    expect(routing.shape).toEqual([16, 256]);
    expect(routing.data[0]).toBe(1);
    const attention = decodeTensor(item.attention);
    expect(attention.shape).toEqual([1, 16, item.token_count]);
  });

  it('verbatim contract examples are abbreviated and rejected', () => {
    const item = example('encode').data[0];
    expect(() => decodeTensor(item.routing)).toThrow(/needs 8192/);
  });

  it('explained score example mixes encodings', () => {
    const exp = expanded('score', 200, 'explained').results[0].explanation;
    expect(decodeTensor(exp.slot_similarity).shape).toEqual([16, 16]);
    expect(decodeTensor(exp.routing_contribution).data[0]).toBeCloseTo(0.71, 3);
  });
});

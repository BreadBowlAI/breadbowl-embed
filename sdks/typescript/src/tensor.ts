// Tensor decoding to Float32Array. float16 is decoded by hand (Float16Array is not assumed).
import { TensorDecodeError } from './errors.js';
import type { Float32Tensor, Tensor, TensorAxis, TensorDtype } from './types.js';

const ITEMSIZE: Record<TensorDtype, number> = { float16: 2, float32: 4 };

export type Nested = number | Nested[];

/** Decode one IEEE 754 binary16 value from its 16 bits. */
export function float16BitsToNumber(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * fraction * 2 ** -24; // zero and subnormals
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

/** Round a number to the nearest binary16 value and return its 16 bits (round half to even). */
export function numberToFloat16Bits(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const abs = Math.abs(value);
  if (abs === Number.POSITIVE_INFINITY) return sign | 0x7c00;
  if (abs < 2 ** -24 / 2) return sign; // rounds to zero
  if (abs < 2 ** -14) {
    // subnormal: value = fraction · 2^-24
    const fraction = roundHalfEven(abs / 2 ** -24);
    return sign | fraction; // fraction 1024 carries into the smallest normal correctly
  }
  let exponent = Math.floor(Math.log2(abs));
  if (abs / 2 ** exponent >= 2) exponent += 1; // guard log2 rounding
  if (abs / 2 ** exponent < 1) exponent -= 1;
  let fraction = roundHalfEven((abs / 2 ** exponent - 1) * 1024);
  if (fraction === 1024) {
    fraction = 0;
    exponent += 1;
  }
  if (exponent > 15) return sign | 0x7c00; // overflow to infinity
  return sign | ((exponent + 15) << 10) | fraction;
}

function roundHalfEven(x: number): number {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Round to the nearest value representable in `dtype`. */
export function roundToDtype(value: number, dtype: TensorDtype): number {
  return dtype === 'float16' ? float16BitsToNumber(numberToFloat16Bits(value)) : Math.fround(value);
}

const B64 = /^[A-Za-z0-9+/]*={0,2}$/;

/** Strict RFC 4648 base64 to bytes, without Node's Buffer (works in browsers, Deno, Bun). */
export function base64ToBytes(data: string): Uint8Array {
  if (data.length % 4 !== 0 || !B64.test(data)) {
    throw new TensorDecodeError('tensor data is not valid padded base64');
  }
  let binary: string;
  try {
    binary = atob(data);
  } catch (err) {
    throw new TensorDecodeError('tensor data is not valid base64', { cause: err });
  }
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

export function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function product(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function flattenChecked(data: unknown, shape: readonly number[], out: number[], path: string): void {
  if (shape.length === 0) {
    if (typeof data !== 'number' || !Number.isFinite(data)) {
      throw new TensorDecodeError(`${path} must be a finite number`);
    }
    out.push(data);
    return;
  }
  if (!Array.isArray(data)) throw new TensorDecodeError(`${path} must be an array of length ${shape[0]}`);
  if (data.length !== shape[0]) {
    throw new TensorDecodeError(`${path} has length ${data.length}, shape says ${shape[0]}`);
  }
  const rest = shape.slice(1);
  data.forEach((item, i) => {
    flattenChecked(item, rest, out, `${path}[${i}]`);
  });
}

/** A decoded tensor: `data` in row-major order with its `shape` and named `axes`. */
export class DecodedTensor {
  constructor(
    readonly data: Float32Array,
    readonly shape: readonly number[],
    readonly axes: readonly TensorAxis[],
    readonly dtype: TensorDtype,
  ) {}

  get size(): number {
    return this.data.length;
  }

  get ndim(): number {
    return this.shape.length;
  }

  /** Position of a named axis, e.g. `attention.axis('token') === 2`. */
  axis(name: TensorAxis): number {
    const i = this.axes.indexOf(name);
    if (i < 0) throw new RangeError(`tensor has no axis ${name}; axes are ${this.axes.join(', ')}`);
    return i;
  }

  /** Flat offset of an index prefix (missing trailing indices are 0). */
  offset(...index: number[]): number {
    if (index.length > this.shape.length) throw new RangeError('too many indices');
    let offset = 0;
    for (let k = 0; k < this.shape.length; k++) {
      const dim = this.shape[k] ?? 0;
      let i = index[k] ?? 0;
      if (i < 0) i += dim;
      if (!Number.isInteger(i) || i < 0 || (i >= dim && k < index.length)) {
        throw new RangeError(`index ${index[k] ?? 0} out of range for axis ${k} (${this.axes[k] ?? '?'}) of size ${dim}`);
      }
      offset = offset * dim + i;
    }
    return offset;
  }

  /** One value: `t.get(slot, dim)`. */
  get(...index: number[]): number {
    if (index.length !== this.shape.length) {
      throw new RangeError(`get() needs ${this.shape.length} indices, got ${index.length}`);
    }
    return this.data[this.offset(...index)] ?? Number.NaN;
  }

  /** A view (no copy) of the values under an index prefix: `attention.slice(head, slot)` is that row. */
  slice(...prefix: number[]): Float32Array {
    const size = product(this.shape.slice(prefix.length));
    const start = this.offset(...prefix);
    return this.data.subarray(start, start + size);
  }

  /** Nested arrays matching `shape`. */
  toNested(): Nested[] {
    const build = (offset: number, dims: readonly number[]): Nested[] => {
      const [first = 0, ...rest] = dims;
      if (rest.length === 0) return Array.from(this.data.subarray(offset, offset + first));
      const step = product(rest);
      return Array.from({ length: first }, (_, i) => build(offset + i * step, rest));
    };
    return build(0, this.shape);
  }
}

function checkHeader(t: Tensor): void {
  if (!(t.dtype in ITEMSIZE)) throw new TensorDecodeError(`unsupported dtype ${t.dtype}`);
  if (!Array.isArray(t.shape) || t.shape.length === 0 || t.shape.some((d) => !Number.isInteger(d) || d < 0)) {
    throw new TensorDecodeError(`invalid shape ${JSON.stringify(t.shape)}`);
  }
  if (!Array.isArray(t.axes) || t.axes.length !== t.shape.length) {
    throw new TensorDecodeError(`axes ${JSON.stringify(t.axes)} do not match shape ${JSON.stringify(t.shape)}`);
  }
}

/** Decode a contract tensor (base64 float16/float32, or nested floats) to a Float32Array. */
export function decodeTensor(tensor: Tensor): DecodedTensor {
  checkHeader(tensor);
  const size = product(tensor.shape);
  const out = new Float32Array(size);
  if (tensor.encoding === 'base64') {
    const byteOrder = (tensor as { byte_order?: unknown }).byte_order;
    if (byteOrder !== undefined && byteOrder !== 'little') {
      throw new TensorDecodeError(`unsupported byte_order ${JSON.stringify(byteOrder)}`);
    }
    if (typeof tensor.data !== 'string') throw new TensorDecodeError('base64 tensor data must be a string');
    const bytes = base64ToBytes(tensor.data);
    const need = size * ITEMSIZE[tensor.dtype];
    if (bytes.length !== need) {
      throw new TensorDecodeError(
        `tensor data has ${bytes.length} bytes; shape ${JSON.stringify(tensor.shape)} of ${tensor.dtype} needs ${need}`,
      );
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (tensor.dtype === 'float16') {
      for (let i = 0; i < size; i++) out[i] = float16BitsToNumber(view.getUint16(i * 2, true));
    } else {
      for (let i = 0; i < size; i++) out[i] = view.getFloat32(i * 4, true);
    }
  } else if ((tensor.encoding as string) === 'float') {
    const flat: number[] = [];
    flattenChecked(tensor.data, tensor.shape, flat, 'data');
    // Round to dtype so both encodings decode to the same values.
    for (let i = 0; i < size; i++) out[i] = roundToDtype(flat[i] ?? 0, tensor.dtype);
  } else {
    throw new TensorDecodeError(`unsupported encoding ${String((tensor as { encoding: unknown }).encoding)}`);
  }
  return new DecodedTensor(out, [...tensor.shape], [...tensor.axes], tensor.dtype);
}

/** Build a contract tensor from values (tests, fixtures, round-tripping). float32 by default. */
export function encodeTensor(
  values: ArrayLike<number>,
  shape: readonly number[],
  axes: readonly TensorAxis[],
  options?: { dtype?: 'float32'; encoding?: 'base64' | 'float' },
): Float32Tensor;
export function encodeTensor(
  values: ArrayLike<number>,
  shape: readonly number[],
  axes: readonly TensorAxis[],
  options: { dtype: TensorDtype; encoding?: 'base64' | 'float' },
): Tensor;
export function encodeTensor(
  values: ArrayLike<number>,
  shape: readonly number[],
  axes: readonly TensorAxis[],
  options: { dtype?: TensorDtype; encoding?: 'base64' | 'float' } = {},
): Tensor {
  const dtype = options.dtype ?? 'float32';
  const size = product(shape);
  if (values.length !== size) throw new TensorDecodeError(`${values.length} values do not fill shape ${JSON.stringify(shape)}`);
  if (options.encoding === 'float') {
    const decoded = new DecodedTensor(
      Float32Array.from(values, (v) => roundToDtype(v, dtype)),
      [...shape],
      [...axes],
      dtype,
    );
    return { encoding: 'float', dtype, shape: [...shape], axes: [...axes], data: decoded.toNested() as number[] };
  }
  const bytes = new Uint8Array(size * ITEMSIZE[dtype]);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < size; i++) {
    const v = values[i] ?? 0;
    if (dtype === 'float16') view.setUint16(i * 2, numberToFloat16Bits(v), true);
    else view.setFloat32(i * 4, v, true);
  }
  return {
    encoding: 'base64',
    dtype,
    shape: [...shape],
    axes: [...axes],
    byte_order: 'little',
    data: bytesToBase64(bytes),
  };
}

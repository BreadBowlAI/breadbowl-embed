import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: true,
  clean: true,
  // ES2020 keeps BigInt and optional chaining native for Node 18, Bun, Deno and browsers.
  target: 'es2020',
  platform: 'neutral',
  treeshake: true,
});

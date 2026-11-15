// Tests must only use the injected mock fetch. The default base URL is a live service.
globalThis.fetch = (input: unknown) => {
  throw new Error(`real network request attempted: ${String(input)}`);
};

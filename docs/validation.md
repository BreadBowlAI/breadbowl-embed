# Validation

The step-4000 KL+replay checkpoint was exported locally and compared on CPU against the original research implementation using two queries and four documents. Query/document routing and value tensors matched exactly. The largest absolute score difference was approximately 1.2e-7, and rankings agreed. See `step4000-parity.json` for the actual smoke-test output.

This is a numerical fidelity check, not a benchmark rerun. It does not establish paper-level accuracy, throughput, memory use at scale, or GPU compatibility. The small exact document-routing index in this toolkit differs from the paper's retrieval pipeline.

The offline unit suite checks independent scoring equations and gradients, preservation of raw document-value magnitudes, reduced-precision scoring inputs, empty inputs, representation mismatch rejection, reranking, index persistence, deterministic ties, integrity failures, model export, and save/load stability. The tiny test model requires no Hub downloads.

The distributable wheel was installed into a separate virtual environment using the existing machine dependencies. All notebook code cells ran against the real exported checkpoint through that installed package. This validates the wheel layout and example code; it is not a clean-machine dependency installation test.

Public Hub revision `d55da971bc24b896a441b3ad466f2dc5d6178d73` was downloaded without authentication into a fresh cache and loaded through the installed toolkit. CPU reranking passed and the representation identifier matched the locally verified export. See `hub-verification.json`. The clean Linux CI installation also passes all 15 tests with the pinned preview dependencies.

The released checkpoint also passed a small Apple Silicon MPS inference test with PyTorch 2.11. Both native precision and float32 ran with automatic CPU fallback disabled and ranked the relevant passage first. Encoding ran on MPS; the convenience API returned slots to CPU for scoring. This does not establish GPU/CPU numerical parity, CUDA support, throughput, or memory behavior at scale.

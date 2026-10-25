# ADR-03: Tensor backend: Candle (Metal) plus custom kernels for hot spots

- **Status:** Accepted
- **Date:** 2026-10-25
- **Module:** #5

## Context
The engine needs tensors on the Apple GPU: matmul, elementwise ops, reductions, indexing and softmax, in f32/f16/bf16 and later quantized formats. Writing all of that against Metal directly means months of plumbing (buffer management, command encoding, shader compilation, a dtype matrix) before the first forward pass. Inference work is the point of the project: model code, KV cache, sampling, batching, and a few kernels that matter. The constraints are in [ADR-01](01-language-rust.md) (Rust) and [ADR-02](02-two-repo-library-design.md) (a stable public API for the harness).

## Decision
Use **`candle-core` with its `metal` feature** as the tensor backend, pinned to an exact version (`=0.11.0`). Write all model code ourselves on top of `candle-core` tensors, and replace hot-spot ops with **hand-written Metal kernels** (#24–#25) plugged in through Candle's custom-op API. `candle-transformers` is never a dependency. Its SmolLM3 and Qwen3 implementations are read only as a cross-check.

## Why
- **Metal ops that work today.** Candle 0.11.0 ships Metal kernels for affine, binary, unary, cast, indexing, reduce, softmax, sort, GEMV, an MLX-derived GEMM, quantized matmul and scaled-dot-product attention (`candle-metal-kernels/src/source.rs`). It compiles them **at runtime** from embedded source (`new_library_with_source` in `candle-metal-kernels/src/kernel.rs`), so building needs only the Xcode Command Line Tools, not full Xcode. Verified locally: an f32 batched matmul on Metal matches the CPU result (gate below).
- **Reference code to cross-check against.** `candle-transformers` 0.11.0 has `models/smol/smollm3.rs`, `models/qwen3.rs` and `models/llama.rs`, the exact architectures we target. When our forward pass disagrees with Transformers, a second independent implementation on the same backend helps tell a backend bug from a model bug.
- **Effort goes into inference, not plumbing.** Device management, buffer allocation and the dtype matrix are already solved, so the modules can focus on the model, cache, sampling and batching.
- **Licensing fits.** Candle is MIT OR Apache-2.0, the same as this project.

## Alternatives considered
- **Raw Metal through objc2 bindings:** full control, but every op, dtype and buffer path is ours to write and test before anything runs.
- **MLX through C bindings:** strong Apple Silicon kernels, but it adds a C++ build and FFI into a Rust project, and it pushes the core of the engine behind a foreign API, against ADR-01's goals.
- **Using `candle-transformers` models directly:** fastest route to output, but it skips the model work the project exists to do, and couples us to its cache and API design.

## Tradeoffs (accepted)
- **Tied to Candle's release cadence.** Recent releases were 0.9.2 (2026-01-24), 0.10.0 (2026-03-31) and 0.11.0 (2026-06-26). Fixes we need may wait for a release, and upgrades can change kernel behaviour.
- **Less control over memory layout.** Tensor storage, strides and buffer allocation are Candle's. KV cache layout, in-place updates and allocation reuse go through its API.
- **Transitive dependencies we don't choose.** candle-core 0.11.0 hard-depends on `zip` 8.6 (for `.npz`) and `tokenizers` (non-wasm targets). As of today this brings three RustSec advisories in `zip`'s `extract()` and one "unmaintained" notice for `paste`, all scoped and justified in `deny.toml`.
- **Numerical defaults.** Candle compiles Metal kernels with fast math enabled unless `CANDLE_METAL_ENABLE_FAST_MATH` is set to false, so Metal results are compared to CPU and to Transformers with documented tolerances, never exact equality.
- **`metal_is_available()` is compile-time only.** It reports the cargo feature, not a usable GPU, so device selection tries `Device::new_metal(0)` and falls back on error.

## Mitigations
- **Custom ops for hot spots.** The `CustomOp1`/`CustomOp2`/`CustomOp3` traits (`candle-core/src/custom_op.rs`) take a `cpu_fwd` and an optional `metal_fwd(&MetalStorage, &Layout)`, applied with `Tensor::apply_op1/2/3`. Our kernels (#24–#25) plug in there, so a slow op can be replaced without leaving Candle.
- **Candle kept out of the public API.** `device.rs` and every module that touches Candle types are crate-private. The harness depends on `lldx-engine` only, so a Candle upgrade, or a backend swap, is never a breaking change for it.
- **Version pinned.** `candle-core = "=0.11.0"` in `[workspace.dependencies]`. Upgrades are deliberate PRs that re-run every gate, including tier 3, and re-check the `deny.toml` ignores.
- **Parity gates.** Every Metal path is checked against the CPU path and against pinned Transformers fixtures ([TESTING.md](../TESTING.md)).

## Gate (#5)
`device::tests::tier3::metal_matmul_matches_cpu`: an f32 `[2, 64, 256] × [2, 256, 96]` matmul on Metal matches CPU within `atol = rtol = 1e-4`. It is tier 3 (local Mac only). On an M4 the observed max difference was 0.0.

## References
- Candle repository: https://github.com/huggingface/candle
- `candle-core` 0.11.0 features (`metal`): https://github.com/huggingface/candle/blob/0.11.0/candle-core/Cargo.toml
- Custom-op traits: https://github.com/huggingface/candle/blob/0.11.0/candle-core/src/custom_op.rs
- Device API (`Device::new_metal`): https://github.com/huggingface/candle/blob/0.11.0/candle-core/src/device.rs
- `metal_is_available()`: https://github.com/huggingface/candle/blob/0.11.0/candle-core/src/utils.rs
- Runtime shader compilation and fast-math default: https://github.com/huggingface/candle/blob/0.11.0/candle-metal-kernels/src/kernel.rs
- Kernel sources list: https://github.com/huggingface/candle/blob/0.11.0/candle-metal-kernels/src/source.rs
- SmolLM3 cross-check: https://github.com/huggingface/candle/blob/0.11.0/candle-transformers/src/models/smol/smollm3.rs
- Qwen3 cross-check: https://github.com/huggingface/candle/blob/0.11.0/candle-transformers/src/models/qwen3.rs
- Release dates: https://crates.io/crates/candle-core

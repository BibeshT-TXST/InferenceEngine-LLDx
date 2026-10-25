# ADR-01: Rust as the implementation language

- **Status:** Accepted
- **Date:** 2026-10-25
- **Module:** #0

## Context
The engine runs LLM inference on Apple Silicon for the LLDx harness. The obvious fast path is Python with MLX, Apple's array framework for Apple Silicon, which already has model implementations and Metal kernels. The project has two goals: ship a durable engine people can depend on, and build skills that carry over to inference infrastructure work.

## Decision
Write the engine in Rust: pinned stable toolchain, edition 2024, target `aarch64-apple-darwin` only.

## Alternatives considered
- **Python + MLX:** fastest iteration and a mature ecosystem. But distribution needs a Python runtime and environment on the user's Mac, and most of the interesting work (kernels, memory layout, scheduling) would already be done by the framework.

## Consequences
- **Single-binary distribution.** The CLI and any harness built on the crate ship as one native binary, with no interpreter or virtualenv.
- **Career value.** Rust inference infrastructure (memory-safe, explicit about allocation and threads) carries over directly to production serving work.
- **Compiler-checked concurrency.** `Send + Sync` on handles is checked at compile time, which the parallelism plan depends on ([ADR-04](04-batch-first-and-parallelism.md)).
- **Accepted cost: slower iteration.** Compile times and a stricter type system slow down experiments compared with Python.
- **Accepted cost: learning two hard things at once.** Rust and inference internals are both being learned at the same time.

## Mitigations
- Python stays where it is cheapest: generating reference fixtures with Transformers (`fixtures/generate.py`), never at runtime.
- Small modules behind gates, so every step is verified before the next one starts.

## References
- MLX: https://github.com/ml-explore/mlx

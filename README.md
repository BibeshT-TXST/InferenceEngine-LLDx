# InferenceEngine-LLDx

A LLM inference engine for Apple Silicon, written in Rust. It powers LLDx (Local Log Doctor), an agent that debugs your logs and code without anything leaving your Mac.

It's a learning project held to durable-software standards. Every module ships behind a test gate, every speedup comes with a metric, and every source is cited.

## Status
Phase 0: foundation. Progress lives in the GitHub issues, one per module.

## What it does
It loads a model, turns text into tokens, runs the transformer on the Metal GPU, samples the next token and streams text back.

It's built for agent loops. The harness gets KV cache handles it can save, restore, fork and trim, so follow-up questions about the same logs skip work the engine already did.

## Target models
| Mac | Model | Context |
|---|---|---|
| 8 GB | SmolLM3-3B | 16–32K |
| 16 GB | Qwen3-4B-Instruct-2507 | 32K |

SmolLM3-3B comes first and stays the only target until it passes every gate. The engine supports one architecture family (Llama-style decoders) with config-driven switches per model.

## Stack
- **Language:** Rust (pinned stable, edition 2024)
- **GPU:** Metal on Apple Silicon
- **Tensor backend:** Candle (Metal), with custom Metal kernels for hot spots ([ADR-03](docs/architecture-decision-records/03-tensor-backend.md))
- **Kernels:** Metal Shading Language
- **Correctness reference:** Hugging Face Transformers, pinned

## How it works
text → chat template → tokenizer → embeddings → decoder blocks (RMSNorm, RoPE/NoPE, GQA attention, SwiGLU) → logits → sampler → streaming detokenizer → text

The KV cache sits beside attention. The prompt runs once (prefill), then each new token reuses the cache (decode). Full design in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Parallelism
- **Prefill:** the whole prompt runs through the model in one pass.
- **Batched decode:** several sequences share each forward pass. Decode is memory-bound, so one read of the weights serves many tokens.
- **Shared-prefix subagents:** branches fork one cached prefix (copy-on-write) and decode side by side.
- **CPU/GPU overlap:** tokenizing, sampling and detokenizing run on the CPU while the GPU computes the next step.
- **Thread-safe handles:** engine and cache handles are `Send + Sync`, checked by the compiler.

Every tensor carries a batch dimension from day one ([ADR-04](docs/architecture-decision-records/04-batch-first-and-parallelism.md)), so batching is never a retrofit.

## Roadmap
0. **Foundation:** workspace, standards, CI, test tiers, backend, reference fixtures
1. **Load and tokenize:** config, safetensors, tokenizer, chat template
2. **Forward pass:** logits match Transformers
3. **Generation:** KV cache, sampling, streaming, public API, CLI, metrics → v0.1.0
4. **Optimization:** quantization, Metal kernels, KV cache v2 (prefix reuse, fork, persist), memory tiers
5. **Harness-facing:** constrained decoding, tool calls, parallelism, benchmark suite
6. **Ship:** signed release, docs, demo video

Then Qwen3-4B-Instruct-2507, repeating every gate.

The LLDx harness (read-only tools, log intelligence, agent loop) lives in its own repo and imports this crate.

## Quality bar
A module is done only when its gate passes:
- **Tier 1:** unit tests (CI)
- **Tier 2:** parity against a tiny reference model (CI)
- **Tier 3:** parity against real SmolLM3-3B on a Mac

Tagged runs record TTFT, tokens/sec, peak memory, cache hit rate and energy in `metrics/runs/`. Details in [docs/TESTING.md](docs/TESTING.md).

## Development
Requires an Apple Silicon Mac. The Rust toolchain is pinned in `rust-toolchain.toml`, and `rustup` installs it on first use.

```sh
cargo build --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace                              # needs cargo-nextest
cargo test --workspace --doc
cargo run -p lldx-engine-cli -- --log-format json
cargo deny check                                           # needs cargo-deny
cargo about generate about.hbs > THIRD_PARTY_LICENSES.md   # needs cargo-about
cargo xtask fixtures --tiny                                # needs uv
```

## Docs
[Architecture](docs/ARCHITECTURE.md) · [Testing](docs/TESTING.md) · [Decisions](docs/architecture-decision-records/) · [References](docs/REFERENCES.md)

## License
MIT OR Apache-2.0

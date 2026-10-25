# Architecture

## Purpose
`lldx-engine` is an LLM inference engine for Apple Silicon. It is a Rust library that the LLDx harness (a separate repo) links and calls in-process. Everything runs locally: model weights, prompts and logs never leave the Mac.

It is a learning project held to durable-software standards: every module ships behind a test gate, every optimization is backed by a metric, and every external source is cited in [REFERENCES.md](REFERENCES.md).

## Non-goals
- **No HTTP server.** The harness imports the crate directly ([ADR-02](architecture-decision-records/02-two-repo-library-design.md)).
- **No other platforms.** The only target is `aarch64-apple-darwin`.
- **No general model zoo.** One architecture family (Llama-style decoders) with config-driven switches. SmolLM3-3B is the only target until it passes every gate, then Qwen3-4B-Instruct-2507.
- **No training or fine-tuning.**
- **No harness logic.** Tools, log intelligence and the agent loop live in the harness repo.

## Data flow
```
text
 → chat template                      (tokenizer/template.rs)
 → tokenizer                          (tokenizer/mod.rs)
 → embeddings                         (model/embedding.rs)
 → N × decoder block                  (model/block.rs)
     RMSNorm → attention (GQA, RoPE or NoPE per layer) → residual
     RMSNorm → SwiGLU MLP → residual
 → final RMSNorm → logits
 → sampler                            (sampling.rs)
 → streaming detokenizer              (tokenizer/stream.rs)
 → text
```
The KV cache (`cache/`) sits beside attention. Prefill runs the whole prompt in one forward pass and fills the cache. Decode then runs one new token per sequence per step and reads the cache.

Model-specific switches, all read from `config.json`:
- **NoPE layers** (SmolLM3): some layers skip rotary position embeddings.
- **QK-norm** (Qwen3): RMSNorm on query and key heads before RoPE.
- **Llama 3 RoPE scaling**: frequency-dependent scaling of RoPE for long context.

## Module map
Each module is one GitHub issue, one branch and one PR. It is done when its gate passes ([TESTING.md](TESTING.md)).

| Module | Files | Phase |
|---|---|---|
| #0 Workspace | `Cargo.toml`, `crates/engine/src/lib.rs`, CI, docs | 0 Foundation |
| #1 Standards | `rustfmt.toml`, workspace lints, `CONTRIBUTING.md` | 0 Foundation |
| #2 Test infrastructure | `.config/nextest.toml`, `crates/engine/tests/gates/main.rs` | 0 Foundation |
| #3 Observability | `crates/cli/` | 0 Foundation |
| #4 Dependency hygiene | `deny.toml`, `about.toml`, `about.hbs` | 0 Foundation |
| #5 Backend | `device.rs`, [ADR-03](architecture-decision-records/03-tensor-backend.md) | 0 Foundation |
| #6 Fixtures | `fixtures/`, `crates/xtask/` | 0 Foundation |
| #7 | `error.rs`, `config.rs`, `tests/gates/common.rs`, `fuzz/` | 1 Load and tokenize |
| #8 | `weights.rs` (safetensors, memory-mapped) | 1 Load and tokenize |
| #9 | `tokenizer/mod.rs` | 1 Load and tokenize |
| #10 | `tokenizer/template.rs` (chat template) | 1 Load and tokenize |
| #11–#15 | `model/{mod,embedding,norm,rope,attention,mlp,block}.rs` | 2 Forward pass |
| #16 | `cache/mod.rs` (KV cache v1, per-sequence slots) | 3 Generation |
| #17 | `sampling.rs` | 3 Generation |
| #18 | `generate.rs` | 3 Generation |
| #19 | `tokenizer/stream.rs` (streaming detokenizer) | 3 Generation |
| #20 | public API finalized | 3 Generation |
| #21 | CLI snapshot tests (insta) | 3 Generation |
| #22 | `metrics.rs`, `benches/inference.rs`, `metrics/runs/` | 3 Generation → v0.1.0 |
| #23 | `quant.rs` | 4 Optimization |
| #24–#25 | `kernels/`, `build.rs` (`.metal` → `.metallib`) | 4 Optimization |
| #26 | `cache/prefix.rs` (prefix reuse, copy-on-write fork) | 4 Optimization |
| #27 | `cache/persist.rs`, `cache/quant.rs` | 4 Optimization |
| #28 | `memory.rs` (RAM tiers) | 4 Optimization |
| #29 | `constrain/{mod,tool_call}.rs` | 5 Harness-facing |
| #30a–#30c | `batch.rs` (batched decode, shared-prefix subagents, CPU/GPU overlap) | 5 Harness-facing |
| #33 | `release.yml`, `dist-workspace.toml` (cargo-dist) | 6 Ship |
| #35 (stretch) | `speculative.rs` | 6 Ship |

## Backend layering
```
public API (lldx-engine types only)
        │
our model code: model/, cache/, sampling, generate
        │
candle-core tensors ──── custom ops (CustomOp1/2/3 → our Metal kernels, #24–#25)
        │
Candle Metal backend → Metal → Apple GPU
```
- We write the model ourselves on top of `candle-core` tensors. `candle-transformers` is read only as a cross-check and is never a dependency.
- Hot-spot ops get hand-written Metal kernels, plugged in through Candle's custom-op traits.
- **Candle types never appear in the public API.** The harness depends on `lldx-engine` only, so swapping or upgrading Candle never breaks it.
- `device.rs` picks the Metal device and falls back to CPU with a logged warning.

Rationale and tradeoffs: [ADR-03](architecture-decision-records/03-tensor-backend.md).

## Intended public API (finalized at #20)
This is a contract to build towards, not code yet. Names will be settled at #20.
- **Load:** load a model from a directory (config, weights, tokenizer, chat template) into an engine handle.
- **Prefill:** run a prompt and get back a cache handle.
- **Generate:** continue from a cache handle with sampling parameters, an optional constraint (grammar or tool-call schema, #29) and a streaming callback that receives text pieces.
- **Cache handles:** save to disk and restore, fork (copy-on-write, #26), and trim to a token count.
- **Batched generate:** several sequences per forward pass (#30a).
- **Metrics:** every call returns per-call metrics (TTFT, tokens/sec, peak memory, cache hits; #22).
- Errors are a typed `thiserror` enum (#7). Engine and cache handles are `Send + Sync`.

## Parallelism model
Summary of [ADR-04](architecture-decision-records/04-batch-first-and-parallelism.md):
- Every model tensor carries a batch dimension from the first layer module (#11) onward.
- KV cache v1 (#16) stores per-sequence slots.
- Engine and cache handles are `Send + Sync`, which the compiler enforces.

| Level | Module | Gate |
|---|---|---|
| Prefill: whole prompt in one forward pass | #15 | logits match Transformers |
| Batched decode: N sequences per forward pass | #30a | each sequence's greedy output matches its sequential run |
| Shared-prefix subagents: fork a cached prefix copy-on-write, decode branches concurrently | #30b (needs #26) | no cross-talk between forks under concurrent use |
| CPU/GPU overlap: tokenize, sample, detokenize on CPU while the GPU runs the next step | #30c | identical output to the non-overlapped path |

Metrics: aggregate tokens/sec at batch 1/2/4/8, per-sequence latency, and memory saved by prefix sharing.

## RAM tiers
| Mac | Model | Context |
|---|---|---|
| 8 GB | SmolLM3-3B | 16–32K |
| 16 GB | Qwen3-4B-Instruct-2507 | 32K |

`memory.rs` (#28) maps the machine's RAM to a tier and sizes the KV cache to fit.

## See also
- [TESTING.md](TESTING.md): test tiers, gates, tolerances, fixtures
- [architecture-decision-records/](architecture-decision-records/): architecture decision records
- [REFERENCES.md](REFERENCES.md): every dependency and external source

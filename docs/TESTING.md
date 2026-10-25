# Testing

## Gates
A **gate** is the check that decides whether a module is done. Every module issue states its gate as one or more commands and pass criteria, including tolerances for numeric comparisons. A module merges only when its gate passes at every tier it covers, and earlier gates must stay green.

Gate tests for the engine live in one integration-test binary, `crates/engine/tests/gates/`. `main.rs` declares one file per gated module (`config.rs`, `weights.rs`, `forward.rs`, ...). Unit tests stay next to the code they test.

## Tiers
| Tier | What | Data | Where it runs |
|---|---|---|---|
| 1 | Unit tests inside modules | none, or inline values | CI and locally |
| 2 | Gates against the tiny reference model | `fixtures/tiny/` (committed, KB-sized) | CI and locally |
| 3 | The same gates against real SmolLM3-3B, **plus every Metal/GPU test** | the model directory in `LLDX_MODEL_DIR` | Local Apple Silicon Mac only |

GitHub's macOS runners are Apple Silicon, but their GPU access is not something to rely on. Every test that touches Metal is tier 3.

### Marking tier-3 tests
Put tier-3 tests in a module named `tier3`:

```rust
#[cfg(test)]
mod tests {
    // tier 1 tests

    mod tier3 {
        // Metal or real-model tests
    }
}
```

The `ci` nextest profile skips any test whose path contains `tier3::` (`default-filter` in `.config/nextest.toml`). The `default` profile runs everything.

## Running tests
```sh
cargo nextest run --workspace               # all tiers (local Mac)
cargo nextest run --workspace --profile ci  # tiers 1 and 2, as CI runs them; writes target/nextest/ci/junit.xml
cargo test --workspace --doc                # doc tests (nextest does not run them)
```

The `ci` profile never retries (`retries = 0`), so a flaky test fails the build and gets fixed instead of hidden. It also runs every test even after a failure (`fail-fast = false`).

## Tolerance policy
Numeric comparisons never use exact equality across devices or implementations. Each gate documents:
- **the dtype** being compared (f32, f16, bf16 or quantized),
- **the metric**: element-wise `|actual − expected| ≤ atol + rtol·|expected|`, unless the gate states otherwise (for example, top-k agreement for quantized logits),
- **`atol` and `rtol`**, with a one-line justification (accumulation length, dtype epsilon, reference precision).

Tolerances are per gate and per dtype. Loosening one needs a note in the PR saying why. Every numeric gate has a row in this table.

| Gate | Test | Compared | dtype | atol | rtol | Why |
|---|---|---|---|---|---|---|
| #5 Backend (tier 3) | `device::tests::tier3::metal_matmul_matches_cpu` | `[2,64,256] × [2,256,96]` matmul, Metal vs CPU | f32 | 1e-4 | 1e-4 | Inputs in [-1, 1], K = 256. Candle compiles Metal kernels with fast math on by default and tiling can change across versions, so exact equality is not guaranteed. The bound sits well above typical f32 rounding error for this K (observed max difference on an M4: 0.0). |

## Reference fixtures
Correctness reference: Hugging Face Transformers, pinned in `fixtures/uv.lock` (transformers 5.20.2, torch 2.14.1, safetensors 0.8.0), on CPU in float32 with `torch.use_deterministic_algorithms(True)`, one thread, eager attention and seed 0.

| Fixture | Where | Model | Used by |
|---|---|---|---|
| `tiny` | `fixtures/tiny/` (committed, under 1 MB) | Random-weight SmolLM3: 4 layers (layer 3 is NoPE), hidden 64, 4 query / 2 KV heads (head dim 16), intermediate 128, vocab 512, tied embeddings, rope theta 5e6 | Tier 2 |
| `smollm3-3b` | `fixtures/out/smollm3-3b/` (gitignored) | `HuggingFaceTB/SmolLM3-3B` at revision `a07cc9a04f16550a088caea529712d1d335b0ac1` | Tier 3 |

Each fixture directory holds:
- `config.json`: the model config as Transformers writes it. The tiny config uses the v5 `rope_parameters` form, while the real checkpoint uses a top-level `rope_theta`.
- `model.safetensors` (tiny only; the real weights stay in `LLDX_MODEL_DIR`). It has the same tensor names as the real checkpoint and no `lm_head.weight` (tied).
- `reference.safetensors`: `input_ids` `[2, L]` (int64; a batch of 2 equal-length sequences, no padding), `embeddings`, then `layers.{i}.attention` (self-attention output before the residual add), `layers.{i}.mlp` (MLP output before the residual add) and `layers.{i}.hidden` (decoder layer output), all `[2, L, hidden]`, and `logits` `[2, L, vocab]`. Captured with forward hooks.

Tiny input ids are seeded random tokens (L = 8). For the real model, the ids come from tokenizing the two `reference_batch` prompts in `fixtures/prompts.json` with the model's tokenizer, truncated to the shorter one.

`fixtures/manifest.json` records, per fixture: model repo and revision, seed, library versions, and the sha256 of every file.

### Regenerating
Requires [uv](https://docs.astral.sh/uv/).

```sh
cargo xtask fixtures --tiny                    # regenerate fixtures/tiny/ and check it against the manifest
cargo xtask fixtures --model /path/to/SmolLM3-3B   # tier 3: real model, ~12 GB RAM in float32
```

`cargo xtask fixtures` runs `fixtures/generate.py` through `uv run --locked`, recomputes every sha256 in Rust, and fails if a file differs from the hashes previously recorded in the manifest. The manifest is then restored, so a failed check never overwrites the record. On the first run for a fixture it only records hashes. If a change is intended (generator edit or version bump), rerun with `--update` and commit the new files and manifest together.

`--model` first checks every file in the model directory against the hashes of the pinned revision (sha256 for LFS files, git blob sha1 for the rest, from the Hugging Face tree API), so a wrong download fails before any compute.

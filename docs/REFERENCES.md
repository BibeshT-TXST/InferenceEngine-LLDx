# References

Every dependency, tool and external source the project relies on, with a link that was opened when the entry was added. Versions are the ones pinned in the repo.

## Toolchain
| What | Version | Link | Used for |
|---|---|---|---|
| Rust stable channel manifest | 1.99.1 (2026-10-13) | https://static.rust-lang.org/dist/channel-rust-stable.toml | Confirming the current stable to pin in `rust-toolchain.toml` |
| Cargo workspaces | | https://doc.rust-lang.org/cargo/reference/workspaces.html | `[workspace.package]`, `[workspace.dependencies]`, `[workspace.lints]` |
| rustfmt configuration | | https://rust-lang.github.io/rustfmt/ | Stable options in `rustfmt.toml` |
| Clippy lint list | | https://rust-lang.github.io/rust-clippy/stable/index.html | `clippy::pedantic` group |
| RFC 430: naming conventions | | https://rust-lang.github.io/rfcs/0430-finalizing-naming-conventions.html | Naming rules in CONTRIBUTING |

## Crates
| Crate | Version | Link | Used by | Used for |
|---|---|---|---|---|
| candle-core | =0.11.0 (`metal`) | https://crates.io/crates/candle-core | engine | Tensor backend ([ADR-03](architecture-decision-records/03-tensor-backend.md)). Features from its [Cargo.toml](https://github.com/huggingface/candle/blob/0.11.0/candle-core/Cargo.toml), device API from [device.rs](https://github.com/huggingface/candle/blob/0.11.0/candle-core/src/device.rs) and [utils.rs](https://github.com/huggingface/candle/blob/0.11.0/candle-core/src/utils.rs), custom ops from [custom_op.rs](https://github.com/huggingface/candle/blob/0.11.0/candle-core/src/custom_op.rs) |
| anyhow | 1.0.104 | https://crates.io/crates/anyhow | cli, xtask | Error handling in binaries |
| clap | 4.6.7 (`derive`) | https://crates.io/crates/clap | cli, xtask | Argument parsing |
| serde_json | 1.0.154 | https://crates.io/crates/serde_json | cli (dev), xtask | Parsing JSON log lines in `tests/cli.rs`. Reading `fixtures/manifest.json` |
| sha2 | 0.11.1 | https://crates.io/crates/sha2 | xtask | Recomputing fixture sha256 hashes independently of Python |
| tracing | 0.1.44 | https://crates.io/crates/tracing | engine, cli | Structured log events |
| tracing-subscriber | 0.3.23 (`fmt`, `ansi`, `json`, `std`) | https://crates.io/crates/tracing-subscriber | cli | Subscriber installed by the CLI. Features checked in its [Cargo.toml](https://github.com/tokio-rs/tracing/blob/tracing-subscriber-0.3.23/tracing-subscriber/Cargo.toml) (`pretty()` needs `ansi`, `json()` needs `json`) |

## Tools
| What | Version | Link | Used for |
|---|---|---|---|
| cargo-nextest | 0.9.149 | https://nexte.st/docs/configuration/reference/ | Test runner, `.config/nextest.toml` keys (`retries`, `fail-fast`, `default-filter`, `junit.path`) |
| nextest filtersets | | https://nexte.st/docs/filtersets/reference/ | `test(/regex/)` matcher used to skip tier-3 tests in CI |
| cargo-deny | 0.20.2 | https://embarkstudios.github.io/cargo-deny/ | `deny.toml`. Ban syntax with `wrappers` from https://embarkstudios.github.io/cargo-deny/checks/bans/cfg.html |
| cargo-about | 0.9.2 | https://embarkstudios.github.io/cargo-about/cli/generate/config.html | `about.toml` keys. `about.hbs` adapted from the template in https://github.com/EmbarkStudios/cargo-about |
| uv | 0.13.5 (local) | https://docs.astral.sh/uv/ | Python environment and lockfile for `fixtures/` |

## Licenses and conventions
| What | Link | Used for |
|---|---|---|
| Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0.txt | `LICENSE-APACHE` (verbatim) |
| Keep a Changelog 1.1.0 | https://keepachangelog.com/en/1.1.0/ | `CHANGELOG.md` format |
| Semantic Versioning 2.0.0 | https://semver.org/spec/v2.0.0.html | Versioning policy |
| Conventional Commits 1.0.0 | https://www.conventionalcommits.org/en/v1.0.0/ | Commit message format |

## CI
| What | Version | Link | Used for |
|---|---|---|---|
| GitHub runner images | `macos-26` (arm64) | https://github.com/actions/runner-images | Apple Silicon CI runner label |
| actions/checkout | v7.0.1 @ `3d3c42e5aac5ba805825da76410c181273ba90b1` | https://github.com/actions/checkout | Checkout in `ci.yml` |
| taiki-e/install-action | v2.87.31 @ `1e29709c0ec63babbe8f266397a465f29a825f91` | https://github.com/taiki-e/install-action | Installing pinned cargo-nextest/cargo-deny in CI (both listed in its TOOLS.md) |
| Swatinem/rust-cache | v2.9.3 @ `d1527c8f67ef9af357290475d4af4b0fab8f12e0` | https://github.com/Swatinem/rust-cache | Cargo cache in `ci.yml` |

## Backend sources
| What | Link | Used for |
|---|---|---|
| Candle repository | https://github.com/huggingface/candle | ADR-03 |
| Metal kernel compilation (runtime, fast-math default) | https://github.com/huggingface/candle/blob/0.11.0/candle-metal-kernels/src/kernel.rs | ADR-03: no full Xcode needed. Tolerance rationale in TESTING |
| Metal kernel source list | https://github.com/huggingface/candle/blob/0.11.0/candle-metal-kernels/src/source.rs | ADR-03: available Metal ops |
| candle-transformers SmolLM3 (cross-check only) | https://github.com/huggingface/candle/blob/0.11.0/candle-transformers/src/models/smol/smollm3.rs | ADR-03. Read-only reference for #11–#15 |
| candle-transformers Qwen3 (cross-check only) | https://github.com/huggingface/candle/blob/0.11.0/candle-transformers/src/models/qwen3.rs | ADR-03. Read-only reference for the second target |

## Models
| What | Revision | Link | Used for |
|---|---|---|---|
| SmolLM3-3B (Apache-2.0) | `a07cc9a04f16550a088caea529712d1d335b0ac1` | https://huggingface.co/HuggingFaceTB/SmolLM3-3B | First target. Tier-3 fixtures |
| SmolLM3-3B file hashes at that revision | | https://huggingface.co/api/models/HuggingFaceTB/SmolLM3-3B/tree/a07cc9a04f16550a088caea529712d1d335b0ac1 | `REAL_FILES` in `fixtures/generate.py` |
| SmolLM3-3B `config.json` at that revision | | https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/a07cc9a04f16550a088caea529712d1d335b0ac1/config.json | NoPE layout, legacy `rope_theta`, tied embeddings |

## Python fixture stack (pinned in `fixtures/uv.lock`)
| Package | Version | Link | Used for |
|---|---|---|---|
| transformers | 5.20.2 | https://pypi.org/project/transformers/5.20.2/ | Correctness reference |
| torch | 2.14.1 | https://pypi.org/project/torch/2.14.1/ | CPU float32 forward pass |
| safetensors | 0.8.0 | https://pypi.org/project/safetensors/0.8.0/ | Fixture file format |
| SmolLM3 config (field names, NoPE rule) | 5.20.2 | https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/smollm3/configuration_smollm3.py | Tiny config. Also checked in the installed copy |
| SmolLM3 modeling (hook points, return types) | 5.20.2 | https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/smollm3/modeling_smollm3.py | Forward hooks in `generate.py`. Also checked in the installed copy |

## Security advisories (ignored in `deny.toml`)
| ID | Link | Why ignored |
|---|---|---|
| RUSTSEC-2026-0367 | https://rustsec.org/advisories/RUSTSEC-2026-0367.html | `zip::ZipArchive::extract()` only. candle-core 0.11.0 never calls `extract` |
| RUSTSEC-2026-0368 | https://rustsec.org/advisories/RUSTSEC-2026-0368.html | Same |
| RUSTSEC-2026-0369 | https://rustsec.org/advisories/RUSTSEC-2026-0369.html | Same |
| RUSTSEC-2024-0436 | https://rustsec.org/advisories/RUSTSEC-2024-0436.html | `paste` unmaintained. Compile-time proc-macro via `gemm`, no safe upgrade |

## Background
| What | Link | Used for |
|---|---|---|
| MLX | https://github.com/ml-explore/mlx | Alternative weighed in ADR-01 |

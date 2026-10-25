# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- #0: Cargo workspace (edition 2024, toolchain pinned to Rust 1.99.1) with the `lldx-engine` library crate exposing `VERSION`.
- #0: CI on an Apple Silicon macOS runner (format, clippy, doc tests), with actions pinned by commit SHA.
- #0: Module issue form, pull request template, README, ARCHITECTURE, ADR-01 (Rust), ADR-02 (library design), ADR-04 (batch-first and parallelism) and REFERENCES.
- #0: Dual license, MIT OR Apache-2.0.
- #1: `rustfmt.toml`, workspace lints (`unsafe_code` denied, `clippy::pedantic` as warnings) and CONTRIBUTING guide.
- #2: nextest profiles (`default`, `ci` with JUnit output, no retries, tier-3 tests skipped), the `gates` integration-test binary with a VERSION smoke test, and TESTING guide. CI now runs nextest.
- #3: `lldx-engine` CLI (`lldx-engine-cli` package) with `--version` and `--log-format json|pretty`. It writes logs to stderr and emits a startup event carrying the engine version.
- #4: `cargo deny` policy (RustSec advisories, license allowlist, crates.io-only sources, `tracing-subscriber` banned outside the CLI) and `cargo about` license report template. CI now runs `cargo deny check`.
- #5: ADR-03 (Accepted): Candle with Metal as the tensor backend, custom Metal kernels for hot spots, Candle kept out of the public API. Adds `candle-core =0.11.0` and internal Metal device selection with CPU fallback, gated by a tier-3 Metal-vs-CPU matmul test.
- #6: Deterministic reference fixtures. `fixtures/generate.py` (Transformers 5.20.2, torch 2.14.1, CPU float32) builds a tiny random-weight SmolLM3 (4 layers incl. one NoPE, committed in `fixtures/tiny/`) and, for tier 3, SmolLM3-3B at revision `a07cc9a0`. `fixtures/manifest.json` records versions and sha256 hashes. `cargo xtask fixtures` regenerates and verifies them.

### Security
- #5: `deny.toml` ignores RUSTSEC-2026-0367/0368/0369 (`zip` `extract()`, not reachable through candle-core 0.11.0) and RUSTSEC-2024-0436 (`paste` unmaintained). Revisit on every Candle upgrade.

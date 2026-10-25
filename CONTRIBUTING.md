# Contributing

## Workflow
- Every piece of work belongs to a **module issue** (use the "Module" issue form). One issue, one branch, one PR.
- Branch names are `<n>-<short-slug>`, for example `11-embedding`.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), with the module number as the scope:
  `feat(#11): add token embedding layer`, `fix(#16): reset cache length on trim`, `docs(#5): accept ADR-03`.
  Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `style`, `chore`.
- Decisions that are hard to reverse get an ADR in `docs/architecture-decision-records/` (copy `00-template.md`).

## Definition of done
A module is done when all of the following hold:
1. Its **gate passes** at every tier it covers ([docs/TESTING.md](docs/TESTING.md)). Tier 3 results are run locally on a Mac and pasted into the PR.
2. **Metrics and/or log events** cover the new code path.
3. **Docs** are updated: ARCHITECTURE, TESTING and ADRs where relevant.
4. **`docs/REFERENCES.md`** has an entry for every new dependency and external source.
5. **`CHANGELOG.md`** has an entry under `[Unreleased]`.

The PR template mirrors this checklist.

## Code conventions
- **Workspace layout:** `crates/engine` (library `lldx-engine`), `crates/cli` (binary `lldx-engine`), `crates/xtask` (dev tasks, run with `cargo xtask`).
- **Naming:** standard Rust naming (RFC 430). Modules are named after the concept they implement (`rope.rs`, `sampling.rs`), not after their type. Tensor shapes are written in doc comments as `[batch, seq, hidden]`.
- **Module boundaries:** one concept per file, following the module map in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Items are private by default. Anything `pub` is part of the API contract.
- **Candle stays internal.** No `candle_core` type appears in a `pub` signature of `lldx-engine`.
- **Formatting and lints:** `cargo fmt` (configured in `rustfmt.toml`) and `cargo clippy --workspace --all-targets -- -D warnings`. Workspace lints enable `clippy::pedantic` as warnings, which CI turns into errors. Silence a lint with `#[expect(lint, reason = "...")]` on the smallest item possible, never crate-wide.
- **Dependencies:** added to `[workspace.dependencies]` in the root `Cargo.toml` with an exact version checked with `cargo add`/`cargo search`, then referenced with `workspace = true`. Every new dependency must pass `cargo deny check`.

## Errors
- **Library (`lldx-engine`):** typed errors with `thiserror`. No `anyhow`, no `unwrap`/`expect` on fallible paths, and no panics across the public API.
- **Binaries (`lldx-engine-cli`, `xtask`):** `anyhow` with context on every fallible call.

## `unsafe`
- `unsafe_code` is denied workspace-wide. It may be allowed only in `weights.rs` (memory-mapped safetensors) and `kernels/` (Metal kernels), with `#[allow(unsafe_code)]` on the smallest item possible.
- Every `unsafe` block has a `// SAFETY:` comment directly above it that states the invariants and why they hold.

## Logging
- The library **only emits** [`tracing`](https://docs.rs/tracing) events and spans. It never installs a subscriber, prints to stdout or stderr, or reads logging environment variables. `cargo deny check` enforces this by allowing `tracing-subscriber` only as a dependency of the CLI.
- Binaries install the subscriber. The CLI writes logs to **stderr** (`--log-format json|pretty`), so stdout stays free for generated text.
- Levels: `error` for failures returned to the caller, `warn` for degraded operation (for example, falling back to CPU), `info` for one-off lifecycle events (model loaded, run summary), `debug` for per-call detail, and `trace` for per-token or per-layer detail.
- Use structured fields (`info!(tokens = n, "prefill done")`), not formatted strings. Never log prompt or log content above `trace`.

# ADR-02: Engine as a library imported in-process by the harness

- **Status:** Accepted
- **Date:** 2026-10-25
- **Module:** #0

## Context
LLDx has two parts: the inference engine (this repo) and the harness (read-only log-debugging tools, log intelligence and an agent loop). Agent loops ask many follow-up questions over the same long context (logs, code). Re-running prefill on that shared context for every turn wastes most of the compute.

## Decision
Two repos. This repo is a library crate, `lldx-engine`, plus a dev/benchmark CLI (`lldx-engine`). The harness depends on the crate and calls it in-process. There is no HTTP server.

## Alternatives considered
- **OpenAI-compatible HTTP server:** language-agnostic and familiar, but cache state hides behind a request/response API. Prefix caching becomes a server heuristic the client cannot steer, and fork or trim has no natural endpoint.
- **One repo for engine and harness:** simpler at first, but it blurs the API boundary and lets harness concerns leak into the engine.

## Consequences
- **Direct cache-handle control.** The harness holds KV cache handles and can save, restore, fork and trim them. Example: fork the cached log context for two parallel subagents, or trim back to before a failed tool call.
- **A real public API boundary.** The crate's API is the contract. It is finalized at #20 and versioned with SemVer.
- **No serialization overhead.** Tokens stream to the caller through a callback, not over a socket.
- **Accepted cost:** the harness must be written in Rust (or reach the crate through FFI later), and a crash in the engine takes the harness process down with it.

## Mitigations
- Candle types stay out of the public API ([ADR-03](03-tensor-backend.md)), so the boundary is stable even when backend internals change.
- Typed errors (`thiserror`) at the boundary, and `unsafe` confined to `weights.rs` and `kernels/`.

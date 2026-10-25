//! `lldx-engine` is a from-scratch LLM inference engine for Apple Silicon.
//!
//! It is a library crate. The harness (a separate repository) imports it and
//! calls it in-process, so it can hold KV cache handles directly instead of
//! going through a server.
//!
//! The crate is in Phase 0 (foundation). The public API is planned in
//! `docs/ARCHITECTURE.md` and is finalized at module #20.

/// The version of this crate, taken from `Cargo.toml` at compile time.
///
/// ```
/// assert!(!lldx_engine::VERSION.is_empty());
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

mod device;

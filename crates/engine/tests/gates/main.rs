//! Gate tests: a single integration-test binary, one file per gated module.
//!
//! See `docs/TESTING.md` for the tier each gate runs at.

#[test]
fn version_matches_package_version() {
    assert_eq!(lldx_engine::VERSION, env!("CARGO_PKG_VERSION"));
}

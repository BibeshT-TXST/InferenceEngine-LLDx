## Module
Closes #<issue> — module #<n>: <title>

## What changed
<!-- One paragraph. Link the ADR if this PR makes or changes a decision. -->

## Gate
<!-- The exact command(s) and their result. -->
```
```

## Definition of done
- [ ] The gate passes at every tier it covers (tier 1 / tier 2 in CI, tier 3 run locally on a Mac and pasted above)
- [ ] Metrics and/or log events added for the new code path
- [ ] `cargo fmt --check` and `cargo clippy --workspace --all-targets -- -D warnings` clean
- [ ] Any `unsafe` is in `weights.rs` or `kernels/` only, each block with a `// SAFETY:` comment
- [ ] Docs updated (ARCHITECTURE / TESTING / ADRs as relevant)
- [ ] `docs/REFERENCES.md` has an entry for every new dependency or external source
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Commits follow Conventional Commits with the module number, e.g. `feat(#11): ...`

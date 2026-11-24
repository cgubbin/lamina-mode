# 0001: Family dependencies carry a version and a sibling path

Status: accepted

## Context

lamina-units, lamina-core, lamina-materials and lamina-mode are published to crates.io in dependency order. A
published crate can't depend on path dependencies, but the crates are developed side by side in sibling
checkouts.

## Decision

Each family dependency names both a version and a sibling path:

```toml
lamina-core = { version = "0.1.0", path = "../lamina-core" }
```

Cargo uses the path in a local build and drops it when it packages the crate,
so the published manifest depends on the version alone. A local build keeps
working against the siblings, and the lockfile doesn't change between the two.

Until the earlier crates are on crates.io, `cargo publish --dry-run` can't
resolve them from the registry. The dry run patches them to the siblings on the
command line, which affects only that invocation:

```sh
cargo publish --dry-run \
  --config 'patch.crates-io.argloc.path="../argloc"' \
  --config 'patch.crates-io.lamina-units.path="../lamina-units"' \
  --config 'patch.crates-io.lamina-core.path="../lamina-core"' \
  --config 'patch.crates-io.lamina-materials.path="../lamina-materials"'
```

The real publish needs no patch: each crate is published after the crates it
depends on, so its versions resolve from the registry.

lamina-materials is a dev-dependency of lamina-mode: only tests and examples
use it. crates.io still requires it to exist, so it is published before
lamina-mode.

## Alternatives

- **Version only, with a committed `[patch.crates-io]`:** the same sibling
  layout is needed, and the lockfile flips between patched and registry
  sources when the patch is removed after publishing.
- **One workspace for the family:** `cargo publish --workspace` would order the
  publish and resolve unpublished members itself, but the crates live in
  separate repositories with separate visibility.

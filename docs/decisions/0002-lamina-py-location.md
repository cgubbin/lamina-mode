# 0002: lamina-py lives in lamina-mode's workspace

Status: accepted

## Context

lamina-py provides Python bindings over the public APIs of the lamina crates and needs a home: its own
repository, or a workspace beside one of the crates it binds.

## Decision

lamina-py is a member of a Cargo workspace rooted at lamina-mode, in
`lamina-py/`. The root package stays lamina-mode, so `cargo build`, `cargo test`
and `cargo publish` at the root act on lamina-mode alone, and lamina-mode's
package include list keeps lamina-py out of its crate.

lamina-py depends on lamina-mode, lamina-core and lamina-units through their
public APIs only (version plus path, as in [0001](0001-family-dependencies.md)),
so it can move to its own repository unchanged if other crates need
bindings too. It is distributed as a Python wheel and is
marked `publish = false` for crates.io.

## Consequences

- The dev shell carries maturin, zig (for manylinux wheels) and Python.
- `cargo deny` at the root covers the bindings' dependencies.
- A change to lamina-mode's public API shows up in the same review as its
  effect on the bindings.

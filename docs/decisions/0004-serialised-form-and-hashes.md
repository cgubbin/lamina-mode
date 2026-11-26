# 0004: Canonical JSON records with SHA-256 identities

Status: accepted (schema version 2: see 0010)

## Context

Results are cached and compared against other calculations, so equal inputs must serialise and hash equally,
and a serialised result must say which schema and conventions it follows. Evidence travels with every answer.

## Decision

- **Format:** JSON. Each record names its `schema` (`lamina.guided_input`,
  `lamina.guided_basis`, `lamina.guided_profile`, `lamina.local_derivatives`)
  and `schema_version` (1). Results also carry the `conventions` they were
  produced under (units, frame, polarisation, phasor sign, normalisation and
  the interface-sample rule) and the analytic `assumption` behind them. A
  change to any of those conventions bumps the schema version.
- **Canonical form:** serialised by the Rust side from typed records, with
  fields in schema order, no whitespace, and shortest round-trip floats
  (serde_json with `float_roundtrip` when parsing). Complex numbers are
  `[re, im]`. Signed zeros in inputs are normalised. The Python classes build
  their JSON and pass it back through the Rust side, so there is one canonical
  form.
- **Identity:** the SHA-256 of the canonical form, as lowercase hex. A basis
  records the `input_hash` of its resolved input; profiles and derivatives
  record the hash of the input they came from.
- **Checks on reading:** unknown fields, another schema, another version or
  different conventions are rejected (`SchemaError`), as is a basis whose
  `input_hash` or `found` doesn't match its contents.
- **Resolved inputs:** a default |β| window is resolved before hashing, and
  the resolved window is what the record stores.
- **Re-solving:** a deserialised basis has no live roots. To sample fields or
  derivatives it re-solves its stored input and refuses to continue unless the
  new record is identical to the stored one.

## Alternatives

- **A binary format (CBOR, MessagePack):** smaller, but not readable in review,
  and JSON already round-trips f64 exactly.
- **Serde derives in lamina-mode itself:** would give Rust consumers the same
  form; deferred until a Rust consumer needs it, at which point these records
  move down unchanged.
- **Hashing Python's own JSON:** float formatting and key order would then
  depend on the Python side.

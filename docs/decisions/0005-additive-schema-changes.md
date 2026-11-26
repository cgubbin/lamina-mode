# 0005: Additive schema changes keep schema version 1

Status: accepted (schema version 2: see 0010)

## Context

Sellmeier materials were added to the input, and two result kinds (`lamina.guided_sensitivity`,
`lamina.group_index`). Serialised results carry a schema version that changes when a convention changes
(decision 0004). An addition that changes no convention needs a rule too.

## Decision

An addition that changes no convention keeps the schema version:

- **New record kinds** (`lamina.guided_sensitivity`, `lamina.group_index`) are version 1 records of their own
  schema.
- **New input variants** extend the existing record compatibly. A material is now an untagged choice whose first
  variant is the existing constant material, so every record written before the change parses to the same value,
  serialises to the same canonical JSON and keeps its hash. The test suite pins two such hashes.
- **Strict parsing stays.** Records reject unknown fields, so a build that predates an addition rejects a record
  that uses it (`SchemaError`) rather than misreading it.

A change to a convention (units, frame, phasor sign, normalisation, the interface-sample rule) still bumps the
version.

## Consequences

- Readers can rely on "version 1 or older build fails loudly", not on "version 1 means a fixed field set".
- Changing a default (for example a tolerance) changes the canonical input of requests that rely on it, and so
  their hash; requests that state the value explicitly keep theirs. See 0003's revision.

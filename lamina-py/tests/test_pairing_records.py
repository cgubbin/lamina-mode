"""Acceptance: pairing results round-trip through their serialised form."""

from __future__ import annotations

import json

import numpy as np
import pytest

import lamina

WAVELENGTH = 1.0e-6


def solve(stack, polarisation="TE", direction="forward"):
    return lamina.enumerate_guided_modes(
        stack, lamina.GuidedRequest(wavelength_m=WAVELENGTH, polarisation=polarisation, direction=direction)
    )


@pytest.fixture(scope="module")
def bases():
    a = solve(lamina.Stack(left=1.0, layers=[(4.0, 0.42e-6), (2.25, 0.1e-6)], right=1.44))
    b = solve(lamina.Stack(
        left=1.0,
        layers=[(lamina.SellmeierMaterial(1.5, [(1.2, 0.7e-6)]), 0.6e-6)],
        right=1.44,
    ))
    return a, b


def gaussian(z, side):
    ey = np.exp(-((z - 0.2e-6) / 0.3e-6) ** 2)
    return (0.0, ey, 0.0), (0.0, 0.0, ey / 376.730313412 * 1.8)


@pytest.fixture(scope="module")
def records(bases):
    a, b = bases
    thickness = lamina.Parameter.thickness(0)
    return {
        "lamina.guided_power_overlap": lamina.power_overlap(a, 0, b, 0, (0.0, 0.05e-6)),
        "lamina.guided_overlap_sensitivity": lamina.power_overlap_sensitivity(
            a, 0, b, 0, None, thickness, (0.0, 0.05e-6)),
        "lamina.guided_reciprocity": lamina.reciprocity(a, 0, b, 1, (0.0, 0.05e-6)),
        "lamina.guided_local_coupling": a.local_coupling(
            lamina.Parameter.interface_position(1), gap_rad_per_m=a.k0_rad_per_m),
        "lamina.guided_projection": a.project_field(gaussian, (-2e-6, 2.5e-6), (0.42e-6,), field_id="gaussian"),
    }


SCHEMAS = [
    "lamina.guided_power_overlap",
    "lamina.guided_overlap_sensitivity",
    "lamina.guided_reciprocity",
    "lamina.guided_local_coupling",
    "lamina.guided_projection",
]


@pytest.mark.parametrize("schema", SCHEMAS)
def test_records_round_trip_with_equal_hashes(records, schema):
    record = records[schema]
    text = record.to_json()
    restored = type(record).from_json(text)
    assert restored == record
    assert restored.to_json() == text
    assert restored.canonical_hash() == record.canonical_hash()
    assert hash(restored) == hash(record)


@pytest.mark.parametrize("schema", SCHEMAS)
def test_records_carry_schema_conventions_definition_and_inputs(records, bases, schema):
    a, b = bases
    record = json.loads(records[schema].to_json())
    assert record["schema"] == schema
    assert record["schema_version"] == lamina.SCHEMA_VERSION == 1
    assert record["conventions"] == dict(lamina.CONVENTIONS)
    assert record["definition"]
    if "input_hash" in record:
        assert record["input_hash"] == a.input_hash
        assert record["assumption"] == a.assumption
    else:
        assert (record["first_input_hash"], record["second_input_hash"]) == (a.input_hash, b.input_hash)
        # Either stack with a Sellmeier material makes the declaration dispersive.
        assert record["assumption"] == b.assumption != a.assumption


def test_optional_parts_serialise_as_null(records):
    coupling = json.loads(records["lamina.guided_local_coupling"].to_json())
    # The wide gap leaves every pair unresolved.
    assert all(v is None for row in coupling["coefficients_real"] for v in row)
    assert coupling["unresolved_groups"] and coupling["parameter"] == {"kind": "interface", "interface": 1}
    overlap = json.loads(records["lamina.guided_overlap_sensitivity"].to_json())
    assert overlap["first_parameter"] is None
    assert overlap["second_parameter"] == {"kind": "thickness", "layer": 0}
    assert overlap["derivative_failure"] is None
    assert overlap["power_rate_w_per_m"] is not None


def test_resolved_coupling_round_trips(bases):
    a, _ = bases
    record = a.local_coupling(lamina.Parameter.permittivity(1))
    assert record.coefficient_resolved.sum() == a.found * (a.found - 1)
    restored = lamina.LocalCoupling.from_json(record.to_json())
    assert restored == record
    np.testing.assert_array_equal(restored.coefficient_resolved, record.coefficient_resolved)


@pytest.mark.parametrize("schema", SCHEMAS)
@pytest.mark.parametrize(
    "tamper",
    [
        lambda r: r.update(unexpected=1),
        lambda r: r.update(schema="lamina.something_else"),
        lambda r: r.update(schema_version=lamina.SCHEMA_VERSION + 1),
        lambda r: r["conventions"].update(power="W"),
        lambda r: r.update(definition=r["definition"] + " "),
    ],
    ids=["unknown_field", "schema", "version", "conventions", "definition"],
)
def test_strict_parsing_rejects_tampered_records(records, schema, tamper):
    record = records[schema]
    d = json.loads(record.to_json())
    tamper(d)
    with pytest.raises(lamina.SchemaError):
        type(record).from_json(json.dumps(d))


def test_record_kinds_reject_each_other(records):
    text = records["lamina.guided_power_overlap"].to_json()
    with pytest.raises(lamina.SchemaError):
        lamina.Reciprocity.from_json(text)


def test_integration_failed_is_a_lamina_error():
    assert issubclass(lamina.IntegrationFailed, lamina.LaminaError)
    assert "IntegrationFailed" in lamina.__all__

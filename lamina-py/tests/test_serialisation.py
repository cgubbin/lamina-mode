"""Acceptance: results round-trip through their serialised form."""

from __future__ import annotations

import json

import numpy as np
import pytest

import lamina


@pytest.fixture(scope="module")
def basis():
    stack = lamina.Stack(left=1.0, layers=[(4.0, 1.0e-6), (2.25, 0.5e-6)], right=2.25)
    return lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(wavelength_m=1.0e-6))


def roundtrip(record):
    text = record.to_json()
    restored = type(record).from_json(text)
    assert restored == record
    assert restored.to_json() == text
    assert restored.canonical_hash() == record.canonical_hash()
    return restored


def test_basis_round_trips(basis):
    restored = roundtrip(basis)
    assert restored.modes == basis.modes
    np.testing.assert_array_equal(restored.n_eff, basis.n_eff)


def test_profile_round_trips(basis):
    z = np.linspace(-1e-6, 2.5e-6, 57)
    profile = basis.sample_fields(1, z)
    restored = roundtrip(profile)
    np.testing.assert_array_equal(restored.z_m, z)
    np.testing.assert_array_equal(restored.electric_v_per_m, profile.electric_v_per_m)
    np.testing.assert_array_equal(restored.magnetic_a_per_m, profile.magnetic_a_per_m)


def test_derivatives_round_trip(basis):
    roundtrip(basis.local_derivatives(0, thickness_layers=[0, 1]))


def test_input_round_trips(basis):
    roundtrip(basis.input)


def test_records_carry_schema_and_conventions(basis):
    record = json.loads(basis.to_json())
    assert record["schema"] == "lamina.guided_basis"
    assert record["schema_version"] == lamina.SCHEMA_VERSION
    assert record["conventions"] == dict(lamina.CONVENTIONS)
    assert record["conventions"]["wavenumber"] == "rad/m"
    assert record["input"]["schema"] == "lamina.guided_input"


@pytest.mark.parametrize(
    "tamper",
    [
        lambda r: r.update(schema_version=lamina.SCHEMA_VERSION + 1),
        lambda r: r.update(schema="lamina.something_else"),
        lambda r: r["conventions"].update(length="um"),
        lambda r: r.update(input_hash="0" * 64),
        lambda r: r.update(found=r["found"] + 1),
        lambda r: r.update(unexpected=1),
    ],
    ids=["version", "schema", "conventions", "input_hash", "found", "unknown_field"],
)
def test_tampered_records_are_rejected(basis, tamper):
    record = json.loads(basis.to_json())
    tamper(record)
    with pytest.raises(lamina.SchemaError):
        lamina.GuidedBasis.from_json(json.dumps(record))


def test_deserialised_basis_samples_identical_fields(basis):
    restored = lamina.GuidedBasis.from_json(basis.to_json())
    z = np.linspace(-0.5e-6, 2e-6, 31)
    assert restored.sample_fields(0, z) == basis.sample_fields(0, z)
    assert restored.local_derivatives(0, [1]) == basis.local_derivatives(0, [1])

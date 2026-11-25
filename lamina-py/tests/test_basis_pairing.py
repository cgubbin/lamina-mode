"""Pairing matrices between modal bases: identity, registration, records and
the error mapping."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import lamina

WAVELENGTH = 1.0e-6
K0 = 2.0 * math.pi / WAVELENGTH
HALF = 0.8e-6
WINDOW = lamina.BetaWindow.forward(2.1 * K0, 2.0 * K0, 0.05 * K0)


def centred(a: float, kind: str = "pec", wavelength: float = WAVELENGTH, right: float = HALF):
    """A core of half-width `a` centred between walls at −HALF and +`right`
    about the core centre; the first interface is at z = 0."""
    stack = lamina.Stack(
        2.25, [(4.0, 2 * a)], 2.25, lamina.Wall(kind, -(HALF - a)), lamina.Wall(kind, a + right)
    )
    return lamina.modal_basis(stack, wavelength, "TE", WINDOW)


def test_a_basis_pairs_with_itself_as_the_identity():
    basis = centred(0.3e-6)
    pairing = lamina.basis_pairing(basis, basis)
    n = len(basis)
    assert pairing.rows == pairing.columns == tuple(range(n))
    assert np.abs(pairing.unconjugated_w_per_m - np.eye(n)).max() < 1e-9
    assert max(pairing.self_defects) < 1e-9
    assert pairing.status == ("complete", "complete") and pairing.scope == ("discrete", "discrete")
    assert np.all(np.isfinite(pairing.conjugated_w_per_m))
    assert all(a is None for row in pairing.conjugated_absent for a in row)
    assert len(pairing.leading_condition_numbers) == n
    np.testing.assert_allclose(pairing.singular_values_w_per_m, 1.0, atol=1e-8)
    assert lamina.BasisPairing.from_json(pairing.to_json()) == pairing
    assert pairing.first_input_hash == pairing.second_input_hash == basis.input_hash


def test_different_cores_pair_under_registered_walls():
    first, second = centred(0.3e-6), centred(0.22e-6)
    # The cores' left interfaces lie 0.08 um apart when the walls coincide.
    pairing = lamina.basis_pairing(first, second, (0.0, 0.08e-6))
    assert pairing.unconjugated_w_per_m.shape == (len(first), len(second))
    assert pairing.origins_m == (0.0, 0.08e-6)
    transposed = lamina.basis_pairing(second, first, (0.08e-6, 0.0))
    np.testing.assert_array_equal(transposed.unconjugated_w_per_m, pairing.unconjugated_w_per_m.T)
    record = json.loads(pairing.to_json())
    record["definitions"]["unconjugated_pairing"] = "(1/4) ∫ (E1 × H2) · x dz"
    with pytest.raises(lamina.SchemaError):
        lamina.BasisPairing.from_json(json.dumps(record))


@pytest.mark.parametrize(
    "second, origins",
    [
        (lambda: centred(0.3e-6, right=0.9e-6), (0.0, 0.0)),
        (lambda: centred(0.3e-6, kind="pmc"), (0.0, 0.0)),
        (lambda: centred(0.3e-6, wavelength=1.01e-6), (0.0, 0.0)),
        (lambda: centred(0.3e-6), (0.0, 1e-9)),
        (lambda: centred(0.3e-6), (0.0, math.inf)),
    ],
    ids=["right wall", "wall kind", "wavelength", "registration", "origin"],
)
def test_incompatible_bases_are_invalid(second, origins):
    with pytest.raises(lamina.InvalidInput):
        lamina.basis_pairing(centred(0.3e-6), second(), origins)


def test_failures_are_lamina_errors():
    for error in (lamina.IntegrationFailed, lamina.FieldsFailed):
        assert issubclass(error, lamina.LaminaError)


def test_a_basis_without_modes_pairs_as_an_empty_matrix():
    stack = lamina.Stack(2.25, [(4.0, 0.6e-6)], 2.25, lamina.Wall("pec", -0.5e-6), lamina.Wall("pec", 1.1e-6))
    empty = lamina.modal_basis(stack, WAVELENGTH, "TE", lamina.BetaWindow((2.05 * K0, 2.09 * K0), (-0.01 * K0, 0.01 * K0)))
    assert empty.is_complete and empty.found == 0
    pairing = lamina.basis_pairing(empty, centred(0.3e-6))
    assert pairing.unconjugated_w_per_m.shape == (0, len(centred(0.3e-6)))
    assert len(pairing.singular_values_w_per_m) == 0 and len(pairing.leading_condition_numbers) == 0

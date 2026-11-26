"""Projection of supplied fields onto a guided basis."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import lamina

WAVELENGTH = 1.0e-6
K0 = 2.0 * math.pi / WAVELENGTH
Z0 = 376.730313412
LEFT, CORE, RIGHT, D = 1.0, 4.0, 2.25, 0.8e-6
COEFFICIENTS = [0.8 + 0.2j, -0.3 + 0.4j, 0.1 - 0.25j, 0.05 + 0.6j]


def solve(polarisation):
    stack = lamina.Stack(left=LEFT, layers=[(CORE, D)], right=RIGHT)
    basis = lamina.enumerate_guided_modes(
        stack, lamina.GuidedRequest(wavelength_m=WAVELENGTH, family=polarisation)
    )
    assert basis.is_complete and basis.found >= 2
    return basis


def region(z, side):
    """0 left exterior, 1 core, 2 right exterior; `side` picks the limit on an interface."""
    edges = (0.0, D)
    if side == "left":
        return sum(1 for e in edges if e < z)
    return sum(1 for e in edges if e <= z)


def slab_mode(beta, polarisation, amplitude):
    """Analytic fields (E in V/m, H in A/m) of an asymmetric slab mode, scaled
    so that the left-interface Ey (TE) or Z0 Hy (TM) is `amplitude`."""
    eps = (LEFT, CORE, RIGHT)
    a_left = math.sqrt(beta * beta - K0 * K0 * LEFT)
    a_right = math.sqrt(beta * beta - K0 * K0 * RIGHT)
    q = math.sqrt(K0 * K0 * CORE - beta * beta)
    # The primary field's slope jumps by eps ratios for TM (continuous dpsi/dz / eps).
    rho = 1.0 if polarisation == "TE" else CORE / LEFT
    core = lambda z: math.cos(q * z) + rho * a_left / q * math.sin(q * z)
    core_slope = lambda z: -q * math.sin(q * z) + rho * a_left * math.cos(q * z)

    def field(z, side):
        r = region(z, side)
        if r == 0:
            psi = math.exp(a_left * z)
            slope = a_left * psi
        elif r == 1:
            psi, slope = core(z), core_slope(z)
        else:
            psi = core(D) * math.exp(-a_right * (z - D))
            slope = -a_right * psi
        psi, slope = amplitude * psi, amplitude * slope
        if polarisation == "TE":
            e = [0.0, psi, 0.0]
            h = [1j * slope / K0 / Z0, 0.0, beta * psi / K0 / Z0]
        else:
            e = [-1j * slope / (K0 * eps[r]), 0.0, -beta * psi / (K0 * eps[r])]
            h = [0.0, psi / Z0, 0.0]
        return np.array(e, dtype=complex), np.array(h, dtype=complex)

    return field, min(a_left, a_right)


def superposition(basis, coefficients):
    pol = basis.input.request.family
    modes = []
    for mode in basis.modes:
        at_zero = basis.sample_fields(mode.index, [0.0])
        amplitude = at_zero.electric_v_per_m[0, 1] if pol == "TE" else Z0 * at_zero.magnetic_a_per_m[0, 1]
        modes.append(slab_mode(mode.beta_rad_per_m.real, pol, amplitude))
    decay = min(a for _, a in modes)

    def field(z, side):
        e = np.zeros(3, dtype=complex)
        h = np.zeros(3, dtype=complex)
        for (f, _), c in zip(modes, coefficients):
            fe, fh = f(z, side)
            e += c * fe
            h += c * fh
        return e, h

    # The truncated tails carry a fraction exp(-50) of the power.
    return field, [m for m, _ in modes], 25.0 / decay


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
def test_analytic_slab_fields_match_the_sampled_profiles(polarisation):
    basis = solve(polarisation)
    _, modes, _ = superposition(basis, COEFFICIENTS)
    z = [-0.3e-6, 0.1e-6, 0.3e-6, D, 0.8e-6]
    for mode, field in zip(basis.modes, modes):
        profile = basis.sample_fields(mode.index, z)
        analytic = [field(x, "right") for x in z]
        e = np.array([a for a, _ in analytic])
        h = np.array([b for _, b in analytic])
        np.testing.assert_allclose(e, profile.electric_v_per_m, rtol=0, atol=1e-6 * np.abs(e).max())
        np.testing.assert_allclose(h, profile.magnetic_a_per_m, rtol=0, atol=1e-6 * np.abs(h).max())


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
def test_projection_reconstructs_a_superposition_of_basis_modes(polarisation):
    basis = solve(polarisation)
    expected = np.array(COEFFICIENTS[: basis.found])
    field, _, tail = superposition(basis, expected)
    calls = {}

    def counted(z, side):
        calls[(z, side)] = calls.get((z, side), 0) + 1
        return field(z, side)

    result = basis.project_field(counted, (-tail, D + tail), interfaces_m=(0.0, D))
    assert result.rank == basis.found and result.is_full_rank
    np.testing.assert_allclose(result.coefficients, expected, rtol=0, atol=1e-7)
    assert result.residual_w_per_m < 1e-9
    assert result.condition_number < 1.000001
    np.testing.assert_allclose(result.gram_w_per_m, np.eye(basis.found), rtol=0, atol=1e-8)
    # Each distinct (position, side) is sampled once, and the interfaces from each side.
    assert set(calls.values()) == {1}
    assert result.field_sample_count == len(calls)
    assert {side for z, side in calls if z == D} == {"left", "right"}
    assert all(side == "interior" for z, side in calls if not (z in (0.0, D, -tail, D + tail)))


def test_rank_deficiency_is_a_result():
    basis = solve("TE")
    field, _, tail = superposition(basis, COEFFICIENTS)
    full = basis.project_field(field, (-tail, D + tail), (0.0, D))
    s = full.singular_values_w_per_m
    assert s.max() > s.min()
    # A threshold between the smallest and largest singular value.
    rtol = 0.5 * (s.min() / s.max() + 1.0)
    deficient = basis.project_field(field, (-tail, D + tail), (0.0, D), rank_rtol=rtol)
    assert deficient.rank == int(np.sum(s > rtol * s.max())) < basis.found
    assert not deficient.is_full_rank
    assert deficient.coefficients is None
    assert deficient.condition_number is None and deficient.residual_w_per_m is None
    np.testing.assert_array_equal(deficient.singular_values_w_per_m, s)
    record = json.loads(deficient.to_json())
    assert record["coefficients_real"] is None and record["rank"] == deficient.rank


def zero_field(z, side):
    return np.zeros(3), np.zeros(3)


@pytest.mark.parametrize(
    "support, interfaces, origin, rank_rtol",
    [
        ((1e-6, 1e-6), (), 0.0, 1e-8),
        ((2e-6, 1e-6), (), 0.0, 1e-8),
        ((math.nan, 1e-6), (), 0.0, 1e-8),
        ((-1e-6, math.inf), (), 0.0, 1e-8),
        ((-1e-6, 1e-6), (1e-6,), 0.0, 1e-8),
        ((-1e-6, 1e-6), (2e-6,), 0.0, 1e-8),
        ((-1e-6, 1e-6), (), math.nan, 1e-8),
        ((-1e-6, 1e-6), (), 0.0, 0.0),
        ((-1e-6, 1e-6), (), 0.0, 1.0),
    ],
    ids=["empty", "reversed", "nan", "infinite", "interface_on_edge", "interface_outside",
         "origin", "rank_rtol_zero", "rank_rtol_one"],
)
def test_invalid_supports_raise_invalid_input(support, interfaces, origin, rank_rtol):
    with pytest.raises(lamina.InvalidInput):
        solve("TE").project_field(zero_field, support, interfaces, origin, rank_rtol)


@pytest.mark.parametrize(
    "field",
    [
        lambda z, side: (np.zeros(2), np.zeros(3)),
        lambda z, side: np.zeros(6),
        lambda z, side: None,
        lambda z, side: (np.zeros(3), [0, math.nan, 0]),
        lambda z, side: ([0, 0, math.inf * 1j], np.zeros(3)),
    ],
    ids=["short", "flat", "none", "nan", "inf"],
)
def test_invalid_field_values_raise_invalid_input(field):
    with pytest.raises(lamina.InvalidInput):
        solve("TE").project_field(field, (-1e-6, 2e-6))


def test_an_exception_from_the_field_propagates_unchanged():
    class Stop(Exception):
        pass

    raised = Stop("from the field")
    seen = []

    def field(z, side):
        seen.append(z)
        if len(seen) > 3:
            raise raised
        return zero_field(z, side)

    with pytest.raises(Stop) as info:
        solve("TE").project_field(field, (-1e-6, 2e-6))
    assert info.value is raised
    assert len(seen) == 4


def test_field_id_enters_the_hash_and_is_null_when_absent():
    basis = solve("TE")
    field, _, tail = superposition(basis, COEFFICIENTS)
    support = (-tail, D + tail)

    def project(field_id):
        return basis.project_field(field, support, (0.0, D), field_id=field_id)

    first, again, other, anonymous = project("slab-superposition"), project("slab-superposition"), project("other"), project(None)
    assert first.field_id == "slab-superposition"
    assert first.canonical_hash() == again.canonical_hash()
    assert first.canonical_hash() != other.canonical_hash()
    assert json.loads(anonymous.to_json())["field_id"] is None
    assert anonymous.field_id is None
    assert first.input_hash == basis.input_hash
    assert first.support_m == support and first.interfaces_m == (0.0, D) and first.origin_m == 0.0

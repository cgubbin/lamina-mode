"""Power overlaps, their derivatives and reciprocity fluxes of guided modes."""

from __future__ import annotations

import math

import numpy as np
import pytest

import lamina

WAVELENGTH = 1.0e-6
LAYERS = [(4.0, 0.6e-6), (2.25, 0.4e-6), (6.0, 0.5e-6)]


def multilayer():
    return lamina.Stack(left=1.0, layers=LAYERS, right=2.25)


def slab(d, right=1.44):
    return lamina.Stack(left=1.0, layers=[(4.0, d)], right=right)


def solve(stack, polarisation="TE", direction="forward", wavelength=WAVELENGTH):
    basis = lamina.enumerate_guided_modes(
        stack,
        lamina.GuidedRequest(wavelength_m=wavelength, family=polarisation, direction=direction),
    )
    assert basis.is_complete, basis.unresolved
    return basis


def overlap(a, i, b, j, origins=(0.0, 0.0)):
    return lamina.power_overlap(a, i, b, j, origins).power_w_per_m


def flux(a, i, b, j, origins=(0.0, 0.0)):
    return lamina.reciprocity(a, i, b, j, origins).flux_w_per_m


def assert_second_order(errors):
    order = math.log10(errors[0] / errors[1])
    assert 1.6 < order < 3.0, errors


# ------------------------------------------------------------ power overlap


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
def test_self_overlap_is_the_signed_unit_power(polarisation):
    forward = solve(multilayer(), polarisation)
    backward = solve(multilayer(), polarisation, "backward")
    for basis, power in ((forward, 1.0), (backward, -1.0)):
        for mode in basis.modes:
            assert abs(overlap(basis, mode.index, basis, mode.index) - power) < 1e-8


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
def test_distinct_modes_of_a_basis_are_orthogonal(polarisation):
    basis = solve(multilayer(), polarisation)
    assert basis.found >= 2
    for i in range(basis.found):
        for j in range(basis.found):
            if i != j:
                assert abs(overlap(basis, i, basis, j)) < 1e-8


def test_te_and_tm_modes_are_orthogonal():
    te, tm = solve(multilayer(), "TE"), solve(multilayer(), "TM")
    for i in range(te.found):
        for j in range(tm.found):
            assert abs(overlap(te, i, tm, j)) < 1e-8


def test_exchange_conjugates_and_common_translation_preserves_the_overlap():
    a, b = solve(slab(0.7e-6)), solve(slab(0.55e-6, right=1.21))
    d = 0.13e-6
    ab = overlap(a, 0, b, 0, (0.0, d))
    ba = overlap(b, 0, a, 0, (d, 0.0))
    translated = overlap(a, 0, b, 0, (2e-6, 2e-6 + d))
    assert abs(ab) > 0.1
    assert abs(ab - ba.conjugate()) < 1e-10
    assert abs(ab - translated) < 1e-10


def test_overlap_record_carries_its_evidence():
    a, b = solve(slab(0.7e-6)), solve(slab(0.55e-6, right=1.21))
    record = lamina.power_overlap(a, 0, b, 0, (0.0, 0.13e-6))
    assert record.first_input_hash == a.input_hash
    assert record.second_input_hash == b.input_hash
    assert (record.first_mode_index, record.second_mode_index) == (0, 0)
    assert record.origins_m == (0.0, 0.13e-6)
    np.testing.assert_allclose(record.partition_m, [0.0, 0.13e-6, 0.55e-6 + 0.13e-6, 0.7e-6], rtol=1e-12)
    assert record.sample_count > 0
    assert 0.0 <= record.quadrature_change_w_per_m <= 1e-9
    assert record.assumption == lamina._native.ASSUMPTION


@pytest.mark.parametrize(
    "call",
    [
        lambda a, b, other: lamina.power_overlap(a, 0, other, 0),
        lambda a, b, other: lamina.power_overlap(a, 0, b, 0, (math.nan, 0.0)),
        lambda a, b, other: lamina.power_overlap(a, 0, b, 0, (0.0, math.inf)),
        lambda a, b, other: lamina.power_overlap(a, a.found, b, 0),
        lambda a, b, other: lamina.power_overlap(a, 0, b, b.found),
        lambda a, b, other: lamina.reciprocity(a, 0, other, 0),
        lambda a, b, other: lamina.reciprocity(a, a.found, b, 0),
    ],
    ids=["wavelength", "nan_origin", "inf_origin", "first_index", "second_index",
         "reciprocity_wavelength", "reciprocity_index"],
)
def test_incompatible_inputs_raise_invalid_input(call):
    a, b = solve(slab(0.7e-6)), solve(slab(0.55e-6, right=1.21))
    other = solve(slab(0.7e-6), wavelength=1.1e-6)
    with pytest.raises(lamina.InvalidInput):
        call(a, b, other)


def test_deserialised_bases_pair_identically():
    a, b = solve(slab(0.7e-6)), solve(slab(0.55e-6, right=1.21))
    restored = lamina.GuidedBasis.from_json(b.to_json())
    assert lamina.power_overlap(a, 0, restored, 0, (0.0, 1e-7)) == lamina.power_overlap(a, 0, b, 0, (0.0, 1e-7))


# ------------------------------------------------------------ overlap derivatives

A_D, B_D, SHIFT = 0.25e-6, 0.30e-6, 0.07e-6


def sensitivity(a, b, first, second, origins=(0.0, SHIFT)):
    return lamina.power_overlap_sensitivity(a, 0, b, 0, first, second, origins)


def converges(exact, of_step, steps, tolerance):
    scale = abs(exact)
    errors = [abs((of_step(h) - of_step(-h)) / (2 * h) - exact) / scale for h in steps]
    assert_second_order(errors[:2])
    assert min(errors) < tolerance, errors


def test_rate_along_the_second_thickness_matches_re_solved_overlaps():
    a, b = solve(slab(A_D)), solve(slab(B_D))
    result = sensitivity(a, b, None, lamina.Parameter.thickness(0))
    assert result.derivative_failure is None
    assert result.parameter_unit == "m"
    assert result.power_w_per_m == overlap(a, 0, b, 0, (0.0, SHIFT))
    converges(
        result.power_rate_w_per_m,
        lambda h: overlap(a, 0, solve(slab(B_D + h)), 0, (0.0, SHIFT)),
        (4e-9, 4e-10, 4e-11),
        1e-6,
    )


def test_rate_under_relative_translation_matches_shifted_overlaps():
    a, b = solve(slab(A_D)), solve(slab(B_D))
    result = sensitivity(a, b, None, lamina.Parameter.translation())
    converges(
        result.power_rate_w_per_m,
        lambda h: overlap(a, 0, b, 0, (0.0, SHIFT + h)),
        (4e-9, 4e-10, 4e-11),
        1e-6,
    )


def test_rate_along_the_shared_wavelength_matches_re_solved_overlaps():
    a, b = solve(slab(A_D)), solve(slab(B_D))
    wavelength = lamina.Parameter.wavelength()
    result = sensitivity(a, b, wavelength, wavelength)

    def at(h):
        return overlap(solve(slab(A_D), wavelength=WAVELENGTH + h), 0,
                       solve(slab(B_D), wavelength=WAVELENGTH + h), 0, (0.0, SHIFT))

    converges(result.power_rate_w_per_m, at, (1e-8, 1e-9, 1e-10), 1e-6)


def test_common_translation_leaves_the_overlap_unchanged():
    a, b = solve(slab(A_D)), solve(slab(B_D))
    translation = lamina.Parameter.translation()
    result = sensitivity(a, b, translation, translation)
    scale = abs(result.power_w_per_m) / A_D
    assert abs(result.power_rate_w_per_m) < 1e-10 * scale


def test_separating_coincident_interfaces_are_a_recorded_failure():
    a = solve(slab(A_D))
    result = sensitivity(a, a, None, lamina.Parameter.translation(), (0.0, 0.0))
    assert result.power_rate_w_per_m is None
    assert result.exterior_rates_w_per_m is None
    assert "coincident interfaces at z = 0 m separate" in result.derivative_failure
    assert abs(result.power_w_per_m - 1.0) < 1e-8
    # Moving both copies together is not a separation.
    together = sensitivity(a, a, lamina.Parameter.thickness(0), lamina.Parameter.thickness(0), (0.0, 0.0))
    assert together.derivative_failure is None
    assert abs(together.power_rate_w_per_m) < 1e-8 / A_D


@pytest.mark.parametrize(
    "first, second",
    [
        (lamina.Parameter.thickness(0), lamina.Parameter.permittivity(1)),
        (lamina.Parameter.wavelength(), None),
        (None, lamina.Parameter.wavelength()),
        (lamina.Parameter.wavelength(), lamina.Parameter.thickness(0)),
        (None, None),
        (lamina.Parameter.thickness(3), None),
    ],
    ids=["units", "wavelength_first_only", "wavelength_second_only", "wavelength_and_thickness",
         "both_fixed", "bad_layer"],
)
def test_mismatched_parameters_raise_invalid_input(first, second):
    a, b = solve(slab(A_D)), solve(slab(B_D))
    with pytest.raises(lamina.InvalidInput):
        sensitivity(a, b, first, second)


# ------------------------------------------------------------ reciprocity


def test_reciprocity_is_antisymmetric():
    forward = solve(multilayer(), "TE")
    backward = solve(multilayer(), "TE", "backward")
    for i in range(forward.found):
        assert abs(flux(forward, i, forward, i)) < 1e-12
        for j in range(forward.found):
            assert abs(flux(forward, i, forward, j) + flux(forward, j, forward, i)) < 1e-12
            assert abs(flux(forward, i, backward, j) + flux(backward, j, forward, i)) < 1e-12


@pytest.mark.parametrize("polarisation, paired", [("TE", -4.0), ("TM", 4.0)])
def test_forward_and_backward_bases_are_bi_orthogonal(polarisation, paired):
    stack = slab(0.7e-6)
    forward = solve(stack, polarisation)
    backward = solve(stack, polarisation, "backward")
    assert forward.found == backward.found >= 2
    for i in range(forward.found):
        for j in range(backward.found):
            expected = paired if i == j else 0.0
            assert abs(flux(forward, i, backward, j) - expected) < 1e-8, (i, j)
            if i != j:
                assert abs(flux(forward, i, forward, j)) < 1e-8


# The unconjugated flux counts a recovered field's residual phase twice, so on a
# stack whose most confined mode is recovered with a boundary defect well above
# round-off the bound scales with the defects each mode records:
#   |flux - expected| <= QUADRATURE_FLOOR + C * 4 W/m * (defect_m + defect_n).
# On the four-region multilayer the defect-dominated pairs (those of the most
# confined mode) have |flux - expected| / (4 W/m * (defect_m + defect_n)) of at
# most 0.07, so C = 1 leaves a margin of about 14.
QUADRATURE_FLOOR = 1e-8
C = 1.0


def defects(basis):
    return [basis.sample_fields(m.index, [0.0]).relative_boundary_defect for m in basis.modes]


@pytest.mark.parametrize("polarisation, paired", [("TE", -4.0), ("TM", 4.0)])
def test_multilayer_bi_orthogonality_within_the_recorded_field_error(polarisation, paired):
    forward = solve(multilayer(), polarisation)
    backward = solve(multilayer(), polarisation, "backward")
    assert forward.found == backward.found >= 2
    df, db = defects(forward), defects(backward)

    def bound(a, b):
        return QUADRATURE_FLOOR + C * 4.0 * (a + b)

    for i in range(forward.found):
        for j in range(backward.found):
            expected = paired if i == j else 0.0
            value = flux(forward, i, backward, j)
            tolerance = bound(df[i], db[j])
            assert abs(value.real - expected) < tolerance, (i, j, value, tolerance)
            assert abs(value - expected) < tolerance, (i, j, value, tolerance)
        for j in range(forward.found):
            if i != j:
                assert abs(flux(forward, i, forward, j)) < bound(df[i], df[j]), (i, j)

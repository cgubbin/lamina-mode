"""Local coupling of a guided basis along a fixed-wavelength parameter."""

from __future__ import annotations

import numpy as np
import pytest

import lamina

WAVELENGTH = 1.0e-6
CORE, CLADDING = (4.0, 0.42e-6), (2.25, 0.10e-6)


def stack(interface=0.0, permittivity=0.0):
    """Interface 1 sits between the two layers; layer 0's permittivity varies."""
    return lamina.Stack(
        left=1.0,
        layers=[(CORE[0] + permittivity, CORE[1] + interface), (CLADDING[0], CLADDING[1] - interface)],
        right=1.44,
    )


def solve(s, polarisation, **request):
    basis = lamina.enumerate_guided_modes(
        s, lamina.GuidedRequest(wavelength_m=WAVELENGTH, polarisation=polarisation, **request)
    )
    assert basis.is_complete, basis.unresolved
    return basis


CASES = [
    ("interface", lamina.Parameter.interface_position(1), lambda h: stack(interface=h), 1e-11),
    ("permittivity", lamina.Parameter.permittivity(1), lambda h: stack(permittivity=h), 1e-5),
]


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: c[0])
def test_diagonal_matches_the_eigenvalue_sensitivity(case, polarisation):
    _, parameter, _, _ = case
    base = solve(stack(), polarisation)
    assert base.found >= 2
    coupling = base.local_coupling(parameter)
    assert coupling.parameter_unit == base.sensitivity(0, parameter).parameter_unit
    for mode in base.modes:
        dbeta = base.sensitivity(mode.index, parameter).dbeta_rad_per_m
        k = coupling.perturbation_matrix_rad_per_m[mode.index, mode.index]
        assert abs(k - dbeta) < 1e-6 * abs(dbeta)


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: c[0])
def test_coefficients_match_derivatives_of_re_solved_overlaps(case, polarisation):
    _, parameter, perturbed, h = case
    base = solve(stack(), polarisation)
    coupling = base.local_coupling(parameter)

    def overlap(step):
        displaced = solve(perturbed(step), polarisation)
        return lamina.power_overlap(displaced, 1, base, 0).power_w_per_m

    fd = (overlap(h) - overlap(-h)) / (2 * h)
    c01, c10 = coupling.coefficients[0, 1], coupling.coefficients[1, 0]
    assert coupling.coefficient_resolved[0, 1] and coupling.coefficient_resolved[1, 0]
    assert abs(fd - c01) < 1e-5 * abs(c01), (fd, c01)
    assert abs(c01 + c10.conjugate()) < 1e-10 * abs(c01)
    # Hermitian perturbation matrix; the diagonal is absent from the coefficients.
    k = coupling.perturbation_matrix_rad_per_m
    np.testing.assert_allclose(k, k.conj().T, rtol=0, atol=1e-12 * np.abs(k).max())
    assert not coupling.coefficient_resolved.diagonal().any()
    assert np.isnan(coupling.coefficients.diagonal()).all()


def test_default_gap_is_relative_to_k0_and_recorded():
    base = solve(stack(), "TE")
    coupling = base.local_coupling(lamina.Parameter.interface_position(1))
    assert coupling.gap_rad_per_m == 1e-6 * base.k0_rad_per_m
    assert coupling.unresolved_groups == ()
    explicit = base.local_coupling(lamina.Parameter.interface_position(1), gap_rad_per_m=10.0)
    assert explicit.gap_rad_per_m == 10.0


def test_a_wide_gap_leaves_modes_unresolved_and_the_matrix_unchanged():
    base = solve(stack(), "TM")
    parameter = lamina.Parameter.permittivity(1)
    resolved = base.local_coupling(parameter)
    unresolved = base.local_coupling(parameter, gap_rad_per_m=base.k0_rad_per_m)
    by_beta = sorted(range(base.found), key=lambda i: base.modes[i].beta_rad_per_m.real)
    assert unresolved.unresolved_groups == (tuple(by_beta),)
    assert not unresolved.coefficient_resolved.any()
    assert np.isnan(unresolved.coefficients).all()
    np.testing.assert_array_equal(
        unresolved.perturbation_matrix_rad_per_m, resolved.perturbation_matrix_rad_per_m
    )


@pytest.mark.parametrize(
    "build, parameter, reason",
    [
        (lambda: lamina.enumerate_guided_modes(
            stack(), lamina.GuidedRequest(wavelength_m=WAVELENGTH, max_subdivisions=0)),
         lamina.Parameter.permittivity(1), "complete basis"),
        (lambda: solve(lamina.Stack(left=1.0, layers=[(lamina.Material(4.0, 1.5), 0.42e-6)], right=1.44), "TE"),
         lamina.Parameter.permittivity(1), "nonmagnetic"),
        (lambda: solve(stack(), "TE"), lamina.Parameter.wavelength(), "wavelength"),
        (lambda: solve(stack(), "TE"), lamina.Parameter.permeability(1), "permeability"),
        (lambda: solve(stack(), "TE"), lamina.Parameter.permittivity(4), "region 4"),
    ],
    ids=["partial", "magnetic", "wavelength", "permeability", "bad_region"],
)
def test_ineligible_requests_raise_invalid_input(build, parameter, reason):
    basis = build()
    with pytest.raises(lamina.InvalidInput, match=reason):
        basis.local_coupling(parameter)


@pytest.mark.parametrize("gap", [-1.0, float("nan"), float("inf")])
def test_invalid_gaps_raise_invalid_input(gap):
    with pytest.raises(lamina.InvalidInput):
        solve(stack(), "TE").local_coupling(lamina.Parameter.permittivity(1), gap_rad_per_m=gap)

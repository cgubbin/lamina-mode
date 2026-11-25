"""Acceptance: analytic slab ladders match to 1e-10 in n_eff through the bindings."""

from __future__ import annotations

import numpy as np
import pytest

import lamina
from slab import box_ladder, ladder

# (name, wavelength, core eps, thickness, left eps, right eps)
CASES = [
    ("symmetric multimode", 1.0e-6, 4.0, 2.0e-6, 1.0, 1.0),
    ("asymmetric multimode", 1.0e-6, 4.0, 2.0e-6, 1.0, 2.25),
    ("symmetric silicon slab", 1.55e-6, 12.11, 2.2e-7, 2.085, 2.085),
    ("asymmetric silicon slab", 1.55e-6, 12.11, 2.2e-7, 1.0, 2.085),
    ("near single-mode cutoff", 1.0e-6, 4.0, 1.01e-6 / (2.0 * 3.0**0.5), 1.0, 1.0),
]


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_ladder_matches_analytic_reference(case, polarisation):
    _, wavelength, core, d, left, right = case
    reference = ladder(wavelength, core, d, left, right, polarisation)
    assert reference, "each case guides at least one mode"

    stack = lamina.Stack(left=left, layers=[(core, d)], right=right)
    basis = lamina.enumerate_guided_modes(
        stack, lamina.GuidedRequest(wavelength_m=wavelength, polarisation=polarisation)
    )

    assert basis.completeness == "complete", basis.unresolved
    assert basis.expected == basis.found == len(reference)
    n_eff = basis.n_eff
    assert np.all(np.abs(n_eff.imag) < 1e-12)
    np.testing.assert_allclose(n_eff.real, reference, rtol=0, atol=1e-10)
    # Evidence travels with every root.
    for mode in basis.modes:
        assert mode.residual <= mode.residual_limit
        assert mode.beta_rad_per_m == pytest.approx(mode.n_eff * basis.k0_rad_per_m, rel=1e-15)


def test_backward_direction_mirrors_forward():
    stack = lamina.Stack(left=1.0, layers=[(4.0, 2e-6)], right=2.25)
    forward = lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(wavelength_m=1e-6))
    backward = lamina.enumerate_guided_modes(
        stack, lamina.GuidedRequest(wavelength_m=1e-6, direction="backward")
    )
    assert backward.is_complete and backward.found == forward.found
    np.testing.assert_allclose(backward.beta_rad_per_m.real, -forward.beta_rad_per_m.real, rtol=1e-12)


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
@pytest.mark.parametrize("wall", ["pec", "pmc"])
def test_walled_ladder_includes_radiation_like_modes(wall, polarisation):
    # A core |z - a| < a in cladding, closed by walls a gap beyond each interface.
    wavelength, core, clad, a, gap = 1.0e-6, 4.0, 2.25, 0.35e-6, 0.45e-6
    stack = lamina.Stack(
        left=clad,
        layers=[(core, 2 * a)],
        right=clad,
        left_termination=lamina.Wall(wall, -gap),
        right_termination=lamina.Wall(wall, 2 * a + gap),
    )
    basis = lamina.enumerate_guided_modes(
        stack, lamina.GuidedRequest(wavelength_m=wavelength, polarisation=polarisation)
    )
    window = basis.input.request.abs_beta_rad_per_m
    reference = box_ladder(wavelength, core, clad, a, gap, wall, polarisation, window)
    assert any(n < clad**0.5 for n in reference), "radiation-like modes in the window"

    assert basis.light_line_rad_per_m == 0.0
    assert basis.completeness == "complete", basis.unresolved
    assert basis.expected == basis.found == len(reference)
    n_eff = basis.n_eff
    assert np.all(np.abs(n_eff.imag) < 1e-12)
    np.testing.assert_allclose(np.sort(n_eff.real), reference, rtol=0, atol=1e-10)

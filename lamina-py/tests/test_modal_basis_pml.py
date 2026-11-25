"""Modal bases of stacks closed by PMLs: completeness, profiles through a PML,
self-pairing, compatibility and the scale refusal."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import lamina

WAVELENGTH = 1.0e-6
K0 = 2.0 * math.pi / WAVELENGTH
A = 0.35e-6
GAP = 0.1e-6
D = 0.75e-6
ORDER = 2.0
STRENGTH = 1 + 3j
WINDOW = lamina.BetaWindow((1.0 * K0, 2.03 * K0), (-0.031 * K0, 1.0 * K0))


def closed(d: float = D, backing: str = "pec") -> lamina.Stack:
    """The core between equal PMLs GAP beyond each interface, d thick."""
    return lamina.Stack(
        2.25,
        [(4.0, 2 * A)],
        2.25,
        lamina.Pml(-GAP, d, ORDER, STRENGTH, backing),
        lamina.Pml(2 * A + GAP, d, ORDER, STRENGTH, backing),
    )


@pytest.fixture(scope="module")
def basis() -> lamina.ModalBasis:
    return lamina.modal_basis(closed(), WAVELENGTH, "TE", WINDOW)


def test_a_pml_closed_basis_is_complete(basis):
    assert basis.status == "complete" and basis.scope == "discrete"
    assert basis.expected == basis.found == len(basis) >= 8
    assert all(m.mode_type == "closed" and m.normalisation == "unit" for m in basis.modes)
    assert lamina.ModalBasis.from_json(basis.to_json()).to_json() == basis.to_json()


def test_a_pml_basis_pairs_with_itself_as_the_identity(basis):
    pairing = lamina.basis_pairing(basis, basis)
    n = len(basis)
    assert np.abs(pairing.unconjugated_w_per_m - np.eye(n)).max() < 1e-8
    assert max(pairing.self_defects) < 1e-8
    assert np.all(np.isfinite(pairing.conjugated_w_per_m))


def test_profiles_through_a_pml_carry_the_stretched_coordinate(basis):
    right = 2 * A + GAP
    z = np.linspace(-GAP - D, right + D, 401)
    profile = basis.profile(0, z)
    regions = np.array(profile.region)
    inside = (z < -GAP) | (z > right)
    assert set(regions[inside]) == {"pml"}
    assert set(regions[~inside]) == {"gap", "layer"}
    assert profile.pml_fields is not None and "continued fields" in profile.pml_fields
    # z̃ = start ± ζ̃(ζ) and s = 1 + c (ζ/d)^m at the outward depth ζ.
    depth = np.where(z < 0, -GAP - z, z - right)
    stretched = depth + STRENGTH * D / (ORDER + 1) * (depth / D) ** (ORDER + 1)
    expected_z = np.where(z < 0, -GAP - stretched, right + stretched)
    stretch = 1 + STRENGTH * (depth / D) ** ORDER
    np.testing.assert_allclose(profile.stretched_z_m[inside], expected_z[inside], rtol=1e-14, atol=0)
    np.testing.assert_allclose(profile.stretch[inside], stretch[inside], rtol=1e-14)
    assert np.all(np.isnan(profile.stretched_z_m[~inside])) and np.all(np.isnan(profile.stretch[~inside]))
    # TE behind PEC backing walls: E_y vanishes at both.
    ey = profile.electric_v_per_m[:, 1]
    assert abs(ey[0]) < 1e-9 * np.abs(ey).max() and abs(ey[-1]) < 1e-9 * np.abs(ey).max()
    restored = lamina.ModalProfile.from_json(profile.to_json())
    assert restored.to_json() == profile.to_json()
    # A profile of a walled stack has no PML entry.
    walls = lamina.Stack(2.25, [(4.0, 2 * A)], 2.25, lamina.Wall("pec", -GAP), lamina.Wall("pec", 2 * A + GAP))
    walled = lamina.modal_basis(walls, WAVELENGTH, "TE", lamina.BetaWindow.forward(2.1 * K0, 1.0 * K0, 0.05 * K0))
    assert "pml" not in json.loads(walled.profile(0, [0.0]).to_json())


def test_a_profile_beyond_the_backing_wall_is_refused(basis):
    with pytest.raises(lamina.InvalidInput, match="right wall"):
        basis.profile(0, [2 * A + GAP + D * 1.01])


def test_differing_pmls_do_not_pair(basis):
    thicker = lamina.modal_basis(closed(D + 0.1e-6), WAVELENGTH, "TE", WINDOW)
    with pytest.raises(lamina.InvalidInput, match="Left"):
        lamina.basis_pairing(basis, thicker)


def test_a_window_beyond_the_pml_scale_is_refused():
    thick = lamina.Stack(
        2.25,
        [(4.0, 2 * A)],
        2.25,
        lamina.Pml(-0.2e-6, 20e-6, 2.0, 2.4j, "pec"),
        lamina.Wall("pmc", 2 * A + 0.3e-6),
    )
    window = lamina.BetaWindow((1.85 * K0, 4.0 * K0), (-0.031 * K0, 0.047 * K0))
    with pytest.raises(lamina.InvalidInput, match=r"Left PML .*exp\(3[0-2]\d\)"):
        lamina.modal_basis(thick, WAVELENGTH, "TE", window)

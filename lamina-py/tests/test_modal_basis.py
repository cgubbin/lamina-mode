"""Modal bases of stacks closed by walls: windows, input records, bases,
profiles and their fixed definitions."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import lamina
from slab import box_ladder

WAVELENGTH = 1.0e-6
K0 = 2.0 * math.pi / WAVELENGTH
A = 0.35e-6
GAP = 0.45e-6
WINDOW = lamina.BetaWindow.forward(2.1 * K0, 3.0 * K0, 0.05 * K0)


def walled(kind: str = "pec", half_width: float = A, gap: float = GAP) -> lamina.Stack:
    return lamina.Stack(
        2.25,
        [(4.0, 2 * half_width)],
        2.25,
        lamina.Wall(kind, -gap),
        lamina.Wall(kind, 2 * half_width + gap),
    )


def test_windows_are_frozen_and_round_trip():
    window = lamina.BetaWindow.forward(10.0, 20.0, 0.5)
    assert window.re_rad_per_m == (-0.5, 10.0) and window.im_rad_per_m == (-0.5, 20.0)
    assert lamina.BetaWindow.from_dict(window.to_dict()) == window
    assert window.to_dict()["kind"] == "wavenumber"
    with pytest.raises(AttributeError):
        window.re_rad_per_m = (0.0, 1.0)  # type: ignore[misc]


def resolve(stack=None, window=WINDOW, **kwargs) -> lamina.ModalBasisInput:
    return lamina.ModalBasisInput(stack or walled(), WAVELENGTH, "TE", window, **kwargs).resolve()


def test_input_records_resolve_the_gap_and_hash_their_window():
    base = resolve()
    assert base.degeneracy_gap_rad_per_m == pytest.approx(1e-6 * K0, rel=1e-15)
    assert lamina.ModalBasisInput.from_json(base.to_json()) == base
    moved = lamina.BetaWindow(WINDOW.re_rad_per_m, (WINDOW.im_rad_per_m[0], 2.9 * K0))
    assert resolve(window=moved).canonical_hash() != base.canonical_hash()
    # Equal inputs in other styles: materials as objects, lists for tuples,
    # the default gap stated, integers for floats.
    styled = lamina.Stack(
        lamina.Material(2.25),
        [lamina.Layer(lamina.Material(4), 2 * A)],
        lamina.Material(2.25, 1),
        lamina.Wall("pec", -GAP),
        lamina.Wall("pec", 2 * A + GAP),
    )
    window = lamina.BetaWindow(list(WINDOW.re_rad_per_m), list(WINDOW.im_rad_per_m))
    same = resolve(styled, window, degeneracy_gap_rad_per_m=base.degeneracy_gap_rad_per_m)
    assert same.to_json() == base.to_json()
    assert same.canonical_hash() == base.canonical_hash()


@pytest.mark.parametrize(
    "change",
    [
        {"self_pairing_rtol": 0.0},
        {"degeneracy_gap_rad_per_m": -1.0},
        {"window": lamina.BetaWindow((1.0, 0.0), (0.0, 1.0))},
    ],
    ids=["threshold", "gap", "window"],
)
def test_invalid_requests_are_refused(change):
    with pytest.raises(lamina.InvalidInput):
        lamina.modal_basis(walled(), WAVELENGTH, "TE", change.pop("window", WINDOW), **change)


def test_an_open_side_is_refused():
    stack = lamina.Stack(2.25, [(4.0, 2 * A)], 2.25, lamina.Wall("pec", -GAP), None)
    with pytest.raises(lamina.InvalidInput):
        lamina.modal_basis(stack, WAVELENGTH, "TE", WINDOW)


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
@pytest.mark.parametrize("kind", ["pec", "pmc"])
def test_a_walled_slab_has_its_whole_ladder(kind, polarisation):
    basis = lamina.modal_basis(walled(kind), WAVELENGTH, polarisation, WINDOW)
    assert basis.is_complete and basis.expected == basis.found == len(basis)
    assert basis.chart == "wavenumber" and basis.scope == "discrete"
    assert dict(basis.definitions) == dict(lamina.DEFINITIONS)
    assert basis.window_beta_image_rad_per_m == (WINDOW.re_rad_per_m, WINDOW.im_rad_per_m)
    real = sorted(m.n_eff.real for m in basis.modes if m.beta_rad_per_m.real > 0.05 * K0)
    expected = box_ladder(WAVELENGTH, 4.0, 2.25, A, GAP, kind, polarisation, (0.05 * K0, 2.1 * K0))
    np.testing.assert_allclose(real, expected, rtol=1e-9)
    assert any(m.beta_rad_per_m.imag > 0.1 * K0 for m in basis.modes)
    for mode in basis.modes:
        assert mode.normalisation == "unit" and mode.mode_type == "closed"
        assert mode.direction == "forward" and mode.sheet == (None, None)
        assert mode.sign_reference == "first_interface" and not mode.sign_ambiguous
    # Round trip, and a parsed basis re-solves to the same record.
    parsed = lamina.ModalBasis.from_json(basis.to_json())
    assert parsed == basis and parsed.input_hash == basis.input.canonical_hash()
    z = np.linspace(-GAP, 2 * A + GAP, 7)
    np.testing.assert_array_equal(
        parsed.profile(0, z).electric_v_per_m, basis.profile(0, z).electric_v_per_m
    )


def test_an_edge_on_a_mode_is_an_incomplete_result():
    basis = lamina.modal_basis(walled(), WAVELENGTH, "TE", WINDOW)
    top = max(m.beta_rad_per_m.real for m in basis.modes)
    edge = lamina.BetaWindow((WINDOW.re_rad_per_m[0], top), WINDOW.im_rad_per_m)
    cut = lamina.modal_basis(walled(), WAVELENGTH, "TE", edge)
    assert cut.status == "count_failed" and cut.expected is None
    assert [r.cause for r in cut.unresolved] == ["boundary_root"]
    assert cut.found == basis.found - 1


@pytest.mark.parametrize("kind, primary", [("pec", 1), ("pmc", 0)])
def test_profiles_name_their_regions_and_meet_the_walls(kind, primary):
    basis = lamina.modal_basis(walled(kind), WAVELENGTH, "TE", WINDOW)
    z = np.linspace(-GAP, 2 * A + GAP, 401)
    for index in range(len(basis)):
        profile = basis.profile(index, z)
        assert profile.region[0] == profile.region[-1] == "gap"
        assert set(profile.region) == {"gap", "layer"}
        assert dict(profile.definitions) == dict(lamina.DEFINITIONS)
        # The wall's tangential field: Ey at a PEC wall, Hx at a PMC wall.
        field = profile.electric_v_per_m if kind == "pec" else profile.magnetic_a_per_m
        peak = np.abs(field[:, primary]).max()
        assert abs(field[0, primary]) < 1e-9 * peak and abs(field[-1, primary]) < 1e-9 * peak
    with pytest.raises(lamina.InvalidInput, match="beyond the left wall"):
        basis.profile(0, [-GAP * 1.01])


def test_an_ill_conditioned_mode_has_no_profile():
    width = 0.5e-6 * (1 + 2e-7)
    box = lamina.Stack(1.0, [(1.0, width)], 1.0, lamina.Wall("pec", 0.0), lamina.Wall("pec", width))
    beta = math.sqrt(K0**2 - (math.pi / width) ** 2)
    window = lamina.BetaWindow.forward(1.1 * K0, 0.5 * K0, 0.1 * beta)
    basis = lamina.modal_basis(box, WAVELENGTH, "TE", window, self_pairing_rtol=1e-2)
    assert basis.status == "incomplete" and basis.found == 1
    assert basis.modes[0].normalisation == "ill_conditioned"
    assert basis.modes[0].self_pairing_ratio < 1e-2 and basis.modes[0].sign_margin is None
    with pytest.raises(lamina.InvalidInput, match="ill-conditioned"):
        basis.profile(0, [0.25e-6])


def test_changed_definitions_are_schema_errors():
    basis = lamina.modal_basis(walled(), WAVELENGTH, "TE", WINDOW)
    record = json.loads(basis.to_json())
    record["definitions"]["ordering"] = "increasing Re β"
    with pytest.raises(lamina.SchemaError):
        lamina.ModalBasis.from_json(json.dumps(record))
    profile = json.loads(basis.profile(0, [0.0]).to_json())
    profile["definitions"]["normalisation"] = "unit power"
    with pytest.raises(lamina.SchemaError):
        lamina.ModalProfile.from_json(json.dumps(profile))

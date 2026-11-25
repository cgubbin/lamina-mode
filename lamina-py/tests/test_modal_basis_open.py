"""Modal bases of stacks with open sides: chart windows, the covering of a β
rectangle, guided and leaky modes with their sheets, the continuum scope and
pairings with absent conjugated entries."""

from __future__ import annotations

import cmath
import json
import math

import numpy as np
import pytest

import lamina

WAVELENGTH = 1.0e-6
K0 = 2.0 * math.pi / WAVELENGTH
A = 0.35e-6
CORE, CLAD = 4.0, 2.25
# Two guided modes on the real z axis and one leaky mode near Im z = −2.6 (TE).
WINDOW = lamina.ChartWindow((-3.0, 0.35), (-2.9, 0.4))


def open_slab(half_width: float = A) -> lamina.Stack:
    return lamina.Stack(CLAD, [(CORE, 2 * half_width)], CLAD)


def even_te(beta: complex) -> complex:
    """The open slab's even TE condition on the improper sheet, independent of lamina."""
    q = cmath.sqrt(CORE * K0**2 - beta**2)
    kappa = -1j * cmath.sqrt(beta**2 - CLAD * K0**2)
    return q * cmath.sin(q * A) + 1j * kappa * cmath.cos(q * A)


def newton(f, z: complex) -> complex:
    for _ in range(100):
        h = 1e-7 * abs(z)
        step = f(z) * 2 * h / (f(z + h) - f(z - h))
        z -= step
        if abs(step) < 1e-14 * abs(z):
            return z
    raise AssertionError("Newton did not converge")


@pytest.fixture(scope="module")
def basis() -> lamina.ModalBasis:
    return lamina.modal_basis(open_slab(), WAVELENGTH, "TE", WINDOW)


def test_chart_windows_are_frozen_round_trip_and_enter_the_hash():
    assert WINDOW.to_dict() == {"kind": "chart", "re": [-3.0, 0.35], "im": [-2.9, 0.4]}
    assert lamina.ChartWindow.from_dict(WINDOW.to_dict()) == WINDOW
    with pytest.raises(AttributeError):
        WINDOW.re = (0.0, 1.0)  # type: ignore[misc]
    first = lamina.ModalBasisInput(open_slab(), WAVELENGTH, "TE", WINDOW).resolve()
    second = lamina.ModalBasisInput(
        open_slab(), WAVELENGTH, "TE", lamina.ChartWindow((-3.0, 0.3), (-2.9, 0.4))
    ).resolve()
    assert first.window == WINDOW
    assert lamina.ModalBasisInput.from_json(first.to_json()).to_json() == first.to_json()
    assert first.to_dict()["window"]["kind"] == "chart"
    assert first.canonical_hash() != second.canonical_hash()


def test_an_open_stack_refuses_a_beta_window_and_a_window_over_a_period():
    beta = lamina.BetaWindow.forward(2.0 * K0, K0, 0.01 * K0)
    with pytest.raises(lamina.InvalidInput, match="ChartWindow"):
        lamina.modal_basis(open_slab(), WAVELENGTH, "TE", beta)
    with pytest.raises(lamina.InvalidInput, match="2π"):
        lamina.modal_basis(open_slab(), WAVELENGTH, "TE", lamina.ChartWindow((-1, 0), (-3.2, 3.1)))


def test_an_open_slab_basis_holds_guided_and_leaky_modes_with_sheets(basis):
    assert basis.status == "complete" and basis.scope == "discrete_without_continuum"
    assert basis.chart == "two_exterior"
    assert basis.expected == basis.found == len(basis) == 3
    types = sorted(m.mode_type for m in basis.modes)
    assert types == ["guided", "guided", "leaky"]
    for mode in basis.modes:
        sheet = "proper" if mode.mode_type == "guided" else "improper"
        assert mode.sheet == (sheet, sheet) and mode.normalisation == "unit"
    assert lamina.ModalBasis.from_json(basis.to_json()).to_json() == basis.to_json()


def test_a_leaky_window_from_a_beta_rectangle_holds_the_leaky_mode():
    rect = ((1.45 * K0, 1.65 * K0), (0.03 * K0, 0.15 * K0))
    window, extra = lamina.chart_window_covering(
        open_slab(), WAVELENGTH, rect, ("improper", "improper")
    )
    assert isinstance(window, lamina.ChartWindow) and extra > 0.0
    basis = lamina.modal_basis(open_slab(), WAVELENGTH, "TE", window)
    assert basis.is_complete
    expected = newton(even_te, complex(1.55 * K0, 0.08 * K0))
    inside = [
        m.beta_rad_per_m
        for m in basis.modes
        if rect[0][0] < abs(m.beta_rad_per_m.real) < rect[0][1]
        and rect[1][0] < abs(m.beta_rad_per_m.imag) < rect[1][1]
    ]
    assert len(inside) == 1
    beta = inside[0] if inside[0].real > 0 else -inside[0]
    assert abs(beta - expected) < 1e-9 * K0


def test_a_rectangle_across_the_cut_is_refused():
    rect = ((1.3 * K0, 1.6 * K0), (-0.01 * K0, 0.1 * K0))
    with pytest.raises(lamina.InvalidInput, match="Left"):
        lamina.chart_window_covering(open_slab(), WAVELENGTH, rect, ("improper", "improper"))
    with pytest.raises(lamina.InvalidInput):
        lamina.chart_window_covering(open_slab(), WAVELENGTH, rect, ("improper", None))


def test_leaky_conjugated_entries_are_absent_and_the_gram_is_the_identity(basis):
    pairing = lamina.basis_pairing(basis, basis)
    n = len(basis)
    assert np.abs(pairing.unconjugated_w_per_m - np.eye(n)).max() < 1e-8
    assert pairing.scope == ("discrete_without_continuum", "discrete_without_continuum")
    leaky = [i for i, m in enumerate(basis.modes) if m.mode_type == "leaky"]
    for i in leaky:
        assert pairing.conjugated_absent[i][i] == "unbounded"
        assert math.isnan(pairing.conjugated_w_per_m[i, i].real)
    guided = [i for i, m in enumerate(basis.modes) if m.mode_type == "guided"]
    for i in guided:
        assert pairing.conjugated_absent[i][i] is None
        assert abs(pairing.conjugated_w_per_m[i, i] - 1.0) < 1e-9
    record = json.loads(pairing.to_json())
    assert record["conjugated_real"][leaky[0]][leaky[0]] is None
    assert lamina.BasisPairing.from_json(pairing.to_json()).to_json() == pairing.to_json()


def test_a_root_at_a_branch_point_is_unresolved():
    # A TE odd mode just above cutoff, κ = i 2e-5 k0.
    alpha = 2e-5 * K0
    q = math.sqrt(1.75 * K0**2 - alpha**2)
    half = (math.pi / 2 + math.atan(alpha / q)) / q
    z = math.log(alpha / K0)
    window = lamina.ChartWindow((z - 1, z + 1), (-0.5, 0.5))
    basis = lamina.modal_basis(open_slab(half), WAVELENGTH, "TE", window)
    assert basis.status == "incomplete" and basis.found == 0 and basis.expected == 1
    assert [r.cause for r in basis.unresolved] == ["branch_point"]

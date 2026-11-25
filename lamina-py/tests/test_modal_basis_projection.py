"""Projection of supplied fields onto modal bases and local coupling on them:
recovery of a guided-plus-leaky combination with its exterior continuation,
the continuum residual of a Gaussian, the coupling diagonal against
re-solved propagation constants, refusals and record round trips."""

from __future__ import annotations

import cmath
import json
import math
from functools import lru_cache

import numpy as np
import pytest

import lamina

WAVELENGTH = 1.0e-6
K0 = 2.0 * math.pi / WAVELENGTH
Z0 = 376.730313412
A = 0.35e-6
CORE, CLAD = 4.0, 2.25
GAP = 0.45e-6
# Two guided modes and one leaky mode of the open slab.
WINDOW = lamina.ChartWindow((-3.0, 0.35), (-2.9, 0.4))
FORWARD = lamina.BetaWindow.forward(2.1 * K0, 3.0 * K0, 0.05 * K0)
SUPPORT = (-0.4e-6, 2 * A + 0.4e-6)


def open_slab() -> lamina.Stack:
    return lamina.Stack(CLAD, [(CORE, 2 * A)], CLAD)


def walled(thickness: float = 2 * A) -> lamina.Stack:
    return lamina.Stack(
        CLAD,
        [(CORE, thickness)],
        CLAD,
        lamina.Wall("pec", -GAP),
        lamina.Wall("pec", 2 * A + GAP),
    )


@pytest.fixture(scope="module")
def basis() -> lamina.ModalBasis:
    b = lamina.modal_basis(open_slab(), WAVELENGTH, "TE", WINDOW)
    assert b.is_complete and b.scope == "discrete_without_continuum"
    assert any(m.mode_type == "leaky" for m in b.modes)
    return b


@pytest.fixture(scope="module")
def walled_basis() -> lamina.ModalBasis:
    b = lamina.modal_basis(walled(), WAVELENGTH, "TE", FORWARD)
    assert b.is_complete
    return b


def coefficients(n: int) -> list[complex]:
    return [complex(0.3 + 0.2 * i, 0.5 - 0.15 * i) for i in range(n)]


def combination(basis: lamina.ModalBasis, c: list[complex]):
    """Σ c_n f_n from the basis's own profiles; TE fields are continuous, so
    the side does not matter."""

    @lru_cache(maxsize=None)
    def at(z: float) -> tuple[np.ndarray, np.ndarray]:
        e = np.zeros(3, dtype=complex)
        h = np.zeros(3, dtype=complex)
        for n, cn in enumerate(c):
            p = basis.profile(n, [z])
            e += cn * p.electric_v_per_m[0]
            h += cn * p.magnetic_a_per_m[0]
        return e, h

    return lambda z, side: at(z)


def kappa(basis: lamina.ModalBasis, n: int, side: int) -> complex:
    """The exterior wavevector of mode n on its recorded sheet, independent of lamina."""
    beta = basis.modes[n].beta_rad_per_m
    k = cmath.sqrt(CLAD * K0**2 - beta**2)
    proper = basis.modes[n].sheet[side] == "proper"
    return k if (k.imag > 0) == proper else -k


def continuation(basis: lamina.ModalBasis, c: list[complex]):
    field = combination(basis, c)
    sides = []
    for side, z in enumerate(SUPPORT):
        waves = []
        for n, cn in enumerate(c):
            e, h = combination(basis, [cn if k == n else 0 for k in range(len(c))])(z, "")
            waves.append(lamina.ExteriorWave(kappa(basis, n, side), tuple(e), tuple(h)))
        sides.append(tuple(waves))
    return field, tuple(sides)


def test_a_guided_and_leaky_combination_is_recovered_with_its_continuation(basis):
    c = coefficients(len(basis))
    field, exterior = continuation(basis, c)
    p = basis.project_field(field, SUPPORT, exterior=exterior, field_id="combination")
    assert p.is_full_rank and p.rank == len(basis)
    assert p.modes == tuple(range(len(basis)))
    assert np.max(np.abs(p.coefficients - np.array(c))) < 1e-8
    assert p.relative_residual < 1e-7 and p.gram_defect < 1e-9
    assert p.residual_includes_continuum and p.scope == "discrete_without_continuum"
    assert p.field_sample_count > 0 and p.status == "complete"
    # Without its continuation the field is cut at the support edges.
    cut = basis.project_field(field, SUPPORT)
    assert np.max(np.abs(cut.coefficients - np.array(c))) > 1e-3


def test_a_gaussian_leaves_a_residual_outside_the_discrete_scope(basis):
    def gaussian(z, side):
        ey = math.exp(-(((z - A) / 0.3e-6) ** 2))
        return (0, ey, 0), (0, 0, 1.8 * ey / Z0)

    p = basis.project_field(gaussian, (-0.75e-6, 2 * A + 0.75e-6))
    assert p.residual_includes_continuum
    assert p.relative_residual > 1e-2
    assert p.equation_residual_w_per_m < 1e-9 * max(1.0, p.field_scale_w_per_m)


def test_projection_records_round_trip_and_check_their_definition(walled_basis):
    def field(z, side):
        return (0, math.cos(z / 1e-6), 0), (0, 0, 0)

    p = walled_basis.project_field(field, (-GAP, 2 * A + GAP))
    assert not p.residual_includes_continuum and p.scope == "discrete"
    again = lamina.BasisProjection.from_json(p.to_json())
    assert again == p and again.canonical_hash() == p.canonical_hash()
    assert p.definitions == lamina.DEFINITIONS
    d = json.loads(p.to_json())
    d["definition"] = "something else"
    with pytest.raises(lamina.SchemaError):
        lamina.BasisProjection.from_json(json.dumps(d))


def test_projection_refusals_and_callable_errors(walled_basis, basis):
    zero = lambda z, side: ((0, 0, 0), (0, 0, 0))
    with pytest.raises(lamina.InvalidInput, match="walls"):
        walled_basis.project_field(zero, (-GAP - 0.1e-6, 0.5e-6))
    wave = lamina.ExteriorWave(1j * K0, (0, 1, 0), (0, 0, 0))
    with pytest.raises(lamina.InvalidInput, match="open side"):
        walled_basis.project_field(zero, (-GAP, 2 * A + GAP), exterior=((wave,), ()))
    with pytest.raises(lamina.InvalidInput):
        basis.project_field(zero, (0.1e-6, 0.5e-6), exterior=((wave,), ()))

    class Boom(Exception):
        pass

    def failing(z, side):
        raise Boom()

    with pytest.raises(Boom):
        walled_basis.project_field(failing, (-GAP, 2 * A + GAP))


def test_the_coupling_diagonal_is_each_modes_beta_rate(walled_basis):
    c = walled_basis.local_coupling(lamina.Parameter.thickness(0))
    assert c.parameter_unit == "m" and c.groups == ()
    n = len(walled_basis)
    assert c.perturbation_matrix_rad_per_m.shape == (n, n)
    assert np.allclose(c.perturbation_matrix_rad_per_m, c.perturbation_matrix_rad_per_m.T, rtol=1e-12, atol=0)
    assert not c.coefficient_resolved.diagonal().any()
    assert c.coefficient_resolved[~np.eye(n, dtype=bool)].all()
    h = 1e-4 * 2 * A
    up = lamina.modal_basis(walled(2 * A + h), WAVELENGTH, "TE", FORWARD).beta_rad_per_m
    down = lamina.modal_basis(walled(2 * A - h), WAVELENGTH, "TE", FORWARD).beta_rad_per_m
    rate = (up - down) / (2 * h)
    diagonal = c.perturbation_matrix_rad_per_m.diagonal()
    # Finite differences of re-solved β carry the same absolute noise for every
    # mode, so compare against the largest rate; lamina-mode checks every mode
    # against the characteristic's implicit slope.
    assert np.max(np.abs(diagonal - rate)) < 1e-6 * np.max(np.abs(rate))
    again = lamina.BasisLocalCoupling.from_json(c.to_json())
    assert again == c


def test_coupling_refuses_wavelength_permeability_and_wall_moving_parameters(walled_basis):
    with pytest.raises(lamina.InvalidInput, match="wavelength"):
        walled_basis.local_coupling(lamina.Parameter.wavelength())
    with pytest.raises(lamina.InvalidInput, match="permeability"):
        walled_basis.local_coupling(lamina.Parameter.permeability(1))
    with pytest.raises(lamina.InvalidInput, match="walls"):
        walled_basis.local_coupling(lamina.Parameter.translation())
    with pytest.raises(lamina.InvalidInput):
        walled_basis.local_coupling(lamina.Parameter.thickness(0), gap_rad_per_m=-1.0)

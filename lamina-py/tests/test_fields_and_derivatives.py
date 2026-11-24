"""Field sampling, unit-power normalisation and local derivatives in SI."""

from __future__ import annotations

import math

import numpy as np
import pytest

import lamina
from slab import ladder

WAVELENGTH = 1.0e-6
CORE, D, LEFT, RIGHT = 4.0, 1.5e-6, 1.0, 2.25


def trapezoid(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


@pytest.fixture(scope="module", params=["TE", "TM"])
def basis(request):
    stack = lamina.Stack(left=LEFT, layers=[(CORE, D)], right=RIGHT)
    return lamina.enumerate_guided_modes(
        stack, lamina.GuidedRequest(wavelength_m=WAVELENGTH, polarisation=request.param)
    )


def test_profiles_carry_unit_power_in_si(basis):
    k0 = basis.k0_rad_per_m
    for mode in basis.modes:
        n = mode.n_eff.real
        # Twenty decay lengths leave a truncated tail of e^-40.
        reach_left = 20.0 / (k0 * math.sqrt(n * n - LEFT))
        reach_right = 20.0 / (k0 * math.sqrt(n * n - RIGHT))
        # Integrate region by region: TM's E_z jumps at each interface, so each
        # region's end points are one-sided limits from inside it.
        regions = [
            np.linspace(-reach_left, np.nextafter(0.0, -1.0), 50_001),
            np.linspace(0.0, np.nextafter(D, 0.0), 50_001),
            np.linspace(D, D + reach_right, 50_001),
        ]
        power = 0.0
        for z in regions:
            profile = basis.sample_fields(mode.index, z)
            e, h = profile.electric_v_per_m, profile.magnetic_a_per_m
            # Time-averaged Poynting flux along x: S_x = Re(E_y H_z* - E_z H_y*) / 2.
            s_x = 0.5 * np.real(e[:, 1] * np.conj(h[:, 2]) - e[:, 2] * np.conj(h[:, 1]))
            power += trapezoid(s_x, z)
        assert power == pytest.approx(1.0, rel=1e-6)
        assert profile.normalisation.signed_power_w_per_m == pytest.approx(1.0, rel=1e-9)


def test_fields_decay_deep_into_both_claddings(basis):
    k0 = basis.k0_rad_per_m
    for mode in basis.modes:
        n = mode.n_eff.real
        for eps, side in ((LEFT, -1.0), (RIGHT, 1.0)):
            kappa = k0 * math.sqrt(n * n - eps)
            depth = np.linspace(0.0, 60.0 / kappa, 601)
            z = depth * side + (D if side > 0 else 0.0)
            if side < 0:
                z[0] = np.nextafter(0.0, -1.0)  # the left exterior side of z = 0
            p = basis.sample_fields(mode.index, z)
            magnitude = np.linalg.norm(np.hstack([p.electric_v_per_m, p.magnetic_a_per_m]), axis=1)
            assert np.all(np.diff(magnitude) < 0)
            assert magnitude[-1] / magnitude[0] == pytest.approx(math.exp(-60.0), rel=1e-6)


def test_phase_rule_and_field_components(basis):
    pol = basis.input.request.polarisation
    for mode in basis.modes:
        at_left = basis.sample_fields(mode.index, [0.0])
        primary = at_left.electric_v_per_m[0, 1] if pol == "TE" else at_left.magnetic_a_per_m[0, 1]
        assert primary.real > 0 and abs(primary.imag) <= 1e-12 * abs(primary)
        # TE has Ey, Hx, Hz; TM has Hy, Ex, Ez.
        p = basis.sample_fields(mode.index, np.linspace(-1e-6, D + 1e-6, 11))
        e, h = p.electric_v_per_m, p.magnetic_a_per_m
        absent = (e[:, [0, 2]], h[:, 1]) if pol == "TE" else (e[:, 1], h[:, [0, 2]])
        assert all(np.all(a == 0) for a in absent)


def test_interface_samples_take_the_limit_from_plus_z(basis):
    # Tangential E is continuous; normal D = eps E_z is continuous across z = D.
    # 1e-16 m from the interface, the fields move by about kappa * 1e-16 ~ 1e-9.
    mode = basis.modes[0]
    eps = 1e-16
    below, on, above = basis.sample_fields(mode.index, [D - eps, D, D + eps]).electric_v_per_m
    scale = np.abs(above).max()
    np.testing.assert_allclose(on, above, rtol=0, atol=1e-8 * scale)
    if basis.input.request.polarisation == "TM":
        # Normal D = eps E_z is continuous, so E_z jumps by eps_core / eps_right.
        assert CORE * below[2] == pytest.approx(RIGHT * on[2], rel=1e-8)
        assert abs(below[2] - on[2]) > 0.1 * scale


def neff(thickness: float, wavelength: float, pol: str) -> np.ndarray:
    return np.array(ladder(wavelength, CORE, thickness, LEFT, RIGHT, pol))


def test_thickness_slope_matches_finite_difference(basis):
    pol = basis.input.request.polarisation
    h = 1e-12
    fd = (neff(D + h, WAVELENGTH, pol) - neff(D - h, WAVELENGTH, pol)) / (2 * h) * basis.k0_rad_per_m
    for mode in basis.modes:
        local = basis.local_derivatives(mode.index, thickness_layers=[0])
        slope = local.dbeta_dthickness(0)
        assert slope is not None and abs(slope.imag) < 1e-6 * abs(slope)
        assert slope.real == pytest.approx(fd[mode.index], rel=1e-5)
        units = {p.parameter: p.unit for p in local.partials}
        assert units == {"k0": "characteristic per rad/m", "beta": "characteristic per rad/m",
                         "thickness": "characteristic per m"}


def test_k0_slope_matches_finite_difference(basis):
    pol = basis.input.request.polarisation
    k0 = basis.k0_rad_per_m
    dk = 1e-6 * k0

    def beta(k):
        return neff(D, 2 * math.pi / k, pol) * k

    fd = (beta(k0 + dk) - beta(k0 - dk)) / (2 * dk)
    for mode in basis.modes:
        slope = basis.local_derivatives(mode.index).dbeta_dk0
        assert slope is not None
        assert slope.real == pytest.approx(fd[mode.index], rel=1e-6)

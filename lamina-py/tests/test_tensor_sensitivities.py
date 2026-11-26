"""Fields and sensitivities of anisotropic stacks and of modes counted on the
coupled channel: tensor components, rotations and per-axis model parameters."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import lamina

WAVELENGTH = 1e-6
K0 = 2 * math.pi / WAVELENGTH
CLAD = (1.0, 2.1)
ORDINARY, EXTRAORDINARY = 4.0, 6.25
THICKNESS = 0.8e-6
Z = np.linspace(-0.3e-6, 1.1e-6, 15)


def solve(core, family="TE", wavelength=WAVELENGTH, clad=CLAD) -> lamina.GuidedBasis:
    stack = lamina.Stack(clad[0], [(core, THICKNESS)], clad[1])
    return lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(wavelength, family=family))


def central(build, h: float, mode: int = 0) -> float:
    """(β(+h) − β(−h)) / 2h of mode `mode`, re-solved. Each root is accurate to
    about 1e-10 β, so h is no smaller than about 1e-3 of its parameter's scale."""
    plus, minus = (build(s * h).beta_rad_per_m[mode].real for s in (1, -1))
    return (plus - minus) / (2 * h)


def rot_x(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# ------------------------------------------------ the coupled channel, isotropic


@pytest.fixture(scope="module")
def pair():
    """An isotropic film counted per polarisation and on the coupled channel."""
    stack = lamina.Stack(CLAD[0], [(4.0, THICKNESS)], CLAD[1])
    te = lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(WAVELENGTH, family="TE"))
    coupled = lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(WAVELENGTH))
    index = coupled.labels.index("TE")
    assert coupled.beta_rad_per_m[index] == pytest.approx(te.beta_rad_per_m[0], rel=1e-10)
    return te, coupled, index


def test_coupled_profiles_reduce_to_the_scalar_phase_rule(pair):
    te, coupled, index = pair
    scalar, hybrid = te.sample_fields(0, Z), coupled.sample_fields(index, Z)
    assert hybrid.normalisation.phase_reference_component == "Ey+Z0Hy"
    assert hybrid.normalisation.bilinear_norm_w_per_m is not None
    assert hybrid.normalisation.seed_exterior_power_w_per_m is None and hybrid.relative_boundary_defect is None
    assert scalar.normalisation.bilinear_norm_w_per_m is None
    scale = np.abs(scalar.electric_v_per_m).max()
    np.testing.assert_allclose(hybrid.electric_v_per_m, scalar.electric_v_per_m, atol=1e-7 * scale)
    np.testing.assert_allclose(hybrid.magnetic_a_per_m, scalar.magnetic_a_per_m, atol=1e-7 * scale / 376.7)
    again = lamina.GuidedProfile.from_json(hybrid.to_json())
    assert again == hybrid and again.canonical_hash() == hybrid.canonical_hash()


@pytest.mark.parametrize(
    "parameter",
    [
        lamina.Parameter.thickness(0),
        lamina.Parameter.interface_position(0),
        lamina.Parameter.wavelength(),
        lamina.Parameter.permittivity(1),
        lamina.Parameter.permittivity(1, (1, 1)),
        lamina.Parameter.permittivity(2, (2, 2)),
    ],
    ids=["thickness", "interface", "wavelength", "permittivity", "eps_yy", "eps_zz_right"],
)
def test_coupled_sensitivities_match_the_scalar_ones(pair, parameter):
    te, coupled, index = pair
    scalar = te.sensitivity(0, parameter)
    hybrid = coupled.sensitivity(index, parameter)
    assert hybrid.power_fractions is None and hybrid.conditioning.denominator_threshold is None
    assert hybrid.dbeta_rad_per_m == pytest.approx(scalar.dbeta_rad_per_m, rel=1e-7, abs=1e-9 * K0)
    again = lamina.GuidedSensitivity.from_json(hybrid.to_json())
    assert again == hybrid and again.assumption == coupled.assumption


def test_coupled_group_index_matches_the_scalar_one(pair):
    te, coupled, index = pair
    g, s = coupled.group_index(index), te.group_index(0)
    assert g.value == pytest.approx(s.value, rel=1e-8)
    assert g.relative_discrepancy <= g.tolerance and g.assumption == coupled.assumption
    assert lamina.GroupIndex.from_json(g.to_json()) == g


def test_coupled_modes_refuse_field_derivatives_and_pairings(pair):
    _, coupled, index = pair
    with pytest.raises(lamina.InvalidInput, match="request no z_m"):
        coupled.sensitivity(index, lamina.Parameter.thickness(0), z_m=[0.0])
    with pytest.raises(lamina.InvalidInput, match="TE or TM"):
        lamina.power_overlap(coupled, 0, coupled, 0)
    with pytest.raises(lamina.InvalidInput, match="TE or TM"):
        coupled.local_derivatives(index, [0])


# ------------------------------------------------ diagonal regions: the scalar path


@pytest.mark.parametrize(
    "family, component", [("TE", (1, 1)), ("TM", (0, 0)), ("TM", (2, 2))], ids=["TE-yy", "TM-xx", "TM-zz"]
)
def test_diagonal_components_match_central_differences(family, component):
    entries = [3.0, 4.0, 5.0]

    def build(h):
        e = list(entries)
        e[component[0]] += h
        return solve(lamina.Diagonal(*e), family)

    base = build(0.0)
    s = base.sensitivity(0, lamina.Parameter.permittivity(1, component), z_m=Z)
    assert s.power_fractions is not None and s.power_fractions.sum() == pytest.approx(1.0, abs=1e-9)
    assert s.power_fraction_rates.sum() == pytest.approx(0.0, abs=1e-9)
    assert s.dbeta_rad_per_m.real == pytest.approx(central(build, 1e-3), rel=1e-5)
    # The derivative of the sampled field, against re-sampled re-solved fields.
    h = 1e-3
    fields = [build(sgn * h).sample_fields(0, Z).electric_v_per_m for sgn in (1, -1)]
    np.testing.assert_allclose(
        s.d_electric_v_per_m, (fields[0] - fields[1]) / (2 * h), atol=1e-5 * np.abs(s.electric_v_per_m).max()
    )


@pytest.mark.parametrize(
    "family, axis, terms",
    [("TE", "b", 1), ("TM", "a", 0), ("TM", "c", 0)],
)
def test_sellmeier_axis_parameters_match_central_differences(family, axis, terms):
    ordinary = lamina.SellmeierMaterial(1.5, [(1.2, 0.7e-6)])
    extraordinary = lamina.SellmeierMaterial(1.8, [(1.4, 0.6e-6)])
    core = lamina.Uniaxial(ordinary, extraordinary)
    base = solve(core, family)
    assert base.regions[1].region_class == "diagonal"
    models = {"a": ordinary, "b": ordinary, "c": extraordinary}

    def build(h):
        m = dict(models)
        old = m[axis]
        m[axis] = lamina.SellmeierMaterial(old.constant, [(old.terms[0][0] + h, old.terms[0][1])])
        return solve(lamina.Biaxial(m["a"], m["b"], m["c"]), family)

    s = base.sensitivity(0, lamina.Parameter.sellmeier_coefficient(1, 0, axis=axis))
    assert s.dbeta_rad_per_m.real == pytest.approx(central(build, 1e-3), rel=1e-5)
    record = json.loads(s.to_json())["parameter"]
    assert record == {"kind": "sellmeier_coefficient", "region": 1, "axis": axis, "term": 0}
    # The uniaxial ordinary rate is the sum over a and b.
    if family == "TE":
        other = base.sensitivity(0, lamina.Parameter.sellmeier_coefficient(1, 0, axis="a"))
        assert other.dbeta_rad_per_m == 0.0


def test_a_missing_axis_model_is_invalid():
    base = solve(lamina.Diagonal(3.0, 4.0, 5.0))
    with pytest.raises(lamina.InvalidInput, match="no Sellmeier model on axis y"):
        base.sensitivity(0, lamina.Parameter.sellmeier_constant(1, axis="y"))
    with pytest.raises(lamina.InvalidInput, match="stack axis"):
        base.sensitivity(0, lamina.Parameter.rotation(1, "a"))


# ------------------------------------------------ rotations


def tilted(theta: float) -> lamina.Biaxial:
    return lamina.Uniaxial(ORDINARY, EXTRAORDINARY, optic_axis=(theta, 0.0))


def test_a_tilt_about_y_matches_central_differences_and_its_components():
    theta = 0.6
    base = solve(tilted(theta), "TM")
    s = base.sensitivity(0, lamina.Parameter.rotation(1, "y"))
    assert s.parameter_unit == "rad" and s.power_fractions is None
    assert s.dbeta_rad_per_m.real == pytest.approx(central(lambda h: solve(tilted(theta + h), "TM"), 1e-3), rel=1e-5)
    # Chain rule through the tensor components: dβ/dθ = Σ (dε/dθ)_ij dβ/dε_ij.
    eps = base.regions[1].relative_permittivity
    g = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    rate = g @ eps - eps @ g
    total = sum(
        rate[i, j] * base.sensitivity(0, lamina.Parameter.permittivity(1, (i, j))).dbeta_rad_per_m
        for i in range(3)
        for j in range(3)
        if rate[i, j] != 0
    )
    assert total == pytest.approx(s.dbeta_rad_per_m, rel=1e-9)
    with pytest.raises(lamina.InvalidInput, match="request no z_m"):
        base.sensitivity(0, lamina.Parameter.rotation(1, "y"), z_m=[0.0])


def test_an_off_diagonal_component_on_an_isotropic_tm_mode_matches_a_tilt():
    # A small tilt of a uniaxial core about y perturbs ε_xz = ε_zx; the rate
    # of an isotropic-looking diagonal core along both components is the tilt's.
    base = solve(tilted(0.0), "TM")
    pair = sum(base.sensitivity(0, lamina.Parameter.permittivity(1, c)).dbeta_rad_per_m for c in ((0, 2), (2, 0)))
    tilt = base.sensitivity(0, lamina.Parameter.rotation(1, "y")).dbeta_rad_per_m
    # At θ = 0 the tilt rate of ε_xz is ε_e − ε_o times d(sin θ cos θ)/dθ = 1, and the
    # diagonal entries do not move at first order.
    assert tilt == pytest.approx((EXTRAORDINARY - ORDINARY) * pair, abs=1e-9 * K0)


@pytest.fixture(scope="module")
def hybrid():
    rotation = lamina.Rotation.from_optic_axis(0.6, 0.7)
    return rotation, solve(lamina.Uniaxial(ORDINARY, EXTRAORDINARY, rotation), "all")


def test_a_hybrid_profile_round_trips_and_names_its_reference(hybrid):
    _, basis = hybrid
    profile = basis.sample_fields(0, Z)
    assert profile.normalisation.phase_reference_component in ("Ey+Z0Hy", "Ex+Z0Hx")
    assert np.all(np.abs(profile.electric_v_per_m[5:10]).max(axis=0) > 0)
    again = lamina.GuidedProfile.from_json(profile.to_json())
    assert again == profile and again.canonical_hash() == profile.canonical_hash()


def test_a_hybrid_turn_about_x_matches_central_differences(hybrid):
    rotation, basis = hybrid
    r = rotation.as_array()

    def build(h):
        turned = lamina.Rotation.from_matrix(rot_x(h) @ r)
        return solve(lamina.Uniaxial(ORDINARY, EXTRAORDINARY, turned), "all")

    s = basis.sensitivity(0, lamina.Parameter.rotation(1, "x"))
    assert s.dbeta_rad_per_m.real == pytest.approx(central(build, 1e-3), rel=1e-5)


def test_a_hybrid_group_index_is_certified(hybrid):
    _, basis = hybrid
    # The coupled energy certificate integrates six components by Simpson sums.
    g = basis.group_index(0, tolerance=1e-6)
    assert g.relative_discrepancy <= g.tolerance
    w = basis.sensitivity(0, lamina.Parameter.wavelength()).dbeta_rad_per_m
    # Constant materials: dβ/dλ = −(k0/λ) dβ/dk0.
    assert w == pytest.approx(-K0 / WAVELENGTH * g.derivative, rel=1e-12)

"""Milestone 2 acceptance through the bindings: sensitivities, fractions and the group index."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import lamina

WAVELENGTH = 1.0e-6
LAYERS = [(4.0, 0.6e-6), (2.25, 0.4e-6), (6.0, 0.5e-6)]
LEFT, RIGHT = 1.0, 2.25

# A dispersive slab: core and substrate Sellmeier, poles away from 1 µm.
CORE = (1.5, ((1.2, 0.7e-6),))
SUBSTRATE = (1.6, ((0.3, 0.42e-6),))
CORE_D = 1.2e-6


def multilayer(layers=LAYERS):
    return lamina.Stack(left=LEFT, layers=layers, right=RIGHT)


def dispersive(core=CORE, substrate=SUBSTRATE, d=CORE_D):
    return lamina.Stack(
        left=1.0,
        layers=[(lamina.SellmeierMaterial(*core), d)],
        right=lamina.SellmeierMaterial(*substrate),
    )


def solve(stack, polarisation, wavelength=WAVELENGTH):
    basis = lamina.enumerate_guided_modes(
        stack,
        lamina.GuidedRequest(wavelength_m=wavelength, polarisation=polarisation,
                             abs_beta_rad_per_m=(0.0, 3.0 * 2 * math.pi / wavelength)),
    )
    assert basis.is_complete, basis.unresolved
    return basis


def central(beta_of, h):
    return (beta_of(h) - beta_of(-h)) / (2 * h)


def assert_second_order(errors):
    """Each tenfold smaller step cuts the error about a hundredfold (at least
    second order; faster where the third derivative is small)."""
    order = math.log10(errors[0] / errors[1])
    assert 1.6 < order < 3.0, errors


# ------------------------------------------------------------ eigenvalue


def geometric_cases():
    def thickness(h):
        layers = list(LAYERS)
        layers[1] = (layers[1][0], layers[1][1] + h)
        return multilayer(layers)

    def interface(h):
        # Interface 1 sits between layers 0 and 1; its neighbours stay fixed.
        layers = list(LAYERS)
        layers[0] = (layers[0][0], layers[0][1] + h)
        layers[1] = (layers[1][0], layers[1][1] - h)
        return multilayer(layers)

    def permittivity(h):
        layers = list(LAYERS)
        layers[1] = (layers[1][0] + h, layers[1][1])
        return multilayer(layers)

    return [
        ("thickness", lamina.Parameter.thickness(1), thickness, 4e-9),
        ("interface", lamina.Parameter.interface_position(1), interface, 4e-9),
        ("permittivity", lamina.Parameter.permittivity(2), permittivity, 1e-2),
    ]


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
@pytest.mark.parametrize("case", geometric_cases(), ids=lambda c: c[0])
def test_beta_derivatives_converge_on_a_multilayer(case, polarisation):
    _, parameter, perturbed, h0 = case
    base = solve(multilayer(), polarisation)
    for mode in base.modes:
        result = base.sensitivity(mode.index, parameter)
        exact = result.dbeta_rad_per_m
        assert abs(exact.imag) < 1e-9 * abs(exact)

        def beta(h):
            return solve(perturbed(h), polarisation).modes[mode.index].beta_rad_per_m.real

        errors = [abs(central(beta, h) - exact.real) / abs(exact.real)
                  for h in (h0, h0 / 10, h0 / 100)]
        assert_second_order(errors[:2])
        assert min(errors) < 1e-5, errors


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
def test_rigid_translation_leaves_beta_unchanged(polarisation):
    base = solve(multilayer(), polarisation)
    for mode in base.modes:
        result = base.sensitivity(mode.index, lamina.Parameter.translation())
        scale = abs(mode.beta_rad_per_m) / LAYERS[0][1]
        assert abs(result.dbeta_rad_per_m) < 1e-10 * scale


def dispersive_cases():
    def wavelength(h):
        return dispersive(), WAVELENGTH + h

    def coefficient(h):
        return dispersive(core=(CORE[0], ((CORE[1][0][0] + h, CORE[1][0][1]),))), WAVELENGTH

    def pole(h):
        return dispersive(core=(CORE[0], ((CORE[1][0][0], CORE[1][0][1] + h),))), WAVELENGTH

    def constant(h):
        return dispersive(core=(CORE[0] + h, CORE[1])), WAVELENGTH

    return [
        ("wavelength", lamina.Parameter.wavelength(), wavelength, 1e-9),
        ("sellmeier_coefficient", lamina.Parameter.sellmeier_coefficient(1, 0), coefficient, 1e-2),
        ("sellmeier_pole_wavelength", lamina.Parameter.sellmeier_pole_wavelength(1, 0), pole, 1e-9),
        ("sellmeier_constant", lamina.Parameter.sellmeier_constant(1), constant, 1e-2),
    ]


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
@pytest.mark.parametrize("case", dispersive_cases(), ids=lambda c: c[0])
def test_beta_derivatives_converge_on_a_dispersive_slab(case, polarisation):
    _, parameter, perturbed, h0 = case
    base = solve(dispersive(), polarisation)
    for mode in base.modes:
        exact = base.sensitivity(mode.index, parameter).dbeta_rad_per_m

        def beta(h):
            stack, wavelength = perturbed(h)
            return solve(stack, polarisation, wavelength).modes[mode.index].beta_rad_per_m.real

        errors = [abs(central(beta, h) - exact.real) / abs(exact.real)
                  for h in (h0, h0 / 10, h0 / 100)]
        assert_second_order(errors[:2])
        assert min(errors) < 1e-5, errors


# ------------------------------------------------------------ fields


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
def test_field_derivatives_converge_in_the_material_attached_frame(polarisation):
    # Varying layer 1's thickness moves interfaces 2 and 3; material-attached
    # samples keep their offset from their region's left interface.
    offsets = [(0, 0.2e-6), (0, 0.45e-6), (1, 0.1e-6), (1, 0.3e-6), (2, 0.25e-6)]
    parameter = lamina.Parameter.thickness(1)

    def positions(layers):
        starts = np.concatenate([[0.0], np.cumsum([d for _, d in layers])])
        return [starts[region] + offset for region, offset in offsets]

    base = solve(multilayer(), polarisation)
    for mode in base.modes:
        result = base.sensitivity(mode.index, parameter, positions(LAYERS))
        assert all(f is None for f in result.sample_failures)
        exact = np.hstack([result.d_electric_v_per_m, result.d_magnetic_a_per_m])
        scale = np.abs(exact).max()

        def fields(h):
            layers = list(LAYERS)
            layers[1] = (layers[1][0], layers[1][1] + h)
            p = solve(multilayer(layers), polarisation).sample_fields(mode.index, positions(layers))
            return np.hstack([p.electric_v_per_m, p.magnetic_a_per_m])

        # Truncation dominates at the larger steps; the smallest approaches the
        # noise of independently normalised fields.
        errors = []
        for h in (4e-8, 4e-9, 4e-10):
            fd = (fields(h) - fields(-h)) / (2 * h)
            errors.append(np.abs(fd - exact).max() / scale)
        assert_second_order(errors[:2])
        assert min(errors) < 2e-5, errors
        # The fields themselves are the unit-power profile.
        profile = base.sample_fields(mode.index, positions(LAYERS))
        np.testing.assert_allclose(result.electric_v_per_m, profile.electric_v_per_m,
                                   rtol=1e-8, atol=1e-8 * np.abs(profile.electric_v_per_m).max())


def test_fixed_lab_sample_on_a_moving_interface_is_unavailable():
    base = solve(multilayer(), "TE")
    interface_2 = LAYERS[0][1] + LAYERS[1][1]
    result = base.sensitivity(0, lamina.Parameter.thickness(1), [0.3e-6, interface_2], "fixed_lab")
    assert result.sample_failures[0] is None
    assert result.sample_failures[1] is not None
    assert np.all(result.d_electric_v_per_m[1] == 0)


# ------------------------------------------------------------ fractions


@pytest.mark.parametrize("parameter", [
    lamina.Parameter.thickness(0),
    lamina.Parameter.interface_position(2),
    lamina.Parameter.permittivity(3),
    lamina.Parameter.wavelength(),
], ids=lambda p: p.kind)
def test_power_fractions_sum_to_one_and_rates_to_zero(parameter):
    for polarisation in ("TE", "TM"):
        base = solve(multilayer(), polarisation)
        for mode in base.modes:
            r = base.sensitivity(mode.index, parameter)
            assert r.power_fractions.shape == (len(LAYERS) + 2,)
            assert r.power_fractions.sum() == pytest.approx(1.0, abs=1e-9)
            assert np.all(r.power_fractions > -1e-12)
            # Regional powers converge to 1e-9 relative, so the sum vanishes to that level.
            assert abs(r.power_fraction_rates.sum()) <= 1e-7 * np.abs(r.power_fraction_rates).max()


def test_power_fraction_rates_match_finite_differences():
    base = solve(multilayer(), "TE")
    exact = base.sensitivity(0, lamina.Parameter.thickness(1)).power_fraction_rates

    def fractions(h):
        layers = list(LAYERS)
        layers[1] = (layers[1][0], layers[1][1] + h)
        return solve(multilayer(layers), "TE").sensitivity(0, lamina.Parameter.thickness(1)).power_fractions

    errors = []
    for h in (4e-9, 4e-10):
        fd = (fractions(h) - fractions(-h)) / (2 * h)
        errors.append(np.abs(fd - exact).max() / np.abs(exact).max())
    assert_second_order(errors)
    assert errors[1] < 3e-5, errors


# ------------------------------------------------------------ group index


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
def test_group_index_matches_finite_difference_on_a_dispersive_slab(polarisation):
    base = solve(dispersive(), polarisation)
    k0 = base.k0_rad_per_m
    for mode in base.modes:
        group = base.group_index(mode.index)
        assert group.relative_discrepancy < group.tolerance == 1e-8
        assert group.value > mode.n_eff.real

        def beta_at(k):
            return solve(dispersive(), polarisation, 2 * math.pi / k).modes[mode.index].beta_rad_per_m.real

        errors = [abs(central(lambda d: beta_at(k0 + d), h * k0) - group.value) / group.value
                  for h in (1e-3, 1e-4)]
        assert_second_order(errors)
        assert errors[1] < 1e-5


# ------------------------------------------------------------ records and inputs


def test_records_round_trip():
    base = solve(dispersive(), "TM")
    for record in (
        base.sensitivity(0, lamina.Parameter.sellmeier_pole_wavelength(1, 0), [0.1e-6, 1.0e-6]),
        base.group_index(0),
        base.input,
    ):
        text = record.to_json()
        again = type(record).from_json(text)
        assert again == record and again.to_json() == text
        assert again.canonical_hash() == record.canonical_hash()


def test_sensitivity_record_carries_its_evidence():
    base = solve(multilayer(), "TE")
    record = json.loads(base.sensitivity(0, lamina.Parameter.thickness(1)).to_json())
    assert record["schema"] == "lamina.guided_sensitivity"
    assert record["schema_version"] == lamina.SCHEMA_VERSION == 1
    assert record["input_hash"] == base.input_hash
    assert record["parameter"] == {"kind": "thickness", "layer": 1}
    assert record["parameter_unit"] == "m"
    assert record["conditioning"]["denominator_threshold"] > 0


def test_constant_material_inputs_keep_their_hashes():
    # Hashes from the build before Sellmeier materials and sensitivities existed.
    # They used the earlier default residual limit, so state it explicitly.
    stack = lamina.Stack(left=1.0, layers=[(4.0, 0.6e-6), (2.25, 0.4e-6)], right=2.25)
    request = lamina.GuidedRequest(wavelength_m=1e-6, polarisation="TM",
                                   refinement=lamina.Refinement(residual_rtol=1e-11))
    assert lamina.GuidedInput.resolve(stack, request).canonical_hash() == (
        "90eefa0254022be48e03e99a25bf6cd724b148e1bc7834da2dd74e66a6198952")
    assert lamina.enumerate_guided_modes(stack, request).canonical_hash() == (
        "5f11cb869cbbb31bb96220eebf013ade3175d54b5718cdd94116063d0d57a9e6")


@pytest.mark.parametrize("parameter", [
    lamina.Parameter.thickness(3),
    lamina.Parameter.interface_position(4),
    lamina.Parameter.permittivity(5),
    lamina.Parameter.sellmeier_constant(1),
], ids=lambda p: p.kind)
def test_invalid_parameters_raise_invalid_input(parameter):
    base = solve(multilayer(), "TE")
    with pytest.raises(lamina.InvalidInput):
        base.sensitivity(0, parameter)


def test_sellmeier_pole_at_the_wavelength_is_invalid():
    stack = dispersive(core=(1.5, ((1.2, WAVELENGTH),)))
    with pytest.raises(lamina.InvalidInput):
        lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(wavelength_m=WAVELENGTH))

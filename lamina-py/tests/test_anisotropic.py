"""Anisotropic regions: tagged tensor materials, rotations, classes and families."""

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


def film(core, family="TE", **request) -> lamina.GuidedBasis:
    stack = lamina.Stack(CLAD[0], [(core, THICKNESS)], CLAD[1])
    return lamina.enumerate_guided_modes(stack, lamina.GuidedRequest(WAVELENGTH, family=family, **request))


def uniaxial_ladder(diagonal: tuple[float, float, float], family: str) -> list[float]:
    """Effective indices, highest first, of a film with a diagonal core
    (εxx, εyy, εzz) between isotropic claddings, from the transverse resonance

        q d − atan(w κ_L / (ε_L q)) − atan(w κ_R / (ε_R q)) = m π,

    TE: q² = εyy k0² − β², w = ε_L = ε_R = 1 (μ = 1);
    TM: q² = (εxx/εzz)(εzz k0² − β²), w = εxx, with κ² = β² − ε_side k0².
    """
    exx, eyy, ezz = diagonal
    top = eyy if family == "TE" else ezz

    def phase(n: float) -> float:
        b2 = (n * K0) ** 2
        q2 = eyy * K0**2 - b2 if family == "TE" else exx / ezz * (ezz * K0**2 - b2)
        q = math.sqrt(max(q2, 0.0))
        total = q * THICKNESS
        for side in CLAD:
            kappa = math.sqrt(max(b2 - side * K0**2, 0.0))
            w = (1.0, 1.0) if family == "TE" else (exx, side)
            total -= math.atan2(w[0] * kappa, w[1] * q)
        return total

    lower, upper = math.sqrt(max(CLAD)), math.sqrt(top)
    roots = []
    for m in range(100):
        lo, hi = lower, upper
        if phase(lo) <= m * math.pi:
            break
        while True:
            mid = 0.5 * (lo + hi)
            if mid in (lo, hi):
                break
            lo, hi = (mid, hi) if phase(mid) > m * math.pi else (lo, mid)
        roots.append(0.5 * (lo + hi))
    return roots


AXES = {
    "x": ((math.pi / 2, 0.0), (EXTRAORDINARY, ORDINARY, ORDINARY)),
    "y": ((math.pi / 2, math.pi / 2), (ORDINARY, EXTRAORDINARY, ORDINARY)),
    "z": ((0.0, 0.0), (ORDINARY, ORDINARY, EXTRAORDINARY)),
}


@pytest.mark.parametrize("family", ["TE", "TM"])
@pytest.mark.parametrize("axis", sorted(AXES))
def test_a_uniaxial_slab_along_a_stack_axis_matches_the_closed_form(axis, family):
    angles, diagonal = AXES[axis]
    basis = film(lamina.Uniaxial(ORDINARY, EXTRAORDINARY, optic_axis=angles), family)
    core = basis.regions[1]
    assert core.region_class == "diagonal" and core.separable
    np.testing.assert_array_equal(core.relative_permittivity, np.diag(diagonal))
    assert basis.count_method == "oscillation" and basis.is_complete
    expected = uniaxial_ladder(diagonal, family)
    assert len(expected) >= 2
    np.testing.assert_allclose(basis.n_eff.real, expected, rtol=1e-11)
    assert set(basis.labels) == {family}
    # The same medium stated on the stack axes solves to the same modes.
    same = film(lamina.Diagonal(*diagonal), family)
    np.testing.assert_allclose(same.n_eff.real, basis.n_eff.real, rtol=1e-13)


def test_an_xz_tilt_is_full_and_separable_with_the_rotated_tensor():
    theta = 0.6
    basis = film(lamina.Uniaxial(ORDINARY, EXTRAORDINARY, optic_axis=(theta, 0.0)), "TM")
    core = basis.regions[1]
    assert core.region_class == "full" and core.separable
    eps = core.relative_permittivity
    expected = (EXTRAORDINARY - ORDINARY) * math.sin(theta) * math.cos(theta)
    assert eps[0, 2] == pytest.approx(expected, rel=1e-14) and eps[2, 0] == pytest.approx(expected, rel=1e-14)
    assert eps[0, 1] == eps[1, 2] == 0.0
    assert basis.count_method == "argument_principle" and basis.is_complete
    assert set(basis.labels) == {"TM"} and len(basis) >= 1


@pytest.fixture(scope="module")
def hybrid():
    return film(lamina.Uniaxial(ORDINARY, EXTRAORDINARY, optic_axis=(0.6, 0.7)), "all")


def test_a_rotated_film_gives_hybrid_modes(hybrid):
    core = hybrid.regions[1]
    assert core.region_class == "full" and not core.separable
    assert hybrid.is_complete and len(hybrid) >= 2
    assert set(hybrid.labels) == {"hybrid"}
    assert hybrid.count_evidence.channel == "coupled"
    assert hybrid.count_evidence.separable_counts is None
    assert all(f == 1.0 for f in hybrid.count_evidence.off_family)
    again = lamina.GuidedBasis.from_json(hybrid.to_json())
    assert again == hybrid


@pytest.mark.parametrize("family", ["TE", "TM"])
def test_te_or_tm_on_a_non_separable_stack_is_invalid(family):
    core = lamina.Uniaxial(ORDINARY, EXTRAORDINARY, optic_axis=(0.6, 0.7))
    with pytest.raises(lamina.InvalidInput, match="couples TE and TM"):
        film(core, family)


def resolve(core) -> lamina.GuidedInput:
    stack = lamina.Stack(CLAD[0], [(core, THICKNESS)], CLAD[1])
    return lamina.GuidedInput.resolve(stack, lamina.GuidedRequest(WAVELENGTH))


MATERIALS = {
    "diagonal": lamina.Diagonal(2.0, lamina.SellmeierMaterial(1.5, [(1.2, 0.7e-6)]), 3.0),
    "biaxial": lamina.Biaxial(2.0, 3.0, 4.0, lamina.Rotation.from_euler(0.4, 0.7, 0.2)),
    "uniaxial": lamina.Uniaxial(ORDINARY, EXTRAORDINARY, optic_axis=(0.6, 0.7)),
    "tensor": lamina.TensorMaterial(
        [[4.0, 0.3, 0.0], [0.3, 5.0, 0.0], [0.0, 0.0, 6.0]], rotation=lamina.Rotation.from_optic_axis(0.3)
    ),
    "lorentz": lamina.Lorentz(2.0, [(1.5, 0.5 * K0, 0.0)]),
    "drude_lorentz": lamina.DrudeLorentz(5.0, 0.3 * K0, 0.0, [(0.5, 0.4 * K0, 0.0)]),
    "tabulated": lamina.Tabulated([0.8e-6, 1.2e-6], [3.0, 3.5 + 0.0j]),
}


@pytest.mark.parametrize("name", sorted(MATERIALS))
def test_every_material_round_trips_with_a_stable_hash(name):
    record = resolve(MATERIALS[name])
    text = record.to_json()
    again = lamina.GuidedInput.from_json(text)
    assert again == record and again.to_json() == text
    assert again.canonical_hash() == record.canonical_hash()
    assert again.stack.layers[0].material.to_dict() == json.loads(text)["stack"]["layers"][0]["material"]


def test_the_optic_axis_helper_resolves_to_its_matrix():
    by_angles = lamina.Uniaxial(ORDINARY, EXTRAORDINARY, optic_axis=(0.6, 0.7))
    record = json.loads(resolve(by_angles).to_json())["stack"]["layers"][0]["material"]
    assert record["kind"] == "full" and record["material"]["kind"] == "principal"
    matrix = record["rotation"]
    c = [row[2] for row in matrix]
    np.testing.assert_allclose(
        c, [math.sin(0.6) * math.cos(0.7), math.sin(0.6) * math.sin(0.7), math.cos(0.6)], rtol=1e-15
    )
    by_matrix = lamina.Uniaxial(ORDINARY, EXTRAORDINARY, lamina.Rotation.from_matrix(matrix))
    assert resolve(by_matrix).to_json() == resolve(by_angles).to_json()
    assert resolve(by_matrix).canonical_hash() == resolve(by_angles).canonical_hash()


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],  # det −1
        [[1.0, 1e-9, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # not orthonormal
    ],
    ids=["improper", "skewed"],
)
def test_an_invalid_rotation_is_invalid_input(matrix):
    with pytest.raises(lamina.InvalidInput, match="rotation"):
        resolve(lamina.Biaxial(2.0, 3.0, 4.0, lamina.Rotation.from_matrix(matrix)))


def test_hashes_follow_rotation_class_and_family():
    base = lamina.Uniaxial(ORDINARY, EXTRAORDINARY, optic_axis=(0.6, 0.0))
    turned = lamina.Uniaxial(ORDINARY, EXTRAORDINARY, optic_axis=(math.nextafter(0.6, 1.0), 0.0))
    forms = [lamina.Material(4.0), lamina.Diagonal(4.0, 4.0, 4.0), lamina.Biaxial(4.0, 4.0, 4.0)]
    hashes = [resolve(m).canonical_hash() for m in (base, turned, *forms)]
    assert len(set(hashes)) == len(hashes)
    stack = lamina.Stack(CLAD[0], [(base, THICKNESS)], CLAD[1])
    by_family = {
        lamina.GuidedInput.resolve(stack, lamina.GuidedRequest(WAVELENGTH, family=f)).canonical_hash()
        for f in ("TE", "TM", "all")
    }
    assert len(by_family) == 3


def test_identical_media_in_three_forms_solve_alike():
    te = [film(m).n_eff.real for m in (lamina.Material(4.0), lamina.Diagonal(4.0, 4.0, 4.0), lamina.Biaxial(4.0, 4.0, 4.0))]
    np.testing.assert_array_equal(te[0], te[1])
    np.testing.assert_array_equal(te[0], te[2])
    classes = [film(m).regions[1].region_class for m in (lamina.Material(4.0), lamina.Biaxial(4.0, 4.0, 4.0))]
    assert classes == ["isotropic", "diagonal"]


def test_a_tabulated_model_is_held_at_its_operating_value():
    table = lamina.Tabulated([0.8e-6, 1.25e-6], [3.0, 5.0])
    basis = film(table)
    eps = complex(basis.regions[1].relative_permittivity[0, 0])
    k = [2 * math.pi / 1.25e-6, 2 * math.pi / 0.8e-6]
    expected = 5.0 + (3.0 - 5.0) * (K0 - k[0]) / (k[1] - k[0])
    assert eps == pytest.approx(expected, rel=1e-14)
    assert eps.real == pytest.approx(table.relative_permittivity_at(WAVELENGTH).real, rel=1e-14)
    np.testing.assert_allclose(basis.n_eff.real, film(eps.real).n_eff.real, rtol=1e-12)
    assert "tabulated" in basis.assumption
    with pytest.raises(lamina.InvalidInput, match="tabulated"):
        basis.sensitivity(0, lamina.Parameter.wavelength())
    with pytest.raises(lamina.InvalidInput, match="tabulated"):
        basis.group_index(0)
    with pytest.raises(lamina.InvalidInput, match="outside the tabulated range"):
        film(lamina.Tabulated([1.1e-6, 1.2e-6], [3.0, 3.5]))


def test_an_undamped_lorentz_model_solves_as_its_operating_permittivity():
    model = lamina.Lorentz(2.0, [(1.5, 0.5 * K0, 0.0)])
    eps = model.relative_permittivity_at(WAVELENGTH)
    assert eps.imag == 0.0
    basis = film(model)
    assert basis.regions[1].relative_permittivity[0, 0] == pytest.approx(eps, rel=1e-14)
    np.testing.assert_allclose(basis.n_eff.real, film(eps.real).n_eff.real, rtol=1e-12)
    with pytest.raises(lamina.InvalidInput, match="pole at the operating wavelength"):
        film(lamina.Lorentz(2.0, [(1.5, K0, 0.0)]))


def test_a_lossy_tensor_cannot_carry_a_guided_count():
    lossy = lamina.TensorMaterial([[4.0 + 0.1j, 0, 0], [0, 4.0, 0], [0, 0, 4.0]])
    with pytest.raises(lamina.InvalidInput, match="NotHermitian"):
        film(lossy, "all")


def test_a_walled_diagonal_stack_has_a_modal_basis():
    walls = (lamina.Wall("pec", -0.45e-6), lamina.Wall("pec", 1.15e-6))
    window = lamina.BetaWindow.forward(2.1 * K0, 3.0 * K0, 0.05 * K0)
    diagonal = lamina.Stack(2.25, [(lamina.Diagonal(3.0, 4.0, 5.0), 0.7e-6)], 2.25, *walls)
    isotropic = lamina.Stack(2.25, [(4.0, 0.7e-6)], 2.25, *walls)
    a = lamina.modal_basis(diagonal, WAVELENGTH, "TE", window)
    b = lamina.modal_basis(isotropic, WAVELENGTH, "TE", window)
    assert a.status == b.status == "complete"
    np.testing.assert_allclose(a.beta_rad_per_m, b.beta_rad_per_m, rtol=1e-10)

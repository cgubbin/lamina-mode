"""Schema version 2: the version policy, conventions, mode families and labels."""

from __future__ import annotations

import json

import numpy as np
import pytest

import lamina

WAVELENGTH = 1e-6


def slab() -> lamina.Stack:
    return lamina.Stack(left=1.0, layers=[(4.0, 0.6e-6)], right=2.25)


@pytest.fixture(scope="module")
def te_basis():
    return lamina.enumerate_guided_modes(slab(), lamina.GuidedRequest(WAVELENGTH, family="TE"))


@pytest.fixture(scope="module")
def tm_basis():
    return lamina.enumerate_guided_modes(slab(), lamina.GuidedRequest(WAVELENGTH, family="TM"))


@pytest.fixture(scope="module")
def all_basis():
    return lamina.enumerate_guided_modes(slab(), lamina.GuidedRequest(WAVELENGTH))


def test_every_record_kind_is_version_2():
    assert lamina.SCHEMA_VERSION == 2


@pytest.mark.parametrize(
    "kind",
    [lamina.GuidedBasis, lamina.GuidedInput],
)
def test_a_version_1_record_is_rejected_naming_both_versions(te_basis, kind):
    record = json.loads(te_basis.to_json() if kind is lamina.GuidedBasis else te_basis.input.to_json())
    record["schema_version"] = 1
    with pytest.raises(lamina.SchemaError, match=r"schema version 1 .*reads 2"):
        kind.from_json(json.dumps(record))


def test_a_version_1_input_inside_a_basis_is_rejected(te_basis):
    record = json.loads(te_basis.to_json())
    record["input"]["schema_version"] = 1
    with pytest.raises(lamina.SchemaError, match="schema version 1"):
        lamina.GuidedBasis.from_json(json.dumps(record))


def test_conventions_state_the_tensor_frame_and_the_phase_rule(te_basis):
    conventions = json.loads(te_basis.to_json())["conventions"]
    assert conventions == dict(lamina.CONVENTIONS)
    assert "T = R T_material R^T" in conventions["tensor_frame"]
    assert "hybrid" in conventions["polarisation"]
    assert "Ex + Z0*Hx" in conventions["normalisation"]


def test_materials_and_scalar_models_are_tagged(te_basis):
    stack = json.loads(te_basis.input.to_json())["stack"]
    assert stack["left"] == {
        "kind": "isotropic",
        "model": {"kind": "constant", "relative_permittivity": 1.0, "relative_permeability": 1.0},
    }
    core = lamina.SellmeierMaterial(1.5, [(1.2, 0.7e-6)]).to_dict()
    assert core["kind"] == "isotropic" and core["model"]["kind"] == "sellmeier"


def test_an_omitted_family_resolves_to_all_with_the_same_hash():
    explicit = lamina.GuidedInput.resolve(slab(), lamina.GuidedRequest(WAVELENGTH, family="all"))
    record = json.loads(explicit.to_json())
    assert record["request"]["family"] == "all"
    del record["request"]["family"]
    text = json.dumps(record)
    assert lamina._native.canonical_json("guided_input", text) == explicit.to_json()
    assert lamina._native.canonical_hash("guided_input", text) == explicit.canonical_hash()
    assert lamina.GuidedRequest(WAVELENGTH).family == "all"


def test_family_changes_the_hash():
    hashes = {
        lamina.GuidedInput.resolve(slab(), lamina.GuidedRequest(WAVELENGTH, family=f)).canonical_hash()
        for f in ("TE", "TM", "all")
    }
    assert len(hashes) == 3


def test_every_mode_of_an_isotropic_slab_is_its_te_and_tm_modes(te_basis, tm_basis, all_basis):
    assert all_basis.is_complete and all_basis.count_method == "argument_principle"
    assert te_basis.count_method == tm_basis.count_method == "oscillation"
    assert te_basis.count_evidence is None
    expected = sorted(
        [(n.real, "TE") for n in te_basis.n_eff] + [(n.real, "TM") for n in tm_basis.n_eff], reverse=True
    )
    found = sorted(zip(all_basis.n_eff.real, all_basis.labels), reverse=True)
    assert [label for _, label in found] == [label for _, label in expected]
    np.testing.assert_allclose([n for n, _ in found], [n for n, _ in expected], rtol=1e-10)
    evidence = all_basis.count_evidence
    assert evidence.channel == "coupled"
    assert evidence.separable_counts == (len(te_basis), len(tm_basis))
    assert len(evidence.off_family) == len(all_basis)
    assert max(evidence.off_family) < 1e-6
    assert all(b in ("Scatter4", "Transfer4") for b in evidence.count_backends)
    assert set(te_basis.labels) == {"TE"} and set(tm_basis.labels) == {"TM"}


def test_modes_alone_have_no_degenerate_group(te_basis, all_basis):
    assert te_basis.degeneracy_gap_rad_per_m == pytest.approx(1e-6 * te_basis.k0_rad_per_m)
    assert all(m.degenerate_group is None for m in te_basis.modes)
    assert all(m.degenerate_group is None for m in all_basis.modes)


def test_the_coupled_basis_round_trips(all_basis):
    text = all_basis.to_json()
    again = lamina.GuidedBasis.from_json(text)
    assert again == all_basis and again.canonical_hash() == all_basis.canonical_hash()


def test_sensitivity_and_group_index_carry_the_basis_assumption(te_basis):
    s = te_basis.sensitivity(0, lamina.Parameter.thickness(0))
    g = te_basis.group_index(0)
    assert s.assumption == g.assumption == te_basis.assumption
    for record in (s, g):
        assert json.loads(record.to_json())["assumption"] == te_basis.assumption
        assert type(record).from_json(record.to_json()) == record


def test_a_modal_basis_needs_te_or_tm():
    stack = lamina.Stack(2.25, [(4.0, 0.7e-6)], 2.25, lamina.Wall("pec", -0.45e-6), lamina.Wall("pec", 1.15e-6))
    k0 = 2 * np.pi / WAVELENGTH
    window = lamina.BetaWindow.forward(2.1 * k0, 3.0 * k0, 0.05 * k0)
    with pytest.raises(lamina.InvalidInput, match="TE or TM"):
        lamina.modal_basis(stack, WAVELENGTH, "all", window)
    basis = lamina.modal_basis(stack, WAVELENGTH, "TE", window)
    assert json.loads(basis.input.to_json())["family"] == "TE"

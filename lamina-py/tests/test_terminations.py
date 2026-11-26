"""Walls: the record and its identity, refusals, and walled fields."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import lamina

WAVELENGTH = 1.0e-6
D = 0.7e-6
GAP = 0.3e-6
REQUEST = lamina.GuidedRequest(family="TE", wavelength_m=WAVELENGTH)


def stack(left=lamina.Wall("pec", -GAP), right=None, thickness=D) -> lamina.Stack:
    return lamina.Stack(2.25, [(4.0, thickness)], 1.0, left, right)


def resolve(s: lamina.Stack, request=REQUEST) -> lamina.GuidedInput:
    return lamina.GuidedInput.resolve(s, request)


def test_walls_are_internally_tagged_and_open_sides_omitted():
    record = json.loads(resolve(stack(right=lamina.Wall("pmc", D + GAP))).to_json())["stack"]
    assert record["left_termination"] == {"kind": "wall", "type": "pec", "z_m": -GAP}
    assert record["right_termination"] == {"kind": "wall", "type": "pmc", "z_m": D + GAP}
    assert "right_termination" not in json.loads(resolve(stack()).to_json())["stack"]


def test_explicit_open_is_the_open_side():
    walled = resolve(stack())
    record = json.loads(walled.to_json())
    record["stack"]["right_termination"] = {"kind": "open"}
    text = json.dumps(record)
    assert lamina.GuidedInput.from_json(text).to_json() == walled.to_json()
    assert lamina._native.canonical_hash("guided_input", text) == walled.canonical_hash()


@pytest.mark.parametrize(
    "termination",
    [{"kind": "absorber"}, {"kind": "wall", "type": "pec"}, {"kind": "wall", "type": "pec", "z_m": 0.0, "x": 1}],
    ids=["unknown kind", "missing position", "unknown field"],
)
def test_malformed_terminations_are_schema_errors(termination):
    record = json.loads(resolve(stack()).to_json())
    record["stack"]["left_termination"] = termination
    with pytest.raises(lamina.SchemaError):
        lamina.GuidedInput.from_json(json.dumps(record))


@pytest.mark.parametrize(
    "left, right",
    [(lamina.Wall("pec", 1e-9), None), (None, lamina.Wall("pmc", D * (1 - 1e-12)))],
    ids=["left inside", "right inside"],
)
def test_walls_inside_the_layers_are_invalid(left, right):
    with pytest.raises(lamina.InvalidInput, match="left_termination" if left else "right_termination"):
        resolve(stack(left, right))


def test_walls_on_the_outermost_interfaces_resolve_and_enumerate():
    layers = [(4.0, 0.3e-6), (2.25, 0.2e-6), (4.0, 0.1e-6)]
    s = lamina.Stack(2.25, layers, 2.25)
    last = float(s.interfaces_m[-1])
    walled = lamina.Stack(2.25, layers, 2.25, lamina.Wall("pmc", -0.0), lamina.Wall("pec", last))
    basis = lamina.enumerate_guided_modes(walled, REQUEST)
    assert basis.completeness == "complete" and basis.found > 0
    assert json.loads(basis.input.to_json())["stack"]["left_termination"]["z_m"] == 0.0


def test_each_change_changes_the_hash():
    base = resolve(stack())
    up = math.nextafter(-GAP, 0.0)
    thicker = math.nextafter(D, 1.0)
    variants = [
        stack(left=None, right=lamina.Wall("pec", D + GAP)),  # side
        stack(left=None),  # termination type
        stack(left=lamina.Wall("pmc", -GAP)),  # wall kind
        stack(left=lamina.Wall("pec", up)),  # wall position
        stack(thickness=thicker, right=lamina.Wall("pec", D + GAP)),  # with the right wall fixed
    ]
    hashes = {base.canonical_hash()} | {resolve(v).canonical_hash() for v in variants}
    assert len(hashes) == len(variants) + 1
    fixed = json.loads(resolve(variants[-1]).to_json())["stack"]
    assert fixed["right_termination"]["z_m"] == D + GAP
    assert fixed["left_termination"]["z_m"] == -GAP


def test_equal_walled_inputs_hash_equally():
    shorthand = lamina.Stack(left=2.25, layers=[(4, D)], right=1, left_termination=lamina.Wall("pec", -GAP))
    explicit = lamina.Stack(
        lamina.Material(2.25, 1.0), (lamina.Layer(lamina.Material(4.0), D),), lamina.Material(1.0),
        left_termination=lamina.Wall(kind="pec", z_m=-GAP),
    )
    restored = lamina.Stack.from_dict(json.loads(json.dumps(explicit.to_dict())))
    assert shorthand == explicit == restored
    assert len({resolve(s).canonical_hash() for s in (shorthand, explicit, restored)}) == 1


@pytest.mark.parametrize("polarisation", ["TE", "TM"])
def test_walled_basis_round_trips_and_resamples_at_its_walls(polarisation):
    s = stack(right=lamina.Wall("pmc", D + GAP))
    request = lamina.GuidedRequest(wavelength_m=WAVELENGTH, family=polarisation)
    basis = lamina.enumerate_guided_modes(s, request)
    assert basis.completeness == "complete" and basis.found > 0
    restored = lamina.GuidedBasis.from_json(basis.to_json())
    assert restored == basis
    z = np.linspace(-GAP, D + GAP, 81)
    profile = restored.sample_fields(0, z)
    assert profile == basis.sample_fields(0, z)
    e, h = profile.electric_v_per_m, profile.magnetic_a_per_m
    # PEC on the left: tangential E vanishes (Ey in TE, Ex in TM). PMC on the
    # right: tangential H vanishes (Hx in TE, Hy in TM).
    left, right = (e[0, 1], h[-1, 0]) if polarisation == "TE" else (e[0, 0], h[-1, 1])
    assert abs(left) < 1e-8 * np.abs(e).max()
    assert abs(right) < 1e-8 * np.abs(h).max()


def test_sampling_beyond_a_wall_is_invalid():
    basis = lamina.enumerate_guided_modes(stack(right=lamina.Wall("pmc", D + GAP)), REQUEST)
    for z in (-GAP * 1.001, (D + GAP) * 1.001):
        with pytest.raises(lamina.InvalidInput):
            basis.sample_fields(0, [z])


@pytest.mark.parametrize(
    "parameter", [lamina.Parameter.translation(), lamina.Parameter.interface_position(0)], ids=lambda p: p.kind
)
def test_moving_the_stack_against_its_walls_is_invalid(parameter):
    basis = lamina.enumerate_guided_modes(stack(), REQUEST)
    with pytest.raises(lamina.InvalidInput):
        basis.sensitivity(0, parameter)
    assert basis.sensitivity(0, lamina.Parameter.thickness(0)).dbeta_rad_per_m is not None

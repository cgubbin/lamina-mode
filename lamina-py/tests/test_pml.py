"""PMLs: the record and its identity, validation, and the enumeration refusal."""

from __future__ import annotations

import json
import math

import pytest

import lamina

WAVELENGTH = 1.0e-6
D = 0.7e-6
GAP = 0.3e-6
THICKNESS = 1.0e-6
REQUEST = lamina.GuidedRequest(wavelength_m=WAVELENGTH)


def pml(start=D + GAP, thickness=THICKNESS, order=2.0, strength=1 + 3j, backing="pec") -> lamina.Pml:
    return lamina.Pml(start, thickness, order, strength, backing)


def stack(left=lamina.Wall("pec", -GAP), right=None, thickness=D) -> lamina.Stack:
    return lamina.Stack(2.25, [(4.0, thickness)], 1.0, left, right if right is not None else pml())


def resolve(s: lamina.Stack, request=REQUEST) -> lamina.GuidedInput:
    return lamina.GuidedInput.resolve(s, request)


def test_a_pml_is_internally_tagged_with_its_profile():
    record = json.loads(resolve(stack()).to_json())["stack"]
    assert record["right_termination"] == {
        "kind": "pml",
        "start_z_m": D + GAP,
        "thickness_m": THICKNESS,
        "profile": {"kind": "polynomial", "order": 2.0, "strength": [1.0, 3.0]},
        "backing": "pec",
    }
    assert record["left_termination"] == {"kind": "wall", "type": "pec", "z_m": -GAP}


def test_pml_stacks_round_trip():
    s = stack(left=pml(start=-GAP, backing="pmc", strength=0.5j))
    restored = lamina.Stack.from_dict(json.loads(json.dumps(s.to_dict())))
    assert restored == s
    assert restored.left_termination == pml(start=-GAP, backing="pmc", strength=0.5j)
    resolved = resolve(s)
    again = lamina.GuidedInput.from_json(resolved.to_json())
    assert again.to_json() == resolved.to_json()
    assert again.canonical_hash() == resolved.canonical_hash()


def test_each_pml_change_changes_the_hash():
    base = resolve(stack()).canonical_hash()
    variants = [
        pml(start=math.nextafter(D + GAP, 1.0)),
        pml(thickness=math.nextafter(THICKNESS, 1.0)),
        pml(order=3.0),
        pml(strength=complex(math.nextafter(1.0, 2.0), 3.0)),
        pml(strength=complex(1.0, math.nextafter(3.0, 4.0))),
        pml(backing="pmc"),
    ]
    hashes = [resolve(stack(right=v)).canonical_hash() for v in variants]
    assert len({base, *hashes}) == len(variants) + 1


def test_walled_and_open_inputs_keep_their_hashes():
    # Hashes of these inputs as the build before PMLs computes them.
    walled = lamina.Stack(2.25, [(4.0, D)], 1.0, lamina.Wall("pec", -GAP), lamina.Wall("pmc", D + GAP))
    assert resolve(walled).canonical_hash() == (
        "af39c0a716e411e187ebee628ebb1f42fce45def8077c43fbf5349db3ef0a4a6")
    assert resolve(lamina.Stack(2.25, [(4.0, D)], 1.0)).canonical_hash() == (
        "d11ec787c79a015c8a40283aad0ee76e1d6c9a5ee67bf989cdeed66d91797370")


@pytest.mark.parametrize(
    "termination, field",
    [
        (pml(thickness=0.0), "thickness_m"),
        (pml(thickness=-1e-7), "thickness_m"),
        (pml(order=-1.0), "order"),
        (pml(strength=1 + 0j), "strength"),
        (pml(strength=-0.5 + 1j), "strength"),
        (pml(start=D * (1 - 1e-12)), "start_z_m"),
    ],
    ids=["zero thickness", "negative thickness", "negative order", "real strength",
         "negative real strength", "start inside"],
)
def test_invalid_pmls_name_their_field(termination, field):
    with pytest.raises(lamina.InvalidInput, match=f"right_termination.*{field}"):
        resolve(stack(right=termination))


def test_missing_profile_fields_are_refused():
    with pytest.raises(TypeError, match="strength"):
        lamina.Pml(D + GAP, THICKNESS, 2.0, backing="pec")
    with pytest.raises(lamina.InvalidInput, match="order"):
        lamina.Pml(D + GAP, THICKNESS, None, 1j, "pec")
    record = json.loads(resolve(stack()).to_json())
    del record["stack"]["right_termination"]["profile"]["strength"]
    with pytest.raises(lamina.SchemaError):
        lamina.GuidedInput.from_json(json.dumps(record))


def test_enumerating_a_pml_terminated_stack_is_refused():
    for s in (stack(), stack(left=pml(start=-GAP), right=lamina.Wall("pmc", D + GAP))):
        with pytest.raises(lamina.UnsupportedTermination, match="PML"):
            lamina.enumerate_guided_modes(s, REQUEST)
    assert issubclass(lamina.UnsupportedTermination, lamina.InvalidInput)

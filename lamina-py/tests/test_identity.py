"""Acceptance: equal inputs give equal hashes, and equal results."""

from __future__ import annotations

import lamina


def make(stack_style: str, window=None) -> tuple[lamina.Stack, lamina.GuidedRequest]:
    if stack_style == "shorthand":
        stack = lamina.Stack(left=1, layers=[(4, 2e-6)], right=2.25)
    else:
        stack = lamina.Stack(
            left=lamina.Material(1.0, 1.0),
            layers=(lamina.Layer(lamina.Material(4.0), 2.0e-6),),
            right=lamina.Material(2.25),
        )
    return stack, lamina.GuidedRequest(family="TE", wavelength_m=1.0e-6, abs_beta_rad_per_m=window)


def test_equal_inputs_give_equal_hashes():
    a = lamina.GuidedInput.resolve(*make("shorthand"))
    b = lamina.GuidedInput.resolve(*make("explicit"))
    assert a == b
    assert a.to_json() == b.to_json()
    assert a.canonical_hash() == b.canonical_hash()
    assert hash(a) == hash(b)


def test_signed_zero_has_one_canonical_form():
    a = lamina.GuidedInput.resolve(*make("explicit", window=(0.0, 2e7)))
    b = lamina.GuidedInput.resolve(*make("explicit", window=(-0.0, 2e7)))
    assert a.canonical_hash() == b.canonical_hash()


def test_different_inputs_give_different_hashes():
    stack, request = make("explicit")
    a = lamina.GuidedInput.resolve(stack, request)
    thicker = lamina.Stack(left=1.0, layers=[(4.0, 2.0e-6 * (1 + 1e-15))], right=2.25)
    b = lamina.GuidedInput.resolve(thicker, request)
    tm = lamina.GuidedInput.resolve(stack, lamina.GuidedRequest(wavelength_m=1e-6, family="TM"))
    assert len({a.canonical_hash(), b.canonical_hash(), tm.canonical_hash()}) == 3


def test_equal_inputs_give_bit_identical_results():
    first = lamina.enumerate_guided_modes(*make("shorthand"))
    second = lamina.enumerate_guided_modes(*make("explicit"))
    assert first.input_hash == second.input_hash == first.input.canonical_hash()
    assert first.to_json() == second.to_json()
    assert first.canonical_hash() == second.canonical_hash()
    z = [-1e-6, 0.0, 1e-6, 2e-6, 3e-6]
    assert first.sample_fields(2, z).to_json() == second.sample_fields(2, z).to_json()

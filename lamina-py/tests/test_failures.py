"""Typed failures, and partial coverage reported as a result."""

from __future__ import annotations

import pytest

import lamina

STACK = lamina.Stack(left=1.0, layers=[(4.0, 2e-6)], right=2.25)
REQUEST = lamina.GuidedRequest(wavelength_m=1e-6)


@pytest.mark.parametrize(
    "stack, request_",
    [
        (lamina.Stack(left=1.0, layers=[(4.0, -1e-6)], right=1.0), REQUEST),
        (lamina.Stack(left=-1.0, layers=[(4.0, 1e-6)], right=1.0), REQUEST),
        (STACK, lamina.GuidedRequest(wavelength_m=0.0)),
        (STACK, lamina.GuidedRequest(wavelength_m=1e-6, abs_beta_rad_per_m=(2e7, 1e7))),
        (STACK, lamina.GuidedRequest(wavelength_m=1e-6, cutoff_margin_rad_per_m=float("nan"))),
    ],
    ids=["thickness", "permittivity", "wavelength", "window", "margin"],
)
def test_invalid_inputs_raise_invalid_input(stack, request_):
    with pytest.raises(lamina.InvalidInput):
        lamina.enumerate_guided_modes(stack, request_)


def test_bad_indices_raise_invalid_input():
    basis = lamina.enumerate_guided_modes(STACK, REQUEST)
    with pytest.raises(lamina.InvalidInput):
        basis.sample_fields(basis.found, [0.0])
    with pytest.raises(lamina.InvalidInput):
        basis.local_derivatives(0, thickness_layers=[1])
    with pytest.raises(lamina.InvalidInput):
        basis.local_derivatives(0, thickness_layers=[0, 0])


def test_partial_coverage_is_a_result_with_its_gaps():
    request = lamina.GuidedRequest(wavelength_m=1e-6, max_subdivisions=0)
    basis = lamina.enumerate_guided_modes(STACK, request)
    assert basis.completeness == "partial"
    assert not basis.is_complete
    assert basis.expected is not None and basis.found < basis.expected
    assert basis.unresolved and all(g.cause for g in basis.unresolved)
    assert sum(g.expected or 0 for g in basis.unresolved) + basis.found == basis.expected


def test_failures_share_a_base_class():
    for error in (lamina.EnumerationFailed, lamina.FieldsFailed, lamina.DerivativesFailed,
                  lamina.IntegrationFailed):
        assert issubclass(error, lamina.LaminaError)
    assert issubclass(lamina.InvalidInput, ValueError)
    assert issubclass(lamina.SchemaError, ValueError)

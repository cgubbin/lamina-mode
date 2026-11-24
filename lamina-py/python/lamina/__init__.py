"""Python bindings over the lamina-mode guided basis.

Enumerate the guided modes of a planar multilayer of constant real materials,
sample their unit-power fields and evaluate their local derivatives. Every
quantity is SI with its unit in its name: lengths in metres, wavenumbers in
rad/m, fields in V/m and A/m, power in W per metre of invariant width.

Results keep their evidence (the window, expected and found counts,
unresolved intervals, completeness, and each root's residual and limits) and
round-trip through a canonical, versioned JSON form. Equal inputs give equal
hashes, and equal inputs give bit-identical results on one platform.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from . import _native
from ._native import (
    DerivativesFailed,
    EnumerationFailed,
    FieldsFailed,
    InvalidInput,
    LaminaError,
    SchemaError,
)

__all__ = [
    "SCHEMA_VERSION",
    "CONVENTIONS",
    "Material",
    "Layer",
    "Stack",
    "Refinement",
    "GuidedRequest",
    "GuidedInput",
    "Mode",
    "Gap",
    "GuidedBasis",
    "PowerNormalisation",
    "GuidedProfile",
    "Partial",
    "Slope",
    "LocalDerivatives",
    "enumerate_guided_modes",
    "LaminaError",
    "SchemaError",
    "InvalidInput",
    "EnumerationFailed",
    "FieldsFailed",
    "DerivativesFailed",
]

SCHEMA_VERSION: int = _native.SCHEMA_VERSION
CONVENTIONS: Mapping[str, str] = MappingProxyType(json.loads(_native.conventions()))

_INPUT_SCHEMA = "lamina.guided_input"


def _c(pair: Sequence[float]) -> complex:
    return complex(pair[0], pair[1])


def _pair(z: complex) -> list[float]:
    return [float(z.real), float(z.imag)]


def _opt_c(pair: Sequence[float] | None) -> complex | None:
    return None if pair is None else _c(pair)


def _opt_pair(z: complex | None) -> list[float] | None:
    return None if z is None else _pair(z)


def _dumps(d: Mapping[str, Any]) -> str:
    try:
        return json.dumps(d, allow_nan=False)
    except ValueError as error:
        raise InvalidInput(f"values must be finite: {error}") from None


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


class _Record:
    """Serialisation, equality and hashing through the canonical form."""

    _kind: str

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_json(self) -> str:
        """Canonical JSON: equal records give equal strings."""
        return _native.canonical_json(self._kind, _dumps(self.to_dict()))

    def canonical_hash(self) -> str:
        """SHA-256 (hex) of the canonical JSON."""
        return _native.canonical_hash(self._kind, _dumps(self.to_dict()))

    @classmethod
    def from_json(cls, text: str):
        """Parse a serialised record, checking its schema, version and conventions."""
        canonical = _native.canonical_json(cls._kind, text)
        return cls.from_dict(json.loads(canonical))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]):
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.to_json() == other.to_json()  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return int(self.canonical_hash()[:16], 16)


# ---------------------------------------------------------------- inputs


@dataclass(frozen=True, eq=False)
class Material:
    """A constant, real, positive relative permittivity and permeability."""

    relative_permittivity: float
    relative_permeability: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_permittivity", float(self.relative_permittivity))
        object.__setattr__(self, "relative_permeability", float(self.relative_permeability))

    @property
    def refractive_index(self) -> float:
        return math.sqrt(self.relative_permittivity * self.relative_permeability)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_permittivity": self.relative_permittivity,
            "relative_permeability": self.relative_permeability,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Material:
        return cls(d["relative_permittivity"], d["relative_permeability"])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Material) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash((self.relative_permittivity, self.relative_permeability))


def _material(m: Material | float) -> Material:
    return m if isinstance(m, Material) else Material(float(m))


@dataclass(frozen=True, eq=False)
class Layer:
    """A finite layer of the stack."""

    material: Material
    thickness_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "material", _material(self.material))
        object.__setattr__(self, "thickness_m", float(self.thickness_m))

    def to_dict(self) -> dict[str, Any]:
        return {"material": self.material.to_dict(), "thickness_m": self.thickness_m}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Layer:
        return cls(Material.from_dict(d["material"]), d["thickness_m"])


@dataclass(frozen=True, eq=False)
class Stack:
    """A planar multilayer: left exterior, finite layers along +z, right exterior.

    Materials may be given as a `Material` or as a relative permittivity.
    Layers may be given as a `Layer` or as a `(material, thickness_m)` pair.
    """

    left: Material
    layers: tuple[Layer, ...]
    right: Material

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", _material(self.left))
        object.__setattr__(self, "right", _material(self.right))
        layers = tuple(
            layer if isinstance(layer, Layer) else Layer(_material(layer[0]), layer[1])
            for layer in self.layers
        )
        object.__setattr__(self, "layers", layers)

    @property
    def interfaces_m(self) -> np.ndarray:
        """Interface positions, with the first at z = 0."""
        return _readonly(np.concatenate([[0.0], np.cumsum([l.thickness_m for l in self.layers])]))

    @property
    def max_refractive_index(self) -> float:
        return max(m.refractive_index for m in (self.left, self.right, *(l.material for l in self.layers)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.to_dict(),
            "layers": [l.to_dict() for l in self.layers],
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Stack:
        return cls(
            Material.from_dict(d["left"]),
            tuple(Layer.from_dict(l) for l in d["layers"]),
            Material.from_dict(d["right"]),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Stack) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(json.dumps(self.to_dict()))


_DEFAULT_REFINEMENT = json.loads(_native.default_refinement())


@dataclass(frozen=True)
class Refinement:
    """Root-correction controls.

    A root is accepted when its Newton correction in lamina-mode's native
    coordinate is within `native_atol + native_rtol * |z|` and its residual is
    within `residual_atol + residual_rtol * k0`. The characteristic is linear
    in wavenumber, so the k0-relative limit means the same at every device
    scale. The native tolerances and budgets default to lamina-mode's.
    """

    residual_atol: float = _DEFAULT_REFINEMENT["residual_atol"]
    residual_rtol: float = _DEFAULT_REFINEMENT["residual_rtol"]
    native_atol: float = _DEFAULT_REFINEMENT["native_atol"]
    native_rtol: float = _DEFAULT_REFINEMENT["native_rtol"]
    max_updates: int = _DEFAULT_REFINEMENT["max_updates"]
    max_backtracks: int = _DEFAULT_REFINEMENT["max_backtracks"]
    max_evaluations: int = _DEFAULT_REFINEMENT["max_evaluations"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_atol": float(self.residual_atol),
            "residual_rtol": float(self.residual_rtol),
            "native_atol": float(self.native_atol),
            "native_rtol": float(self.native_rtol),
            "max_updates": int(self.max_updates),
            "max_backtracks": int(self.max_backtracks),
            "max_evaluations": int(self.max_evaluations),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Refinement:
        return cls(**d)


@dataclass(frozen=True)
class GuidedRequest:
    """What to enumerate: one wavelength, polarisation and direction.

    `abs_beta_rad_per_m` is the open interval of |beta| to search. When it is
    None, `enumerate_guided_modes` searches from 0 to just above the highest
    material light line and records the resolved window in the result's input.
    `phase_tolerance` is in cycles of pi, as in lamina-mode.
    """

    wavelength_m: float
    polarisation: Literal["TE", "TM"] = "TE"
    direction: Literal["forward", "backward"] = "forward"
    abs_beta_rad_per_m: tuple[float, float] | None = None
    cutoff_margin_rad_per_m: float = 1e-5
    phase_tolerance: float = 1e-10
    max_subdivisions: int = 1000
    refinement: Refinement = field(default_factory=Refinement)

    @property
    def k0_rad_per_m(self) -> float:
        return 2.0 * math.pi / self.wavelength_m

    def to_dict(self) -> dict[str, Any]:
        if self.abs_beta_rad_per_m is None:
            raise InvalidInput("abs_beta_rad_per_m must be resolved before serialisation")
        lo, hi = self.abs_beta_rad_per_m
        return {
            "wavelength_m": float(self.wavelength_m),
            "polarisation": self.polarisation,
            "direction": self.direction,
            "abs_beta_rad_per_m": [float(lo), float(hi)],
            "cutoff_margin_rad_per_m": float(self.cutoff_margin_rad_per_m),
            "phase_tolerance": float(self.phase_tolerance),
            "max_subdivisions": int(self.max_subdivisions),
            "refinement": self.refinement.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GuidedRequest:
        return cls(
            wavelength_m=d["wavelength_m"],
            polarisation=d["polarisation"],
            direction=d["direction"],
            abs_beta_rad_per_m=tuple(d["abs_beta_rad_per_m"]),
            cutoff_margin_rad_per_m=d["cutoff_margin_rad_per_m"],
            phase_tolerance=d["phase_tolerance"],
            max_subdivisions=d["max_subdivisions"],
            refinement=Refinement.from_dict(d["refinement"]),
        )


@dataclass(frozen=True, eq=False)
class GuidedInput(_Record):
    """A stack and a resolved request: the canonical input of an enumeration."""

    stack: Stack
    request: GuidedRequest
    _kind = "guided_input"

    @classmethod
    def resolve(cls, stack: Stack, request: GuidedRequest) -> GuidedInput:
        """Fill a default |beta| window and validate."""
        if request.abs_beta_rad_per_m is None:
            try:
                upper = 1.01 * request.k0_rad_per_m * stack.max_refractive_index
            except (ValueError, ZeroDivisionError, OverflowError):
                # Left to the native validation below, which names the bad value.
                upper = 1.0
            request = GuidedRequest(**{**request.__dict__, "abs_beta_rad_per_m": (0.0, upper)})
        resolved = cls(stack, request)
        canonical = _native.normalise_input(_dumps(resolved.to_dict()))
        return cls.from_dict(json.loads(canonical))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _INPUT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "stack": self.stack.to_dict(),
            "request": self.request.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GuidedInput:
        return cls(Stack.from_dict(d["stack"]), GuidedRequest.from_dict(d["request"]))


# ---------------------------------------------------------------- results


@dataclass(frozen=True)
class Mode:
    """One accepted guided root, with its numerical evidence."""

    index: int
    beta_rad_per_m: complex
    n_eff: complex
    native_coordinate: complex
    characteristic_value: complex
    characteristic_derivative: complex
    residual: float
    residual_limit: float
    correction: float | None
    correction_limit: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "beta_rad_per_m": _pair(self.beta_rad_per_m),
            "n_eff": _pair(self.n_eff),
            "native_coordinate": _pair(self.native_coordinate),
            "characteristic_value": _pair(self.characteristic_value),
            "characteristic_derivative": _pair(self.characteristic_derivative),
            "residual": self.residual,
            "residual_limit": self.residual_limit,
            "correction": self.correction,
            "correction_limit": self.correction_limit,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Mode:
        return cls(
            d["index"],
            _c(d["beta_rad_per_m"]),
            _c(d["n_eff"]),
            _c(d["native_coordinate"]),
            _c(d["characteristic_value"]),
            _c(d["characteristic_derivative"]),
            d["residual"],
            d["residual_limit"],
            d["correction"],
            d["correction_limit"],
        )


@dataclass(frozen=True)
class Gap:
    """An unresolved |beta| interval and its cause."""

    abs_beta_rad_per_m: tuple[float, float]
    expected: int | None
    cause: str
    detail: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "abs_beta_rad_per_m": list(self.abs_beta_rad_per_m),
            "expected": self.expected,
            "cause": self.cause,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Gap:
        return cls(tuple(d["abs_beta_rad_per_m"]), d["expected"], d["cause"], d["detail"])


@dataclass(frozen=True, eq=False)
class GuidedBasis(_Record):
    """The guided modes of one stack in one window, with their evidence.

    `completeness` is "complete" when every counted state in the eligible
    window has one accepted root, "partial" when some are unresolved, and
    "count_unknown" when an endpoint phase was undecidable.
    """

    conventions: Mapping[str, str]
    assumption: str
    input: GuidedInput
    input_hash: str
    k0_rad_per_m: float
    light_line_rad_per_m: float
    upper_bound_rad_per_m: float
    eligible_rad_per_m: tuple[float, float] | None
    expected: int | None
    found: int
    completeness: Literal["complete", "partial", "count_unknown"]
    modes: tuple[Mode, ...]
    unresolved: tuple[Gap, ...]
    _kind = "guided_basis"

    @property
    def is_complete(self) -> bool:
        return self.completeness == "complete"

    @property
    def beta_rad_per_m(self) -> np.ndarray:
        return _readonly(np.array([m.beta_rad_per_m for m in self.modes], dtype=complex))

    @property
    def n_eff(self) -> np.ndarray:
        return _readonly(np.array([m.n_eff for m in self.modes], dtype=complex))

    def __len__(self) -> int:
        return len(self.modes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "lamina.guided_basis",
            "schema_version": SCHEMA_VERSION,
            "conventions": dict(self.conventions),
            "assumption": self.assumption,
            "input": self.input.to_dict(),
            "input_hash": self.input_hash,
            "k0_rad_per_m": self.k0_rad_per_m,
            "light_line_rad_per_m": self.light_line_rad_per_m,
            "upper_bound_rad_per_m": self.upper_bound_rad_per_m,
            "eligible_rad_per_m": None if self.eligible_rad_per_m is None else list(self.eligible_rad_per_m),
            "expected": self.expected,
            "found": self.found,
            "completeness": self.completeness,
            "modes": [m.to_dict() for m in self.modes],
            "unresolved": [g.to_dict() for g in self.unresolved],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GuidedBasis:
        return cls(
            MappingProxyType(dict(d["conventions"])),
            d["assumption"],
            GuidedInput.from_dict(d["input"]),
            d["input_hash"],
            d["k0_rad_per_m"],
            d["light_line_rad_per_m"],
            d["upper_bound_rad_per_m"],
            None if d["eligible_rad_per_m"] is None else tuple(d["eligible_rad_per_m"]),
            d["expected"],
            d["found"],
            d["completeness"],
            tuple(Mode.from_dict(m) for m in d["modes"]),
            tuple(Gap.from_dict(g) for g in d["unresolved"]),
        )

    def _solver(self) -> Any:
        native = self.__dict__.get("_native")
        if native is None:
            # A deserialised basis: re-solve its input and insist on the same result.
            native = _native.enumerate_guided_modes(self.input.to_json())
            if native.record_json() != self.to_json():
                raise LaminaError("re-solving the stored input did not reproduce this basis")
            object.__setattr__(self, "_native", native)
        return native

    def sample_fields(self, mode_index: int, z_m: Sequence[float] | np.ndarray) -> GuidedProfile:
        """Unit-power E and H of one mode at positions z (m) through the stack."""
        z = np.ascontiguousarray(z_m, dtype=float).ravel()
        text = self._solver().sample_fields(int(mode_index), z.tolist())
        return GuidedProfile.from_dict(json.loads(text))

    def local_derivatives(
        self, mode_index: int, thickness_layers: Sequence[int] = ()
    ) -> LocalDerivatives:
        """Characteristic partials in k0, beta and the chosen layer thicknesses,
        and the implicit slopes dbeta/dk0 and dbeta/d(thickness)."""
        text = self._solver().local_derivatives(int(mode_index), [int(i) for i in thickness_layers])
        return LocalDerivatives.from_dict(json.loads(text))


@dataclass(frozen=True)
class PowerNormalisation:
    """How a profile was scaled to unit longitudinal power."""

    signed_power_w_per_m: float
    scale: complex
    phase_reference: complex
    seed_power_w_per_m: float
    seed_exterior_power_w_per_m: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "signed_power_w_per_m": self.signed_power_w_per_m,
            "scale": _pair(self.scale),
            "phase_reference": _pair(self.phase_reference),
            "seed_power_w_per_m": self.seed_power_w_per_m,
            "seed_exterior_power_w_per_m": list(self.seed_exterior_power_w_per_m),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PowerNormalisation:
        return cls(
            d["signed_power_w_per_m"],
            _c(d["scale"]),
            _c(d["phase_reference"]),
            d["seed_power_w_per_m"],
            tuple(d["seed_exterior_power_w_per_m"]),
        )


def _components(re: Sequence[Sequence[float]], im: Sequence[Sequence[float]]) -> np.ndarray:
    a = np.asarray(re, dtype=float).reshape(-1, 3) + 1j * np.asarray(im, dtype=float).reshape(-1, 3)
    return _readonly(a)


@dataclass(frozen=True, eq=False)
class GuidedProfile(_Record):
    """Unit-power fields of one mode. Arrays have shape (n, 3): (x, y, z)."""

    conventions: Mapping[str, str]
    input_hash: str
    mode_index: int
    beta_rad_per_m: complex
    z_m: np.ndarray
    electric_v_per_m: np.ndarray
    magnetic_a_per_m: np.ndarray
    normalisation: PowerNormalisation
    relative_boundary_defect: float
    _kind = "guided_profile"

    def to_dict(self) -> dict[str, Any]:
        e, h = self.electric_v_per_m, self.magnetic_a_per_m
        return {
            "schema": "lamina.guided_profile",
            "schema_version": SCHEMA_VERSION,
            "conventions": dict(self.conventions),
            "input_hash": self.input_hash,
            "mode_index": self.mode_index,
            "beta_rad_per_m": _pair(self.beta_rad_per_m),
            "z_m": self.z_m.tolist(),
            "electric_real": e.real.tolist(),
            "electric_imag": e.imag.tolist(),
            "magnetic_real": h.real.tolist(),
            "magnetic_imag": h.imag.tolist(),
            "normalisation": self.normalisation.to_dict(),
            "relative_boundary_defect": self.relative_boundary_defect,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GuidedProfile:
        return cls(
            MappingProxyType(dict(d["conventions"])),
            d["input_hash"],
            d["mode_index"],
            _c(d["beta_rad_per_m"]),
            _readonly(np.asarray(d["z_m"], dtype=float)),
            _components(d["electric_real"], d["electric_imag"]),
            _components(d["magnetic_real"], d["magnetic_imag"]),
            PowerNormalisation.from_dict(d["normalisation"]),
            d["relative_boundary_defect"],
        )


@dataclass(frozen=True)
class Partial:
    """One characteristic partial, or the reason it is unavailable."""

    parameter: Literal["k0", "beta", "thickness"]
    layer: int | None
    value: complex | None
    unit: str
    held_fixed: str | None
    failure: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "layer": self.layer,
            "value": _opt_pair(self.value),
            "unit": self.unit,
            "held_fixed": self.held_fixed,
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Partial:
        return cls(d["parameter"], d["layer"], _opt_c(d["value"]), d["unit"], d["held_fixed"], d["failure"])


@dataclass(frozen=True)
class Slope:
    """One implicit simple-root slope, or the reason it is unavailable."""

    name: Literal["dbeta_dk0", "dbeta_dthickness"]
    layer: int | None
    value: complex | None
    unit: str
    failure: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "value": _opt_pair(self.value),
            "unit": self.unit,
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Slope:
        return cls(d["name"], d["layer"], _opt_c(d["value"]), d["unit"], d["failure"])


@dataclass(frozen=True, eq=False)
class LocalDerivatives(_Record):
    """Local derivatives at one accepted root.

    The characteristic and its partials are in lamina-core's backend gauge;
    the implicit slopes are gauge free. These are complex local slopes, not a
    certified group velocity.
    """

    conventions: Mapping[str, str]
    assumption: str
    input_hash: str
    mode_index: int
    characteristic: complex | None
    characteristic_failure: str | None
    partials: tuple[Partial, ...]
    slopes: tuple[Slope, ...]
    _kind = "local_derivatives"

    @property
    def dbeta_dk0(self) -> complex | None:
        return next(s.value for s in self.slopes if s.name == "dbeta_dk0")

    def dbeta_dthickness(self, layer: int) -> complex | None:
        """d beta / d(thickness of `layer`), in rad/m per m."""
        for s in self.slopes:
            if s.name == "dbeta_dthickness" and s.layer == layer:
                return s.value
        raise KeyError(f"layer {layer} was not requested")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "lamina.local_derivatives",
            "schema_version": SCHEMA_VERSION,
            "conventions": dict(self.conventions),
            "assumption": self.assumption,
            "input_hash": self.input_hash,
            "mode_index": self.mode_index,
            "characteristic": _opt_pair(self.characteristic),
            "characteristic_failure": self.characteristic_failure,
            "partials": [p.to_dict() for p in self.partials],
            "slopes": [s.to_dict() for s in self.slopes],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LocalDerivatives:
        return cls(
            MappingProxyType(dict(d["conventions"])),
            d["assumption"],
            d["input_hash"],
            d["mode_index"],
            _opt_c(d["characteristic"]),
            d["characteristic_failure"],
            tuple(Partial.from_dict(p) for p in d["partials"]),
            tuple(Slope.from_dict(s) for s in d["slopes"]),
        )


# ---------------------------------------------------------------- entry point


def enumerate_guided_modes(stack: Stack, request: GuidedRequest) -> GuidedBasis:
    """Enumerate the guided modes of `stack` for `request`.

    Partial coverage is a result, not an error: read `completeness`,
    `expected`, `found` and `unresolved` together. Invalid inputs raise
    `InvalidInput`; a failed enumeration raises `EnumerationFailed`.
    """
    resolved = GuidedInput.resolve(stack, request)
    native = _native.enumerate_guided_modes(resolved.to_json())
    basis = GuidedBasis.from_dict(json.loads(native.record_json()))
    object.__setattr__(basis, "_native", native)
    return basis

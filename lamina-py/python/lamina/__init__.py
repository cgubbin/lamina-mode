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
    "SellmeierMaterial",
    "Parameter",
    "GuidedSensitivity",
    "GroupIndex",
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

    def refractive_index_at(self, wavelength_m: float) -> float:
        return self.refractive_index

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


@dataclass(frozen=True)
class SellmeierMaterial:
    """A dispersive dielectric: ε(λ) = constant + Σ Bⱼ λ² / (λ² − λⱼ²), μ = 1.

    `terms` holds `(Bⱼ, λⱼ in metres)` pairs. No pole may sit at the operating
    wavelength.
    """

    constant: float
    terms: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "constant", float(self.constant))
        object.__setattr__(self, "terms", tuple((float(b), float(p)) for b, p in self.terms))

    def relative_permittivity_at(self, wavelength_m: float) -> float:
        l2 = wavelength_m * wavelength_m
        return self.constant + sum(b * l2 / (l2 - p * p) for b, p in self.terms)

    def refractive_index_at(self, wavelength_m: float) -> float:
        return math.sqrt(self.relative_permittivity_at(wavelength_m))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sellmeier": {
                "constant": self.constant,
                "terms": [{"coefficient": b, "pole_wavelength_m": p} for b, p in self.terms],
            },
            "relative_permeability": 1.0,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SellmeierMaterial:
        s = d["sellmeier"]
        return cls(s["constant"], tuple((t["coefficient"], t["pole_wavelength_m"]) for t in s["terms"]))


AnyMaterial = Material | SellmeierMaterial


def _material(m: AnyMaterial | float) -> AnyMaterial:
    return m if isinstance(m, (Material, SellmeierMaterial)) else Material(float(m))


def _material_from_dict(d: Mapping[str, Any]) -> AnyMaterial:
    return SellmeierMaterial.from_dict(d) if "sellmeier" in d else Material.from_dict(d)


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
        return cls(_material_from_dict(d["material"]), d["thickness_m"])


@dataclass(frozen=True, eq=False)
class Stack:
    """A planar multilayer: left exterior, finite layers along +z, right exterior.

    Materials may be given as a `Material`, a `SellmeierMaterial`, or a relative permittivity.
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

    def max_refractive_index(self, wavelength_m: float) -> float:
        materials = (self.left, self.right, *(l.material for l in self.layers))
        return max(m.refractive_index_at(wavelength_m) for m in materials)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.to_dict(),
            "layers": [l.to_dict() for l in self.layers],
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Stack:
        return cls(
            _material_from_dict(d["left"]),
            tuple(Layer.from_dict(l) for l in d["layers"]),
            _material_from_dict(d["right"]),
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
                upper = 1.01 * request.k0_rad_per_m * stack.max_refractive_index(request.wavelength_m)
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

    def sensitivity(
        self,
        mode_index: int,
        parameter: Parameter,
        z_m: Sequence[float] | np.ndarray = (),
        frame: Literal["material_attached", "fixed_lab"] = "material_attached",
    ) -> GuidedSensitivity:
        """Total derivatives of one mode with respect to `parameter`: dβ/dp, the
        unit-power fields at z (m) and their derivatives in `frame`, and the
        regional power fractions with their derivatives."""
        z = np.ascontiguousarray(z_m, dtype=float).ravel()
        text = self._solver().sensitivity(
            int(mode_index), _dumps(parameter.to_dict()), z.tolist(), frame
        )
        return GuidedSensitivity.from_dict(json.loads(text))

    def group_index(self, mode_index: int, tolerance: float = 1e-8) -> GroupIndex:
        """The group index of one mode, certified by two independent estimates
        agreeing within `tolerance`; raises `DerivativesFailed` otherwise."""
        text = self._solver().group_index(int(mode_index), float(tolerance))
        return GroupIndex.from_dict(json.loads(text))


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


@dataclass(frozen=True)
class Parameter:
    """A stack or material parameter to differentiate with respect to.

    Regions run left exterior (0), layers (1..n), right exterior (n + 1).
    Interfaces run 0..n from the left. Build one with the class methods.
    """

    kind: str
    layer: int | None = None
    interface: int | None = None
    region: int | None = None
    term: int | None = None

    @classmethod
    def thickness(cls, layer: int) -> Parameter:
        """A layer's thickness, with the interfaces to its left fixed (per m)."""
        return cls("thickness", layer=layer)

    @classmethod
    def interface_position(cls, interface: int) -> Parameter:
        """One interface's position, with its neighbours fixed (per m)."""
        return cls("interface", interface=interface)

    @classmethod
    def translation(cls) -> Parameter:
        """A rigid translation of the whole stack along z (per m)."""
        return cls("translation")

    @classmethod
    def wavelength(cls) -> Parameter:
        """The vacuum wavelength, with every material's dispersion (per m)."""
        return cls("wavelength")

    @classmethod
    def permittivity(cls, region: int) -> Parameter:
        """A region's relative permittivity at the operating wavelength (per unit)."""
        return cls("permittivity", region=region)

    @classmethod
    def permeability(cls, region: int) -> Parameter:
        """A region's relative permeability at the operating wavelength (per unit)."""
        return cls("permeability", region=region)

    @classmethod
    def sellmeier_constant(cls, region: int) -> Parameter:
        return cls("sellmeier_constant", region=region)

    @classmethod
    def sellmeier_coefficient(cls, region: int, term: int) -> Parameter:
        return cls("sellmeier_coefficient", region=region, term=term)

    @classmethod
    def sellmeier_pole_wavelength(cls, region: int, term: int) -> Parameter:
        """A Sellmeier pole wavelength (per m)."""
        return cls("sellmeier_pole_wavelength", region=region, term=term)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind}
        for key in ("layer", "interface", "region", "term"):
            value = getattr(self, key)
            if value is not None:
                d[key] = int(value)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Parameter:
        return cls(**d)


@dataclass(frozen=True)
class Conditioning:
    """Conditioning of the implicit derivative at a simple root."""

    characteristic: complex
    characteristic_direction: complex
    characteristic_beta: complex
    denominator_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "characteristic": _pair(self.characteristic),
            "characteristic_direction": _pair(self.characteristic_direction),
            "characteristic_beta": _pair(self.characteristic_beta),
            "denominator_threshold": self.denominator_threshold,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Conditioning:
        return cls(
            _c(d["characteristic"]),
            _c(d["characteristic_direction"]),
            _c(d["characteristic_beta"]),
            d["denominator_threshold"],
        )


@dataclass(frozen=True, eq=False)
class GuidedSensitivity(_Record):
    """Total derivatives of one mode with respect to one parameter.

    `dbeta_rad_per_m` is in rad/m per `parameter_unit`. Field arrays have shape
    (n, 3); derivative arrays are per `parameter_unit` and are zero where
    `sample_failures` names a cause. Fractions run left exterior, layers, right
    exterior; they sum to one and their rates to zero.
    """

    conventions: Mapping[str, str]
    input_hash: str
    mode_index: int
    parameter: Parameter
    parameter_unit: str
    frame: str
    beta_rad_per_m: complex
    dbeta_rad_per_m: complex
    conditioning: Conditioning
    z_m: np.ndarray
    electric_v_per_m: np.ndarray
    magnetic_a_per_m: np.ndarray
    d_electric_v_per_m: np.ndarray
    d_magnetic_a_per_m: np.ndarray
    sample_failures: tuple[str | None, ...]
    power_fractions: np.ndarray
    power_fraction_rates: np.ndarray
    _kind = "guided_sensitivity"

    def to_dict(self) -> dict[str, Any]:
        e, h = self.electric_v_per_m, self.magnetic_a_per_m
        de, dh = self.d_electric_v_per_m, self.d_magnetic_a_per_m
        return {
            "schema": "lamina.guided_sensitivity",
            "schema_version": SCHEMA_VERSION,
            "conventions": dict(self.conventions),
            "input_hash": self.input_hash,
            "mode_index": self.mode_index,
            "parameter": self.parameter.to_dict(),
            "parameter_unit": self.parameter_unit,
            "frame": self.frame,
            "beta_rad_per_m": _pair(self.beta_rad_per_m),
            "dbeta_rad_per_m": _pair(self.dbeta_rad_per_m),
            "conditioning": self.conditioning.to_dict(),
            "z_m": self.z_m.tolist(),
            "electric_real": e.real.tolist(),
            "electric_imag": e.imag.tolist(),
            "magnetic_real": h.real.tolist(),
            "magnetic_imag": h.imag.tolist(),
            "d_electric_real": de.real.tolist(),
            "d_electric_imag": de.imag.tolist(),
            "d_magnetic_real": dh.real.tolist(),
            "d_magnetic_imag": dh.imag.tolist(),
            "sample_failures": list(self.sample_failures),
            "power_fractions": self.power_fractions.tolist(),
            "power_fraction_rates": self.power_fraction_rates.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GuidedSensitivity:
        return cls(
            MappingProxyType(dict(d["conventions"])),
            d["input_hash"],
            d["mode_index"],
            Parameter.from_dict(d["parameter"]),
            d["parameter_unit"],
            d["frame"],
            _c(d["beta_rad_per_m"]),
            _c(d["dbeta_rad_per_m"]),
            Conditioning.from_dict(d["conditioning"]),
            _readonly(np.asarray(d["z_m"], dtype=float)),
            _components(d["electric_real"], d["electric_imag"]),
            _components(d["magnetic_real"], d["magnetic_imag"]),
            _components(d["d_electric_real"], d["d_electric_imag"]),
            _components(d["d_magnetic_real"], d["d_magnetic_imag"]),
            tuple(d["sample_failures"]),
            _readonly(np.asarray(d["power_fractions"], dtype=float)),
            _readonly(np.asarray(d["power_fraction_rates"], dtype=float)),
        )


@dataclass(frozen=True, eq=False)
class GroupIndex(_Record):
    """A certified group index: `value` = Re(dβ/dk0), signed like β, agreeing with
    the energy-velocity estimate c W / P within `tolerance`."""

    conventions: Mapping[str, str]
    input_hash: str
    mode_index: int
    value: float
    derivative: complex
    energy_estimate: float
    relative_discrepancy: float
    tolerance: float
    conditioning: Conditioning
    _kind = "group_index"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "lamina.group_index",
            "schema_version": SCHEMA_VERSION,
            "conventions": dict(self.conventions),
            "input_hash": self.input_hash,
            "mode_index": self.mode_index,
            "value": self.value,
            "derivative": _pair(self.derivative),
            "energy_estimate": self.energy_estimate,
            "relative_discrepancy": self.relative_discrepancy,
            "tolerance": self.tolerance,
            "conditioning": self.conditioning.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GroupIndex:
        return cls(
            MappingProxyType(dict(d["conventions"])),
            d["input_hash"],
            d["mode_index"],
            d["value"],
            _c(d["derivative"]),
            d["energy_estimate"],
            d["relative_discrepancy"],
            d["tolerance"],
            Conditioning.from_dict(d["conditioning"]),
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

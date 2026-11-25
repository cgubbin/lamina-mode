"""Python bindings over the lamina-mode guided basis.

Enumerate the guided modes of a planar multilayer of constant real materials,
sample their unit-power fields, evaluate their local derivatives and pair
modes with each other and with supplied fields. For a stack closed by walls,
find every mode in a window of the complex β plane as a counted, normalised
modal basis, and pair two such bases. Every quantity is SI with its
unit in its name: lengths in metres, wavenumbers in rad/m, fields in V/m and
A/m, power in W per metre of invariant width.

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
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from . import _native
from ._native import (
    DerivativesFailed,
    EnumerationFailed,
    FieldsFailed,
    IntegrationFailed,
    InvalidInput,
    LaminaError,
    SchemaError,
    UnsupportedTermination,
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
    "Wall",
    "Pml",
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
    "PowerOverlap",
    "PowerOverlapSensitivity",
    "Reciprocity",
    "LocalCoupling",
    "FieldProjection",
    "DEFINITIONS",
    "BetaWindow",
    "ChartWindow",
    "ModalBasisInput",
    "UnresolvedRegion",
    "BasisMode",
    "ModalBasis",
    "ModalProfile",
    "BasisPairing",
    "ExteriorWave",
    "BasisProjection",
    "BasisLocalCoupling",
    "enumerate_guided_modes",
    "modal_basis",
    "chart_window_covering",
    "basis_pairing",
    "power_overlap",
    "power_overlap_sensitivity",
    "reciprocity",
    "LaminaError",
    "SchemaError",
    "InvalidInput",
    "UnsupportedTermination",
    "EnumerationFailed",
    "FieldsFailed",
    "DerivativesFailed",
    "IntegrationFailed",
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


@dataclass(frozen=True)
class Wall:
    """A perfect electric (`"pec"`) or perfect magnetic (`"pmc"`) wall at absolute
    `z_m` in the stack frame (z = 0 at the first interface).

    A PEC wall makes the tangential electric field vanish, a PMC wall the tangential
    magnetic field. The exterior material fills the gap between the wall and the stack.
    """

    kind: Literal["pec", "pmc"]
    z_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "z_m", float(self.z_m))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "wall", "type": self.kind, "z_m": self.z_m}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> Wall | None:
        """A wall, or None for an open side (absent or `{"kind": "open"}`)."""
        if d is None or d.get("kind") == "open":
            return None
        return cls(d["type"], d["z_m"])


@dataclass(frozen=True)
class Pml:
    """A perfectly matched layer of the exterior material, backed by a `"pec"` or
    `"pmc"` wall.

    It starts at absolute `start_z_m` in the stack frame and extends outward, away from
    the layers, by `thickness_m`. With ζ the outward distance into it, the complex
    stretch is s(ζ) = 1 + strength (ζ/thickness)^order, with order ≥ 0, Re strength ≥ 0
    and Im strength > 0. Every field is required.
    """

    start_z_m: float
    thickness_m: float
    order: float
    strength: complex
    backing: Literal["pec", "pmc"]

    def __post_init__(self) -> None:
        for name in ("start_z_m", "thickness_m", "order", "strength"):
            if getattr(self, name) is None:
                raise InvalidInput(f"Pml {name} is required")
        object.__setattr__(self, "start_z_m", float(self.start_z_m))
        object.__setattr__(self, "thickness_m", float(self.thickness_m))
        object.__setattr__(self, "order", float(self.order))
        object.__setattr__(self, "strength", complex(self.strength))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "pml",
            "start_z_m": self.start_z_m,
            "thickness_m": self.thickness_m,
            "profile": {"kind": "polynomial", "order": self.order, "strength": _pair(self.strength)},
            "backing": self.backing,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Pml:
        profile = d["profile"]
        return cls(d["start_z_m"], d["thickness_m"], profile["order"], _c(profile["strength"]), d["backing"])


def _termination_from_dict(d: Mapping[str, Any] | None) -> Wall | Pml | None:
    """A wall, a PML, or None for an open side."""
    return Pml.from_dict(d) if d is not None and d.get("kind") == "pml" else Wall.from_dict(d)


@dataclass(frozen=True, eq=False)
class Stack:
    """A planar multilayer: left exterior, finite layers along +z, right exterior.

    Materials may be given as a `Material`, a `SellmeierMaterial`, or a relative permittivity.
    Layers may be given as a `Layer` or as a `(material, thickness_m)` pair. Each side is
    open (None) or closed by a `Wall` or a `Pml` on or outside its outermost interface.
    """

    left: Material
    layers: tuple[Layer, ...]
    right: Material
    left_termination: Wall | Pml | None = None
    right_termination: Wall | Pml | None = None

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
        d = {
            "left": self.left.to_dict(),
            "layers": [l.to_dict() for l in self.layers],
            "right": self.right.to_dict(),
        }
        for side in ("left_termination", "right_termination"):
            termination = getattr(self, side)
            if termination is not None:
                d[side] = termination.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Stack:
        return cls(
            _material_from_dict(d["left"]),
            tuple(Layer.from_dict(l) for l in d["layers"]),
            _material_from_dict(d["right"]),
            _termination_from_dict(d.get("left_termination")),
            _termination_from_dict(d.get("right_termination")),
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

    def local_coupling(self, parameter: Parameter, gap_rad_per_m: float | None = None) -> LocalCoupling:
        """The perturbation matrix and coupling coefficients of this basis along a
        fixed-wavelength `parameter`.

        The basis must be complete, its regions nonmagnetic with positive
        permittivity, and `parameter` neither the wavelength nor a permeability;
        otherwise `InvalidInput`. Modes whose propagation constants differ by no
        more than `gap_rad_per_m` (default 1e-6 k0) form an unresolved group
        whose mutual coefficients are absent.
        """
        gap = None if gap_rad_per_m is None else float(gap_rad_per_m)
        text = self._solver().local_coupling(_dumps(parameter.to_dict()), gap)
        return LocalCoupling.from_dict(json.loads(text))

    def project_field(
        self,
        field: Callable[[float, str], tuple[Any, Any]],
        support_m: tuple[float, float],
        interfaces_m: Sequence[float] = (),
        origin_m: float = 0.0,
        rank_rtol: float = 1e-8,
        field_id: str | None = None,
    ) -> FieldProjection:
        """Project a supplied field onto this complete basis.

        `field(z_m, side)` returns `(E, H)`: E in V/m and H in A/m, each three
        complex components (x, y, z), at position `z_m` in the field's own
        coordinate. `side` is "left" (the limit from -z), "right" (the limit from
        +z) or "interior". The field must vanish outside `support_m`;
        `interfaces_m` lists the points strictly inside the support where it may
        be discontinuous. `origin_m` places the basis's first interface in the
        field's coordinate.

        The callable runs once per distinct (position, side), from Python, inside
        an adaptive quadrature for each mode, so its cost sets the projection's.
        Each interval is integrated with uniform panels: a support far wider than
        the field's decay length costs samples and may exceed the sampling budget
        (`IntegrationFailed`). An exception the callable raises propagates
        unchanged; a wrong shape or a non-finite value raises `InvalidInput`.

        The result has the Gram matrix, the right-hand side, the numerical rank
        under `rank_rtol` and the singular values; at full rank, the coefficients,
        condition number and residual. `field_id` identifies the field in the
        record and so enters its hash; without it the field is unidentified.
        """
        lo, hi = support_m
        text = self._solver().project_field(
            _field_sampler(field),
            [float(lo), float(hi)],
            [float(z) for z in interfaces_m],
            float(origin_m),
            float(rank_rtol),
            None if field_id is None else str(field_id),
        )
        return FieldProjection.from_dict(json.loads(text))


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


# ---------------------------------------------------------------- pairings


def _origins(origins_m: tuple[float, float]) -> list[float]:
    first, second = origins_m
    return [float(first), float(second)]


def _complex_matrix(re: Sequence[Sequence[float]], im: Sequence[Sequence[float]]) -> np.ndarray:
    return _readonly(np.asarray(re, dtype=float) + 1j * np.asarray(im, dtype=float))


def _complex_vector(re: Sequence[float], im: Sequence[float]) -> np.ndarray:
    return _readonly(np.asarray(re, dtype=float) + 1j * np.asarray(im, dtype=float))


def _optional_matrix(rows: Sequence[Sequence[float | None]]) -> np.ndarray:
    return np.array([[math.nan if v is None else v for v in row] for row in rows], dtype=float)


def _optional_rows(a: np.ndarray, present: np.ndarray) -> list[list[float | None]]:
    return [
        [float(v) if ok else None for v, ok in zip(row, mask)]
        for row, mask in zip(a.tolist(), present.tolist())
    ]


class _TwoModeRecord(_Record):
    """Fields shared by the records of two registered modes."""

    def _identity(self, schema: str) -> dict[str, Any]:
        return {
            "schema": schema,
            "schema_version": SCHEMA_VERSION,
            "conventions": dict(self.conventions),  # type: ignore[attr-defined]
            "assumption": self.assumption,  # type: ignore[attr-defined]
            "definition": self.definition,  # type: ignore[attr-defined]
            "first_input_hash": self.first_input_hash,  # type: ignore[attr-defined]
            "first_mode_index": self.first_mode_index,  # type: ignore[attr-defined]
            "second_input_hash": self.second_input_hash,  # type: ignore[attr-defined]
            "second_mode_index": self.second_mode_index,  # type: ignore[attr-defined]
        }


@dataclass(frozen=True, eq=False)
class PowerOverlap(_TwoModeRecord):
    """The power pairing (1/4) ∫ (E1 × H2* + E2* × H1) · x dz of two unit-power
    modes, linear in the first, in W/m.

    `origins_m` places each basis's first interface in a common frame.
    `exterior_power_w_per_m` holds the analytic tails beyond the outermost
    interfaces (left, right); `partition_m` is the union of both stacks'
    interfaces; `quadrature_change_w_per_m` is the change between the last two
    composite estimates.
    """

    conventions: Mapping[str, str]
    assumption: str
    definition: str
    first_input_hash: str
    first_mode_index: int
    second_input_hash: str
    second_mode_index: int
    origins_m: tuple[float, float]
    power_w_per_m: complex
    exterior_power_w_per_m: tuple[complex, complex]
    partition_m: np.ndarray
    sample_count: int
    quadrature_change_w_per_m: float
    _kind = "guided_power_overlap"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity("lamina.guided_power_overlap"),
            "origins_m": list(self.origins_m),
            "power_w_per_m": _pair(self.power_w_per_m),
            "exterior_power_w_per_m": [_pair(z) for z in self.exterior_power_w_per_m],
            "partition_m": self.partition_m.tolist(),
            "sample_count": self.sample_count,
            "quadrature_change_w_per_m": self.quadrature_change_w_per_m,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PowerOverlap:
        return cls(
            MappingProxyType(dict(d["conventions"])),
            d["assumption"],
            d["definition"],
            d["first_input_hash"],
            d["first_mode_index"],
            d["second_input_hash"],
            d["second_mode_index"],
            tuple(d["origins_m"]),
            _c(d["power_w_per_m"]),
            tuple(_c(z) for z in d["exterior_power_w_per_m"]),
            _readonly(np.asarray(d["partition_m"], dtype=float)),
            d["sample_count"],
            d["quadrature_change_w_per_m"],
        )


@dataclass(frozen=True, eq=False)
class PowerOverlapSensitivity(_TwoModeRecord):
    """The power overlap of two modes and its derivative, each stack along its own
    parameter (None holds it fixed), per one shared `parameter_unit`.

    The base overlap and its evidence are always present. `power_rate_w_per_m`
    is in W/m per `parameter_unit`; when the derivative cannot be formed it is
    None and `derivative_failure` names the cause.
    """

    conventions: Mapping[str, str]
    assumption: str
    definition: str
    first_input_hash: str
    first_mode_index: int
    second_input_hash: str
    second_mode_index: int
    first_parameter: Parameter | None
    second_parameter: Parameter | None
    parameter_unit: str
    origins_m: tuple[float, float]
    power_w_per_m: complex
    exterior_power_w_per_m: tuple[complex, complex]
    partition_m: np.ndarray
    sample_count: int
    quadrature_change_w_per_m: float
    power_rate_w_per_m: complex | None
    exterior_rates_w_per_m: tuple[complex, complex] | None
    rate_sample_count: int | None
    rate_quadrature_change_w_per_m: float | None
    derivative_failure: str | None
    _kind = "guided_overlap_sensitivity"

    def to_dict(self) -> dict[str, Any]:
        rates = self.exterior_rates_w_per_m
        return {
            **self._identity("lamina.guided_overlap_sensitivity"),
            "first_parameter": None if self.first_parameter is None else self.first_parameter.to_dict(),
            "second_parameter": None if self.second_parameter is None else self.second_parameter.to_dict(),
            "parameter_unit": self.parameter_unit,
            "origins_m": list(self.origins_m),
            "power_w_per_m": _pair(self.power_w_per_m),
            "exterior_power_w_per_m": [_pair(z) for z in self.exterior_power_w_per_m],
            "partition_m": self.partition_m.tolist(),
            "sample_count": self.sample_count,
            "quadrature_change_w_per_m": self.quadrature_change_w_per_m,
            "power_rate_w_per_m": _opt_pair(self.power_rate_w_per_m),
            "exterior_rates_w_per_m": None if rates is None else [_pair(z) for z in rates],
            "rate_sample_count": self.rate_sample_count,
            "rate_quadrature_change_w_per_m": self.rate_quadrature_change_w_per_m,
            "derivative_failure": self.derivative_failure,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PowerOverlapSensitivity:
        def parameter(p: Mapping[str, Any] | None) -> Parameter | None:
            return None if p is None else Parameter.from_dict(p)

        rates = d["exterior_rates_w_per_m"]
        return cls(
            MappingProxyType(dict(d["conventions"])),
            d["assumption"],
            d["definition"],
            d["first_input_hash"],
            d["first_mode_index"],
            d["second_input_hash"],
            d["second_mode_index"],
            parameter(d["first_parameter"]),
            parameter(d["second_parameter"]),
            d["parameter_unit"],
            tuple(d["origins_m"]),
            _c(d["power_w_per_m"]),
            tuple(_c(z) for z in d["exterior_power_w_per_m"]),
            _readonly(np.asarray(d["partition_m"], dtype=float)),
            d["sample_count"],
            d["quadrature_change_w_per_m"],
            _opt_c(d["power_rate_w_per_m"]),
            None if rates is None else tuple(_c(z) for z in rates),
            d["rate_sample_count"],
            d["rate_quadrature_change_w_per_m"],
            d["derivative_failure"],
        )


@dataclass(frozen=True, eq=False)
class Reciprocity(_TwoModeRecord):
    """The unconjugated, antisymmetric flux ∫ (E1 × H2 − E2 × H1) · x dz of two
    unit-power modes, in W/m, registered as for `PowerOverlap`."""

    conventions: Mapping[str, str]
    assumption: str
    definition: str
    first_input_hash: str
    first_mode_index: int
    second_input_hash: str
    second_mode_index: int
    origins_m: tuple[float, float]
    flux_w_per_m: complex
    exterior_flux_w_per_m: tuple[complex, complex]
    partition_m: np.ndarray
    sample_count: int
    quadrature_change_w_per_m: float
    _kind = "guided_reciprocity"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity("lamina.guided_reciprocity"),
            "origins_m": list(self.origins_m),
            "flux_w_per_m": _pair(self.flux_w_per_m),
            "exterior_flux_w_per_m": [_pair(z) for z in self.exterior_flux_w_per_m],
            "partition_m": self.partition_m.tolist(),
            "sample_count": self.sample_count,
            "quadrature_change_w_per_m": self.quadrature_change_w_per_m,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Reciprocity:
        return cls(
            MappingProxyType(dict(d["conventions"])),
            d["assumption"],
            d["definition"],
            d["first_input_hash"],
            d["first_mode_index"],
            d["second_input_hash"],
            d["second_mode_index"],
            tuple(d["origins_m"]),
            _c(d["flux_w_per_m"]),
            tuple(_c(z) for z in d["exterior_flux_w_per_m"]),
            _readonly(np.asarray(d["partition_m"], dtype=float)),
            d["sample_count"],
            d["quadrature_change_w_per_m"],
        )


@dataclass(frozen=True, eq=False)
class LocalCoupling(_Record):
    """The local coupling of a basis along one fixed-wavelength parameter.

    `perturbation_matrix_rad_per_m` is Hermitian, in rad/m per `parameter_unit`;
    on a forward basis its diagonal is each mode's dβ/dp. `coefficients` are per
    `parameter_unit`, row the receiving mode and column the differentiated mode;
    entries on the diagonal and within an unresolved group are absent: nan there,
    with `coefficient_resolved` False. `unresolved_groups` lists the modes whose
    propagation constants lie within `gap_rad_per_m` of each other.
    """

    conventions: Mapping[str, str]
    assumption: str
    definition: str
    input_hash: str
    parameter: Parameter
    parameter_unit: str
    gap_rad_per_m: float
    perturbation_matrix_rad_per_m: np.ndarray
    coefficients: np.ndarray
    coefficient_resolved: np.ndarray
    unresolved_groups: tuple[tuple[int, ...], ...]
    quadrature_change_rad_per_m: float
    _kind = "guided_local_coupling"

    def to_dict(self) -> dict[str, Any]:
        m, c, ok = self.perturbation_matrix_rad_per_m, self.coefficients, self.coefficient_resolved
        return {
            "schema": "lamina.guided_local_coupling",
            "schema_version": SCHEMA_VERSION,
            "conventions": dict(self.conventions),
            "assumption": self.assumption,
            "definition": self.definition,
            "input_hash": self.input_hash,
            "parameter": self.parameter.to_dict(),
            "parameter_unit": self.parameter_unit,
            "gap_rad_per_m": self.gap_rad_per_m,
            "perturbation_matrix_real": m.real.tolist(),
            "perturbation_matrix_imag": m.imag.tolist(),
            "coefficients_real": _optional_rows(c.real, ok),
            "coefficients_imag": _optional_rows(c.imag, ok),
            "unresolved_groups": [list(g) for g in self.unresolved_groups],
            "quadrature_change_rad_per_m": self.quadrature_change_rad_per_m,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LocalCoupling:
        re = _optional_matrix(d["coefficients_real"])
        im = _optional_matrix(d["coefficients_imag"])
        resolved = np.array(
            [[v is not None for v in row] for row in d["coefficients_real"]], dtype=bool
        ).reshape(re.shape)
        return cls(
            MappingProxyType(dict(d["conventions"])),
            d["assumption"],
            d["definition"],
            d["input_hash"],
            Parameter.from_dict(d["parameter"]),
            d["parameter_unit"],
            d["gap_rad_per_m"],
            _complex_matrix(d["perturbation_matrix_real"], d["perturbation_matrix_imag"]),
            _readonly(re + 1j * im),
            _readonly(resolved),
            tuple(tuple(g) for g in d["unresolved_groups"]),
            d["quadrature_change_rad_per_m"],
        )


@dataclass(frozen=True, eq=False)
class FieldProjection(_Record):
    """The projection of a supplied field onto a complete basis.

    Solves G c = b, with `gram_w_per_m[i, j]` the power overlap of mode j with
    mode i and `right_hand_side_w_per_m[i]` that of the supplied field with
    mode i over the support. `rank` counts the singular values above
    `rank_rtol` times the largest. Below full rank, `coefficients`,
    `condition_number` and `residual_w_per_m` are None.
    `field_sample_count` is the number of distinct samples of the field.
    """

    conventions: Mapping[str, str]
    assumption: str
    definition: str
    input_hash: str
    field_id: str | None
    support_m: tuple[float, float]
    interfaces_m: tuple[float, ...]
    origin_m: float
    rank_rtol: float
    gram_w_per_m: np.ndarray
    right_hand_side_w_per_m: np.ndarray
    rank: int
    singular_values_w_per_m: np.ndarray
    field_sample_count: int
    quadrature_change_w_per_m: float
    coefficients: np.ndarray | None
    condition_number: float | None
    residual_w_per_m: float | None
    _kind = "guided_projection"

    @property
    def is_full_rank(self) -> bool:
        return self.coefficients is not None

    def to_dict(self) -> dict[str, Any]:
        g, b, c = self.gram_w_per_m, self.right_hand_side_w_per_m, self.coefficients
        return {
            "schema": "lamina.guided_projection",
            "schema_version": SCHEMA_VERSION,
            "conventions": dict(self.conventions),
            "assumption": self.assumption,
            "definition": self.definition,
            "input_hash": self.input_hash,
            "field_id": self.field_id,
            "support_m": list(self.support_m),
            "interfaces_m": list(self.interfaces_m),
            "origin_m": self.origin_m,
            "rank_rtol": self.rank_rtol,
            "gram_real": g.real.tolist(),
            "gram_imag": g.imag.tolist(),
            "right_hand_side_real": b.real.tolist(),
            "right_hand_side_imag": b.imag.tolist(),
            "rank": self.rank,
            "singular_values_w_per_m": self.singular_values_w_per_m.tolist(),
            "field_sample_count": self.field_sample_count,
            "quadrature_change_w_per_m": self.quadrature_change_w_per_m,
            "coefficients_real": None if c is None else c.real.tolist(),
            "coefficients_imag": None if c is None else c.imag.tolist(),
            "condition_number": self.condition_number,
            "residual_w_per_m": self.residual_w_per_m,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FieldProjection:
        c_re, c_im = d["coefficients_real"], d["coefficients_imag"]
        coefficients = None if c_re is None or c_im is None else _complex_vector(c_re, c_im)
        return cls(
            MappingProxyType(dict(d["conventions"])),
            d["assumption"],
            d["definition"],
            d["input_hash"],
            d["field_id"],
            tuple(d["support_m"]),
            tuple(d["interfaces_m"]),
            d["origin_m"],
            d["rank_rtol"],
            _complex_matrix(d["gram_real"], d["gram_imag"]),
            _complex_vector(d["right_hand_side_real"], d["right_hand_side_imag"]),
            d["rank"],
            _readonly(np.asarray(d["singular_values_w_per_m"], dtype=float)),
            d["field_sample_count"],
            d["quadrature_change_w_per_m"],
            coefficients,
            d["condition_number"],
            d["residual_w_per_m"],
        )


def _field_sampler(field: Callable[[float, str], tuple[Any, Any]]) -> Callable[[float, str], list[float]]:
    """Adapt `field(z_m, side) -> (E, H)` to the twelve floats the native module reads."""

    def sample(z_m: float, side: str) -> list[float]:
        result = field(z_m, side)
        try:
            e, h = result
            e = np.asarray(e, dtype=complex)
            h = np.asarray(h, dtype=complex)
        except (TypeError, ValueError) as error:
            raise InvalidInput(
                f"the field must return (E, H), each three complex components: {error}"
            ) from None
        if e.shape != (3,) or h.shape != (3,):
            raise InvalidInput(
                f"the field must return (E, H), each three complex components; got shapes {e.shape} and {h.shape}"
            )
        return [*e.real.tolist(), *e.imag.tolist(), *h.real.tolist(), *h.imag.tolist()]

    return sample


def power_overlap(
    first: GuidedBasis,
    first_mode: int,
    second: GuidedBasis,
    second_mode: int,
    origins_m: tuple[float, float] = (0.0, 0.0),
) -> PowerOverlap:
    """The power overlap of mode `first_mode` of `first` with mode `second_mode`
    of `second`, linear in the first, in W/m.

    The two bases may come from different stacks at one wavelength.
    `origins_m` places each basis's first interface in a common frame; the
    default aligns them. A different wavelength, a non-finite origin or a mode
    index out of range raises `InvalidInput`; field recovery raises
    `FieldsFailed`, and a quadrature over its budget `IntegrationFailed`.
    """
    text = _native.power_overlap(
        first._solver(), int(first_mode), second._solver(), int(second_mode), _origins(origins_m)
    )
    return PowerOverlap.from_dict(json.loads(text))


def power_overlap_sensitivity(
    first: GuidedBasis,
    first_mode: int,
    second: GuidedBasis,
    second_mode: int,
    first_parameter: Parameter | None,
    second_parameter: Parameter | None,
    origins_m: tuple[float, float] = (0.0, 0.0),
) -> PowerOverlapSensitivity:
    """The power overlap of two modes and its derivative when the first stack
    varies along `first_parameter` and the second along `second_parameter`
    (None holds that stack fixed), at one shared rate.

    At least one parameter must be set; two set parameters must share their
    unit; the wavelength varies on both sides or neither. Otherwise
    `InvalidInput`. Each stack's first interface moves with its parameter, so
    `(None, Parameter.translation())` is a relative translation and
    `(Parameter.translation(), Parameter.translation())` a common one. A
    derivative that cannot be formed (for example where coincident interfaces
    of the two stacks separate) is reported in `derivative_failure`, with the
    base overlap present.
    """

    def parameter(p: Parameter | None) -> str | None:
        return None if p is None else _dumps(p.to_dict())

    text = _native.power_overlap_sensitivity(
        first._solver(),
        int(first_mode),
        second._solver(),
        int(second_mode),
        parameter(first_parameter),
        parameter(second_parameter),
        _origins(origins_m),
    )
    return PowerOverlapSensitivity.from_dict(json.loads(text))


def reciprocity(
    first: GuidedBasis,
    first_mode: int,
    second: GuidedBasis,
    second_mode: int,
    origins_m: tuple[float, float] = (0.0, 0.0),
) -> Reciprocity:
    """The unconjugated, antisymmetric flux ∫ (E1 × H2 − E2 × H1) · x dz of two
    unit-power modes, in W/m, registered and checked as for `power_overlap`."""
    text = _native.reciprocity(
        first._solver(), int(first_mode), second._solver(), int(second_mode), _origins(origins_m)
    )
    return Reciprocity.from_dict(json.loads(text))


# ---------------------------------------------------------------- entry point


def enumerate_guided_modes(stack: Stack, request: GuidedRequest) -> GuidedBasis:
    """Enumerate the guided modes of `stack` for `request`.

    Partial coverage is a result, not an error: read `completeness`,
    `expected`, `found` and `unresolved` together. Invalid inputs raise
    `InvalidInput`, and a stack closed by a `Pml` raises its subclass
    `UnsupportedTermination`; a failed enumeration raises `EnumerationFailed`.
    """
    resolved = GuidedInput.resolve(stack, request)
    native = _native.enumerate_guided_modes(resolved.to_json())
    basis = GuidedBasis.from_dict(json.loads(native.record_json()))
    object.__setattr__(basis, "_native", native)
    return basis


# ---------------------------------------------------------------- modal bases

DEFINITIONS: Mapping[str, str] = MappingProxyType(json.loads(_native.definitions()))
"""The pairing, normalisation, sign, ordering and near-degenerate definitions
every modal basis, profile and basis pairing carries."""

_Rect = tuple[tuple[float, float], tuple[float, float]]


def _rect(r: Sequence[Sequence[float]]) -> _Rect:
    return ((float(r[0][0]), float(r[0][1])), (float(r[1][0]), float(r[1][1])))


def _bounds(pair: Sequence[float]) -> tuple[float, float]:
    low, high = pair
    return (float(low), float(high))


@dataclass(frozen=True)
class BetaWindow:
    """A rectangle of the complex β plane in rad/m, for stacks closed on both
    sides: Re β in `re_rad_per_m`, Im β in `im_rad_per_m`, each ascending."""

    re_rad_per_m: tuple[float, float]
    im_rad_per_m: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "re_rad_per_m", _bounds(self.re_rad_per_m))
        object.__setattr__(self, "im_rad_per_m", _bounds(self.im_rad_per_m))

    @classmethod
    def forward(
        cls, beta_max_rad_per_m: float, decay_max_rad_per_m: float, margin_rad_per_m: float
    ) -> BetaWindow:
        """Re β in [−margin, beta_max], Im β in [−margin, decay_max]: real and
        imaginary roots of a lossless stack lie strictly inside, and the window
        meets its mirror image only in the square |Re β|, |Im β| ≤ margin."""
        return cls((-margin_rad_per_m, beta_max_rad_per_m), (-margin_rad_per_m, decay_max_rad_per_m))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "wavenumber",
            "re_rad_per_m": list(self.re_rad_per_m),
            "im_rad_per_m": list(self.im_rad_per_m),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BetaWindow:
        if d.get("kind") != "wavenumber":
            raise InvalidInput(f"not a wavenumber window: {d.get('kind')!r}")
        return cls(tuple(d["re_rad_per_m"]), tuple(d["im_rad_per_m"]))


@dataclass(frozen=True)
class ChartWindow:
    """A rectangle in the dimensionless conformal coordinate z, for stacks
    with an open side: Re z in `re`, Im z in `im`, each ascending, less than
    one period, 2π, tall. With one open side β² = k0² (χ + e^{2z}) and
    κ = i k0 e^z; with two, β² = k0² (s + e^{2z} + r² e^{−2z}),
    κ_L,R = i k0 (e^z ∓ r e^{−z}), r = (χ_L − χ_R)/4, s = (χ_L + χ_R)/2.
    `chart_window_covering` builds one from a β rectangle on stated sheets."""

    re: tuple[float, float]
    im: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "re", _bounds(self.re))
        object.__setattr__(self, "im", _bounds(self.im))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "chart", "re": list(self.re), "im": list(self.im)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ChartWindow:
        if d.get("kind") != "chart":
            raise InvalidInput(f"not a chart window: {d.get('kind')!r}")
        return cls(tuple(d["re"]), tuple(d["im"]))


def _window(d: Mapping[str, Any]) -> BetaWindow | ChartWindow:
    return ChartWindow.from_dict(d) if d.get("kind") == "chart" else BetaWindow.from_dict(d)


@dataclass(frozen=True, eq=False)
class ModalBasisInput(_Record):
    """The canonical input of a modal basis, with the near-degeneracy gap resolved."""

    stack: Stack
    wavelength_m: float
    polarisation: Literal["TE", "TM"]
    window: BetaWindow | ChartWindow
    degeneracy_gap_rad_per_m: float | None = None
    self_pairing_rtol: float = 1e-8
    _kind = "modal_basis_input"

    def resolve(self) -> ModalBasisInput:
        """Validate, resolving the default gap (1e-6 k0)."""
        canonical = _native.normalise_modal_basis_input(_dumps(self.to_dict()))
        return ModalBasisInput.from_dict(json.loads(canonical))

    def to_dict(self) -> dict[str, Any]:
        gap = self.degeneracy_gap_rad_per_m
        return {
            "schema": "lamina.modal_basis_input",
            "schema_version": SCHEMA_VERSION,
            "stack": self.stack.to_dict(),
            "wavelength_m": float(self.wavelength_m),
            "polarisation": self.polarisation,
            "window": self.window.to_dict(),
            "degeneracy_gap_rad_per_m": None if gap is None else float(gap),
            "self_pairing_rtol": float(self.self_pairing_rtol),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ModalBasisInput:
        return cls(
            Stack.from_dict(d["stack"]),
            d["wavelength_m"],
            d["polarisation"],
            _window(d["window"]),
            d["degeneracy_gap_rad_per_m"],
            d["self_pairing_rtol"],
        )


@dataclass(frozen=True)
class UnresolvedRegion:
    """A region of the window that did not end as accepted modes, and why:
    "boundary_root", "non_integer_count", "budget", "refinement",
    "multiplicity", "cutoff", "scale" (a PML scale out of range) or
    "branch_point" (a root whose exterior wavevector on an open side is
    within 1e-4 k0 of zero, where its self-pairing tail diverges). `bounds`
    is in the chart's coordinate (rad/m on the wavenumber chart,
    dimensionless z on a conformal chart); `count` is the number of zeros
    counted inside."""

    bounds: _Rect
    beta_image_rad_per_m: _Rect
    count: int | None
    cause: str
    detail: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounds": [list(b) for b in self.bounds],
            "beta_image_rad_per_m": [list(b) for b in self.beta_image_rad_per_m],
            "count": self.count,
            "cause": self.cause,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> UnresolvedRegion:
        return cls(_rect(d["bounds"]), _rect(d["beta_image_rad_per_m"]), d["count"], d["cause"], d["detail"])


@dataclass(frozen=True)
class BasisMode:
    """One mode of a modal basis. `normalisation` is "unit" (⟨f, f⟩ = 1 W/m)
    or "ill_conditioned" (|⟨u, u⟩| below the threshold of the field scale: no
    normalised fields). `sign_margin` is |Re ℓ| / max |Ey + Z0 Hy|. `sheet`
    gives each open side's sheet, "proper" (Im κ > 0, decaying) or
    "improper" (growing), None for a closed side; `mode_type` is "closed"
    without open sides, else "guided" or "leaky"."""

    index: int
    beta_rad_per_m: complex
    n_eff: complex
    mode_type: str
    direction: Literal["forward", "backward"]
    sheet: tuple[str | None, str | None]
    residual: float
    self_pairing_raw_w_per_m: complex
    normalisation: Literal["unit", "ill_conditioned"]
    self_pairing_ratio: float
    sign_reference: Literal["first_interface", "wall_on_first_interface"]
    sign_margin: float | None
    sign_ambiguous: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "beta_rad_per_m": _pair(self.beta_rad_per_m),
            "n_eff": _pair(self.n_eff),
            "sheet": list(self.sheet),
            "self_pairing_raw_w_per_m": _pair(self.self_pairing_raw_w_per_m),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BasisMode:
        return cls(**{
            **d,
            "beta_rad_per_m": _c(d["beta_rad_per_m"]),
            "n_eff": _c(d["n_eff"]),
            "sheet": tuple(d["sheet"]),
            "self_pairing_raw_w_per_m": _c(d["self_pairing_raw_w_per_m"]),
        })


@dataclass(frozen=True, eq=False)
class ModalBasis(_Record):
    """Every mode of a stack in a window, with its evidence: a β window for a
    stack closed by walls or PMLs ("wavenumber" chart, scope "discrete"), a
    chart window for one with an open side ("one_exterior" or
    "two_exterior" chart, scope "discrete_without_continuum": the guided and
    leaky modes in the window, never the radiation continuum, so they do not
    span the stack's fields).

    `expected` is the count from the argument principle over the window
    boundary and `found` the distinct accepted modes. `status` is "complete"
    when they agree, nothing is unresolved and every mode is normalised;
    "count_failed" when the boundary count failed; else "incomplete". Modes
    are in the order of `definitions["ordering"]`; `groups` lists the
    near-degenerate groups, whose members form an unconjugated Löwdin basis.
    """

    conventions: Mapping[str, str]
    definitions: Mapping[str, str]
    assumption: str
    input: ModalBasisInput
    input_hash: str
    k0_rad_per_m: float
    chart: str
    window_beta_image_rad_per_m: _Rect
    scope: str
    status: Literal["complete", "incomplete", "count_failed"]
    expected: int | None
    found: int
    unresolved: tuple[UnresolvedRegion, ...]
    groups: tuple[tuple[int, ...], ...]
    modes: tuple[BasisMode, ...]
    _kind = "modal_basis"

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    @property
    def beta_rad_per_m(self) -> np.ndarray:
        return _readonly(np.array([m.beta_rad_per_m for m in self.modes], dtype=complex))

    def __len__(self) -> int:
        return len(self.modes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "lamina.modal_basis",
            "schema_version": SCHEMA_VERSION,
            "conventions": dict(self.conventions),
            "definitions": dict(self.definitions),
            "assumption": self.assumption,
            "input": self.input.to_dict(),
            "input_hash": self.input_hash,
            "k0_rad_per_m": self.k0_rad_per_m,
            "chart": self.chart,
            "window_beta_image_rad_per_m": [list(b) for b in self.window_beta_image_rad_per_m],
            "scope": self.scope,
            "status": self.status,
            "expected": self.expected,
            "found": self.found,
            "unresolved": [r.to_dict() for r in self.unresolved],
            "groups": [list(g) for g in self.groups],
            "modes": [m.to_dict() for m in self.modes],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ModalBasis:
        return cls(
            MappingProxyType(dict(d["conventions"])),
            MappingProxyType(dict(d["definitions"])),
            d["assumption"],
            ModalBasisInput.from_dict(d["input"]),
            d["input_hash"],
            d["k0_rad_per_m"],
            d["chart"],
            _rect(d["window_beta_image_rad_per_m"]),
            d["scope"],
            d["status"],
            d["expected"],
            d["found"],
            tuple(UnresolvedRegion.from_dict(r) for r in d["unresolved"]),
            tuple(tuple(g) for g in d["groups"]),
            tuple(BasisMode.from_dict(m) for m in d["modes"]),
        )

    def _solver(self) -> Any:
        native = self.__dict__.get("_native")
        if native is None:
            # A deserialised basis: re-solve its input and insist on the same result.
            native = _native.modal_basis(self.input.to_json())
            if native.record_json() != self.to_json():
                raise LaminaError("re-solving the stored input did not reproduce this basis")
            object.__setattr__(self, "_native", native)
        return native

    def profile(self, mode: int, z_m: Sequence[float] | np.ndarray) -> ModalProfile:
        """Normalised E and H of one mode at positions z (m), with each sample's
        region. A position beyond a wall, or an ill-conditioned mode, raises
        `InvalidInput`."""
        z = np.ascontiguousarray(z_m, dtype=float).ravel()
        return ModalProfile.from_dict(json.loads(self._solver().profile(int(mode), z.tolist())))

    def project_field(
        self,
        field: Callable[[float, str], tuple[Any, Any]],
        support_m: tuple[float, float],
        interfaces_m: Sequence[float] = (),
        origin_m: float = 0.0,
        rank_rtol: float = 1e-8,
        field_id: str | None = None,
        exterior: tuple[Sequence[ExteriorWave], Sequence[ExteriorWave]] = ((), ()),
    ) -> BasisProjection:
        """Expand a supplied field F in this basis's normalised modes f_m.

        Solves G a = b with G[m, n] = ⟨f_m, f_n⟩ and b[m] = ⟨f_m, F⟩, both under
        the unconjugated pairing of `definitions`, on the basis's partition with
        the support ends and `interfaces_m` added: along the stretched
        coordinate in a PML and with continued tails on open sides.

        `field(z_m, side)` returns `(E, H)` as for `GuidedBasis.project_field`.
        Inside a PML it is the exterior solution continued to the stretched
        coordinate, as `profile` gives it; a field of the equivalent medium in
        real z has its normal components divided by the stretch first. It
        vanishes outside `support_m`, which must stay within the walls. On an
        open side whose support edge lies at or beyond the outermost interface,
        `exterior` may state the field beyond the edge as `ExteriorWave`s,
        (left, right); a field built from leaky modes needs them.

        The record holds the Gram matrix, its singular values, condition number
        and defect, and at full rank the coefficients, the equation residual
        and the reconstruction residual: the field scale of F − Σ a_m f_m over
        the bounded partition. `residual_includes_continuum` is True for a
        basis of a stack with an open side: its modes never span the fields,
        so the residual then includes F's radiation content and is physical,
        not numerical. An incomplete basis is accepted; read `status`.
        """
        lo, hi = support_m
        left, right = exterior
        text = self._solver().project_field(
            _field_sampler(field),
            [float(lo), float(hi)],
            [float(z) for z in interfaces_m],
            float(origin_m),
            float(rank_rtol),
            _dumps([[w.to_dict() for w in left], [w.to_dict() for w in right]]),
            None if field_id is None else str(field_id),
        )
        return BasisProjection.from_dict(json.loads(text))

    def local_coupling(self, parameter: Parameter, gap_rad_per_m: float | None = None) -> BasisLocalCoupling:
        """The perturbation matrix and coupling coefficients of this basis's
        normalised modes along a fixed-wavelength `parameter`.

        `perturbation_matrix_rad_per_m` is complex symmetric, its diagonal each
        mode's dβ/dp; `coefficients[m, n]` = K[m, n] / (β_n − β_m) is the rate of
        the pairing of mode m with the moving mode n. Modes whose propagation
        constants differ by at most `gap_rad_per_m` (default: the basis's
        degeneracy gap) form `groups` whose mutual coefficients, like the
        diagonal, are absent. A wavelength or permeability parameter, or one
        moving an interface against a wall or PML, raises `InvalidInput`.
        """
        gap = None if gap_rad_per_m is None else float(gap_rad_per_m)
        text = self._solver().local_coupling(_dumps(parameter.to_dict()), gap)
        return BasisLocalCoupling.from_dict(json.loads(text))


@dataclass(frozen=True)
class ExteriorWave:
    """One exterior wave of a supplied field beyond its support on an open
    side: E and H at the support edge (three complex components each, V/m and
    A/m) times exp(iκζ) at outward distance ζ, κ in rad/m on either sheet."""

    kappa_rad_per_m: complex
    electric_v_per_m: tuple[complex, complex, complex]
    magnetic_a_per_m: tuple[complex, complex, complex]

    def to_dict(self) -> dict[str, Any]:
        e = np.asarray(self.electric_v_per_m, dtype=complex)
        h = np.asarray(self.magnetic_a_per_m, dtype=complex)
        if e.shape != (3,) or h.shape != (3,):
            raise InvalidInput("an ExteriorWave's E and H each have three complex components")
        return {
            "kappa_rad_per_m": _pair(complex(self.kappa_rad_per_m)),
            "electric_real": e.real.tolist(),
            "electric_imag": e.imag.tolist(),
            "magnetic_real": h.real.tolist(),
            "magnetic_imag": h.imag.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ExteriorWave:
        e = tuple(complex(r, i) for r, i in zip(d["electric_real"], d["electric_imag"]))
        h = tuple(complex(r, i) for r, i in zip(d["magnetic_real"], d["magnetic_imag"]))
        return cls(_c(d["kappa_rad_per_m"]), e, h)  # type: ignore[arg-type]


@dataclass(frozen=True, eq=False)
class ModalProfile(_Record):
    """Normalised fields of one basis mode; arrays have shape (n, 3): (x, y, z).
    `region` names each sample's region: "layer", "gap", "pml" or "exterior".

    For a stack with a PML side, `stretched_z_m` and `stretch` give each
    sample's stretched coordinate z̃ and stretch s (nan outside a PML), and
    `pml_fields` states what the fields are there: the exterior solution
    continued to z̃; the equivalent medium in real z has the same tangential
    components and normal components s E_z and s H_z. All three are None for
    a stack without a PML."""

    conventions: Mapping[str, str]
    definitions: Mapping[str, str]
    input_hash: str
    mode_index: int
    beta_rad_per_m: complex
    z_m: np.ndarray
    region: tuple[str, ...]
    electric_v_per_m: np.ndarray
    magnetic_a_per_m: np.ndarray
    stretched_z_m: np.ndarray | None = None
    stretch: np.ndarray | None = None
    pml_fields: str | None = None
    _kind = "modal_profile"

    def to_dict(self) -> dict[str, Any]:
        e, h = self.electric_v_per_m, self.magnetic_a_per_m
        return {
            "schema": "lamina.modal_profile",
            "schema_version": SCHEMA_VERSION,
            "conventions": dict(self.conventions),
            "definitions": dict(self.definitions),
            "input_hash": self.input_hash,
            "mode_index": self.mode_index,
            "beta_rad_per_m": _pair(self.beta_rad_per_m),
            "z_m": self.z_m.tolist(),
            "region": list(self.region),
            "electric_real": e.real.tolist(),
            "electric_imag": e.imag.tolist(),
            "magnetic_real": h.real.tolist(),
            "magnetic_imag": h.imag.tolist(),
            **(
                {}
                if self.pml_fields is None
                else {
                    "pml": {
                        "fields": self.pml_fields,
                        "stretched_z_m": _pml_samples(self.stretched_z_m),
                        "stretch": _pml_samples(self.stretch),
                    }
                }
            ),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ModalProfile:
        pml = d.get("pml")
        return cls(
            MappingProxyType(dict(d["conventions"])),
            MappingProxyType(dict(d["definitions"])),
            d["input_hash"],
            d["mode_index"],
            _c(d["beta_rad_per_m"]),
            _readonly(np.asarray(d["z_m"], dtype=float)),
            tuple(d["region"]),
            _components(d["electric_real"], d["electric_imag"]),
            _components(d["magnetic_real"], d["magnetic_imag"]),
            None if pml is None else _pml_array(pml["stretched_z_m"]),
            None if pml is None else _pml_array(pml["stretch"]),
            None if pml is None else pml["fields"],
        )


def _pml_array(values: Sequence[Sequence[float] | None]) -> np.ndarray:
    """Complex samples with nan where the record has null."""
    nan = complex(math.nan, math.nan)
    return _readonly(np.array([nan if v is None else _c(v) for v in values], dtype=complex))


def _pml_samples(values: np.ndarray | None) -> list[list[float] | None]:
    return [None if np.isnan(v) else _pair(v) for v in (values if values is not None else [])]


@dataclass(frozen=True, eq=False)
class BasisPairing(_Record):
    """Both pairing matrices between the normalised modes of two bases, in W/m:
    row i is mode `rows[i]` of the first basis, column j mode `columns[j]` of
    the second. Absent conjugated entries are nan, with their cause in
    `conjugated_absent`. `leading_condition_numbers[k]` is the condition number
    of the unconjugated matrix's leading (k + 1) × (k + 1) block and
    `self_defects` each basis's max |G − I| on the same grid."""

    conventions: Mapping[str, str]
    definitions: Mapping[str, str]
    first_input_hash: str
    second_input_hash: str
    origins_m: tuple[float, float]
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    unconjugated_w_per_m: np.ndarray
    conjugated_w_per_m: np.ndarray
    conjugated_absent: tuple[tuple[str | None, ...], ...]
    singular_values_w_per_m: np.ndarray
    leading_condition_numbers: np.ndarray
    self_defects: tuple[float, float]
    sample_count: int
    quadrature_change_w_per_m: float
    status: tuple[str, str]
    scope: tuple[str, str]
    _kind = "basis_pairing"

    def to_dict(self) -> dict[str, Any]:
        u, c = self.unconjugated_w_per_m, self.conjugated_w_per_m
        present = np.array([[a is None for a in row] for row in self.conjugated_absent], dtype=bool)
        present = present.reshape(c.shape)
        return {
            "schema": "lamina.basis_pairing",
            "schema_version": SCHEMA_VERSION,
            "conventions": dict(self.conventions),
            "definitions": dict(self.definitions),
            "first_input_hash": self.first_input_hash,
            "second_input_hash": self.second_input_hash,
            "origins_m": list(self.origins_m),
            "rows": list(self.rows),
            "columns": list(self.columns),
            "unconjugated_real": u.real.tolist(),
            "unconjugated_imag": u.imag.tolist(),
            "conjugated_real": _optional_rows(c.real, present),
            "conjugated_imag": _optional_rows(c.imag, present),
            "conjugated_absent": [list(row) for row in self.conjugated_absent],
            "singular_values_w_per_m": self.singular_values_w_per_m.tolist(),
            "leading_condition_numbers": self.leading_condition_numbers.tolist(),
            "self_defects": list(self.self_defects),
            "sample_count": self.sample_count,
            "quadrature_change_w_per_m": self.quadrature_change_w_per_m,
            "status": list(self.status),
            "scope": list(self.scope),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BasisPairing:
        shape = (len(d["rows"]), len(d["columns"]))
        conjugated = _optional_matrix(d["conjugated_real"]).reshape(shape) + 1j * _optional_matrix(
            d["conjugated_imag"]
        ).reshape(shape)
        return cls(
            MappingProxyType(dict(d["conventions"])),
            MappingProxyType(dict(d["definitions"])),
            d["first_input_hash"],
            d["second_input_hash"],
            tuple(d["origins_m"]),
            tuple(d["rows"]),
            tuple(d["columns"]),
            _readonly(_complex_matrix(d["unconjugated_real"], d["unconjugated_imag"]).reshape(shape)),
            _readonly(conjugated),
            tuple(tuple(row) for row in d["conjugated_absent"]),
            _readonly(np.asarray(d["singular_values_w_per_m"], dtype=float)),
            _readonly(np.asarray(d["leading_condition_numbers"], dtype=float)),
            tuple(d["self_defects"]),
            d["sample_count"],
            d["quadrature_change_w_per_m"],
            tuple(d["status"]),
            tuple(d["scope"]),
        )


@dataclass(frozen=True, eq=False)
class BasisProjection(_Record):
    """A supplied field expanded in the normalised modes of a modal basis.

    `modes` are the basis indices of the rows and columns of `gram_w_per_m`
    (⟨f_m, f_n⟩) and of `right_hand_side_w_per_m` (⟨f_m, F⟩). `rank` counts
    the singular values above `rank_rtol` times the largest; below full rank
    `coefficients`, `condition_number`, `equation_residual_w_per_m`,
    `residual_scale_w_per_m` and `relative_residual` are None. The field and
    residual scales are (1/4Z0) ∫ (|E|² + |Z0 H|²) dz over the bounded
    partition; `relative_residual` is the square root of their ratio (None
    for a zero field). `residual_includes_continuum` states that the basis
    has no radiation continuum, so the residual holds F's radiation content.
    """

    conventions: Mapping[str, str]
    definitions: Mapping[str, str]
    definition: str
    assumption: str
    input_hash: str
    field_id: str | None
    support_m: tuple[float, float]
    interfaces_m: tuple[float, ...]
    origin_m: float
    exterior: tuple[tuple[ExteriorWave, ...], tuple[ExteriorWave, ...]]
    rank_rtol: float
    modes: tuple[int, ...]
    gram_w_per_m: np.ndarray
    right_hand_side_w_per_m: np.ndarray
    rank: int
    singular_values_w_per_m: np.ndarray
    gram_defect: float
    partition_m: np.ndarray
    field_sample_count: int
    sample_count: int
    quadrature_change_w_per_m: float
    coefficients: np.ndarray | None
    condition_number: float | None
    equation_residual_w_per_m: float | None
    field_scale_w_per_m: float
    residual_scale_w_per_m: float | None
    relative_residual: float | None
    residual_includes_continuum: bool
    status: str
    scope: str
    _kind = "basis_projection"

    @property
    def is_full_rank(self) -> bool:
        return self.coefficients is not None

    def to_dict(self) -> dict[str, Any]:
        g, b, c = self.gram_w_per_m, self.right_hand_side_w_per_m, self.coefficients
        return {
            "schema": "lamina.basis_projection",
            "schema_version": SCHEMA_VERSION,
            "conventions": dict(self.conventions),
            "definitions": dict(self.definitions),
            "definition": self.definition,
            "assumption": self.assumption,
            "input_hash": self.input_hash,
            "field_id": self.field_id,
            "support_m": list(self.support_m),
            "interfaces_m": list(self.interfaces_m),
            "origin_m": self.origin_m,
            "exterior": [[w.to_dict() for w in side] for side in self.exterior],
            "rank_rtol": self.rank_rtol,
            "modes": list(self.modes),
            "gram_real": g.real.tolist(),
            "gram_imag": g.imag.tolist(),
            "right_hand_side_real": b.real.tolist(),
            "right_hand_side_imag": b.imag.tolist(),
            "rank": self.rank,
            "singular_values_w_per_m": self.singular_values_w_per_m.tolist(),
            "gram_defect": self.gram_defect,
            "partition_m": self.partition_m.tolist(),
            "field_sample_count": self.field_sample_count,
            "sample_count": self.sample_count,
            "quadrature_change_w_per_m": self.quadrature_change_w_per_m,
            "coefficients_real": None if c is None else c.real.tolist(),
            "coefficients_imag": None if c is None else c.imag.tolist(),
            "condition_number": self.condition_number,
            "equation_residual_w_per_m": self.equation_residual_w_per_m,
            "field_scale_w_per_m": self.field_scale_w_per_m,
            "residual_scale_w_per_m": self.residual_scale_w_per_m,
            "relative_residual": self.relative_residual,
            "residual_includes_continuum": self.residual_includes_continuum,
            "status": self.status,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BasisProjection:
        c_re, c_im = d["coefficients_real"], d["coefficients_imag"]
        coefficients = None if c_re is None or c_im is None else _complex_vector(c_re, c_im)
        left, right = d["exterior"]
        return cls(
            MappingProxyType(dict(d["conventions"])),
            MappingProxyType(dict(d["definitions"])),
            d["definition"],
            d["assumption"],
            d["input_hash"],
            d["field_id"],
            tuple(d["support_m"]),
            tuple(d["interfaces_m"]),
            d["origin_m"],
            (
                tuple(ExteriorWave.from_dict(w) for w in left),
                tuple(ExteriorWave.from_dict(w) for w in right),
            ),
            d["rank_rtol"],
            tuple(d["modes"]),
            _complex_matrix(d["gram_real"], d["gram_imag"]),
            _complex_vector(d["right_hand_side_real"], d["right_hand_side_imag"]),
            d["rank"],
            _readonly(np.asarray(d["singular_values_w_per_m"], dtype=float)),
            d["gram_defect"],
            _readonly(np.asarray(d["partition_m"], dtype=float)),
            d["field_sample_count"],
            d["sample_count"],
            d["quadrature_change_w_per_m"],
            coefficients,
            d["condition_number"],
            d["equation_residual_w_per_m"],
            d["field_scale_w_per_m"],
            d["residual_scale_w_per_m"],
            d["relative_residual"],
            d["residual_includes_continuum"],
            d["status"],
            d["scope"],
        )


@dataclass(frozen=True, eq=False)
class BasisLocalCoupling(_Record):
    """The local coupling of a modal basis along one fixed-wavelength parameter.

    Rows and columns are the basis modes `modes`. `perturbation_matrix_rad_per_m`
    is complex symmetric, in rad/m per `parameter_unit`, its diagonal each mode's
    dβ/dp. `coefficients` are per `parameter_unit`, row the receiving mode and
    column the differentiated mode; entries on the diagonal and within a group
    are absent: nan there, with `coefficient_resolved` False. `groups` lists
    the modes, by basis index, whose propagation constants lie within
    `gap_rad_per_m` of each other.
    """

    conventions: Mapping[str, str]
    definitions: Mapping[str, str]
    definition: str
    assumption: str
    input_hash: str
    parameter: Parameter
    parameter_unit: str
    gap_rad_per_m: float
    modes: tuple[int, ...]
    perturbation_matrix_rad_per_m: np.ndarray
    coefficients: np.ndarray
    coefficient_resolved: np.ndarray
    groups: tuple[tuple[int, ...], ...]
    partition_m: np.ndarray
    sample_count: int
    quadrature_change_rad_per_m: float
    status: str
    scope: str
    _kind = "basis_local_coupling"

    def to_dict(self) -> dict[str, Any]:
        m, c, ok = self.perturbation_matrix_rad_per_m, self.coefficients, self.coefficient_resolved
        return {
            "schema": "lamina.basis_local_coupling",
            "schema_version": SCHEMA_VERSION,
            "conventions": dict(self.conventions),
            "definitions": dict(self.definitions),
            "definition": self.definition,
            "assumption": self.assumption,
            "input_hash": self.input_hash,
            "parameter": self.parameter.to_dict(),
            "parameter_unit": self.parameter_unit,
            "gap_rad_per_m": self.gap_rad_per_m,
            "modes": list(self.modes),
            "perturbation_matrix_real": m.real.tolist(),
            "perturbation_matrix_imag": m.imag.tolist(),
            "coefficients_real": _optional_rows(c.real, ok),
            "coefficients_imag": _optional_rows(c.imag, ok),
            "groups": [list(g) for g in self.groups],
            "partition_m": self.partition_m.tolist(),
            "sample_count": self.sample_count,
            "quadrature_change_rad_per_m": self.quadrature_change_rad_per_m,
            "status": self.status,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BasisLocalCoupling:
        n = len(d["modes"])
        re = _optional_matrix(d["coefficients_real"]).reshape(n, n)
        im = _optional_matrix(d["coefficients_imag"]).reshape(n, n)
        resolved = np.array(
            [[v is not None for v in row] for row in d["coefficients_real"]], dtype=bool
        ).reshape(n, n)
        return cls(
            MappingProxyType(dict(d["conventions"])),
            MappingProxyType(dict(d["definitions"])),
            d["definition"],
            d["assumption"],
            d["input_hash"],
            Parameter.from_dict(d["parameter"]),
            d["parameter_unit"],
            d["gap_rad_per_m"],
            tuple(d["modes"]),
            _readonly(
                _complex_matrix(d["perturbation_matrix_real"], d["perturbation_matrix_imag"]).reshape(n, n)
            ),
            _readonly(re + 1j * im),
            _readonly(resolved),
            tuple(tuple(g) for g in d["groups"]),
            _readonly(np.asarray(d["partition_m"], dtype=float)),
            d["sample_count"],
            d["quadrature_change_rad_per_m"],
            d["status"],
            d["scope"],
        )


def modal_basis(
    stack: Stack,
    wavelength_m: float,
    polarisation: Literal["TE", "TM"],
    window: BetaWindow | ChartWindow,
    *,
    degeneracy_gap_rad_per_m: float | None = None,
    self_pairing_rtol: float = 1e-8,
) -> ModalBasis:
    """Every mode of `stack` in `window`: a `BetaWindow` for a stack closed by
    walls or PMLs on both sides, a `ChartWindow` for one with an open side.

    An incomplete basis is a result, not an error: read `status`, `expected`,
    `found` and `unresolved` together; a cell whose PML scale leaves the
    representable range is unresolved with cause "scale". Invalid inputs,
    including a `BetaWindow` on a stack with an open side, a `ChartWindow`
    taller than one period, and a window on which a PML's scale leaves the
    representable range (naming the side and the log-magnitude), raise
    `InvalidInput`; a self-pairing quadrature over its budget or a singular
    open tail raises `IntegrationFailed` and field recovery `FieldsFailed`.
    """
    resolved = ModalBasisInput(
        stack, wavelength_m, polarisation, window, degeneracy_gap_rad_per_m, self_pairing_rtol
    ).resolve()
    native = _native.modal_basis(resolved.to_json())
    basis = ModalBasis.from_dict(json.loads(native.record_json()))
    object.__setattr__(basis, "_native", native)
    return basis


def chart_window_covering(
    stack: Stack,
    wavelength_m: float,
    beta_rect_rad_per_m: tuple[tuple[float, float], tuple[float, float]],
    sheets: tuple[Literal["proper", "improper"] | None, Literal["proper", "improper"] | None],
) -> tuple[ChartWindow, float]:
    """The smallest chart window whose image covers the β rectangle
    `((re_low, re_high), (im_low, im_high))` (rad/m) on `sheets`, one per side
    (left, right), None for a closed side, and the extra area of its β image
    (rad²/m²). A rectangle meeting a stated sheet's cut, where a leaky mode
    would be reached across a branch cut, raises `InvalidInput` naming the
    side. The basis of the window holds every mode in it, each with its sheet.
    """
    (re, im), extra = _native.chart_window_covering(
        _dumps(stack.to_dict()),
        float(wavelength_m),
        [list(_bounds(beta_rect_rad_per_m[0])), list(_bounds(beta_rect_rad_per_m[1]))],
        list(sheets),
    )
    return ChartWindow(tuple(re), tuple(im)), float(extra)


def basis_pairing(
    first: ModalBasis, second: ModalBasis, origins_m: tuple[float, float] = (0.0, 0.0)
) -> BasisPairing:
    """Both pairing matrices between two modal bases at one wavelength.

    `origins_m` places each stack's first interface in a common frame; the
    walls must then coincide in kind and position, and PMLs in start,
    thickness, profile and backing, and an open side must be open in both
    (materials may differ). Beyond the stacks an open side contributes a
    closed-form tail: unconjugated by analytic continuation for guided and
    leaky modes alike; a conjugated entry whose tail diverges is nan with
    cause "unbounded" in `conjugated_absent`.
    Otherwise, or at different wavelengths, `InvalidInput`; a quadrature over
    its budget raises `IntegrationFailed` and field recovery `FieldsFailed`.
    """
    text = _native.basis_pairing(first._solver(), second._solver(), _origins(origins_m))
    return BasisPairing.from_dict(json.loads(text))

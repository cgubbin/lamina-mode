//! Physical local characteristic derivatives at accepted modal points.
//!
//! A source search assumption does not implicitly authorize a physical
//! derivative request; callers must supply the additional local declaration:
//!
//! ```compile_fail
//! use lamina_core::Scatter2;
//! use lamina_mode::{LocalDerivativeSettings, ModePoint, PhysicalParameter};
//! fn missing_local_declaration(point: &ModePoint) {
//!     let _ = point.local_derivatives_with(
//!         Scatter2::new(),
//!         &[PhysicalParameter::K0],
//!         LocalDerivativeSettings::default(),
//!     );
//! }
//! ```
use super::*;
use lamina_core::{
    AnalyticalMaterialHandle, CanonicalCoordinates, DerivativeOrder,
    EvaluateDifferentiableMeromorphicMaterial, EvaluateMeromorphicMaterial, ExteriorWavevectors,
    FiniteLayerIndex, ModalDerivativeBackend, ModeJet1, Scalar, ScalarAlgebra, SeedJet, Stack,
    ValidationConfig, compile_canonical_constant_stack, compile_canonical_thickness_stack,
};
use lamina_units::LengthUnit;
use ndarray::{Ix0, arr0};
use std::sync::Arc;

/// Caller declaration covering local complex-coordinate and selected geometry
/// variations. A search-time assumption alone does not authorize these
/// additional physical directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalAnalyticityAssumption {
    /// The caller asserts a holomorphic, pole-free neighbourhood in k0 and beta
    /// and regularity under the selected real finite-layer thickness changes.
    CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
}

/// A physical parameter whose characteristic partial may be requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicalParameter {
    /// Vacuum angular wavenumber k0 in cm^-1.
    K0,
    /// In-plane angular wavenumber beta in cm^-1.
    Beta,
    /// One finite-layer thickness in its caller-stored length unit.
    LayerThickness(FiniteLayerIndex),
}

/// Validated controls for branch cutoffs and implicit-slope denominators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalDerivativeSettings {
    /// Minimum accepted exterior |kappa| in cm^-1 for coordinate derivatives.
    pub exterior_kappa_cutoff: f64,
    /// Relative tolerance for checking stored exterior roots against their
    /// source constitutive relation.
    pub branch_relation_rtol: f64,
}
impl Default for LocalDerivativeSettings {
    fn default() -> Self {
        Self {
            exterior_kappa_cutoff: 1e-10,
            branch_relation_rtol: 1e-7,
        }
    }
}
impl LocalDerivativeSettings {
    pub fn validate(self) -> Result<Self, LocalDerivativeSettingsError> {
        if ![self.exterior_kappa_cutoff, self.branch_relation_rtol]
            .into_iter()
            .all(f64::is_finite)
            || self.exterior_kappa_cutoff < 0.0
            || self.branch_relation_rtol < 0.0
        {
            return Err(LocalDerivativeSettingsError);
        }
        Ok(self)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "local derivative settings must be finite and nonnegative, with a positive reference scale"
)]
pub struct LocalDerivativeSettingsError;

/// Units for a physical characteristic partial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartialUnits {
    /// Characteristic per cm^-1, with the other physical coordinate fixed.
    PerInverseCentimetre,
    /// Characteristic per caller-stored layer-thickness unit.
    PerLength(LengthUnit),
}

/// Successful physical partial and its held-fixed convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalPartial {
    pub parameter: PhysicalParameter,
    pub value: Complex,
    pub units: PartialUnits,
    pub held_fixed: &'static str,
}

/// One independently fallible parameter outcome.
#[derive(Clone, Debug)]
pub struct PartialOutcome {
    pub parameter: PhysicalParameter,
    pub result: Result<PhysicalPartial, LocalDerivativeFailure>,
}

/// Failure for one physical derivative without discarding other outcomes.
#[derive(Clone, Debug, thiserror::Error)]
pub enum LocalDerivativeFailure {
    #[error("accepted exterior branch evidence is unavailable")]
    Branch(#[source] Diagnostic),
    #[error("exterior {side} branch is at or below the derivative cutoff: |kappa|={magnitude}")]
    ExteriorCutoff { side: &'static str, magnitude: f64 },
    #[error(
        "stored {side} exterior branch is inconsistent with its source materials (defect {defect})"
    )]
    ExteriorRelation { side: &'static str, defect: f64 },
    #[error("selected thickness layer {0:?} is outside the finite stack")]
    LayerOutOfBounds(FiniteLayerIndex),
    #[error("core stack compilation failed")]
    Compilation(#[source] Arc<dyn std::error::Error + Send + Sync>),
    #[error("checked physical derivative evaluation failed")]
    Evaluation(#[source] Diagnostic),
    #[error("physical derivative contains non-finite components")]
    NonFinite,
}

/// Owned derivatives associated with an accepted point and explicit backend.
#[derive(Clone, Debug)]
pub struct LocalDerivativeReport {
    pub point: ModePoint,
    pub local_assumption: LocalAnalyticityAssumption,
    pub source_backend: &'static str,
    pub evaluation_backend: &'static str,
    pub settings: LocalDerivativeSettings,
    pub characteristic: Result<Complex, Diagnostic>,
    pub outcomes: Vec<PartialOutcome>,
}
impl LocalDerivativeReport {
    /// Form one guarded first-order implicit slope from already evaluated
    /// physical partials. Thresholds use the denominator partial's units, so
    /// supply a separate setting when requested slopes have different units.
    pub fn implicit_slope(
        &self,
        request: ImplicitSlopeRequest,
        settings: ImplicitSlopeSettings,
    ) -> Result<ImplicitSlope, ImplicitSlopeFailure> {
        let settings = settings.validate()?;
        let (numerator_parameter, denominator_parameter, units) = match request {
            ImplicitSlopeRequest::BetaPerK0 => (
                PhysicalParameter::K0,
                PhysicalParameter::Beta,
                SlopeUnits::Dimensionless,
            ),
            ImplicitSlopeRequest::K0PerBeta => (
                PhysicalParameter::Beta,
                PhysicalParameter::K0,
                SlopeUnits::Dimensionless,
            ),
            ImplicitSlopeRequest::BetaPerThickness(layer)
            | ImplicitSlopeRequest::K0PerThickness(layer) => {
                let numerator = PhysicalParameter::LayerThickness(layer);
                let denominator = match request {
                    ImplicitSlopeRequest::BetaPerThickness(_) => PhysicalParameter::Beta,
                    ImplicitSlopeRequest::K0PerThickness(_) => PhysicalParameter::K0,
                    _ => unreachable!(),
                };
                let PartialUnits::PerLength(unit) = self.partial(numerator)?.units else {
                    unreachable!("thickness partial carries its stored length unit")
                };
                (
                    numerator,
                    denominator,
                    SlopeUnits::InverseCentimetrePerLength(unit),
                )
            }
        };
        let numerator = self.partial(numerator_parameter)?;
        let denominator = self.partial(denominator_parameter)?;
        let threshold = settings.denominator_atol
            + settings.denominator_rtol * settings.denominator_reference_scale;
        let magnitude = denominator.value.norm();
        if !magnitude.is_finite() || magnitude <= threshold {
            return Err(ImplicitSlopeFailure::DegenerateDenominator {
                value: denominator.value,
                threshold,
            });
        }
        let value = -numerator.value / denominator.value;
        if !finite(value) {
            return Err(ImplicitSlopeFailure::NonFinite);
        }
        Ok(ImplicitSlope {
            request,
            value,
            units,
            numerator,
            denominator,
            exclusion_threshold: threshold,
        })
    }

    fn partial(
        &self,
        parameter: PhysicalParameter,
    ) -> Result<PhysicalPartial, ImplicitSlopeFailure> {
        let outcome = self
            .outcomes
            .iter()
            .find(|outcome| outcome.parameter == parameter)
            .ok_or(ImplicitSlopeFailure::MissingPartial(parameter))?;
        match &outcome.result {
            Ok(partial) => Ok(*partial),
            Err(error) => Err(ImplicitSlopeFailure::PartialUnavailable(error.clone())),
        }
    }
}

/// Globally invalid requests fail before any selected parameter is evaluated.
#[derive(Debug, thiserror::Error)]
pub enum LocalDerivativeRequestError {
    #[error(transparent)]
    Settings(#[from] LocalDerivativeSettingsError),
    #[error("at least one unique physical parameter must be requested")]
    EmptyOrDuplicateRequest,
}

/// Global request failure retaining the accepted source point.
#[derive(Debug, thiserror::Error)]
#[error("local derivative request failed: {cause}")]
pub struct LocalDerivativeRequestFailure {
    pub point: ModePoint,
    #[source]
    pub cause: LocalDerivativeRequestError,
}

/// Requested conditional simple-root implicit slope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImplicitSlopeRequest {
    BetaPerK0,
    K0PerBeta,
    BetaPerThickness(FiniteLayerIndex),
    K0PerThickness(FiniteLayerIndex),
}

/// Dimensioned nondegeneracy check for one slope denominator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImplicitSlopeSettings {
    /// Absolute threshold in the denominator partial's units.
    pub denominator_atol: f64,
    /// Relative threshold, dimensionless.
    pub denominator_rtol: f64,
    /// Reference scale in the denominator partial's units.
    pub denominator_reference_scale: f64,
}
impl Default for ImplicitSlopeSettings {
    fn default() -> Self {
        Self {
            denominator_atol: 1e-12,
            denominator_rtol: 1e-10,
            denominator_reference_scale: 1.0,
        }
    }
}
impl ImplicitSlopeSettings {
    pub fn validate(self) -> Result<Self, LocalDerivativeSettingsError> {
        if ![
            self.denominator_atol,
            self.denominator_rtol,
            self.denominator_reference_scale,
        ]
        .into_iter()
        .all(f64::is_finite)
            || self.denominator_atol < 0.0
            || self.denominator_rtol < 0.0
            || self.denominator_reference_scale <= 0.0
        {
            return Err(LocalDerivativeSettingsError);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlopeUnits {
    Dimensionless,
    /// cm^-1 per unit of stored layer thickness.
    InverseCentimetrePerLength(LengthUnit),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImplicitSlope {
    pub request: ImplicitSlopeRequest,
    pub value: Complex,
    pub units: SlopeUnits,
    pub numerator: PhysicalPartial,
    pub denominator: PhysicalPartial,
    pub exclusion_threshold: f64,
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum ImplicitSlopeFailure {
    #[error("a required physical partial was not requested")]
    MissingPartial(PhysicalParameter),
    #[error("a required physical partial is unavailable")]
    PartialUnavailable(#[source] LocalDerivativeFailure),
    #[error("slope denominator {value} is within its exclusion threshold {threshold}")]
    DegenerateDenominator { value: Complex, threshold: f64 },
    #[error("implicit slope is non-finite or unrepresentable")]
    NonFinite,
    #[error(transparent)]
    InvalidSettings(#[from] LocalDerivativeSettingsError),
}

type DerivativeJet = ModeJet1<Complex, Ix0>;
type DerivativeStack = Stack<AnalyticalMaterialHandle<Complex>, f64>;

impl ModePoint {
    /// Evaluate requested physical partial derivatives at this accepted point.
    ///
    /// `k0` and `beta` are differentiated independently in cm^-1. A thickness
    /// derivative uses the selected layer's stored caller unit. The explicit
    /// local declaration covers variations beyond those authorized by the
    /// original search. In particular, a fixed-frequency search assumption
    /// alone does not authorize a physical spectral partial. No root correction
    /// or discovery is performed.
    pub fn local_derivatives_with<B>(
        &self,
        backend: B,
        parameters: &[PhysicalParameter],
        local_assumption: LocalAnalyticityAssumption,
        settings: LocalDerivativeSettings,
    ) -> Result<LocalDerivativeReport, Box<LocalDerivativeRequestFailure>>
    where
        B: ModalDerivativeBackend,
    {
        let fail = |cause| {
            Box::new(LocalDerivativeRequestFailure {
                point: self.clone(),
                cause,
            })
        };
        let settings = settings
            .validate()
            .map_err(|error| fail(LocalDerivativeRequestError::Settings(error)))?;
        if parameters.is_empty() {
            return Err(fail(LocalDerivativeRequestError::EmptyOrDuplicateRequest));
        }
        for (i, parameter) in parameters.iter().enumerate() {
            if parameters[..i].contains(parameter) {
                return Err(fail(LocalDerivativeRequestError::EmptyOrDuplicateRequest));
            }
        }

        let signature = match self.signature() {
            Ok(signature) => *signature,
            Err(error) => {
                // Preserve the branch diagnostic as a per-parameter failure below.
                let outcomes = parameters
                    .iter()
                    .copied()
                    .map(|parameter| PartialOutcome {
                        parameter,
                        result: Err(LocalDerivativeFailure::Branch(error.clone())),
                    })
                    .collect();
                return Ok(LocalDerivativeReport {
                    point: self.clone(),
                    local_assumption,
                    source_backend: self.backend(),
                    evaluation_backend: std::any::type_name::<B>(),
                    settings,
                    characteristic: Err(error.clone()),
                    outcomes,
                });
            }
        };

        let provenance = self.provenance();
        let stack = provenance.stack();
        let characteristic = match compile_canonical_constant_stack::<_, DerivativeJet>(
            stack,
            Ix0(),
            &ValidationConfig::permissive(),
        ) {
            Ok(canonical) => {
                let coordinates = CanonicalCoordinates::new(
                    <DerivativeJet as SeedJet>::constant(arr0(signature.k0)),
                    <DerivativeJet as SeedJet>::constant(arr0(signature.k_parallel)),
                );
                let exterior = constant_exteriors(&signature);
                backend
                    .checked_characteristic(
                        canonical,
                        &coordinates,
                        &exterior,
                        provenance.polarisation(),
                    )
                    .map(|value| value.value()[()])
                    .map_err(|error| Diagnostic::new(Stage::Evaluation, error))
            }
            Err(error) => Err(Diagnostic::new(Stage::Evaluation, error)),
        };
        let outcomes = parameters
            .iter()
            .copied()
            .map(|parameter| PartialOutcome {
                parameter,
                result: self.evaluate_partial(&backend, stack, signature, parameter, settings),
            })
            .collect();
        Ok(LocalDerivativeReport {
            point: self.clone(),
            local_assumption,
            source_backend: self.backend(),
            evaluation_backend: std::any::type_name::<B>(),
            settings,
            characteristic,
            outcomes,
        })
    }

    fn evaluate_partial<B>(
        &self,
        backend: &B,
        stack: &DerivativeStack,
        signature: BranchSignature,
        parameter: PhysicalParameter,
        settings: LocalDerivativeSettings,
    ) -> Result<PhysicalPartial, LocalDerivativeFailure>
    where
        B: ModalDerivativeBackend,
    {
        if let PhysicalParameter::LayerThickness(layer) = parameter {
            if layer.get() >= stack.len() {
                return Err(LocalDerivativeFailure::LayerOutOfBounds(layer));
            }
            let canonical = compile_canonical_thickness_stack::<_, DerivativeJet>(
                stack,
                Ix0(),
                &ValidationConfig::permissive(),
                layer,
            )
            .map_err(|e| LocalDerivativeFailure::Compilation(Arc::new(e)))?;
            let coords = CanonicalCoordinates::new(
                <DerivativeJet as SeedJet>::constant(arr0(signature.k0)),
                <DerivativeJet as SeedJet>::constant(arr0(signature.k_parallel)),
            );
            let exterior = constant_exteriors(&signature);
            let result = backend
                .checked_characteristic(
                    canonical,
                    &coords,
                    &exterior,
                    self.provenance().polarisation(),
                )
                .map_err(|e| {
                    LocalDerivativeFailure::Evaluation(Diagnostic::new(Stage::Evaluation, e))
                })?;
            let derivative = result.first()[()];
            if !finite(derivative) {
                return Err(LocalDerivativeFailure::NonFinite);
            }
            let unit = stack.layers_left_to_right()[layer.get()].thickness().unit();
            return Ok(PhysicalPartial {
                parameter,
                value: derivative,
                units: PartialUnits::PerLength(unit),
                held_fixed: "k0, beta, all other layer thicknesses, and exterior branches",
            });
        }

        let canonical = compile_canonical_constant_stack::<_, DerivativeJet>(
            stack,
            Ix0(),
            &ValidationConfig::permissive(),
        )
        .map_err(|e| LocalDerivativeFailure::Compilation(Arc::new(e)))?;
        let k0 = signature.k0;
        let beta = signature.k_parallel;
        let k0_variable = parameter == PhysicalParameter::K0;
        let k0_jet = if k0_variable {
            DerivativeJet::variable(arr0(k0), 0).expect("ModeJet1 has slot zero")
        } else {
            <DerivativeJet as SeedJet>::constant(arr0(k0))
        };
        let beta_jet = if parameter == PhysicalParameter::Beta {
            DerivativeJet::variable(arr0(beta), 0).expect("ModeJet1 has slot zero")
        } else {
            <DerivativeJet as SeedJet>::constant(arr0(beta))
        };
        let exteriors = [stack.left_exterior(), stack.right_exterior()];
        let references = signature.exteriors();
        let mut kappas = [None, None];
        for (side_index, (material, reference)) in exteriors.into_iter().zip(references).enumerate()
        {
            // A side closed by a wall or a PML carries no exterior wavevector.
            let Some(reference) = reference else { continue };
            let side = if side_index == 0 { "left" } else { "right" };
            let magnitude = reference.norm();
            if !magnitude.is_finite() || magnitude <= settings.exterior_kappa_cutoff {
                return Err(LocalDerivativeFailure::ExteriorCutoff { side, magnitude });
            }
            let epsilon = material.evaluate_relative_permittivity_complex(Scalar(k0));
            let mu = material.evaluate_relative_permeability_complex(Scalar(k0));
            let epsilon_prime = material.evaluate_relative_permittivity_complex_derivative(
                Scalar(k0),
                DerivativeOrder::First,
            );
            let mu_prime = material.evaluate_relative_permeability_complex_derivative(
                Scalar(k0),
                DerivativeOrder::First,
            );
            let q = epsilon * mu * k0 * k0 - beta * beta;
            let q_defect = (reference * reference - q).norm();
            let relation_scale = q
                .norm()
                .max((reference * reference).norm())
                .max(f64::MIN_POSITIVE);
            if !q_defect.is_finite() || q_defect > settings.branch_relation_rtol * relation_scale {
                return Err(LocalDerivativeFailure::ExteriorRelation {
                    side,
                    defect: q_defect,
                });
            }
            let q_prime = match parameter {
                PhysicalParameter::K0 => {
                    (epsilon_prime * mu + epsilon * mu_prime) * k0 * k0
                        + Complex::new(2.0, 0.0) * epsilon * mu * k0
                }
                PhysicalParameter::Beta => Complex::new(-2.0, 0.0) * beta,
                PhysicalParameter::LayerThickness(_) => unreachable!(),
            };
            let kappa_prime = exterior_branch_derivative(reference, q_prime);
            if !finite(kappa_prime) {
                return Err(LocalDerivativeFailure::NonFinite);
            }
            let axis = if k0_variable { k0 } else { beta };
            let axis_jet = if k0_variable { &k0_jet } else { &beta_jet };
            let offset = reference - kappa_prime * axis;
            kappas[side_index] = Some(
                axis_jet
                    .scale(kappa_prime)
                    .add(&<DerivativeJet as SeedJet>::constant(arr0(offset))),
            );
        }
        let coords = CanonicalCoordinates::new(k0_jet, beta_jet);
        let [left, right] = kappas;
        let exterior = ExteriorWavevectors::from_options(left, right);
        let result = backend
            .checked_characteristic(
                canonical,
                &coords,
                &exterior,
                self.provenance().polarisation(),
            )
            .map_err(|e| {
                LocalDerivativeFailure::Evaluation(Diagnostic::new(Stage::Evaluation, e))
            })?;
        let derivative = result.first()[()];
        if !finite(derivative) {
            return Err(LocalDerivativeFailure::NonFinite);
        }
        Ok(PhysicalPartial {
            parameter,
            value: derivative,
            units: PartialUnits::PerInverseCentimetre,
            held_fixed: if k0_variable {
                "beta, layer thicknesses, and accepted exterior branch labels"
            } else {
                "k0, layer thicknesses, and accepted exterior branch labels"
            },
        })
    }
}

/// The accepted exterior wavevectors as constants, none on a closed side.
fn constant_exteriors(signature: &BranchSignature) -> ExteriorWavevectors<DerivativeJet> {
    let constant = |kappa| <DerivativeJet as SeedJet>::constant(arr0(kappa));
    ExteriorWavevectors::from_options(signature.left.map(constant), signature.right.map(constant))
}

fn finite(value: Complex) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn exterior_branch_derivative(reference: Complex, q_prime: Complex) -> Complex {
    q_prime / (Complex::new(2.0, 0.0) * reference)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivative_settings_validate_cutoffs_and_branch_tolerance() {
        assert!(LocalDerivativeSettings::default().validate().is_ok());
        for settings in [
            LocalDerivativeSettings {
                exterior_kappa_cutoff: f64::NAN,
                ..Default::default()
            },
            LocalDerivativeSettings {
                branch_relation_rtol: -1.0,
                ..Default::default()
            },
        ] {
            assert!(settings.validate().is_err());
        }
    }

    #[test]
    fn slope_settings_validate_dimensioned_threshold_inputs() {
        assert!(ImplicitSlopeSettings::default().validate().is_ok());
        assert!(
            ImplicitSlopeSettings {
                denominator_reference_scale: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn negating_the_stored_exterior_root_negates_its_derivative() {
        let root = Complex::new(2.0, -3.0);
        let q_prime = Complex::new(5.0, 7.0);
        assert_eq!(
            exterior_branch_derivative(-root, q_prime),
            -exterior_branch_derivative(root, q_prime)
        );
    }
}

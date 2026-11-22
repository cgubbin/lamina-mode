//! Optional interpretation of accepted roots. Raw values and thresholds are
//! snapshots on the originating chart, not globally unique mode identities.
//! Interpretation never changes numerical acceptance, grouping or coverage.
use super::orchestration::{BatchReport, BranchSignature};
use super::radiation::{RadiationProvenance, exterior_factors, finite, positive_dielectric};
use super::*;
use crate::refinement::AcceptedRoot;
use lamina_core::ScalarAlgebra;
use ndarray::arr0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpretationConfig {
    /// Wavenumber absolute band, cm^-1; no dimensionless floor in scaling.
    pub absolute: f64,
    pub relative: f64,
    /// Dimensionless outward-flux coefficient bands.
    pub flux_absolute: f64,
    pub flux_relative: f64,
    /// Positive wavenumber cutoff band, cm^-1.
    pub cutoff: f64,
}
impl Default for InterpretationConfig {
    fn default() -> Self {
        Self {
            absolute: 1e-9,
            relative: 1e-9,
            flux_absolute: 1e-9,
            flux_relative: 1e-9,
            cutoff: 1e-9,
        }
    }
}
impl InterpretationConfig {
    pub(crate) fn validate(self) -> Result<(), InputError> {
        if [
            self.absolute,
            self.relative,
            self.flux_absolute,
            self.flux_relative,
            self.cutoff,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
            || self.cutoff == 0.0
        {
            Err(InputError::InvalidConfig)
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InputError {
    #[error("invalid interpretation tolerances or unrepresentable thresholds")]
    InvalidConfig,
    #[error("non-finite physical point data")]
    NonFinite,
    #[error("physical interpretation belongs to another problem")]
    ForeignProblem,
    #[error("physical interpretation belongs to another domain")]
    ForeignDomain,
}
#[derive(Debug)]
pub enum PointError<E> {
    Input(InputError),
    Unavailable(E),
}
impl<E> PointError<E> {
    pub fn evaluation_error(&self) -> Option<&E> {
        if let Self::Unavailable(e) = self {
            Some(e)
        } else {
            None
        }
    }
}
impl<E: std::fmt::Display> std::fmt::Display for PointError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input(e) => e.fmt(f),
            Self::Unavailable(e) => write!(f, "physical point unavailable: {e}"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for PointError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Input(e) => Some(e),
            Self::Unavailable(e) => Some(e),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalData {
    pub k0: C,
    pub beta: C,
    pub kappa: [C; 2],
    pub epsilon: [C; 2],
    pub mu: [C; 2],
}
impl PhysicalData {
    pub fn alpha(&self) -> [C; 2] {
        self.kappa.map(|k| C::new(0.0, -1.0) * k)
    }
    fn scale(&self) -> f64 {
        [self.k0, self.beta, self.kappa[0], self.kappa[1]]
            .into_iter()
            .map(|x| x.norm())
            .fold(0.0, f64::max)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    Negative,
    Boundary,
    Positive,
}
fn sign(x: f64, band: f64) -> Sign {
    if x > band {
        Sign::Positive
    } else if x < -band {
        Sign::Negative
    } else {
        Sign::Boundary
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExteriorBehaviour {
    Decay,
    Growth,
    Boundary,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evolution {
    Attenuation,
    Amplification,
    TemporalDecay,
    TemporalGrowth,
    Boundary,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unsupported {
    SolveDirection,
    ExteriorMaterial,
    PropagationDirection,
    Frequency,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Flux {
    /// Dimensionless coefficient per squared primary field amplitude, with
    /// the positive unit prefactor suppressed. Not integrated modal power.
    Supported {
        coefficient: f64,
        direction: Sign,
        threshold: f64,
    },
    Unsupported(Unsupported),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeType {
    Guided,
    LeftLeaky,
    RightLeaky,
    LeftSource,
    RightSource,
    TwoSidedLeaky,
    TwoSidedSource,
    Absorptive,
    Gain,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryReason {
    Cutoff,
    NearSectorBoundary,
    ImproperRealAxis,
    InconsistentEvidence,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Classification {
    Classified(ModeType),
    Boundary(BoundaryReason),
    Unsupported(Unsupported),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadiationQualification {
    Unspecified,
    VerifiedOther,
    OutgoingContinuation,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Interpretation {
    pub exterior: [ExteriorBehaviour; 2],
    pub evolution: Evolution,
    pub flux: [Flux; 2],
    pub classification: Classification,
    pub radiation: RadiationQualification,
    pub wavenumber_threshold: f64,
    pub config: InterpretationConfig,
}
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalDetails {
    pub data: PhysicalData,
    pub interpretation: Interpretation,
}
/// Retains the accepted root even when physical data are unavailable. Its
/// provenance supplies polarization, solve direction and exact domain identity.
#[derive(Debug)]
pub struct PointDescription<'a, E> {
    pub root: &'a AcceptedRoot,
    pub observation: Option<(usize, usize)>,
    pub result: Result<PhysicalDetails, PointError<E>>,
}

fn interpret(
    data: PhysicalData,
    coordinate: SearchCoordinate,
    pol: Polarisation,
    radiation: &RadiationProvenance,
    config: InterpretationConfig,
) -> Result<PhysicalDetails, InputError> {
    config.validate()?;
    if ![
        data.k0,
        data.beta,
        data.kappa[0],
        data.kappa[1],
        data.epsilon[0],
        data.epsilon[1],
        data.mu[0],
        data.mu[1],
    ]
    .into_iter()
    .all(finite)
    {
        return Err(InputError::NonFinite);
    }
    let band = config.absolute + config.relative * data.scale();
    if !band.is_finite() {
        return Err(InputError::InvalidConfig);
    }
    let exterior = data.kappa.map(|k| match sign(k.im, band) {
        Sign::Positive => ExteriorBehaviour::Decay,
        Sign::Negative => ExteriorBehaviour::Growth,
        Sign::Boundary => ExteriorBehaviour::Boundary,
    });
    let spectral = matches!(coordinate, SearchCoordinate::Spectral { .. });
    let evolution = match (
        spectral,
        sign(if spectral { data.k0.im } else { data.beta.im }, band),
    ) {
        (false, Sign::Positive) => Evolution::Attenuation,
        (false, Sign::Negative) => Evolution::Amplification,
        (true, Sign::Negative) => Evolution::TemporalDecay,
        (true, Sign::Positive) => Evolution::TemporalGrowth,
        (_, Sign::Boundary) => Evolution::Boundary,
    };
    let supported_material = (0..2).all(|j| positive_dielectric(data.epsilon[j], data.mu[j]));
    let support = if spectral {
        Some(Unsupported::SolveDirection)
    } else if !supported_material {
        Some(Unsupported::ExteriorMaterial)
    } else if data.k0.im != 0.0 || data.k0.re <= 0.0 {
        Some(Unsupported::Frequency)
    } else {
        None
    };
    let mut flux = [Flux::Unsupported(support.unwrap_or(Unsupported::Frequency)); 2];
    let mut directions = [Sign::Boundary; 2];
    if support.is_none() {
        let coefficients = [0, 1].map(|j| {
            let factor = match pol {
                Polarisation::TransverseElectric => data.mu[j],
                Polarisation::TransverseMagnetic => data.epsilon[j],
            };
            (data.kappa[j] / data.k0 / factor).re
        });
        let threshold = config.flux_absolute
            + config.flux_relative * coefficients.iter().map(|x| x.abs()).fold(0.0, f64::max);
        if !threshold.is_finite() || coefficients.iter().any(|x| !x.is_finite()) {
            return Err(InputError::NonFinite);
        }
        for j in 0..2 {
            directions[j] = sign(coefficients[j], threshold);
            flux[j] = Flux::Supported {
                coefficient: coefficients[j],
                direction: directions[j],
                threshold,
            };
        }
    }
    let classification = if let Some(reason) = support {
        Classification::Unsupported(reason)
    } else if data.beta.re <= band {
        Classification::Unsupported(Unsupported::PropagationDirection)
    } else {
        classify(&data, exterior, evolution, directions, band, config.cutoff)
    };
    let qualification = match radiation {
        RadiationProvenance::Unspecified => RadiationQualification::Unspecified,
        RadiationProvenance::Verified(r) if r.all_outgoing() => {
            RadiationQualification::OutgoingContinuation
        }
        RadiationProvenance::Verified(_) => RadiationQualification::VerifiedOther,
    };
    Ok(PhysicalDetails {
        data,
        interpretation: Interpretation {
            exterior,
            evolution,
            flux,
            classification,
            radiation: qualification,
            wavenumber_threshold: band,
            config,
        },
    })
}
fn classify(
    data: &PhysicalData,
    exterior: [ExteriorBehaviour; 2],
    evolution: Evolution,
    flux: [Sign; 2],
    band: f64,
    cutoff: f64,
) -> Classification {
    use BoundaryReason::*;
    use Classification::{Boundary, Classified};
    use Evolution::{Amplification, Attenuation};
    use ExteriorBehaviour::{Decay, Growth};
    use ModeType::*;
    use Sign::{Boundary as Zero, Negative as I, Positive as O};
    if data.kappa.iter().any(|k| k.norm() <= cutoff) {
        return Boundary(Cutoff);
    }
    // Check dispersion independently of the proposed labels. A generous roundoff
    // band in squared units is derived from the same dimensional scale.
    let scale = data.scale();
    let limit = (2.0 * band + 64.0 * f64::EPSILON * scale) * scale;
    if !limit.is_finite() {
        return Boundary(InconsistentEvidence);
    }
    for j in 0..2 {
        let residual = data.kappa[j] * data.kappa[j] - data.epsilon[j] * data.k0 * data.k0
            + data.beta * data.beta;
        if !finite(residual) || residual.norm() > limit {
            return Boundary(InconsistentEvidence);
        }
    }
    if evolution == Evolution::Boundary {
        if exterior == [Decay; 2]
            && flux == [Zero; 2]
            && (0..2).all(|j| data.beta.re > data.epsilon[j].re.sqrt() * data.k0.re + band)
        {
            return Classified(Guided);
        }
        return Boundary(ImproperRealAxis);
    }
    let label = match (exterior, flux, evolution) {
        ([Growth, Decay], [O, I], Attenuation) => LeftLeaky,
        ([Decay, Growth], [O, I], Amplification) => RightSource,
        ([Decay, Growth], [I, O], Attenuation) => RightLeaky,
        ([Growth, Decay], [I, O], Amplification) => LeftSource,
        ([Growth, Growth], [I, I], Amplification) => TwoSidedSource,
        ([Decay, Decay], [I, I], Attenuation) => Absorptive,
        ([Decay, Decay], [O, O], Amplification) => Gain,
        ([Growth, Growth], [O, O], Attenuation) => TwoSidedLeaky,
        _ => {
            return Boundary(
                if exterior.contains(&ExteriorBehaviour::Boundary) || flux.contains(&Zero) {
                    NearSectorBoundary
                } else {
                    InconsistentEvidence
                },
            );
        }
    };
    Classified(label)
}
pub(crate) fn from_signature(
    signature: BranchSignature,
    provenance: &DomainProvenance,
    config: InterpretationConfig,
) -> Result<PhysicalDetails, InputError> {
    config.validate()?;
    if ![
        signature.k0,
        signature.k_parallel,
        signature.left,
        signature.right,
    ]
    .into_iter()
    .all(finite)
    {
        return Err(InputError::NonFinite);
    }
    let factors = exterior_factors(provenance.problem(), signature.k0);
    interpret(
        PhysicalData {
            k0: signature.k0,
            beta: signature.k_parallel,
            kappa: [signature.left, signature.right],
            epsilon: factors.map(|v| v.0),
            mu: factors.map(|v| v.1),
        },
        provenance.problem().coordinate(),
        provenance.problem().polarisation(),
        provenance.radiation(),
        config,
    )
}
impl<M, S> ModalProblem<M, S>
where
    M: Clone,
    S: ModeDeterminantSolver<J> + Clone,
{
    /// Snapshot physical interpretation on the exact originating chart. Invalid
    /// ownership/configuration is checked before evaluating chart or materials.
    pub fn describe_root<'a, H: AnalyticChart<J>>(
        &self,
        domain: &ProblemDomain<M, S, H>,
        root: &'a AcceptedRoot,
        config: InterpretationConfig,
    ) -> PointDescription<'a, ExecutionError<NativeOpticalError<H, S>>> {
        let result = (|| {
            use PointError::Input;
            config.validate().map_err(Input)?;
            if !root.provenance().belongs_to(self) || !domain.provenance.belongs_to(self) {
                return Err(Input(InputError::ForeignProblem));
            }
            if !root.provenance().same_domain(&domain.provenance) {
                return Err(Input(InputError::ForeignDomain));
            }
            let point = domain
                .search
                .dispersion()
                .chart()
                .map(J::lift_constant(arr0(root.evidence().z)))
                .map_err(|e| {
                    PointError::Unavailable(ExecutionError::Optical(Box::new(
                        ChartDispersionError::Chart(e),
                    )))
                })?;
            from_signature(
                BranchSignature {
                    k0: point.coordinates().vacuum_angular_wavenumber().value()[()],
                    k_parallel: point.coordinates().parallel_angular_wavenumber().value()[()],
                    left: point.exterior_wavevectors().left().value()[()],
                    right: point.exterior_wavevectors().right().value()[()],
                },
                root.provenance(),
                config,
            )
            .map_err(Input)
        })();
        PointDescription {
            root,
            observation: None,
            result,
        }
    }
}
impl<E, D> BatchReport<'_, E, D> {
    /// One outcome per accepted observation. Signature failures are borrowed with
    /// their original type and cause; no re-evaluation or hidden fallback occurs.
    pub fn describe_points(&self, config: InterpretationConfig) -> Vec<PointDescription<'_, &E>> {
        self.observations()
            .iter()
            .map(|observation| {
                let root = observation.root();
                let result = (|| {
                    config.validate().map_err(PointError::Input)?;
                    let provenance = self
                        .charts()
                        .get(observation.chart())
                        .ok_or(PointError::Input(InputError::ForeignDomain))?
                        .provenance();
                    if !root.provenance().same_domain(provenance) {
                        return Err(PointError::Input(InputError::ForeignDomain));
                    }
                    let signature = *observation.signature().map_err(PointError::Unavailable)?;
                    from_signature(signature, provenance, config).map_err(PointError::Input)
                })();
                PointDescription {
                    root,
                    observation: Some((observation.chart(), observation.candidate())),
                    result,
                }
            })
            .collect()
    }
}
#[cfg(test)]
mod tests;

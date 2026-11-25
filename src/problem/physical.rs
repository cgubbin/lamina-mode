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
/// Physical coordinates and exterior data of a point. `kappa` holds the
/// exterior wavevector of each open side; a side closed by a wall or a PML has none.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalData {
    pub k0: C,
    pub beta: C,
    pub kappa: [Option<C>; 2],
    pub epsilon: [C; 2],
    pub mu: [C; 2],
}
impl PhysicalData {
    pub fn alpha(&self) -> [Option<C>; 2] {
        self.kappa.map(|k| k.map(|k| C::new(0.0, -1.0) * k))
    }
    fn scale(&self) -> f64 {
        [self.k0, self.beta]
            .into_iter()
            .chain(self.kappa.into_iter().flatten())
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
    /// Closed by a wall or a PML: no exterior wave.
    Closed,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Evolution {
    Attenuation,
    Amplification,
    TemporalDecay,
    TemporalGrowth,
    Boundary,
    /// A side is closed by a PML, an analytic continuation rather than a
    /// passive medium: the sign of Im β at the level of its truncation says
    /// nothing about gain or loss, and none is inferred. `truncation` is the
    /// largest |exp(2iκ̃ℓ)| over the PML sides, ℓ = g + τ the complex length
    /// from the outermost interface to the backing wall and κ̃ the root of
    /// κ² = εμk0² − β² decaying over it: the relative size of the PML's
    /// perturbation of the open stack's characteristic.
    PmlTruncation {
        truncation: f64,
    },
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
    /// Closed by a wall or a PML: no power crosses it.
    Closed,
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
    /// A mode of a stack closed on both sides, by walls or PMLs.
    Closed,
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
    if ![data.k0, data.beta]
        .into_iter()
        .chain(data.kappa.into_iter().flatten())
        .chain(data.epsilon)
        .chain(data.mu)
        .all(finite)
    {
        return Err(InputError::NonFinite);
    }
    let band = config.absolute + config.relative * data.scale();
    if !band.is_finite() {
        return Err(InputError::InvalidConfig);
    }
    let exterior = data.kappa.map(|k| match k.map(|k| sign(k.im, band)) {
        None => ExteriorBehaviour::Closed,
        Some(Sign::Positive) => ExteriorBehaviour::Decay,
        Some(Sign::Negative) => ExteriorBehaviour::Growth,
        Some(Sign::Boundary) => ExteriorBehaviour::Boundary,
    });
    let open: Vec<usize> = (0..2).filter(|&j| data.kappa[j].is_some()).collect();
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
    let supported_material = open
        .iter()
        .all(|&j| positive_dielectric(data.epsilon[j], data.mu[j]));
    let support = if spectral {
        Some(Unsupported::SolveDirection)
    } else if !supported_material {
        Some(Unsupported::ExteriorMaterial)
    } else if data.k0.im != 0.0 || data.k0.re <= 0.0 {
        Some(Unsupported::Frequency)
    } else {
        None
    };
    let mut flux = data.kappa.map(|k| match k {
        None => Flux::Closed,
        Some(_) => Flux::Unsupported(support.unwrap_or(Unsupported::Frequency)),
    });
    let mut directions = [Sign::Boundary; 2];
    if support.is_none() {
        let coefficients = [0, 1].map(|j| {
            data.kappa[j].map(|kappa| {
                let factor = match pol {
                    Polarisation::TransverseElectric => data.mu[j],
                    Polarisation::TransverseMagnetic => data.epsilon[j],
                };
                (kappa / data.k0 / factor).re
            })
        });
        let open_coefficients = coefficients.iter().flatten();
        let threshold = config.flux_absolute
            + config.flux_relative
                * open_coefficients
                    .clone()
                    .map(|x| x.abs())
                    .fold(0.0, f64::max);
        if !threshold.is_finite() || open_coefficients.clone().any(|x| !x.is_finite()) {
            return Err(InputError::NonFinite);
        }
        for &j in &open {
            let coefficient = coefficients[j].expect("open side");
            directions[j] = sign(coefficient, threshold);
            flux[j] = Flux::Supported {
                coefficient,
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
        classify(
            &data,
            &open,
            exterior,
            evolution,
            directions,
            band,
            config.cutoff,
        )
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
/// Classify from the open sides alone; `open` lists them in left/right order.
/// With no open side the type is `Closed`.
fn classify(
    data: &PhysicalData,
    open: &[usize],
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
    if open.is_empty() {
        return Classified(Closed);
    }
    let kappa = |j: usize| data.kappa[j].expect("open side");
    if open.iter().any(|&j| kappa(j).norm() <= cutoff) {
        return Boundary(Cutoff);
    }
    // Check dispersion independently of the proposed labels. A generous roundoff
    // band in squared units is derived from the same dimensional scale.
    let scale = data.scale();
    let limit = (2.0 * band + 64.0 * f64::EPSILON * scale) * scale;
    if !limit.is_finite() {
        return Boundary(InconsistentEvidence);
    }
    for &j in open {
        let residual =
            kappa(j) * kappa(j) - data.epsilon[j] * data.k0 * data.k0 + data.beta * data.beta;
        if !finite(residual) || residual.norm() > limit {
            return Boundary(InconsistentEvidence);
        }
    }
    // Beside a PML the evolution says nothing, so the open side's exterior
    // and flux decide.
    if let (Evolution::PmlTruncation { .. }, &[j]) = (evolution, open) {
        return match (exterior[j], flux[j]) {
            (Decay, _) => Classified(Guided),
            (Growth, O) => Classified([LeftLeaky, RightLeaky][j]),
            (Growth, I) => Classified([LeftSource, RightSource][j]),
            _ => Boundary(NearSectorBoundary),
        };
    }
    if evolution == Evolution::Boundary {
        if open.iter().all(|&j| {
            exterior[j] == Decay
                && flux[j] == Zero
                && data.beta.re > data.epsilon[j].re.sqrt() * data.k0.re + band
        }) {
            return Classified(Guided);
        }
        return Boundary(ImproperRealAxis);
    }
    let label = match *open {
        [j] => match (exterior[j], flux[j], evolution) {
            (Growth, O, Attenuation) => Some([LeftLeaky, RightLeaky][j]),
            (Growth, I, Amplification) => Some([LeftSource, RightSource][j]),
            (Decay, I, Attenuation) => Some(Absorptive),
            (Decay, O, Amplification) => Some(Gain),
            _ => None,
        },
        _ => match (exterior, flux, evolution) {
            ([Growth, Decay], [O, I], Attenuation) => Some(LeftLeaky),
            ([Decay, Growth], [O, I], Amplification) => Some(RightSource),
            ([Decay, Growth], [I, O], Attenuation) => Some(RightLeaky),
            ([Growth, Decay], [I, O], Amplification) => Some(LeftSource),
            ([Growth, Growth], [I, I], Amplification) => Some(TwoSidedSource),
            ([Decay, Decay], [I, I], Attenuation) => Some(Absorptive),
            ([Decay, Decay], [O, O], Amplification) => Some(Gain),
            ([Growth, Growth], [O, O], Attenuation) => Some(TwoSidedLeaky),
            _ => None,
        },
    };
    let Some(label) = label else {
        let near = open
            .iter()
            .any(|&j| exterior[j] == ExteriorBehaviour::Boundary || flux[j] == Zero);
        return Boundary(if near {
            NearSectorBoundary
        } else {
            InconsistentEvidence
        });
    };
    Classified(label)
}
pub(crate) fn from_signature(
    signature: BranchSignature,
    provenance: &DomainProvenance,
    config: InterpretationConfig,
) -> Result<PhysicalDetails, InputError> {
    config.validate()?;
    if ![signature.k0, signature.k_parallel]
        .into_iter()
        .chain(signature.exteriors().into_iter().flatten())
        .all(finite)
    {
        return Err(InputError::NonFinite);
    }
    let stack = provenance.problem().stack();
    let factors = exterior_factors(stack, signature.k0);
    let mut details = interpret(
        PhysicalData {
            k0: signature.k0,
            beta: signature.k_parallel,
            kappa: signature.exteriors(),
            epsilon: factors.map(|v| v.0),
            mu: factors.map(|v| v.1),
        },
        provenance.problem().coordinate(),
        provenance.problem().polarisation(),
        provenance.radiation(),
        config,
    )?;
    let (k0, beta) = (signature.k0, signature.k_parallel);
    let truncation = pmls(stack)
        .into_iter()
        .zip(factors)
        .filter_map(|(pml, (epsilon, mu))| {
            let (gap, tau) = pml?;
            let kappa = (epsilon * mu * k0 * k0 - beta * beta).sqrt();
            Some((-2.0 * (kappa * (gap + tau)).im.abs()).exp())
        })
        .reduce(f64::max);
    if let Some(truncation) = truncation {
        let interpretation = &mut details.interpretation;
        interpretation.evolution = Evolution::PmlTruncation { truncation };
        // Beside an open side, the type follows from that side alone.
        if !matches!(
            interpretation.classification,
            Classification::Unsupported(_)
        ) {
            let open: Vec<usize> = (0..2)
                .filter(|&j| details.data.kappa[j].is_some())
                .collect();
            let directions = interpretation.flux.map(|flux| match flux {
                Flux::Supported { direction, .. } => direction,
                _ => Sign::Boundary,
            });
            interpretation.classification = classify(
                &details.data,
                &open,
                interpretation.exterior,
                interpretation.evolution,
                directions,
                interpretation.wavenumber_threshold,
                config.cutoff,
            );
        }
    }
    Ok(details)
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
            from_signature(BranchSignature::of(&point), root.provenance(), config).map_err(Input)
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

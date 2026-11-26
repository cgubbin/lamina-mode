//! Direct frequency charts over a supplied core stack. Caller assumptions apply
//! to the whole evaluation neighbourhood; pointwise checks do not certify it.
//! Both exterior roots are continued from fixed anchors independently of sample
//! order. Constitutive derivatives enter jets analytically, not through quadrature.
use super::radiation::{RadiationChoice, RadiationProvenance, RadiationTolerance, finite};
use super::*;
use crate::conformal::ConformalPoint;
use lamina_core::{AnalyticalMaterialHandle, SeedJet};
use lamina_core::{CanonicalCoordinates, ConstitutiveLift, ExteriorWavevectors, ScalarAlgebra};
use ndarray::arr0;
use std::convert::Infallible;

pub type FrequencyProblem<B> = ModalProblem<(), Compiled<B>>;
pub type FrequencyDomain<S> = ProblemDomain<(), S, FrequencyChart>;

/// Bounds in physical vacuum angular wavenumber (cm^-1), never dimensionless z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrequencyRegion(ConformalRegion<f64>);
impl FrequencyRegion {
    pub fn new(
        real_min: f64,
        real_max: f64,
        imag_min: f64,
        imag_max: f64,
    ) -> Result<Self, FrequencyError> {
        ConformalRegion::new(real_min, real_max, imag_min, imag_max)
            .map(Self)
            .map_err(|_| FrequencyError::InvalidBounds)
    }
    pub fn bounds(&self) -> ConformalRegion<f64> {
        self.0
    }
    pub(super) fn from_native(bounds: ConformalRegion<f64>) -> Self {
        Self(bounds)
    }
}
/// Additional, explicit obligation on both q_j, not a certificate from samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExteriorAnalyticAssumption {
    CallerAssumedHolomorphicNonzero,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExclusionReason {
    MaterialSingularity,
    ExteriorThreshold { side: usize },
    CallerDeclared(String),
}
#[derive(Clone, Debug, PartialEq)]
pub struct FrequencyExclusion {
    pub region: FrequencyRegion,
    pub reason: ExclusionReason,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContinuationConfig {
    /// Absolute tolerance of the dimensionless logarithmic integral.
    pub integral_tolerance: f64,
    pub consistency_tolerance: f64,
    /// Positive wavevector cutoff band, cm^-1.
    pub cutoff: f64,
    pub max_depth: u32,
    /// Per exterior and point, counting local and quadrature material jets.
    pub max_evaluations: usize,
}
impl Default for ContinuationConfig {
    fn default() -> Self {
        Self {
            integral_tolerance: 1e-10,
            consistency_tolerance: 1e-8,
            cutoff: 1e-10,
            max_depth: 20,
            max_evaluations: 4096,
        }
    }
}
impl ContinuationConfig {
    fn validate(self) -> Result<(), FrequencyError> {
        if [
            self.integral_tolerance,
            self.consistency_tolerance,
            self.cutoff,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.0)
            || self.max_evaluations == 0
            || self.max_depth > 32
        {
            Err(FrequencyError::InvalidConfig)
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExteriorReferenceValues {
    pub frequency: C,
    pub kappa: [C; 2],
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrequencyReference {
    Explicit(ExteriorReferenceValues),
    RealRadiation {
        frequency: f64,
        choices: [RadiationChoice; 2],
        tolerance: RadiationTolerance,
    },
}
#[derive(Clone, Debug)]
pub struct FrequencyDomainInput {
    pub bounds: FrequencyRegion,
    pub reference: FrequencyReference,
    pub continuation: ContinuationConfig,
    pub exterior_assumption: ExteriorAnalyticAssumption,
    /// Known guard rectangles, not a discovered or certified complete list.
    pub exclusions: Vec<FrequencyExclusion>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct FrequencyEvidence {
    pub reference: ExteriorReferenceValues,
    pub continuation: ContinuationConfig,
    pub exterior_assumption: ExteriorAnalyticAssumption,
    pub exclusions: Vec<FrequencyExclusion>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ExteriorFailure {
    #[error("non-finite constitutive value")]
    NonFiniteMaterial,
    #[error("non-finite constitutive or wavevector derivative")]
    NonFiniteDerivative,
    #[error("sampled exterior wavevector is within the cutoff band")]
    Cutoff,
    #[error("exterior evaluation budget exhausted")]
    EvaluationBudget,
    #[error("exterior continuation quadrature did not converge")]
    IntegrationDepth,
    #[error("non-finite exterior continuation result")]
    NonFiniteResult,
    #[error("exterior reference or continuation is inconsistent with its dispersion relation")]
    InconsistentValue,
    #[error("exterior jet derivative is inconsistent with continuation")]
    InconsistentDerivative,
    #[error("exterior is not isotropic")]
    Anisotropic,
}
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum FrequencyError {
    #[error("invalid frequency bounds")]
    InvalidBounds,
    #[error("invalid continuation settings")]
    InvalidConfig,
    #[error("frequency reference must be finite and strictly inside the requested region")]
    InvalidReference,
    #[error("frequency lies outside the evaluation region")]
    OutsideRegion,
    #[error("cannot construct a finite frequency evaluation halo")]
    InvalidMargin,
    #[error("invalid frequency exclusion")]
    InvalidExclusion,
    #[error("frequency chart intersects declared exclusion: {0:?}")]
    Excluded(FrequencyExclusion),
    #[error("exterior {side} failed after {evaluations} material evaluations: {source}")]
    Exterior {
        side: usize,
        #[source]
        source: ExteriorFailure,
        evaluations: usize,
    },
    #[error("radiation reference failed: {0}")]
    Radiation(#[source] radiation::RadiationReferenceError<Infallible>),
}
#[derive(Clone, Copy, Debug)]
pub struct ContinuedWavevector {
    pub value: C,
    pub derivative: C,
    pub evaluations: usize,
}
#[derive(Clone, Debug)]
pub struct FrequencyChart {
    region: ConformalRegion<f64>,
    materials: [AnalyticalMaterialHandle<C>; 2],
    beta: f64,
    evidence: FrequencyEvidence,
}
pub(super) fn contains(r: ConformalRegion<f64>, k: C) -> bool {
    let (x0, x1) = r.real_bounds();
    let (y0, y1) = r.imaginary_bounds();
    finite(k) && k.re >= x0 && k.re <= x1 && k.im >= y0 && k.im <= y1
}
fn overlaps(a: ConformalRegion<f64>, b: ConformalRegion<f64>) -> bool {
    let (a0, a1) = a.real_bounds();
    let (a2, a3) = a.imaginary_bounds();
    let (b0, b1) = b.real_bounds();
    let (b2, b3) = b.imaginary_bounds();
    a0 <= b1 && b0 <= a1 && a2 <= b3 && b2 <= a3
}
impl FrequencyChart {
    fn error(&self, side: usize, source: ExteriorFailure, evaluations: usize) -> FrequencyError {
        FrequencyError::Exterior {
            side,
            source,
            evaluations,
        }
    }
    fn check_exclusions(&self, region: ConformalRegion<f64>) -> Result<(), FrequencyError> {
        for e in &self.evidence.exclusions {
            if matches!(e.reason,ExclusionReason::ExteriorThreshold{side} if side>1) {
                return Err(FrequencyError::InvalidExclusion);
            }
            if overlaps(region, e.region.0) {
                return Err(FrequencyError::Excluded(e.clone()));
            }
        }
        Ok(())
    }
    fn q_jet(&self, side: usize, k: J) -> Result<J, FrequencyError> {
        use ExteriorFailure::*;
        let material = &self.materials[side];
        let epsilon=<J as ConstitutiveLift<ComplexPlane,AnalyticalMaterialHandle<C>>>::relative_permittivity(material,&k);
        let mu=<J as ConstitutiveLift<ComplexPlane,AnalyticalMaterialHandle<C>>>::relative_permeability(material,&k);
        if !finite(epsilon.value()[()]) || !finite(mu.value()[()]) {
            return Err(self.error(side, NonFiniteMaterial, 0));
        }
        if !finite(epsilon.first()[()]) || !finite(mu.first()[()]) {
            return Err(self.error(side, NonFiniteDerivative, 0));
        }
        let q = epsilon
            .multiply(&mu)
            .multiply(&k.square())
            .subtract(&k.constant(C::new(self.beta * self.beta, 0.0)));
        if !finite(q.value()[()]) {
            return Err(self.error(side, NonFiniteResult, 0));
        }
        if !finite(q.first()[()]) {
            return Err(self.error(side, NonFiniteDerivative, 0));
        }
        if q.value()[()].norm().sqrt() <= self.evidence.continuation.cutoff {
            return Err(self.error(side, Cutoff, 0));
        }
        Ok(q)
    }
    fn q(&self, side: usize, k: C) -> Result<(C, C), FrequencyError> {
        let q = self.q_jet(
            side,
            J::variable(arr0(k), 0).expect("one spectral variable"),
        )?;
        Ok((q.value()[()], q.first()[()]))
    }
    fn consistent(&self, value: C, q: C) -> bool {
        let scale = (value * value).norm().max(q.norm());
        let limit = self.evidence.continuation.consistency_tolerance * scale
            + self.evidence.continuation.cutoff.powi(2);
        finite(value)
            && scale.is_finite()
            && limit.is_finite()
            && (value * value - q).norm() <= limit
    }
    pub fn continuation_at(
        &self,
        side: usize,
        k: C,
    ) -> Result<ContinuedWavevector, FrequencyError> {
        use ExteriorFailure::*;
        if side > 1 {
            return Err(FrequencyError::InvalidReference);
        }
        if !contains(self.region, k) {
            return Err(FrequencyError::OutsideRegion);
        }
        let cfg = self.evidence.continuation;
        let mut evaluations = 0;
        let mut sample = |point: C| {
            if evaluations >= cfg.max_evaluations {
                return Err(self.error(side, EvaluationBudget, evaluations));
            }
            evaluations += 1;
            self.q(side, point).map_err(|e| match e {
                FrequencyError::Exterior { side, source, .. } => FrequencyError::Exterior {
                    side,
                    source,
                    evaluations,
                },
                other => other,
            })
        };
        let (q, dq) = sample(k)?;
        let reference = self.evidence.reference;
        let delta = k - reference.frequency;
        let integral = if delta == C::new(0.0, 0.0) {
            C::new(0.0, 0.0)
        } else {
            let mut integrand = |t: f64| {
                let (q, p) = sample(reference.frequency + delta * t)?;
                let value = p / q * delta;
                if !finite(value) {
                    return Err(self.error(side, NonFiniteResult, 0));
                }
                Ok(value)
            };
            integrate(&mut integrand, cfg.integral_tolerance, cfg.max_depth).map_err(
                |e| match e {
                    FrequencyError::Exterior {
                        source,
                        evaluations: 0,
                        ..
                    } => self.error(side, source, evaluations),
                    FrequencyError::InvalidConfig => {
                        self.error(side, IntegrationDepth, evaluations)
                    }
                    other => other,
                },
            )?
        };
        let value = reference.kappa[side] * (integral / 2.0).exp();
        if !finite(value) {
            return Err(self.error(side, NonFiniteResult, evaluations));
        }
        if !self.consistent(value, q) {
            return Err(self.error(side, InconsistentValue, evaluations));
        }
        let derivative = dq / (2.0 * value);
        if !finite(derivative) {
            return Err(self.error(side, NonFiniteDerivative, evaluations));
        }
        Ok(ContinuedWavevector {
            value,
            derivative,
            evaluations,
        })
    }
    pub fn evidence(&self) -> &FrequencyEvidence {
        &self.evidence
    }
}
// Fallible adaptive Simpson with caller-controlled sample budget. A reference
// path is fixed for every evaluation, so quadrature order cannot choose sheets.
fn integrate<F>(f: &mut F, tolerance: f64, depth: u32) -> Result<C, FrequencyError>
where
    F: FnMut(f64) -> Result<C, FrequencyError>,
{
    fn recurse<F>(
        f: &mut F,
        interval: [f64; 2],
        values: [C; 3],
        whole: C,
        tol: f64,
        depth: u32,
    ) -> Result<C, FrequencyError>
    where
        F: FnMut(f64) -> Result<C, FrequencyError>,
    {
        let [a, b] = interval;
        let m = (a + b) / 2.0;
        let l = f((a + m) / 2.0)?;
        let r = f((m + b) / 2.0)?;
        let left = (m - a) / 6.0 * (values[0] + 4.0 * l + values[1]);
        let right = (b - m) / 6.0 * (values[1] + 4.0 * r + values[2]);
        let difference = left + right - whole;
        if !finite(left) || !finite(right) || !finite(difference) {
            return Err(FrequencyError::Exterior {
                side: 0,
                source: ExteriorFailure::NonFiniteResult,
                evaluations: 0,
            });
        }
        if difference.norm() <= 15.0 * tol {
            return Ok(left + right + difference / 15.0);
        }
        if depth == 0 {
            return Err(FrequencyError::InvalidConfig);
        }
        Ok(recurse(
            f,
            [a, m],
            [values[0], l, values[1]],
            left,
            tol / 2.0,
            depth - 1,
        )? + recurse(
            f,
            [m, b],
            [values[1], r, values[2]],
            right,
            tol / 2.0,
            depth - 1,
        )?)
    }
    let values = [f(0.0)?, f(0.5)?, f(1.0)?];
    recurse(
        f,
        [0.0, 1.0],
        values,
        (values[0] + 4.0 * values[1] + values[2]) / 6.0,
        tolerance,
        depth,
    )
}
impl AnalyticChart<J> for FrequencyChart {
    type Error = FrequencyError;
    fn region(&self) -> &ConformalRegion<f64> {
        &self.region
    }
    fn reference(&self) -> C {
        self.evidence.reference.frequency
    }
    fn reference_coordinate(&self) -> C {
        self.reference()
    }
    fn coordinate_kind(&self) -> CoordinateKind {
        CoordinateKind::Frequency
    }
    fn coordinate(&self, k: C) -> Result<C, FrequencyError> {
        if contains(self.region, k) {
            Ok(k)
        } else {
            Err(FrequencyError::OutsideRegion)
        }
    }
    fn map(&self, k: J) -> Result<ConformalPoint<J>, FrequencyError> {
        let point = k.value()[()];
        self.coordinate(point)?;
        let mut jets = Vec::with_capacity(2);
        for side in 0..2 {
            let continued = self.continuation_at(side, point)?;
            if continued.evaluations >= self.evidence.continuation.max_evaluations {
                return Err(self.error(
                    side,
                    ExteriorFailure::EvaluationBudget,
                    continued.evaluations,
                ));
            }
            let q = self.q_jet(side, k.clone()).map_err(|e| match e {
                FrequencyError::Exterior { side, source, .. } => {
                    self.error(side, source, continued.evaluations + 1)
                }
                other => other,
            })?;
            let local = q.sqrt();
            let primal = local.value()[()];
            let selected = if (primal - continued.value).norm() <= (primal + continued.value).norm()
            {
                local
            } else {
                local.scale(C::new(-1.0, 0.0))
            };
            let expected = continued.derivative * k.first()[()];
            let cfg = self.evidence.continuation;
            let limit = cfg.consistency_tolerance
                * expected.norm().max(selected.first()[()].norm())
                + 64.0 * f64::EPSILON;
            if !finite(selected.first()[()])
                || !finite(expected)
                || !limit.is_finite()
                || (selected.first()[()] - expected).norm() > limit
            {
                return Err(self.error(
                    side,
                    ExteriorFailure::InconsistentDerivative,
                    continued.evaluations + 1,
                ));
            }
            if (selected.value()[()] - continued.value).norm()
                > cfg.consistency_tolerance * continued.value.norm() + cfg.cutoff
            {
                return Err(self.error(
                    side,
                    ExteriorFailure::InconsistentValue,
                    continued.evaluations + 1,
                ));
            }
            jets.push(selected);
        }
        Ok(ConformalPoint::new(
            CanonicalCoordinates::new(k.clone(), k.constant(C::new(self.beta, 0.0))),
            ExteriorWavevectors::new(jets.remove(0), jets.remove(0)),
        ))
    }
    fn with_evaluation_margin(&self) -> Result<Self, FrequencyError> {
        let region = halo(self.region).ok_or(FrequencyError::InvalidMargin)?;
        self.check_exclusions(region)?;
        let mut chart = self.clone();
        chart.region = region;
        Ok(chart)
    }
}
/// `region` grown by a few ulps on every side, absorbing boundary roundoff.
pub(super) fn halo(region: ConformalRegion<f64>) -> Option<ConformalRegion<f64>> {
    let expand = |(lo, hi): (f64, f64)| {
        let margin =
            64.0 * f64::EPSILON * lo.abs().max(hi.abs()).max(hi - lo).max(f64::MIN_POSITIVE);
        let a = lo - margin;
        let b = hi + margin;
        (a.is_finite() && b.is_finite() && a < lo && b > hi).then_some((a, b))
    };
    let (x0, x1) = expand(region.real_bounds())?;
    let (y0, y1) = expand(region.imaginary_bounds())?;
    ConformalRegion::new(x0, x1, y0, y1).ok()
}
impl<B: Backend<J, ComplexPlane>> FrequencyProblem<B> {
    /// Reuse a complete core analytical stack. No sampled exterior replacement.
    pub fn frequency(
        stack: impl Into<ModeStack>,
        beta: f64,
        pol: impl Into<lamina_core::Channel>,
        backend: B,
    ) -> Result<Self, ProblemError> {
        let pol = pol.into();
        let stack = stack.into();
        if !beta.is_finite() {
            return Err(ProblemError::Spectral(
                WavenumberMapError::InvalidInPlaneWavenumber,
            ));
        }
        check_terminations(&stack)?;
        if !stack.is_open() {
            return Err(ProblemError::ClosedSide);
        }
        super::exterior_polarisation(&stack, pol)?;
        material::isotropic_exteriors(&stack)?;
        let solver = Compiled::compile(&stack, backend, pol)?;
        Ok(Self {
            context: Arc::new(ProblemContext {
                constants: None,
                stack,
                channel: pol,
                coordinate: SearchCoordinate::Spectral {
                    fixed_k_parallel: beta,
                },
            }),
            map: (),
            solver,
        })
    }
}
impl<S: ModeDeterminantSolver<J> + Clone> ModalProblem<(), S> {
    pub fn frequency_domain(
        &self,
        input: FrequencyDomainInput,
        assumption: AnalyticAssumption,
    ) -> Result<FrequencyDomain<S>, FrequencyError> {
        input.continuation.validate()?;
        let (reference, radiation) = match input.reference {
            FrequencyReference::Explicit(r) => (r, RadiationProvenance::Unspecified),
            FrequencyReference::RealRadiation {
                frequency,
                choices,
                tolerance,
            } => {
                let verified =
                    radiation::frequency_reference(&self.context, frequency, choices, tolerance)
                        .map_err(FrequencyError::Radiation)?;
                (
                    ExteriorReferenceValues {
                        frequency: C::new(frequency, 0.0),
                        kappa: verified.observed().map(|k| k.expect("open stack")),
                    },
                    RadiationProvenance::Verified(verified),
                )
            }
        };
        let r = input.bounds.0;
        let (x0, x1) = r.real_bounds();
        let (y0, y1) = r.imaginary_bounds();
        if !contains(r, reference.frequency)
            || reference.frequency.re <= x0
            || reference.frequency.re >= x1
            || reference.frequency.im <= y0
            || reference.frequency.im >= y1
            || reference
                .kappa
                .iter()
                .any(|k| !finite(*k) || k.norm() <= input.continuation.cutoff)
        {
            return Err(FrequencyError::InvalidReference);
        }
        let SearchCoordinate::Spectral {
            fixed_k_parallel: beta,
        } = self.context.coordinate
        else {
            return Err(FrequencyError::InvalidReference);
        };
        let chart = FrequencyChart {
            region: r,
            materials: material::isotropic_exteriors(&self.context.stack).map_err(|e| {
                FrequencyError::Exterior {
                    side: usize::from(matches!(
                        e,
                        lamina_core::ConstitutiveError::Unsupported {
                            region: lamina_core::StackRegion::RightExterior,
                            ..
                        }
                    )),
                    source: ExteriorFailure::Anisotropic,
                    evaluations: 0,
                }
            })?,
            beta,
            evidence: FrequencyEvidence {
                reference,
                continuation: input.continuation,
                exterior_assumption: input.exterior_assumption,
                exclusions: input.exclusions,
            },
        };
        chart.check_exclusions(r)?;
        for side in 0..2 {
            let (q, _) = chart.q(side, reference.frequency)?;
            if !chart.consistent(reference.kappa[side], q) {
                return Err(chart.error(side, ExteriorFailure::InconsistentValue, 1));
            }
        }
        let search = SingleChartModalSearch::new(
            ConformalDispersion::<_, (), C, FrequencyChart>::with_chart(self.solver.clone(), chart),
        )?;
        let evaluation = *search.dispersion().chart().region();
        let frequency = Some(search.dispersion().chart().evidence.clone());
        let provenance = DomainProvenance {
            domain_identity: Arc::new(()),
            context: self.context.clone(),
            requested: r,
            evaluation,
            selection: ReferenceSelection::Coordinate {
                reference: reference.frequency,
                coordinate: reference.frequency,
            },
            reference: reference.frequency,
            reference_coordinate: reference.frequency,
            assumption,
            radiation,
            coordinate_kind: CoordinateKind::Frequency,
            frequency,
        };
        Ok(ProblemDomain { search, provenance })
    }
}

#[cfg(test)]
mod tests;

impl<S> FrequencyDomain<S> {
    pub(crate) fn frequency_chart(&self) -> &FrequencyChart {
        self.search.dispersion().chart()
    }
}

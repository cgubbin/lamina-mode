//! Direct frequency charts over a supplied core stack. Caller assumptions apply
//! to the whole evaluation neighbourhood; pointwise checks do not certify it.
//! Both exterior roots are continued from fixed anchors independently of sample
//! order. Constitutive derivatives enter jets analytically, not through quadrature.
use super::radiation::{RadiationChoice, RadiationProvenance, RadiationTolerance, finite};
use super::*;
use crate::conformal::ConformalPoint;
use lamina_core::{AnalyticalMaterialHandle, SeedJet};
use lamina_core::{CanonicalCoordinates, ConstitutiveLift, ExteriorWavevectors, ScalarAlgebra};
use lamina_core::{ConstitutiveQuantity, SingularityKind, StackRegion};
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
#[derive(Clone, Debug, PartialEq)]
pub enum ExclusionReason {
    MaterialSingularity,
    /// A declared material singularity, from
    /// [`FrequencyProblem::singularity_exclusions`](crate::FrequencyProblem::singularity_exclusions).
    KnownSingularity(KnownSingularity),
    ExteriorThreshold {
        side: usize,
    },
    CallerDeclared(String),
}
/// A declared pole or zero of one region's constitutive entry at which this
/// channel's characteristic is singular.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KnownSingularity {
    pub region: StackRegion,
    /// Complex vacuum angular wavenumber, cm^-1.
    pub k0: C,
    pub kind: SingularityKind,
    pub quantity: ConstitutiveQuantity,
    /// Stack-frame diagonal entry (0, 1, 2 for xx, yy, zz); `None` for every
    /// entry of an isotropic region.
    pub axis: Option<usize>,
}
impl std::fmt::Display for KnownSingularity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.kind {
            SingularityKind::Pole => "pole",
            SingularityKind::Zero => "zero",
        };
        let symbol = match self.quantity {
            ConstitutiveQuantity::Permittivity => "ε",
            ConstitutiveQuantity::Permeability => "μ",
        };
        let entry = match self.axis {
            Some(0) => "_xx",
            Some(1) => "_yy",
            Some(2) => "_zz",
            _ => "",
        };
        let region = match self.region {
            StackRegion::LeftExterior => "the left exterior".to_string(),
            StackRegion::RightExterior => "the right exterior".to_string(),
            StackRegion::Layer(layer) => format!("layer {}", layer.get()),
        };
        write!(
            f,
            "{kind} of {symbol}{entry} of {region} at k0 = {} {:+}i cm^-1",
            self.k0.re, self.k0.im
        )
    }
}
/// Declared singularities of a problem's regions and the regions that
/// declare none, whose analyticity remains the caller's assumption.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialSingularities {
    pub known: Vec<KnownSingularity>,
    pub undeclared: Vec<StackRegion>,
}
/// Whether a declared point of `quantity` on `axis` is singular for `channel`.
/// The scalar channel's field obeys `(1/t) ψ″ + (v k0² − β²/n) ψ = 0`, with
/// `t, n` = ε_xx, ε_zz (TM) or μ_xx, μ_zz (TE): a pole of any entry and a zero
/// of `t` or `n` are singular, a zero of `v` is not. Every zero counts for the
/// coupled channel.
fn singular_for(
    channel: lamina_core::Channel,
    kind: SingularityKind,
    quantity: ConstitutiveQuantity,
    axis: Option<usize>,
) -> bool {
    use lamina_core::Polarisation::*;
    let divides = match channel.polarisation() {
        None => return true,
        Some(TransverseMagnetic) => ConstitutiveQuantity::Permittivity,
        Some(TransverseElectric) => ConstitutiveQuantity::Permeability,
    };
    kind == SingularityKind::Pole || (quantity == divides && axis != Some(1))
}
pub(crate) fn known_singularities(
    stack: &ModeStack,
    channel: lamina_core::Channel,
) -> MaterialSingularities {
    let mut known = Vec::new();
    let mut undeclared = Vec::new();
    for (region, material) in material::regions(stack) {
        match material.spectral_singularities() {
            None => undeclared.push(region),
            Some(points) => known.extend(
                points
                    .into_iter()
                    .filter(|p| singular_for(channel, p.kind, p.quantity, p.axis))
                    .map(|p| KnownSingularity {
                        region,
                        k0: p.k0,
                        kind: p.kind,
                        quantity: p.quantity,
                        axis: p.axis,
                    }),
            ),
        }
    }
    MaterialSingularities { known, undeclared }
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
    /// Exclusions inside the box this domain was cut from, which the domain
    /// abuts but does not overlap; see
    /// [`FrequencyProblem::domains`](crate::FrequencyProblem::domains).
    pub holes: Vec<FrequencyExclusion>,
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
    #[error(
        "frequency chart intersects declared exclusion: {0:?}; exclusions inside a box \
         need FrequencyProblem::domains"
    )]
    Excluded(FrequencyExclusion),
    #[error(
        "the evaluation region holds a {0}, where the characteristic is not \
         holomorphic; exclude it (FrequencyProblem::singularity_exclusions) and cut \
         the box around it (FrequencyProblem::domains)"
    )]
    KnownSingularity(KnownSingularity),
    #[error("the frequency reference lies inside an exclusion of the box")]
    ReferenceExcluded,
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
fn interiors_overlap(a: ConformalRegion<f64>, b: ConformalRegion<f64>) -> bool {
    let (a0, a1) = a.real_bounds();
    let (a2, a3) = a.imaginary_bounds();
    let (b0, b1) = b.real_bounds();
    let (b2, b3) = b.imaginary_bounds();
    a0 < b1 && b0 < a1 && a2 < b3 && b2 < a3
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
        for e in self.evidence.exclusions.iter().chain(&self.evidence.holes) {
            if matches!(e.reason,ExclusionReason::ExteriorThreshold{side} if side>1) {
                return Err(FrequencyError::InvalidExclusion);
            }
        }
        for e in &self.evidence.exclusions {
            if overlaps(region, e.region.0) {
                return Err(FrequencyError::Excluded(e.clone()));
            }
        }
        // A hole is abutted, not overlapped: compare interiors with the
        // requested region, not the halo.
        for e in &self.evidence.holes {
            if interiors_overlap(self.region, e.region.0) {
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
    /// The declared singularities of this problem's regions for its channel.
    pub fn known_singularities(&self) -> MaterialSingularities {
        known_singularities(&self.context.stack, self.context.channel)
    }
    /// One square exclusion of half-width `half_width` (cm^-1) around each
    /// known singularity whose square meets `bounds`.
    pub fn singularity_exclusions(
        &self,
        bounds: FrequencyRegion,
        half_width: f64,
    ) -> Result<Vec<FrequencyExclusion>, FrequencyError> {
        if !half_width.is_finite() || half_width <= 0.0 {
            return Err(FrequencyError::InvalidExclusion);
        }
        let mut exclusions: Vec<FrequencyExclusion> = Vec::new();
        for point in self.known_singularities().known {
            let k = point.k0;
            let region = FrequencyRegion::new(
                k.re - half_width,
                k.re + half_width,
                k.im - half_width,
                k.im + half_width,
            )
            .map_err(|_| FrequencyError::InvalidExclusion)?;
            if !overlaps(region.0, bounds.0) || exclusions.iter().any(|e| e.region == region) {
                continue;
            }
            exclusions.push(FrequencyExclusion {
                region,
                reason: ExclusionReason::KnownSingularity(point),
            });
        }
        Ok(exclusions)
    }
    fn resolve_reference(
        &self,
        input: &FrequencyDomainInput,
    ) -> Result<(ExteriorReferenceValues, RadiationProvenance), FrequencyError> {
        input.continuation.validate()?;
        Ok(match input.reference {
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
        })
    }
    fn chart(
        &self,
        region: ConformalRegion<f64>,
        reference: ExteriorReferenceValues,
        input: &FrequencyDomainInput,
        exclusions: Vec<FrequencyExclusion>,
        holes: Vec<FrequencyExclusion>,
    ) -> Result<FrequencyChart, FrequencyError> {
        let SearchCoordinate::Spectral {
            fixed_k_parallel: beta,
        } = self.context.coordinate
        else {
            return Err(FrequencyError::InvalidReference);
        };
        Ok(FrequencyChart {
            region,
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
                exclusions,
                holes,
            },
        })
    }
    pub fn frequency_domain(
        &self,
        input: FrequencyDomainInput,
        assumption: AnalyticAssumption,
    ) -> Result<FrequencyDomain<S>, FrequencyError> {
        let (reference, radiation) = self.resolve_reference(&input)?;
        let exclusions = input.exclusions.clone();
        self.build_domain(
            input.bounds.0,
            reference,
            radiation,
            &input,
            exclusions,
            vec![],
            assumption,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn build_domain(
        &self,
        r: ConformalRegion<f64>,
        reference: ExteriorReferenceValues,
        radiation: RadiationProvenance,
        input: &FrequencyDomainInput,
        exclusions: Vec<FrequencyExclusion>,
        holes: Vec<FrequencyExclusion>,
        assumption: AnalyticAssumption,
    ) -> Result<FrequencyDomain<S>, FrequencyError> {
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
        let chart = self.chart(r, reference, input, exclusions, holes)?;
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
        let frequency = search.dispersion().chart().evidence.clone();
        // Under the pole-free assumption, no declared singularity may lie in
        // the evaluation region outside the declared exclusions.
        let AnalyticAssumption::CallerAssumedHolomorphicPoleFree = assumption;
        let excluded = |k: C| {
            frequency
                .exclusions
                .iter()
                .chain(&frequency.holes)
                .any(|e| contains(e.region.0, k))
        };
        if let Some(point) = self
            .known_singularities()
            .known
            .into_iter()
            .find(|p| contains(evaluation, p.k0) && !excluded(p.k0))
        {
            return Err(FrequencyError::KnownSingularity(point));
        }
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
            frequency: Some(frequency),
        };
        Ok(ProblemDomain { search, provenance })
    }
    /// The requested box less the exclusions inside it, as rectangular
    /// domains in a stable order. Each tile's exterior reference is
    /// continued from the requested one through the tiles, never across an
    /// exclusion; the exclusions inside the box are each tile's holes.
    /// Without an exclusion inside the box this is [`Self::frequency_domain`].
    pub fn frequency_domains(
        &self,
        input: FrequencyDomainInput,
        assumption: AnalyticAssumption,
    ) -> Result<Vec<FrequencyDomain<S>>, FrequencyError> {
        let bounds = input.bounds.0;
        let (inside, outside): (Vec<_>, Vec<_>) = input
            .exclusions
            .iter()
            .cloned()
            .partition(|e| interiors_overlap(bounds, e.region.0));
        if inside.is_empty() {
            return Ok(vec![self.frequency_domain(input, assumption)?]);
        }
        let (reference, radiation) = self.resolve_reference(&input)?;
        let tiles = tiles(bounds, inside.iter().map(|e| e.region.0));
        let start = tiles
            .iter()
            .position(|t| contains(*t, reference.frequency))
            .ok_or(FrequencyError::ReferenceExcluded)?;
        // Continue κ from the reference to the start tile's centre, then from
        // tile to tile through the midpoint of their shared edge.
        let parent = self.chart(bounds, reference, &input, vec![], vec![])?;
        let at = |chart: &FrequencyChart, k: C| -> Result<[C; 2], FrequencyError> {
            Ok([
                chart.continuation_at(0, k)?.value,
                chart.continuation_at(1, k)?.value,
            ])
        };
        let mut references: Vec<Option<ExteriorReferenceValues>> = vec![None; tiles.len()];
        let first = centre_of(tiles[start]);
        references[start] = Some(ExteriorReferenceValues {
            frequency: first,
            kappa: at(&parent, first)?,
        });
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(a) = queue.pop_front() {
            let from = references[a].expect("visited");
            for b in 0..tiles.len() {
                if references[b].is_some() {
                    continue;
                }
                let Some(midpoint) = shared_edge_midpoint(tiles[a], tiles[b]) else {
                    continue;
                };
                let mut leg = parent.clone();
                leg.region = tiles[a];
                leg.evidence.reference = from;
                let crossing = ExteriorReferenceValues {
                    frequency: midpoint,
                    kappa: at(&leg, midpoint)?,
                };
                leg.region = tiles[b];
                leg.evidence.reference = crossing;
                let centre = centre_of(tiles[b]);
                references[b] = Some(ExteriorReferenceValues {
                    frequency: centre,
                    kappa: at(&leg, centre)?,
                });
                queue.push_back(b);
            }
        }
        tiles
            .iter()
            .zip(references)
            .map(|(tile, reference)| {
                self.build_domain(
                    *tile,
                    reference.ok_or(FrequencyError::InvalidBounds)?,
                    radiation.clone(),
                    &input,
                    outside.clone(),
                    inside.clone(),
                    assumption,
                )
            })
            .collect()
    }
}
fn centre_of(r: ConformalRegion<f64>) -> C {
    let (x0, x1) = r.real_bounds();
    let (y0, y1) = r.imaginary_bounds();
    C::new(0.5 * (x0 + x1), 0.5 * (y0 + y1))
}
/// The midpoint of the edge segment two rectangles share, if it has length.
fn shared_edge_midpoint(a: ConformalRegion<f64>, b: ConformalRegion<f64>) -> Option<C> {
    let ((a0, a1), (a2, a3)) = (a.real_bounds(), a.imaginary_bounds());
    let ((b0, b1), (b2, b3)) = (b.real_bounds(), b.imaginary_bounds());
    let (lo_x, hi_x) = (a0.max(b0), a1.min(b1));
    let (lo_y, hi_y) = (a2.max(b2), a3.min(b3));
    if (a1 == b0 || b1 == a0) && lo_y < hi_y {
        Some(C::new(if a1 == b0 { a1 } else { a0 }, 0.5 * (lo_y + hi_y)))
    } else if (a3 == b2 || b3 == a2) && lo_x < hi_x {
        Some(C::new(0.5 * (lo_x + hi_x), if a3 == b2 { a3 } else { a2 }))
    } else {
        None
    }
}
/// Rectangles covering `bounds` less `holes`: the grid of all their edges,
/// less the cells inside a hole, merged into horizontal runs and then equal
/// runs of consecutive rows; ordered by imaginary, then real, lower bound.
pub(crate) fn tiles(
    bounds: ConformalRegion<f64>,
    holes: impl Iterator<Item = ConformalRegion<f64>> + Clone,
) -> Vec<ConformalRegion<f64>> {
    let ((x0, x1), (y0, y1)) = (bounds.real_bounds(), bounds.imaginary_bounds());
    let lines = |outer: (f64, f64), edges: Vec<(f64, f64)>| {
        let mut lines = vec![outer.0, outer.1];
        for (lo, hi) in edges {
            lines.extend(
                [lo, hi]
                    .into_iter()
                    .filter(|v| *v > outer.0 && *v < outer.1),
            );
        }
        lines.sort_by(f64::total_cmp);
        lines.dedup();
        lines
    };
    let xs = lines((x0, x1), holes.clone().map(|h| h.real_bounds()).collect());
    let ys = lines(
        (y0, y1),
        holes.clone().map(|h| h.imaginary_bounds()).collect(),
    );
    let kept = |i: usize, j: usize| {
        let c = C::new(0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1]));
        !holes.clone().any(|h| {
            let ((h0, h1), (h2, h3)) = (h.real_bounds(), h.imaginary_bounds());
            c.re > h0 && c.re < h1 && c.im > h2 && c.im < h3
        })
    };
    // Open rectangles: (x_min, x_max, y_min, y_max).
    let mut done: Vec<[f64; 4]> = Vec::new();
    let mut open: Vec<[f64; 4]> = Vec::new();
    for j in 0..ys.len() - 1 {
        let mut runs = Vec::new();
        let mut i = 0;
        while i < xs.len() - 1 {
            if !kept(i, j) {
                i += 1;
                continue;
            }
            let start = i;
            while i < xs.len() - 1 && kept(i, j) {
                i += 1;
            }
            runs.push((xs[start], xs[i]));
        }
        let mut next = Vec::new();
        for (a, b) in runs {
            match open.iter().position(|r| r[0] == a && r[1] == b) {
                Some(k) => {
                    let mut r = open.remove(k);
                    r[3] = ys[j + 1];
                    next.push(r);
                }
                None => next.push([a, b, ys[j], ys[j + 1]]),
            }
        }
        done.append(&mut open);
        open = next;
    }
    done.append(&mut open);
    done.sort_by(|p, q| p[2].total_cmp(&q[2]).then(p[0].total_cmp(&q[0])));
    done.into_iter()
        .filter_map(|[a, b, c, d]| ConformalRegion::new(a, b, c, d).ok())
        .collect()
}

#[cfg(test)]
mod tests;

impl<S> FrequencyDomain<S> {
    pub(crate) fn frequency_chart(&self) -> &FrequencyChart {
        self.search.dispersion().chart()
    }
}

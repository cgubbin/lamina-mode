use super::*;
use crate::{
    conformal::{in_plane::ComplexInPlaneMap, wavenumber::ComplexWavenumberMap},
    problem as p,
};
use lamina_core::{
    Backend, ComplexPlane, Layer, PlaneWaveDeterminant, Polarisation,
    ProjectPlaneWaveModeDeterminant, Scatter2,
};
use std::sync::Arc;
type J = p::J;
type I = ComplexInPlaneMap<J>;
type W = ComplexWavenumberMap<J>;
type S<B> = p::Compiled<B>;
type D = p::direct::DirectChart;
/// Fixed real k0 propagation solver over a supplied core analytical stack.
#[derive(Clone, Debug)]
pub struct PropagationProblem<B = Scatter2> {
    inner: p::InPlaneProblem<B>,
}
/// Constant-exterior conformal spectral strategy. Constancy comes from explicit
/// construction evidence; analytical finite layers retain their dispersion.
#[derive(Clone, Debug)]
pub struct ConformalSpectralProblem<B = Scatter2> {
    inner: p::SpectralProblem<B>,
}
/// Direct k0 spectral strategy over an unchanged analytical core stack.
#[derive(Clone, Debug)]
pub struct FrequencyProblem<B = Scatter2> {
    inner: p::frequency::FrequencyProblem<B>,
}
#[derive(Clone, Debug)]
enum DomainKind<B> {
    Propagation(p::ProblemDomain<I, S<B>>),
    Direct(p::direct::DirectDomain<S<B>>),
    Conformal(p::ProblemDomain<W, S<B>>),
    Frequency(p::frequency::FrequencyDomain<S<B>>),
}
/// Problem-owned executable domain. Its private strategy and identity are checked
/// before evaluation; cloning retains the exact reference and halo.
#[derive(Clone, Debug)]
pub struct Domain<B = Scatter2> {
    inner: DomainKind<B>,
}
impl<B> Domain<B> {
    pub fn provenance(&self) -> Provenance {
        let inner = match &self.inner {
            DomainKind::Propagation(d) => d.provenance(),
            DomainKind::Direct(d) => d.provenance(),
            DomainKind::Conformal(d) => d.provenance(),
            DomainKind::Frequency(d) => d.provenance(),
        };
        Provenance {
            inner: inner.clone(),
        }
    }
}
#[derive(Clone, Debug)]
enum PlanKind<B> {
    Propagation(p::orchestration::ChartPlan<I, S<B>>),
    Direct(p::orchestration::ChartPlan<I, S<B>, D>),
    Conformal(p::orchestration::ChartPlan<W, S<B>>),
    Frequency(p::orchestration::ChartPlan<(), S<B>, p::frequency::FrequencyChart>),
}
/// Finite ordered domains of one physical problem and strategy. Use explicit
/// domains or an existing canonical conformal builder. No implicit global scope.
#[derive(Clone, Debug)]
pub struct Plan<B = Scatter2> {
    inner: PlanKind<B>,
}
/// Characteristic and native derivative from one seeded checked evaluation.
#[derive(Clone, Copy, Debug)]
pub struct Evaluation {
    pub value: Complex,
    pub derivative: Complex,
    pub coordinate_kind: CoordinateKind,
}
macro_rules! common_problem {
    ($name:ident, $($variant:ident),+) => {
        impl<B, E> $name<B>
        where
            B: Backend<J, ComplexPlane, Entries = E> + Clone,
            E: ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<J>>,
        {
            /// Explicit finite plan. Empty, repeated exact and foreign domains fail
            /// before numerical execution. Declared exclusions are coverage data.
            pub fn plan(
                &self,
                domains: Vec<Domain<B>>,
                exclusions: Vec<CoverageExclusion>,
            ) -> Result<Plan<B>, Diagnostic> {
                $(
                    if domains.iter().all(|d| matches!(d.inner, DomainKind::$variant(_))) {
                        let domains = domains
                            .into_iter()
                            .filter_map(|d| match d.inner {
                                DomainKind::$variant(d) => Some(d),
                                _ => None,
                            })
                            .collect();
                        return p::orchestration::ChartPlan::explicit(
                            self.inner.clone(),
                            domains,
                            exclusions,
                        )
                        .map(|inner| Plan {
                            inner: PlanKind::$variant(inner),
                        })
                        .map_err(|e| Diagnostic::new(Stage::Domain, e));
                    }
                )+
                Err(Diagnostic::new(Stage::Domain, StrategyMismatch))
            }
            /// Checked value and derivative on the domain's existing native branch.
            pub fn evaluate(
                &self,
                domain: &Domain<B>,
                native_coordinate: Complex,
            ) -> Result<Evaluation, Diagnostic> {
                let (b, coordinate_kind) = match &domain.inner {
                    $(
                        DomainKind::$variant(d) => (
                            self.inner.evaluate(d, native_coordinate).map_err(|e| {
                                let foreign = matches!(e, p::ExecutionError::ForeignDomain);
                                let mut diagnostic = Diagnostic::checked::<
                                    _,
                                    <B as Backend<J, ComplexPlane>>::Error,
                                >(Stage::Evaluation, e);
                                diagnostic.foreign |= foreign;
                                diagnostic
                            })?,
                            d.provenance().coordinate_kind(),
                        ),
                    )+
                    _ => return Err(Diagnostic::new(Stage::Evaluation, StrategyMismatch)),
                };
                Ok(Evaluation {
                    value: *b.value(),
                    derivative: *b.derivative(),
                    coordinate_kind,
                })
            }
        }
    };
}
// Generate explicit-plan calls directly, without a new backend/chart trait.
common_problem!(PropagationProblem, Propagation, Direct);
common_problem!(ConformalSpectralProblem, Conformal);
common_problem!(FrequencyProblem, Frequency);
impl PropagationProblem {
    pub fn new(
        stack: impl Into<ModeStack>,
        fixed_k0: f64,
        pol: Polarisation,
    ) -> Result<Self, Diagnostic> {
        Self::with_backend(stack, fixed_k0, pol, Scatter2::new())
    }
}
impl FrequencyProblem {
    pub fn new(
        stack: impl Into<ModeStack>,
        fixed_beta: f64,
        pol: Polarisation,
    ) -> Result<Self, Diagnostic> {
        Self::with_backend(stack, fixed_beta, pol, Scatter2::new())
    }
}
impl ConformalSpectralProblem {
    pub fn new(
        left: ConstantExterior,
        right: ConstantExterior,
        layers: impl IntoIterator<Item = Layer<impl Into<Region>, f64>>,
        fixed_beta: f64,
        pol: Polarisation,
    ) -> Result<Self, Diagnostic> {
        Self::with_backend(left, right, layers, fixed_beta, pol, Scatter2::new())
    }
}
macro_rules! conformal_domains {
    ($name:ident,$variant:ident) => {
        impl<B, E> $name<B>
        where
            B: Backend<J, ComplexPlane, Entries = E> + Clone,
            E: ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<J>>,
        {
            /// Dimensionless conformal rectangle. An explicit declaration covers
            /// materials and characteristic on the entire evaluation neighbourhood.
            pub fn domain(
                &self,
                bounds: ConformalRegion<f64>,
                reference: ReferenceSelection,
                config: ConformalChartConfig<f64>,
                assumption: AnalyticAssumption,
            ) -> Result<Domain<B>, Diagnostic> {
                self.inner
                    .domain(bounds, reference, config, assumption)
                    .map(|d| Domain {
                        inner: DomainKind::$variant(d),
                    })
                    .map_err(|e| Diagnostic::new(Stage::Domain, e))
            }
            /// Verify actual dielectric radiation branches at a real physical
            /// anchor, distinct from choosing the linear coordinate sign.
            pub fn radiation_domain(
                &self,
                bounds: ConformalRegion<f64>,
                reference: RadiationReference,
                config: ConformalChartConfig<f64>,
                assumption: AnalyticAssumption,
            ) -> Result<Domain<B>, Diagnostic> {
                self.inner
                    .domain_with_radiation_reference(bounds, reference.into(), config, assumption)
                    .map(|d| Domain {
                        inner: DomainKind::$variant(d),
                    })
                    .map_err(|e| Diagnostic::new(Stage::Domain, e))
            }
        }
    };
}
conformal_domains!(PropagationProblem, Propagation);
conformal_domains!(ConformalSpectralProblem, Conformal);
impl<B, E> PropagationProblem<B>
where
    B: Backend<J, ComplexPlane, Entries = E> + Clone,
    E: ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<J>>,
{
    pub fn with_backend(
        stack: impl Into<ModeStack>,
        fixed_k0: f64,
        pol: impl Into<lamina_core::Channel>,
        backend: B,
    ) -> Result<Self, Diagnostic> {
        p::InPlaneProblem::in_plane(stack, fixed_k0, pol, backend)
            .map(|inner| Self { inner })
            .map_err(|e| Diagnostic::new(Stage::Construction, e))
    }
    pub fn canonical_plan(
        &self,
        extent: InPlaneSearchExtent<f64>,
        reference: ReferenceSelection,
        config: ConformalChartConfig<f64>,
        assumption: AnalyticAssumption,
    ) -> Result<Plan<B>, Diagnostic> {
        self.inner
            .canonical_in_plane_plan(extent, reference, config, assumption)
            .map(|inner| Plan {
                inner: PlanKind::Propagation(inner),
            })
            .map_err(|e| Diagnostic::new(Stage::Domain, e))
    }
    /// The canonical strip of [`Self::canonical_plan`] and the same strip
    /// shifted by iπ, covering every exterior sheet: proper, leaky and
    /// incoming improper. Use it to follow a lossy mode across the light
    /// line, where it may leave the proper sheet; each point names its exterior sectors
    /// ([`BranchSignature::sectors`]).
    pub fn full_period_plan(
        &self,
        extent: InPlaneSearchExtent<f64>,
        reference: ReferenceSelection,
        config: ConformalChartConfig<f64>,
        assumption: AnalyticAssumption,
    ) -> Result<Plan<B>, Diagnostic> {
        self.inner
            .full_period_in_plane_plan(extent, reference, config, assumption)
            .map(|inner| Plan {
                inner: PlanKind::Propagation(inner),
            })
            .map_err(|e| Diagnostic::new(Stage::Domain, e))
    }
    /// Two domains over `real` bounds of Re z: the strip
    /// `|Im z| < π/2 − margin` and the same strip shifted by iπ, which
    /// together hold every exterior sheet; see [`Self::full_period_plan`].
    /// Unlike the canonical plans, this also serves identical exteriors,
    /// whose chart has no finite automatic truncation.
    pub fn full_period_domains(
        &self,
        real: [f64; 2],
        imaginary_margin: f64,
        reference: ReferenceSelection,
        config: ConformalChartConfig<f64>,
        assumption: AnalyticAssumption,
    ) -> Result<Vec<Domain<B>>, Diagnostic> {
        let half = std::f64::consts::FRAC_PI_2;
        if !(imaginary_margin > 0.0 && imaginary_margin < half) {
            return Err(Diagnostic::new(
                Stage::Domain,
                InPlaneSearchError::InvalidImaginaryMargin,
            ));
        }
        [0.0, std::f64::consts::PI]
            .into_iter()
            .map(|shift| {
                let bounds = ConformalRegion::new(
                    real[0],
                    real[1],
                    -half + imaginary_margin + shift,
                    half - imaginary_margin + shift,
                )
                .map_err(|e| Diagnostic::new(Stage::Domain, e))?;
                self.domain(bounds, reference, config, assumption)
            })
            .collect()
    }
    /// β rectangle in cm^-1 for a stack closed on both sides, by walls or PMLs,
    /// whose characteristic is entire and even in β: no sheet or radiation
    /// reference. The declaration covers the characteristic on the whole
    /// evaluation halo. A rectangle on which a PML's normalisation scale
    /// cannot be represented is refused ([`DirectChartError::PmlScale`]).
    pub fn direct_domain(
        &self,
        bounds: ConformalRegion<f64>,
        assumption: AnalyticAssumption,
    ) -> Result<Domain<B>, Diagnostic> {
        self.inner
            .direct_domain(bounds, assumption)
            .map(|d| Domain {
                inner: DomainKind::Direct(d),
            })
            .map_err(|e| Diagnostic::new(Stage::Domain, e))
    }
}
impl<B, E> ConformalSpectralProblem<B>
where
    B: Backend<J, ComplexPlane, Entries = E> + Clone,
    E: ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<J>>,
{
    pub fn with_backend(
        left: ConstantExterior,
        right: ConstantExterior,
        layers: impl IntoIterator<Item = Layer<impl Into<Region>, f64>>,
        fixed_beta: f64,
        pol: impl Into<lamina_core::Channel>,
        backend: B,
    ) -> Result<Self, Diagnostic> {
        p::SpectralProblem::spectral(left, right, layers, fixed_beta, pol, backend)
            .map(|inner| Self { inner })
            .map_err(|e| Diagnostic::new(Stage::Construction, e))
    }
    pub fn canonical_plan(
        &self,
        extent: WavenumberSearchExtent<f64>,
        references: [ReferenceSelection; 2],
        config: ConformalChartConfig<f64>,
        assumption: AnalyticAssumption,
    ) -> Result<Plan<B>, Diagnostic> {
        self.inner
            .canonical_spectral_plan(extent, references, config, assumption)
            .map(|inner| Plan {
                inner: PlanKind::Conformal(inner),
            })
            .map_err(|e| Diagnostic::new(Stage::Domain, e))
    }
}
impl<B, E> FrequencyProblem<B>
where
    B: Backend<J, ComplexPlane, Entries = E> + Clone,
    E: ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<J>>,
{
    pub fn with_backend(
        stack: impl Into<ModeStack>,
        fixed_beta: f64,
        pol: impl Into<lamina_core::Channel>,
        backend: B,
    ) -> Result<Self, Diagnostic> {
        p::frequency::FrequencyProblem::frequency(stack, fixed_beta, pol, backend)
            .map(|inner| Self { inner })
            .map_err(|e| Diagnostic::new(Stage::Construction, e))
    }
    /// The declared poles and zeros of the stack's regions at which this
    /// channel's characteristic is singular, and the regions that declare
    /// none. A domain refuses a box that holds a known one outside its
    /// exclusions.
    pub fn known_singularities(&self) -> MaterialSingularities {
        self.inner.known_singularities()
    }
    /// One square exclusion of half-width `half_width` (cm^-1) around each
    /// known singularity whose square meets `bounds`, for [`Self::domains`].
    pub fn singularity_exclusions(
        &self,
        bounds: FrequencyRegion,
        half_width: f64,
    ) -> Result<Vec<FrequencyExclusion>, Diagnostic> {
        self.inner
            .singularity_exclusions(bounds, half_width)
            .map_err(|e| Diagnostic::new(Stage::Domain, e))
    }
    /// The requested box less the exclusions inside it, as rectangular
    /// domains for one plan. Each tile's exterior reference is continued
    /// from the input's through adjacent tiles, never across an exclusion,
    /// and the exclusions inside the box are recorded as its holes. Without
    /// an exclusion inside the box this is one [`Self::domain`].
    pub fn domains(
        &self,
        input: FrequencyDomainInput,
        assumption: AnalyticAssumption,
    ) -> Result<Vec<Domain<B>>, Diagnostic> {
        self.inner
            .frequency_domains(input, assumption)
            .map(|domains| {
                domains
                    .into_iter()
                    .map(|d| Domain {
                        inner: DomainKind::Frequency(d),
                    })
                    .collect()
            })
            .map_err(|e| Diagnostic::new(Stage::Domain, e))
    }
    /// Physical k0 bounds in cm^-1. Caller explicitly declares characteristic
    /// analyticity and nonzero holomorphic exterior q on the evaluation region.
    /// A declared exclusion may not meet the evaluation region (see
    /// [`Self::domains`]), and neither may a known singularity outside it.
    pub fn domain(
        &self,
        input: FrequencyDomainInput,
        assumption: AnalyticAssumption,
    ) -> Result<Domain<B>, Diagnostic> {
        self.inner
            .frequency_domain(input, assumption)
            .map(|d| Domain {
                inner: DomainKind::Frequency(d),
            })
            .map_err(|e| Diagnostic::new(Stage::Domain, e))
    }
}
macro_rules! each_plan {
    ($self:ident,$plan:ident,$body:expr) => {
        match &$self.inner {
            PlanKind::Propagation($plan) => $body,
            PlanKind::Direct($plan) => $body,
            PlanKind::Conformal($plan) => $body,
            PlanKind::Frequency($plan) => $body,
        }
    };
}
impl<B, E> Plan<B>
where
    B: Backend<J, ComplexPlane, Entries = E> + Clone,
    E: ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<J>>,
{
    pub fn domains(&self) -> Vec<Provenance> {
        each_plan!(
            self,
            p,
            p.domains()
                .iter()
                .map(|d| Provenance {
                    inner: d.provenance().clone()
                })
                .collect()
        )
    }
    pub fn geometry(&self) -> &PlanGeometry {
        each_plan!(self, p, p.geometry())
    }
    pub fn exclusions(&self) -> &[CoverageExclusion] {
        each_plan!(self, p, p.exclusions())
    }
    /// One original result or typed failure per chart, in plan order. Numerical
    /// success preserves unresolved estimates and makes no acceptance claim.
    pub fn discover(&self, config: DiscoveryConfig) -> Result<Discovery, Diagnostic> {
        each_plan!(
            self,
            p,
            p.discover(config)
                .map(|batch| Discovery {
                    inner: Arc::new(batch.map_errors(|e| Diagnostic::checked::<
                        _,
                        <B as Backend<J, ComplexPlane>>::Error,
                    >(
                        Stage::Discovery, e
                    )))
                })
                .map_err(|e| Diagnostic::new(Stage::Discovery, e))
        )
    }
    /// Refinement runs once on the exact discovery's plan. The resulting report
    /// owns its outcomes and shares discovery once; it never borrows this plan.
    pub fn refine(
        &self,
        discovery: &Discovery,
        config: RefinementConfig,
        matching: MatchConfig,
    ) -> Result<SolveReport, Diagnostic> {
        each_plan!(self, p, {
            let report = p
                .refine(
                    discovery.inner.as_ref(),
                    config.resolve(p.wavenumber()),
                    matching.into(),
                )
                .map_err(|e| Diagnostic::new(Stage::Refinement, e))?;
            Ok(super::report::owned_report::<
                _,
                <B as Backend<J, ComplexPlane>>::Error,
            >(
                &report,
                discovery.clone(),
                config,
                matching,
                std::any::type_name::<B>(),
            ))
        })
    }
    /// Convenience discovery + refinement with the same algorithms and owned
    /// evidence as staged execution. Settings are validated before optical work.
    pub fn solve(
        &self,
        discovery: DiscoveryConfig,
        refinement: RefinementConfig,
        matching: MatchConfig,
    ) -> Result<SolveReport, Diagnostic> {
        let internal = refinement.resolve(each_plan!(self, p, p.wavenumber()));
        internal
            .validate()
            .map_err(|e| Diagnostic::new(Stage::Refinement, e))?;
        let internal: crate::problem::orchestration::MatchConfig = matching.into();
        internal
            .validate()
            .map_err(|e| Diagnostic::new(Stage::Refinement, e))?;
        let discovered = self.discover(discovery)?;
        self.refine(&discovered, refinement, matching)
    }
}

impl FrequencyProblem {
    pub(super) fn correct_frequency_seed(
        &self,
        domain: &Domain,
        seed: Complex,
        settings: RefinementConfig,
    ) -> Result<ModePoint, BranchFailure> {
        let DomainKind::Frequency(domain) = &domain.inner else {
            return Err(BranchFailure::InvalidRequest);
        };
        let settings = settings
            .resolve(domain.provenance().wavenumber())
            .validate()
            .map_err(|_| BranchFailure::InvalidRequest)?;
        let (root, signature, enclosure) = self.inner.corrected_point(domain, seed, settings)?;
        Ok(ModePoint {
            root,
            signature: Ok(signature),
            enclosure,
            chart: 0,
            candidate: 0,
            backend: std::any::type_name::<Scatter2>(),
        })
    }
}

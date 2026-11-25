//! Sequential finite chart orchestration. Coverage is relative to listed domains
//! and caller assumptions. Failures/exclusions are never erased by overlapping
//! success. Discovery owns results; refinement borrows them, avoiding references
//! into an owning result. Duplicate groups are numerical observations, not a
//! certified globally unique mode count or a material-domain certificate.
use super::*;
use crate::{
    conformal::{
        in_plane::region::{InPlaneSearchError, InPlaneSearchExtent},
        wavenumber::region::{WavenumberSearchError, WavenumberSearchExtent},
    },
    refinement::{
        AcceptedRoot, CandidateStatus, RefinementConfig, RefinementInputError, RefinementReport,
    },
};
use argloc::Rect;
use lamina_core::ScalarAlgebra;
use ndarray::arr0;

#[derive(Clone, Debug, PartialEq)]
pub enum CoverageExclusion {
    Frequency(frequency::FrequencyExclusion),
    OutsideFrequencyRegion {
        bounds: frequency::FrequencyRegion,
    },
    /// Outside these finite real-z truncation bounds is not searched.
    FiniteRealTruncation {
        min: f64,
        max: f64,
    },
    ImaginaryStrip {
        min: f64,
        max: f64,
    },
    /// Unsearched sheets in a full exterior period, independent of margins.
    /// Bounds describe imaginary z intervals; endpoints do not claim seam coverage.
    OmittedExteriorSheetSectors {
        selected_min: f64,
        selected_max: f64,
        exterior_period: f64,
        omitted: [(f64, f64); 2],
    },
    /// Caller-declared exclusions, not a computed geometric complement.
    Declared {
        description: String,
        region: Option<ConformalRegion<f64>>,
    },
}
#[derive(Clone, Debug, PartialEq)]
pub enum PlanGeometry {
    Explicit,
    ExplicitFrequency,
    CanonicalInPlane(InPlaneSearchExtent<f64>),
    CanonicalSpectral(WavenumberSearchExtent<f64>),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlanError {
    #[error("a chart plan must contain at least one domain")]
    Empty,
    #[error("chart domain belongs to another problem")]
    ForeignDomain,
    #[error("the same exact domain was supplied twice")]
    DuplicateDomain,
    #[error("discovery batch belongs to another plan")]
    ForeignBatch,
    #[error("invalid global discovery configuration")]
    InvalidConfig,
}
#[derive(Debug, thiserror::Error)]
pub enum CanonicalPlanError<G: std::error::Error + 'static, E: std::error::Error + 'static> {
    #[error("canonical geometry requires valid supported inputs or explicit domains: {0}")]
    Geometry(#[source] G),
    #[error("canonical chart construction failed: {0}")]
    Domain(#[source] ConformalChartError<E>),
    #[error("canonical plan construction failed: {0}")]
    Plan(#[from] PlanError),
}
#[derive(Clone, Debug)]
pub struct ChartPlan<M, S, H = ConformalChart<M, J>> {
    identity: Arc<()>,
    problem: ModalProblem<M, S>,
    domains: Vec<ProblemDomain<M, S, H>>,
    geometry: PlanGeometry,
    exclusions: Vec<CoverageExclusion>,
}
/// Safe discover settings are validated before any job. An opaque upstream
/// integrator config has no getters for validation; build it from these finite
/// fields instead, retaining its default quadrature rule and undeformed contour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscoveryConfig {
    pub score_tolerance: f64,
    pub zero_tolerance: f64,
    pub residual_tolerance: f64,
    pub boundary_shift_fraction: f64,
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
    pub max_iterations: usize,
    pub max_depth: usize,
    pub max_leaves: usize,
    pub max_function_evaluations: usize,
}
impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            score_tolerance: 1e-6,
            zero_tolerance: 1e-12,
            residual_tolerance: 1e-8,
            boundary_shift_fraction: 0.1,
            relative_tolerance: 1.49e-8,
            absolute_tolerance: 1.49e-8,
            max_iterations: 1000,
            max_depth: 32,
            max_leaves: 100000,
            max_function_evaluations: 5000,
        }
    }
}
impl DiscoveryConfig {
    pub(crate) fn argument_config(self) -> Result<ArgumentConfig<f64>, PlanError> {
        if [
            self.score_tolerance,
            self.zero_tolerance,
            self.residual_tolerance,
            self.relative_tolerance,
            self.absolute_tolerance,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
            || !self.boundary_shift_fraction.is_finite()
            || self.boundary_shift_fraction <= 0.0
            || self.boundary_shift_fraction >= 0.5
            || self.max_leaves == 0
            || self.max_function_evaluations == 0
        {
            return Err(PlanError::InvalidConfig);
        }
        let mut config = ArgumentConfig::new(self.score_tolerance);
        config.zero_tol = self.zero_tolerance;
        config.residual_tolerance = self.residual_tolerance;
        config.boundary_shift_fraction = self.boundary_shift_fraction;
        config.quad_tree.max_iter = self.max_iterations;
        config.quad_tree.max_depth = self.max_depth;
        config.quad_tree.max_leaves = self.max_leaves;
        config.integrator = config
            .integrator
            .with_relative_tolerance(self.relative_tolerance)
            .with_absolute_tolerance(self.absolute_tolerance)
            .with_max_function_evalutions(self.max_function_evaluations);
        Ok(config)
    }
}
#[derive(Debug)]
pub struct DiscoveryBatch<E> {
    identity: Arc<()>,
    entries: Vec<Result<ProblemSearchResult, E>>,
    config: DiscoveryConfig,
}
impl<E> DiscoveryBatch<E> {
    pub(crate) fn map_errors<T>(self, mut f: impl FnMut(E) -> T) -> DiscoveryBatch<T> {
        DiscoveryBatch {
            identity: self.identity,
            config: self.config,
            entries: self
                .entries
                .into_iter()
                .map(|e| e.map_err(&mut f))
                .collect(),
        }
    }
    pub fn entries(&self) -> &[Result<ProblemSearchResult, E>] {
        &self.entries
    }
    pub fn config(&self) -> DiscoveryConfig {
        self.config
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchTolerance {
    pub absolute: f64,
    pub relative: f64,
}
impl MatchTolerance {
    fn valid(self) -> bool {
        self.absolute.is_finite()
            && self.relative.is_finite()
            && self.absolute >= 0.0
            && self.relative >= 0.0
    }
    fn close(self, a: C, b: C) -> bool {
        let limit = self.absolute + self.relative * a.norm().max(b.norm());
        limit.is_finite() && (a - b).norm() <= limit
    }
}
/// Native absolute tolerances follow the plan's recorded units: dimensionless
/// conformal z or frequency k0 in cm^-1. Physical/kappa tolerances use cm^-1.
/// Relative tolerances are dimensionless. No uncertainty certificate is implied.
/// The legacy field name `conformal` also denotes native k0 in a frequency plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchConfig {
    pub conformal: MatchTolerance,
    pub physical: MatchTolerance,
    pub exterior: MatchTolerance,
}
impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            conformal: MatchTolerance {
                absolute: 1e-8,
                relative: 1e-8,
            },
            physical: MatchTolerance {
                absolute: 1e-8,
                relative: 1e-8,
            },
            exterior: MatchTolerance {
                absolute: 1e-8,
                relative: 1e-8,
            },
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("matching tolerances must be finite and nonnegative")]
pub struct InvalidMatchConfig;
impl MatchConfig {
    pub(crate) fn validate(self) -> Result<(), InvalidMatchConfig> {
        if self.conformal.valid() && self.physical.valid() && self.exterior.valid() {
            Ok(())
        } else {
            Err(InvalidMatchConfig)
        }
    }
}
#[derive(Debug, thiserror::Error)]
pub enum BatchRefinementError {
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error(transparent)]
    Refinement(#[from] RefinementInputError),
    #[error(transparent)]
    Matching(#[from] InvalidMatchConfig),
}
/// Physical coordinates of a point and the exterior wavevector of each open
/// side; a side closed by a wall or a PML carries none.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BranchSignature {
    pub k0: C,
    pub k_parallel: C,
    pub left: Option<C>,
    pub right: Option<C>,
}
impl BranchSignature {
    pub(crate) fn of(point: &crate::conformal::ConformalPoint<J>) -> Self {
        let value = |a: &J| a.value()[()];
        let exterior = point.exterior_wavevectors();
        Self {
            k0: value(point.coordinates().vacuum_angular_wavenumber()),
            k_parallel: value(point.coordinates().parallel_angular_wavenumber()),
            left: exterior.left().map(value),
            right: exterior.right().map(value),
        }
    }
    /// The exterior wavevectors in left/right order, `None` on a closed side.
    pub fn exteriors(&self) -> [Option<C>; 2] {
        [self.left, self.right]
    }
    /// Both exterior wavevectors, when both sides are open.
    pub fn open_exteriors(&self) -> Option<[C; 2]> {
        Some([self.left?, self.right?])
    }
}
#[derive(Debug)]
pub struct AcceptedObservation<E> {
    chart: usize,
    candidate: usize,
    root: AcceptedRoot,
    enclosure: Rect<f64>,
    signature: Result<BranchSignature, E>,
}
impl<E> AcceptedObservation<E> {
    pub fn chart(&self) -> usize {
        self.chart
    }
    pub fn candidate(&self) -> usize {
        self.candidate
    }
    pub fn root(&self) -> &AcceptedRoot {
        &self.root
    }
    pub fn enclosure(&self) -> Rect<f64> {
        self.enclosure
    }
    pub fn signature(&self) -> Result<&BranchSignature, &E> {
        self.signature.as_ref()
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DuplicateGroup {
    members: Vec<usize>,
}
impl DuplicateGroup {
    pub fn members(&self) -> &[usize] {
        &self.members
    }
    pub fn representative(&self) -> usize {
        self.members[0]
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ambiguity {
    UnsupportedOverlap,
    IncompatibleEnclosure,
    IncompatibleBranch,
    ExteriorSignAmbiguous,
    SignatureFailed,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReconciliationDiagnostic {
    pub first: usize,
    pub second: Option<usize>,
    pub reason: Ambiguity,
}
#[derive(Debug)]
pub struct ChartCoverage<'a, E, D> {
    index: usize,
    provenance: &'a DomainProvenance,
    discovery: &'a Result<ProblemSearchResult, D>,
    refinement: Option<RefinementReport<'a, E>>,
}
impl<'a, E, D> ChartCoverage<'a, E, D> {
    pub fn index(&self) -> usize {
        self.index
    }
    pub fn provenance(&self) -> &DomainProvenance {
        self.provenance
    }
    pub fn discovery(&self) -> &'a Result<ProblemSearchResult, D> {
        self.discovery
    }
    pub fn refinement(&self) -> Option<&RefinementReport<'a, E>> {
        self.refinement.as_ref()
    }
}
/// Ledger entries preserve original leaves rather than guessing that every leaf
/// denotes an unresolved cell. No global `complete` flag is inferred.
#[derive(Debug)]
pub struct BatchReport<'a, E, D> {
    batch: &'a DiscoveryBatch<D>,
    geometry: &'a PlanGeometry,
    exclusions: &'a [CoverageExclusion],
    charts: Vec<ChartCoverage<'a, E, D>>,
    observations: Vec<AcceptedObservation<E>>,
    groups: Vec<DuplicateGroup>,
    diagnostics: Vec<ReconciliationDiagnostic>,
    matching: MatchConfig,
}
impl<'a, E, D> BatchReport<'a, E, D> {
    pub fn discovery(&self) -> &'a DiscoveryBatch<D> {
        self.batch
    }
    pub fn geometry(&self) -> &PlanGeometry {
        self.geometry
    }
    pub fn exclusions(&self) -> &[CoverageExclusion] {
        self.exclusions
    }
    pub fn charts(&self) -> &[ChartCoverage<'a, E, D>] {
        &self.charts
    }
    pub fn observations(&self) -> &[AcceptedObservation<E>] {
        &self.observations
    }
    pub fn groups(&self) -> &[DuplicateGroup] {
        &self.groups
    }
    pub fn diagnostics(&self) -> &[ReconciliationDiagnostic] {
        &self.diagnostics
    }
    pub fn matching(&self) -> MatchConfig {
        self.matching
    }
}
type OpticalFailure<M, S> = ExecutionError<OpticalError<M, S>>;
type DiscoveryFailure<M, S> = ExecutionError<SearchError<M, S>>;
type BoundReport<'a, H, S> = BatchReport<
    'a,
    ExecutionError<NativeOpticalError<H, S>>,
    ExecutionError<NativeSearchError<H, S>>,
>;
impl<M, S, H> ChartPlan<M, S, H>
where
    M: Clone,
    H: AnalyticChart<J>,
    S: ModeDeterminantSolver<J> + Clone,
{
    /// Scope is exactly these finite domains; omitted space is not certified.
    pub fn explicit(
        problem: ModalProblem<M, S>,
        domains: Vec<ProblemDomain<M, S, H>>,
        exclusions: Vec<CoverageExclusion>,
    ) -> Result<Self, PlanError> {
        if domains.is_empty() {
            return Err(PlanError::Empty);
        }
        for (i, domain) in domains.iter().enumerate() {
            if !domain.provenance.belongs_to(&problem) {
                return Err(PlanError::ForeignDomain);
            }
            if domains[..i]
                .iter()
                .any(|prior| prior.provenance.same_domain(&domain.provenance))
            {
                return Err(PlanError::DuplicateDomain);
            }
        }
        let geometry = if domains[0].provenance.coordinate_kind() == CoordinateKind::Frequency {
            PlanGeometry::ExplicitFrequency
        } else {
            PlanGeometry::Explicit
        };
        let mut exclusions = exclusions;
        for domain in &domains {
            if let Some(evidence) = domain.provenance.frequency() {
                for exclusion in &evidence.exclusions {
                    let entry = CoverageExclusion::Frequency(exclusion.clone());
                    if !exclusions.contains(&entry) {
                        exclusions.push(entry);
                    }
                }
                exclusions.push(CoverageExclusion::OutsideFrequencyRegion {
                    bounds: frequency::FrequencyRegion::from_native(domain.provenance.requested()),
                });
            }
        }
        Ok(Self {
            identity: Arc::new(()),
            problem,
            domains,
            geometry,
            exclusions,
        })
    }
    pub fn domains(&self) -> &[ProblemDomain<M, S, H>] {
        &self.domains
    }
    pub fn geometry(&self) -> &PlanGeometry {
        &self.geometry
    }
    pub fn exclusions(&self) -> &[CoverageExclusion] {
        &self.exclusions
    }
    pub fn discover(
        &self,
        config: DiscoveryConfig,
    ) -> Result<DiscoveryBatch<ExecutionError<NativeSearchError<H, S>>>, PlanError> {
        let argument = config.argument_config()?;
        let entries = self
            .domains
            .iter()
            .map(|domain| self.problem.search(domain, argument.clone()))
            .collect();
        Ok(DiscoveryBatch {
            identity: self.identity.clone(),
            entries,
            config,
        })
    }
    pub fn refine<'a, D>(
        &'a self,
        batch: &'a DiscoveryBatch<D>,
        config: RefinementConfig,
        matching: MatchConfig,
    ) -> Result<BatchReport<'a, ExecutionError<NativeOpticalError<H, S>>, D>, BatchRefinementError>
    {
        config.validate()?;
        matching.validate()?;
        if !Arc::ptr_eq(&self.identity, &batch.identity) {
            return Err(PlanError::ForeignBatch.into());
        }
        let mut charts = Vec::new();
        let mut observations = Vec::new();
        for (index, (domain, discovery)) in self.domains.iter().zip(&batch.entries).enumerate() {
            let refinement = match discovery {
                Ok(result) => Some(self.problem.refine(domain, result, config)?),
                Err(_) => None,
            };
            if let Some(report) = &refinement {
                for outcome in report.outcomes() {
                    if let CandidateStatus::Accepted(root) = outcome.status() {
                        let z = root.evidence().z;
                        let signature = domain
                            .search
                            .dispersion()
                            .chart()
                            .map(J::lift_constant(arr0(z)))
                            .map_err(|error| {
                                ExecutionError::Optical(Box::new(ChartDispersionError::Chart(
                                    error,
                                )))
                            });
                        let signature = signature.map(|point| BranchSignature::of(&point));
                        observations.push(AcceptedObservation {
                            chart: index,
                            candidate: outcome.index(),
                            root: (**root).clone(),
                            enclosure: report.discovery().result().estimates()[outcome.index()]
                                .estimate()
                                .enclosure,
                            signature,
                        });
                    }
                }
            }
            charts.push(ChartCoverage {
                index,
                provenance: &domain.provenance,
                discovery,
                refinement,
            });
        }
        let (groups, diagnostics) = reconcile(&observations, matching);
        Ok(BatchReport {
            batch,
            geometry: &self.geometry,
            exclusions: &self.exclusions,
            charts,
            observations,
            groups,
            diagnostics,
            matching,
        })
    }
}
// A constant jet suffices for a physical branch signature: no optical solve or
// exterior principal-root reconstruction is performed during reconciliation.
impl<S> ModalProblem<ComplexInPlaneMap<J>, S>
where
    S: ModeDeterminantSolver<J> + Clone,
{
    pub fn canonical_in_plane_plan(
        &self,
        extent: InPlaneSearchExtent<f64>,
        selection: ReferenceSelection,
        config: ConformalChartConfig<f64>,
        assumption: AnalyticAssumption,
    ) -> Result<
        ChartPlan<ComplexInPlaneMap<J>, S>,
        CanonicalPlanError<
            InPlaneSearchError,
            crate::conformal::in_plane::region::InPlaneRegionError,
        >,
    > {
        let bounds = self
            .map
            .search_region(extent)
            .map_err(CanonicalPlanError::Geometry)?
            .into_inner();
        let domain = self
            .domain(bounds, selection, config, assumption)
            .map_err(CanonicalPlanError::Domain)?;
        let mut plan = ChartPlan::explicit(
            self.clone(),
            vec![domain],
            canonical_exclusions(bounds, extent.imaginary_margin, false),
        )?;
        plan.geometry = PlanGeometry::CanonicalInPlane(extent);
        Ok(plan)
    }
}
impl<S> ModalProblem<ComplexWavenumberMap<J>, S>
where
    S: ModeDeterminantSolver<J> + Clone,
{
    pub fn canonical_spectral_plan(
        &self,
        extent: WavenumberSearchExtent<f64>,
        selections: [ReferenceSelection; 2],
        config: ConformalChartConfig<f64>,
        assumption: AnalyticAssumption,
    ) -> Result<
        ChartPlan<ComplexWavenumberMap<J>, S>,
        CanonicalPlanError<
            WavenumberSearchError,
            crate::conformal::wavenumber::region::WavenumberRegionError,
        >,
    > {
        let regions = self
            .map
            .search_regions(extent)
            .map_err(CanonicalPlanError::Geometry)?
            .into_regions();
        let mut domains = Vec::new();
        for (bounds, selection) in regions.into_iter().zip(selections) {
            domains.push(
                self.domain(bounds, selection, config, assumption)
                    .map_err(CanonicalPlanError::Domain)?,
            )
        }
        let mut plan = ChartPlan::explicit(
            self.clone(),
            domains,
            canonical_exclusions(regions[0], extent.imaginary_margin, true),
        )?;
        plan.geometry = PlanGeometry::CanonicalSpectral(extent);
        Ok(plan)
    }
}
fn canonical_exclusions(
    bounds: ConformalRegion<f64>,
    margin: f64,
    spectral: bool,
) -> Vec<CoverageExclusion> {
    let (min, max) = bounds.real_bounds();
    let half = std::f64::consts::FRAC_PI_2;
    let mut exclusions = vec![
        CoverageExclusion::FiniteRealTruncation { min, max },
        CoverageExclusion::ImaginaryStrip {
            min: -half,
            max: -half + margin,
        },
        CoverageExclusion::ImaginaryStrip {
            min: half - margin,
            max: half,
        },
        CoverageExclusion::OmittedExteriorSheetSectors {
            selected_min: -half,
            selected_max: half,
            exterior_period: 2.0 * std::f64::consts::PI,
            omitted: [(-std::f64::consts::PI, -half), (half, std::f64::consts::PI)],
        },
    ];
    if spectral {
        exclusions.push(CoverageExclusion::ImaginaryStrip {
            min: -margin,
            max: margin,
        })
    }
    exclusions
}
fn inside(rect: Rect<f64>, z: C) -> bool {
    z.re > rect.x_min && z.re < rect.x_max && z.im > rect.y_min && z.im < rect.y_max
}
fn overlapping(a: ConformalRegion<f64>, b: ConformalRegion<f64>) -> bool {
    let (ax, bx) = a.real_bounds();
    let (ay, by) = a.imaginary_bounds();
    let (cx, dx) = b.real_bounds();
    let (cy, dy) = b.imaginary_bounds();
    ax.max(cx) < bx.min(dx) && ay.max(cy) < by.min(dy)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compatibility {
    Separate,
    Compatible,
    Ambiguous(Ambiguity),
}
fn compatibility<E>(
    a: &AcceptedObservation<E>,
    b: &AcceptedObservation<E>,
    cfg: MatchConfig,
) -> Compatibility {
    if a.chart == b.chart {
        return Compatibility::Separate;
    }
    if !cfg.physical.close(a.root.coordinate(), b.root.coordinate()) {
        return Compatibility::Separate;
    }
    if !overlapping(
        a.root.provenance().requested(),
        b.root.provenance().requested(),
    ) || !cfg
        .conformal
        .close(a.root.evidence().z, b.root.evidence().z)
    {
        return Compatibility::Ambiguous(Ambiguity::UnsupportedOverlap);
    }
    if ![a.root.evidence().z, b.root.evidence().z]
        .iter()
        .all(|z| inside(a.enclosure, *z) && inside(b.enclosure, *z))
    {
        return Compatibility::Ambiguous(Ambiguity::IncompatibleEnclosure);
    }
    let (Ok(a), Ok(b)) = (&a.signature, &b.signature) else {
        return Compatibility::Separate;
    };
    if !cfg.physical.close(a.k0, b.k0) || !cfg.physical.close(a.k_parallel, b.k_parallel) {
        return Compatibility::Ambiguous(Ambiguity::IncompatibleBranch);
    }
    // Both observations belong to one problem, so they share their open sides.
    let open = a.exteriors().into_iter().zip(b.exteriors());
    for (left, right) in open.filter_map(|(a, b)| Some((a?, b?))) {
        if !cfg.exterior.close(left, right) {
            return Compatibility::Ambiguous(Ambiguity::IncompatibleBranch);
        }
        if cfg.exterior.close(left, -right) {
            return Compatibility::Ambiguous(Ambiguity::ExteriorSignAmbiguous);
        }
    }
    Compatibility::Compatible
}
fn reconcile<E>(
    observations: &[AcceptedObservation<E>],
    cfg: MatchConfig,
) -> (Vec<DuplicateGroup>, Vec<ReconciliationDiagnostic>) {
    let mut diagnostics = Vec::new();
    for (i, observation) in observations.iter().enumerate() {
        if observation.signature.is_err() {
            diagnostics.push(ReconciliationDiagnostic {
                first: i,
                second: None,
                reason: Ambiguity::SignatureFailed,
            })
        }
    }
    for i in 0..observations.len() {
        for j in i + 1..observations.len() {
            if let Compatibility::Ambiguous(reason) =
                compatibility(&observations[i], &observations[j], cfg)
            {
                diagnostics.push(ReconciliationDiagnostic {
                    first: i,
                    second: Some(j),
                    reason,
                })
            }
        }
    }
    let mut groups: Vec<DuplicateGroup> = Vec::new();
    for index in 0..observations.len() {
        if let Some(group) = groups.iter_mut().find(|group| {
            group.members.iter().all(|prior| {
                compatibility(&observations[*prior], &observations[index], cfg)
                    == Compatibility::Compatible
            })
        }) {
            group.members.push(index)
        } else {
            groups.push(DuplicateGroup {
                members: vec![index],
            })
        }
    }
    (groups, diagnostics)
}
#[cfg(test)]
mod tests;

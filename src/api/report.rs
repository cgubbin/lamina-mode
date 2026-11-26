use super::*;
use crate::{problem as p, refinement as r};
use std::sync::Arc;

/// Exact source domain, assumptions and branch data. Identity survives cloning.
#[derive(Clone, Debug)]
pub struct Provenance {
    pub(super) inner: p::DomainProvenance,
}
impl Provenance {
    pub fn strategy(&self) -> Strategy {
        match self.direction() {
            SearchCoordinate::InPlane { .. } => Strategy::Propagation,
            SearchCoordinate::Spectral { .. }
                if self.coordinate_kind() == CoordinateKind::Conformal =>
            {
                Strategy::ConformalSpectral
            }
            _ => Strategy::DirectFrequency,
        }
    }

    pub fn same_domain(&self, other: &Self) -> bool {
        self.inner.same_domain(&other.inner)
    }
    pub fn coordinate_kind(&self) -> CoordinateKind {
        self.inner.coordinate_kind()
    }
    pub fn units(&self) -> &'static str {
        self.coordinate_kind().units()
    }
    pub fn requested_bounds(&self) -> ConformalRegion<f64> {
        self.inner.requested()
    }
    pub fn evaluation_bounds(&self) -> ConformalRegion<f64> {
        self.inner.evaluation()
    }
    pub fn reference(&self) -> Complex {
        self.inner.reference()
    }
    pub fn reference_coordinate(&self) -> Complex {
        self.inner.reference_coordinate()
    }
    pub fn selection(&self) -> ReferenceSelection {
        self.inner.selection()
    }
    pub fn assumption(&self) -> AnalyticAssumption {
        self.inner.assumption()
    }
    pub fn frequency(&self) -> Option<&FrequencyEvidence> {
        self.inner.frequency()
    }
    pub fn radiation(&self) -> RadiationProvenance {
        self.inner.radiation().into()
    }
    pub fn direction(&self) -> SearchCoordinate {
        self.inner.problem().coordinate()
    }
    /// The polarisation of a scalar channel; `None` for the coupled one.
    pub fn polarisation(&self) -> Option<lamina_core::Polarisation> {
        self.inner.problem().polarisation()
    }
    /// The field subspace solved.
    pub fn channel(&self) -> lamina_core::Channel {
        self.inner.problem().channel()
    }
    pub fn stack(&self) -> &ModeStack {
        self.inner.problem().stack()
    }
}
/// Owned discovery, shared once by reports rather than copying trees per point.
#[derive(Clone, Debug)]
pub struct Discovery {
    pub(super) inner: Arc<p::orchestration::DiscoveryBatch<Diagnostic>>,
}
impl Discovery {
    pub fn config(&self) -> DiscoveryConfig {
        self.inner.config()
    }
    pub fn entries(
        &self,
    ) -> impl ExactSizeIterator<Item = Result<ChartDiscovery<'_>, &Diagnostic>> {
        self.inner
            .entries()
            .iter()
            .map(|entry| entry.as_ref().map(|inner| ChartDiscovery { inner }))
    }
}
/// Borrowed inspection of an original successful chart discovery, never acceptance.
#[derive(Clone, Copy, Debug)]
pub struct ChartDiscovery<'a> {
    inner: &'a p::ProblemSearchResult,
}
impl ChartDiscovery<'_> {
    pub fn provenance(&self) -> Provenance {
        Provenance {
            inner: self.inner.provenance().clone(),
        }
    }
    pub fn estimates(&self) -> impl ExactSizeIterator<Item = Estimate<'_>> {
        self.inner
            .result()
            .estimates()
            .iter()
            .map(|inner| Estimate { inner })
    }
    pub fn leaves(&self) -> &[ArgumentLeaf<f64>] {
        self.inner.result().leaves()
    }
    pub fn summary(&self) -> &RunSummary<f64> {
        self.inner.result().summary()
    }
    pub fn termination(&self) -> &Termination {
        self.inner.result().termination()
    }
}
/// Native/physical projection of a discovery estimate, including unresolved clusters.
#[derive(Clone, Copy, Debug)]
pub struct Estimate<'a> {
    inner: &'a crate::search::ModalEstimate<Complex>,
}
impl Estimate<'_> {
    pub fn raw(&self) -> &SingularPointEstimate<Complex> {
        self.inner.estimate()
    }
    pub fn native_coordinate(&self) -> Complex {
        self.inner.mode().z()
    }
    pub fn physical_coordinate(&self) -> Complex {
        self.inner.mode().coordinate()
    }
    pub fn residual(&self) -> Complex {
        self.inner.mode().residual()
    }
}
/// Fresh checked numerical evidence. Derivative and correction are native-coordinate
/// quantities, with units recorded in source provenance; residual uses core's gauge.
#[derive(Clone, Copy, Debug)]
pub struct NumericalEvidence {
    pub native_coordinate: Complex,
    pub value: Complex,
    pub derivative: Complex,
    pub correction: Option<f64>,
    pub residual_limit: f64,
    pub correction_limit: f64,
}
impl From<r::NumericalEvidence> for NumericalEvidence {
    fn from(v: r::NumericalEvidence) -> Self {
        Self {
            native_coordinate: v.z,
            value: v.value,
            derivative: v.derivative,
            correction: v.correction,
            residual_limit: v.residual_limit,
            correction_limit: v.correction_limit,
        }
    }
}
/// One outcome per original estimate. Accepted point values live in the report's
/// observations and are linked by chart and candidate indices.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum CandidateStatus {
    Accepted,
    UnresolvedCluster,
    IneligibleEstimate,
    InvalidEnclosure,
    BoundaryLimited,
    ZeroDerivative,
    NonFiniteCorrection,
    NonFiniteEvaluation,
    NonFiniteThreshold,
    NoProgress,
    Stagnated,
    BudgetExceeded,
    EvaluationFailed(Diagnostic),
    ProjectionFailed(Diagnostic),
}
#[derive(Clone, Debug)]
pub struct CandidateOutcome {
    pub index: usize,
    pub status: CandidateStatus,
    pub last_valid: Option<NumericalEvidence>,
    pub work: RefinementWork,
}
/// Numerical point value, not a global mode/branch identity or normalized QNM.
/// Can be cloned independently of the report; source context stays shared.
#[derive(Clone, Debug)]
pub struct ModePoint {
    pub(super) root: r::AcceptedRoot,
    pub(super) backend: &'static str,
    pub(super) chart: usize,
    pub(super) candidate: usize,
    pub(super) enclosure: Rect<f64>,
    pub(super) signature: Result<BranchSignature, Diagnostic>,
}
impl ModePoint {
    /// Selected core backend type name, for diagnostics rather than serialization.
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    pub fn chart(&self) -> usize {
        self.chart
    }
    pub fn candidate(&self) -> usize {
        self.candidate
    }
    pub fn native_coordinate(&self) -> Complex {
        self.root.evidence().z
    }
    pub fn physical_coordinate(&self) -> Complex {
        self.root.coordinate()
    }
    pub fn evidence(&self) -> NumericalEvidence {
        (*self.root.evidence()).into()
    }
    pub fn provenance(&self) -> Provenance {
        Provenance {
            inner: self.root.provenance().clone(),
        }
    }
    pub fn enclosure(&self) -> Rect<f64> {
        self.enclosure
    }
    pub fn config(&self) -> RefinementConfig {
        self.root.config().into()
    }
    pub fn work(&self) -> RefinementWork {
        self.root.work()
    }
    pub fn signature(&self) -> Result<&BranchSignature, &Diagnostic> {
        self.signature.as_ref()
    }
    /// Optional interpretation using retained branch data and actual source
    /// materials. Failure preserves this accepted point. Spectral named types and
    /// complex-frequency flux remain unsupported; no normalization is computed.
    pub fn describe(&self, config: InterpretationConfig) -> Result<PhysicalDetails, Diagnostic> {
        config
            .validate()
            .map_err(|e| Diagnostic::new(Stage::Interpretation, e))?;
        let signature = *self
            .signature
            .as_ref()
            .map_err(|e| e.at_stage(Stage::Interpretation))?;
        p::physical::from_signature(signature, self.root.provenance(), config)
            .map_err(|e| Diagnostic::new(Stage::Interpretation, e))
    }
}
#[derive(Clone, Debug)]
pub struct ChartOutcomes {
    pub provenance: Provenance,
    pub candidates: Option<Vec<CandidateOutcome>>,
}
/// Why part of a plan's charts lacks complete evidence.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum UnresolvedCause {
    /// Discovery of the chart failed: none of its region was searched.
    Discovery(Diagnostic),
    /// Discovery stopped before it converged.
    Stopped(Termination),
    /// A discovery leaf kept a score above the tolerance, or a non-finite one.
    Leaf { score: f64 },
    /// Refinement did not accept this estimate.
    Candidate {
        index: usize,
        status: CandidateStatus,
    },
}
/// A region of one chart, in its native coordinates, without complete evidence.
#[derive(Clone, Debug)]
pub struct UnresolvedRegion {
    pub chart: usize,
    pub bounds: Rect<f64>,
    pub cause: UnresolvedCause,
}
/// Owned result of one finite plan. Discovery is shared once; chart/candidate and
/// group indices refer to the original stable order. A failed or partial chart
/// stays in the report and is listed by [`SolveReport::unresolved`].
#[derive(Clone, Debug)]
pub struct SolveReport {
    backend: &'static str,
    pub(super) discovery: Discovery,
    pub(super) charts: Vec<ChartOutcomes>,
    pub(super) points: Vec<ModePoint>,
    pub(super) groups: Vec<DuplicateGroup>,
    pub(super) diagnostics: Vec<ReconciliationDiagnostic>,
    pub(super) exclusions: Vec<CoverageExclusion>,
    pub(super) geometry: PlanGeometry,
    pub(super) refinement: RefinementConfig,
    pub(super) matching: MatchConfig,
}
impl SolveReport {
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    pub fn discovery(&self) -> &Discovery {
        &self.discovery
    }
    pub fn charts(&self) -> &[ChartOutcomes] {
        &self.charts
    }
    pub fn observations(&self) -> &[ModePoint] {
        &self.points
    }
    pub fn groups(&self) -> &[DuplicateGroup] {
        &self.groups
    }
    pub fn diagnostics(&self) -> &[ReconciliationDiagnostic] {
        &self.diagnostics
    }
    pub fn exclusions(&self) -> &[CoverageExclusion] {
        &self.exclusions
    }
    pub fn geometry(&self) -> &PlanGeometry {
        &self.geometry
    }
    pub fn refinement_config(&self) -> RefinementConfig {
        self.refinement
    }
    pub fn matching_config(&self) -> MatchConfig {
        self.matching
    }
    /// Every failed chart, stopped discovery, unresolved leaf and unaccepted
    /// estimate, in chart order.
    pub fn unresolved(&self) -> Vec<UnresolvedRegion> {
        let tolerance = self.discovery.config().score_tolerance;
        let mut unresolved = Vec::new();
        for (chart, (entry, outcomes)) in self.discovery.entries().zip(&self.charts).enumerate() {
            let region = outcomes.provenance.requested_bounds();
            let ((x_min, x_max), (y_min, y_max)) =
                (region.real_bounds(), region.imaginary_bounds());
            let requested = Rect {
                x_min,
                x_max,
                y_min,
                y_max,
            };
            let mut push = |bounds, cause| {
                unresolved.push(UnresolvedRegion {
                    chart,
                    bounds,
                    cause,
                })
            };
            let entry = match entry {
                Ok(entry) => entry,
                Err(diagnostic) => {
                    push(requested, UnresolvedCause::Discovery(diagnostic.clone()));
                    continue;
                }
            };
            if *entry.termination() != Termination::Converged {
                push(requested, UnresolvedCause::Stopped(*entry.termination()));
            }
            for leaf in entry.leaves() {
                let score = leaf.data.score;
                if !score.is_finite() || score > tolerance {
                    push(leaf.bounds, UnresolvedCause::Leaf { score });
                }
            }
            for outcome in outcomes.candidates.iter().flatten() {
                if !matches!(outcome.status, CandidateStatus::Accepted) {
                    let bounds = entry
                        .estimates()
                        .nth(outcome.index)
                        .map_or(requested, |estimate| estimate.raw().enclosure);
                    let cause = UnresolvedCause::Candidate {
                        index: outcome.index,
                        status: outcome.status.clone(),
                    };
                    push(bounds, cause);
                }
            }
        }
        unresolved
    }
    /// Numerical coverage of every chart's requested region at the stated
    /// tolerances: nothing unresolved and no reconciliation diagnostic. Exclusions
    /// outside the charts are not covered.
    pub fn is_complete(&self) -> bool {
        self.diagnostics.is_empty() && self.unresolved().is_empty()
    }
    pub fn describe_points(
        &self,
        config: InterpretationConfig,
    ) -> Vec<Result<PhysicalDetails, Diagnostic>> {
        self.points
            .iter()
            .map(|point| point.describe(config))
            .collect()
    }
}
pub(super) fn owned_report<
    E: std::error::Error + Clone + Send + Sync + 'static,
    BE: std::error::Error + 'static,
>(
    report: &p::orchestration::BatchReport<'_, E, Diagnostic>,
    discovery: Discovery,
    refinement: RefinementConfig,
    matching: MatchConfig,
    backend: &'static str,
) -> SolveReport {
    let charts = report
        .charts()
        .iter()
        .map(|chart| ChartOutcomes {
            provenance: Provenance {
                inner: chart.provenance().clone(),
            },
            candidates: chart.refinement().map(|refinement| {
                refinement
                    .outcomes()
                    .iter()
                    .map(|o| {
                        use r::CandidateStatus as I;
                        let status = match o.status() {
                            I::Accepted(_) => CandidateStatus::Accepted,
                            I::UnresolvedCluster => CandidateStatus::UnresolvedCluster,
                            I::IneligibleEstimate => CandidateStatus::IneligibleEstimate,
                            I::InvalidEnclosure => CandidateStatus::InvalidEnclosure,
                            I::BoundaryLimited => CandidateStatus::BoundaryLimited,
                            I::ZeroDerivative => CandidateStatus::ZeroDerivative,
                            I::NonFiniteCorrection => CandidateStatus::NonFiniteCorrection,
                            I::NonFiniteEvaluation => CandidateStatus::NonFiniteEvaluation,
                            I::NonFiniteThreshold => CandidateStatus::NonFiniteThreshold,
                            I::NoProgress => CandidateStatus::NoProgress,
                            I::Stagnated => CandidateStatus::Stagnated,
                            I::BudgetExceeded => CandidateStatus::BudgetExceeded,
                            I::EvaluationFailed(e) => CandidateStatus::EvaluationFailed(
                                Diagnostic::checked::<_, BE>(Stage::Refinement, e.clone()),
                            ),
                            I::ProjectionFailed(e) => CandidateStatus::ProjectionFailed(
                                Diagnostic::checked::<_, BE>(Stage::Refinement, e.clone()),
                            ),
                        };
                        CandidateOutcome {
                            index: o.index(),
                            status,
                            last_valid: o.last_valid().copied().map(Into::into),
                            work: o.work(),
                        }
                    })
                    .collect()
            }),
        })
        .collect();
    let points = report
        .observations()
        .iter()
        .map(|o| ModePoint {
            root: o.root().clone(),
            backend,
            chart: o.chart(),
            candidate: o.candidate(),
            enclosure: o.enclosure(),
            signature: o
                .signature()
                .copied()
                .map_err(|e| Diagnostic::checked::<_, BE>(Stage::Evaluation, e.clone())),
        })
        .collect();
    SolveReport {
        backend,
        discovery,
        charts,
        points,
        groups: report.groups().to_vec(),
        diagnostics: report.diagnostics().to_vec(),
        exclusions: report.exclusions().to_vec(),
        geometry: report.geometry().clone(),
        refinement,
        matching,
    }
}

//! Optional numerical polishing of isolated discovery estimates. Acceptance is
//! conditional on the retained caller assumption, not an isolation certificate.
//! Raw search results and coverage diagnostics remain untouched. Residual units
//! follow the core characteristic gauge; the fixed scale is never inferred from
//! a nearly zero starting value. Clusters require separate future work.
use crate::problem::{C, DomainProvenance, ProblemSearchResult};
use argloc::{
    FunctionBundle, Rect, SearchTarget, SingularPointEstimate, SingularPointEstimateKind,
};

/// Budgets are per candidate. max_backtracks is the number of halvings after
/// the full step (zero disables work); trials count even unevaluated proposals.
/// Evaluations include initial, evaluated trials, and fresh acceptance checks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefinementConfig {
    pub residual_atol: f64,
    pub residual_rtol: f64,
    pub residual_scale: f64,
    /// Native correction units: dimensionless z or k0 in cm^-1, per provenance.
    pub z_atol: f64,
    pub z_rtol: f64,
    pub max_updates: usize,
    pub max_backtracks: usize,
    pub max_evaluations: usize,
}
impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            residual_atol: 1e-10,
            residual_rtol: 0.0,
            residual_scale: 1.0,
            z_atol: 1e-10,
            z_rtol: 1e-10,
            max_updates: 32,
            max_backtracks: 12,
            max_evaluations: 256,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RefinementInputError {
    #[error("invalid refinement tolerances, scale, or non-finite threshold")]
    InvalidConfig,
    #[error("discovery and refinement belong to different physical problems")]
    ForeignProblem,
    #[error("discovery and refinement belong to different domains")]
    ForeignDomain,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct ValidatedConfig {
    config: RefinementConfig,
    residual_limit: f64,
}
impl RefinementConfig {
    pub(crate) fn validate(self) -> Result<ValidatedConfig, RefinementInputError> {
        let tolerances = [
            self.residual_atol,
            self.residual_rtol,
            self.z_atol,
            self.z_rtol,
        ];
        let limit = self.residual_atol + self.residual_rtol * self.residual_scale;
        if tolerances.iter().any(|x| !x.is_finite() || *x < 0.0)
            || !self.residual_scale.is_finite()
            || self.residual_scale <= 0.0
            || !limit.is_finite()
            || limit <= 0.0
            || !(self.z_atol + self.z_rtol).is_finite()
            || self.z_atol + self.z_rtol <= 0.0
        {
            return Err(RefinementInputError::InvalidConfig);
        }
        Ok(ValidatedConfig {
            config: self,
            residual_limit: limit,
        })
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefinementWork {
    pub updates: usize,
    pub trials: usize,
    pub evaluations: usize,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericalEvidence {
    pub z: C,
    pub value: C,
    pub derivative: C,
    pub correction: Option<f64>,
    pub residual_limit: f64,
    pub correction_limit: f64,
}
/// Constructed only after a fresh checked evaluation passes both criteria.
#[derive(Clone, Debug)]
pub struct AcceptedRoot {
    index: usize,
    evidence: NumericalEvidence,
    coordinate: C,
    provenance: DomainProvenance,
    config: RefinementConfig,
    work: RefinementWork,
}
impl AcceptedRoot {
    pub fn index(&self) -> usize {
        self.index
    }
    pub fn evidence(&self) -> &NumericalEvidence {
        &self.evidence
    }
    pub fn coordinate(&self) -> C {
        self.coordinate
    }
    pub fn provenance(&self) -> &DomainProvenance {
        &self.provenance
    }
    pub fn config(&self) -> RefinementConfig {
        self.config
    }
    pub fn work(&self) -> RefinementWork {
        self.work
    }
}
#[derive(Debug)]
pub enum CandidateStatus<E> {
    Accepted(Box<AcceptedRoot>),
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
    EvaluationFailed(E),
    ProjectionFailed(E),
}
#[derive(Debug)]
pub struct CandidateOutcome<E> {
    index: usize,
    status: CandidateStatus<E>,
    last_valid: Option<NumericalEvidence>,
    work: RefinementWork,
}
impl<E> CandidateOutcome<E> {
    pub fn index(&self) -> usize {
        self.index
    }
    pub fn status(&self) -> &CandidateStatus<E> {
        &self.status
    }
    pub fn last_valid(&self) -> Option<&NumericalEvidence> {
        self.last_valid.as_ref()
    }
    pub fn work(&self) -> RefinementWork {
        self.work
    }
    pub fn error(&self) -> Option<&E> {
        match &self.status {
            CandidateStatus::EvaluationFailed(e) | CandidateStatus::ProjectionFailed(e) => Some(e),
            _ => None,
        }
    }
}
/// Borrows the complete original successful discovery, including elapsed time.
/// Per-candidate failures are explicit outcomes, not partial failed discoveries.
#[derive(Debug)]
pub struct RefinementReport<'a, E> {
    discovery: &'a ProblemSearchResult,
    config: RefinementConfig,
    outcomes: Vec<CandidateOutcome<E>>,
}
impl<'a, E> RefinementReport<'a, E> {
    pub(crate) fn new(
        discovery: &'a ProblemSearchResult,
        config: RefinementConfig,
        outcomes: Vec<CandidateOutcome<E>>,
    ) -> Self {
        Self {
            discovery,
            config,
            outcomes,
        }
    }
    pub fn discovery(&self) -> &'a ProblemSearchResult {
        self.discovery
    }
    pub fn config(&self) -> RefinementConfig {
        self.config
    }
    pub fn outcomes(&self) -> &[CandidateOutcome<E>] {
        &self.outcomes
    }
}
fn finite(z: C) -> bool {
    z.re.is_finite() && z.im.is_finite()
}
fn inside(r: Rect<f64>, z: C) -> bool {
    finite(z) && z.re > r.x_min && z.re < r.x_max && z.im > r.y_min && z.im < r.y_max
}
fn valid_enclosure(r: Rect<f64>, outer: Rect<f64>) -> bool {
    [r.x_min, r.x_max, r.y_min, r.y_max]
        .iter()
        .all(|v| v.is_finite())
        && r.x_min < r.x_max
        && r.y_min < r.y_max
        && r.x_min >= outer.x_min
        && r.x_max <= outer.x_max
        && r.y_min >= outer.y_min
        && r.y_max <= outer.y_max
}
fn evidence<E>(
    z: C,
    b: &FunctionBundle<C>,
    config: ValidatedConfig,
) -> Result<(NumericalEvidence, C), CandidateStatus<E>> {
    let f = *b.value();
    let derivative = *b.derivative();
    if !finite(f) || !finite(derivative) {
        return Err(CandidateStatus::NonFiniteEvaluation);
    }
    if derivative == C::new(0.0, 0.0) {
        return Err(CandidateStatus::ZeroDerivative);
    }
    let delta = f / derivative;
    let correction = delta.norm();
    if !finite(delta) || !correction.is_finite() {
        return Err(CandidateStatus::NonFiniteCorrection);
    }
    let correction_limit = config.config.z_atol + config.config.z_rtol * z.norm().max(1.0);
    if !correction_limit.is_finite() {
        return Err(CandidateStatus::NonFiniteThreshold);
    }
    Ok((
        NumericalEvidence {
            z,
            value: f,
            derivative,
            correction: Some(correction),
            residual_limit: config.residual_limit,
            correction_limit,
        },
        delta,
    ))
}
fn passes(e: &NumericalEvidence) -> bool {
    e.value.norm() <= e.residual_limit && e.correction.is_some_and(|v| v <= e.correction_limit)
}

pub(crate) fn refine_candidate<E>(
    estimate: &SingularPointEstimate<C>,
    index: usize,
    provenance: &DomainProvenance,
    config: ValidatedConfig,
    mut evaluate: impl FnMut(C) -> Result<FunctionBundle<C>, E>,
    mut project: impl FnMut(C) -> Result<C, E>,
) -> CandidateOutcome<E> {
    let mut out = CandidateOutcome {
        index,
        status: CandidateStatus::IneligibleEstimate,
        last_valid: None,
        work: RefinementWork::default(),
    };
    macro_rules! finish {
        ($status:expr) => {{
            out.status = $status;
            return out;
        }};
    }
    // Precedence: target/inconsistent metadata, cluster, geometry, budgets,
    // evaluation, derivative/correction, convergence, update/trial termination.
    if estimate.target != SearchTarget::Zeros || estimate.multiplicity == 0 {
        return out;
    }
    match (estimate.kind, estimate.multiplicity) {
        (SingularPointEstimateKind::ClusterCentroid, n) if n > 1 => {
            finish!(CandidateStatus::UnresolvedCluster)
        }
        (SingularPointEstimateKind::SinglePoint, 1) => {}
        _ => return out,
    }
    let requested = provenance.requested();
    let (xmin, xmax) = requested.real_bounds();
    let (ymin, ymax) = requested.imaginary_bounds();
    let outer = Rect {
        x_min: xmin,
        x_max: xmax,
        y_min: ymin,
        y_max: ymax,
    };
    let enclosure = estimate.enclosure;
    if !valid_enclosure(enclosure, outer) {
        finish!(CandidateStatus::InvalidEnclosure)
    }
    let mut z = estimate.location;
    if !inside(enclosure, z) || !inside(outer, z) {
        finish!(CandidateStatus::BoundaryLimited)
    }
    if config.config.max_updates == 0
        || config.config.max_backtracks == 0
        || config.config.max_evaluations == 0
    {
        finish!(CandidateStatus::BudgetExceeded)
    }
    out.work.evaluations += 1;
    let mut bundle = match evaluate(z) {
        Ok(value) => value,
        Err(error) => finish!(CandidateStatus::EvaluationFailed(error)),
    };
    loop {
        let (current, delta) = match evidence(z, &bundle, config) {
            Ok(value) => value,
            Err(status) => {
                if finite(*bundle.value()) && finite(*bundle.derivative()) {
                    out.last_valid = Some(NumericalEvidence {
                        z,
                        value: *bundle.value(),
                        derivative: *bundle.derivative(),
                        correction: None,
                        residual_limit: config.residual_limit,
                        correction_limit: config.config.z_atol
                            + config.config.z_rtol * z.norm().max(1.0),
                    });
                }
                finish!(status)
            }
        };
        out.last_valid = Some(current);
        if passes(&current) {
            if out.work.evaluations >= config.config.max_evaluations {
                finish!(CandidateStatus::BudgetExceeded)
            }
            out.work.evaluations += 1;
            let fresh = match evaluate(z) {
                Ok(value) => value,
                Err(error) => finish!(CandidateStatus::EvaluationFailed(error)),
            };
            let (fresh, _) = match evidence(z, &fresh, config) {
                Ok(value) => value,
                Err(status) => finish!(status),
            };
            out.last_valid = Some(fresh);
            if !passes(&fresh) {
                finish!(CandidateStatus::NoProgress)
            }
            let coordinate = match project(z) {
                Ok(value) => value,
                Err(error) => finish!(CandidateStatus::ProjectionFailed(error)),
            };
            if !finite(coordinate) {
                finish!(CandidateStatus::NonFiniteEvaluation)
            }
            let accepted = AcceptedRoot {
                index,
                evidence: fresh,
                coordinate,
                provenance: provenance.clone(),
                config: config.config,
                work: out.work,
            };
            finish!(CandidateStatus::Accepted(Box::new(accepted)))
        }
        if out.work.updates >= config.config.max_updates {
            finish!(CandidateStatus::BudgetExceeded)
        }
        let mut lambda = 1.0;
        let mut admissible = false;
        let mut progressed = false;
        let mut stagnated = false;
        for _ in 0..=config.config.max_backtracks {
            out.work.trials = out.work.trials.saturating_add(1);
            let trial = z - delta * lambda;
            lambda *= 0.5;
            if !finite(trial) {
                finish!(CandidateStatus::NonFiniteCorrection)
            }
            if trial == z {
                stagnated = true;
                break;
            }
            if !inside(enclosure, trial) || !inside(outer, trial) {
                continue;
            }
            admissible = true;
            if out.work.evaluations >= config.config.max_evaluations {
                finish!(CandidateStatus::BudgetExceeded)
            }
            out.work.evaluations += 1;
            let next = match evaluate(trial) {
                Ok(value) => value,
                Err(error) => finish!(CandidateStatus::EvaluationFailed(error)),
            };
            if !finite(*next.value()) || !finite(*next.derivative()) {
                finish!(CandidateStatus::NonFiniteEvaluation)
            }
            if next.value().norm() < bundle.value().norm() {
                z = trial;
                bundle = next;
                out.work.updates += 1;
                progressed = true;
                break;
            }
        }
        if !progressed {
            if stagnated {
                finish!(CandidateStatus::Stagnated)
            }
            if !admissible {
                finish!(CandidateStatus::BoundaryLimited)
            }
            finish!(CandidateStatus::NoProgress)
        }
    }
}
#[cfg(test)]
mod tests;

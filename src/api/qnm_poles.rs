//! Finite-region discovery and checked local correspondence of scattering poles.

use super::*;
use lamina_core::{Polarisation, Scatter2};

#[derive(Clone, Debug)]
pub struct QnmPoleRequest {
    pub beta_per_cm: f64,
    pub polarisation: Polarisation,
    pub domain: FrequencyDomainInput,
    pub assumption: AnalyticAssumption,
    pub discovery: DiscoveryConfig,
    pub refinement: RefinementConfig,
    pub matching: MatchConfig,
    pub normalization: PoleNormalizationSettings,
    pub recovery: FieldRecoverySettings,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QnmPoleCoverage {
    /// Resolved numerical winding counts agree with the selected simple poles.
    Reconciled,
    Unresolved,
    /// Only previously selected poles were corrected at this station.
    TrackedOnly,
}

#[derive(Debug)]
pub struct QnmPoleSet {
    pub request: QnmPoleRequest,
    pub poles: Vec<QnmScatteringResidue>,
    pub coverage: QnmPoleCoverage,
    pub winding_count: Option<usize>,
    pub issues: Vec<QnmPoleIssue>,
    pub search: Option<SolveReport>,
}

#[derive(Debug)]
pub struct QnmPoleIssue {
    pub k0_per_cm: Complex,
    pub reason: QnmPoleIssueReason,
}

#[derive(Debug)]
pub enum QnmPoleIssueReason {
    OutsideOutgoingHalfPlane,
    Normalization(Box<NormalizationFailure>),
    Residue(QnmScatteringError),
}

#[derive(Debug, thiserror::Error)]
pub enum QnmPoleSearchError {
    #[error("expected a positive-frequency outgoing reference and lower-half-plane search extent")]
    InvalidRequest,
    #[error("pole search setup or execution failed")]
    Search(#[from] Diagnostic),
}

fn setup(
    stack: ModeStack,
    request: &QnmPoleRequest,
) -> Result<(FrequencyProblem, Domain), QnmPoleSearchError> {
    let bounds = request.domain.bounds.bounds();
    let (lo, _) = bounds.real_bounds();
    let (bottom, _) = bounds.imaginary_bounds();
    if lo <= 0.0
        || bottom >= 0.0
        || !matches!(
            request.domain.reference,
            FrequencyReference::RealRadiation {
                choices: [
                    RadiationChoice::OutgoingOrDecaying,
                    RadiationChoice::OutgoingOrDecaying
                ],
                ..
            }
        )
    {
        return Err(QnmPoleSearchError::InvalidRequest);
    }
    let problem = FrequencyProblem::new(stack, request.beta_per_cm, request.polarisation)?;
    let domain = problem.domain(request.domain.clone(), request.assumption)?;
    Ok((problem, domain))
}

fn residue(point: &ModePoint, req: &QnmPoleRequest) -> Result<QnmScatteringResidue, QnmPoleIssue> {
    let k = point.physical_coordinate();
    let result = if k.re > 0.0 && k.im < 0.0 {
        point
            .normalize_with(
                Scatter2::new(),
                PoleVariable::K0AtFixedBeta,
                req.assumption,
                req.normalization,
                req.recovery,
            )
            .map_err(QnmPoleIssueReason::Normalization)
            .and_then(|mode| {
                mode.scattering_residue()
                    .map_err(QnmPoleIssueReason::Residue)
            })
    } else {
        Err(QnmPoleIssueReason::OutsideOutgoingHalfPlane)
    };
    result.map_err(|reason| QnmPoleIssue {
        k0_per_cm: k,
        reason,
    })
}

fn sort_poles(poles: &mut [QnmScatteringResidue]) {
    poles.sort_by(|a, b| {
        a.signature
            .k0
            .re
            .total_cmp(&b.signature.k0.re)
            .then(a.signature.k0.im.total_cmp(&b.signature.k0.im))
    });
}

pub fn select_qnm_poles(
    stack: impl Into<ModeStack>,
    request: QnmPoleRequest,
) -> Result<QnmPoleSet, QnmPoleSearchError> {
    let (problem, domain) = setup(stack.into(), &request)?;
    discover(problem, domain, request)
}

fn discover(
    problem: FrequencyProblem,
    domain: Domain,
    request: QnmPoleRequest,
) -> Result<QnmPoleSet, QnmPoleSearchError> {
    let search = problem.plan(vec![domain], vec![])?.solve(
        request.discovery,
        request.refinement,
        request.matching,
    )?;
    let mut poles = Vec::new();
    let mut issues = Vec::new();
    for group in search.groups() {
        match residue(&search.observations()[group.representative()], &request) {
            Ok(pole) => poles.push(pole),
            Err(issue) => issues.push(issue),
        }
    }
    sort_poles(&mut poles);
    let winding_count = search
        .discovery()
        .entries()
        .next()
        .and_then(Result::ok)
        .and_then(|entry| {
            if *entry.termination() != Termination::Converged
                || entry.leaves().is_empty()
                || entry.leaves().iter().any(|leaf| {
                    leaf.data.score > request.discovery.score_tolerance
                        || !leaf.data.score.is_finite()
                })
            {
                return None;
            }
            entry.leaves().iter().try_fold(0usize, |total, leaf| {
                total.checked_add(usize::try_from(leaf.data.root_count).ok()?)
            })
        });
    let coverage = if winding_count == Some(poles.len())
        && issues.is_empty()
        && search.diagnostics().is_empty()
        && search.charts().iter().all(|chart| {
            chart.candidates.as_ref().is_some_and(|candidates| {
                candidates
                    .iter()
                    .all(|candidate| matches!(candidate.status, CandidateStatus::Accepted))
            })
        }) {
        QnmPoleCoverage::Reconciled
    } else {
        QnmPoleCoverage::Unresolved
    };
    Ok(QnmPoleSet {
        request,
        poles,
        coverage,
        winding_count,
        issues,
        search: Some(search),
    })
}

#[derive(Clone, Copy, Debug)]
pub struct QnmPoleTrackingSettings {
    /// Maximum distance from a source pole for a correspondence, in inverse cm.
    pub match_radius_per_cm: f64,
    /// Run independent discovery at this station.
    pub audit_coverage: bool,
}

#[derive(Debug)]
pub struct QnmPoleTrack {
    pub destination: QnmPoleSet,
    pub matches: Vec<(usize, usize)>,
    pub ambiguous_source: Vec<usize>,
    pub unmatched_source: Vec<usize>,
    pub unmatched_destination: Vec<usize>,
    /// Source indices for which the local correction or residue check failed.
    pub failed_local_sources: Vec<usize>,
}

/// Correct the previous poles locally or refresh the destination with discovery.
pub fn track_qnm_poles(
    source: &QnmPoleSet,
    stack: impl Into<ModeStack>,
    request: QnmPoleRequest,
    settings: QnmPoleTrackingSettings,
) -> Result<QnmPoleTrack, QnmPoleSearchError> {
    let radius = settings.match_radius_per_cm;
    if !radius.is_finite()
        || radius <= 0.0
        || source.request.beta_per_cm != request.beta_per_cm
        || source.request.polarisation != request.polarisation
    {
        return Err(QnmPoleSearchError::InvalidRequest);
    }
    let (problem, domain) = setup(stack.into(), &request)?;
    let distinct_seeds = source.poles.iter().enumerate().all(|(i, a)| {
        source.poles[..i]
            .iter()
            .all(|b| (a.signature.k0 - b.signature.k0).norm() > 2.0 * radius)
    });
    let mut failed_local_sources = Vec::new();
    let mut poles = Vec::new();
    if !settings.audit_coverage && distinct_seeds && source.coverage != QnmPoleCoverage::Unresolved
    {
        for (i, pole) in source.poles.iter().enumerate() {
            let k = pole.signature.k0;
            let result = problem
                .correct_frequency_seed(&domain, k, request.refinement)
                .ok()
                .filter(|point| (point.physical_coordinate() - k).norm() <= radius)
                .and_then(|point| residue(&point, &request).ok());
            if let Some(result) = result {
                poles.push(result);
            } else {
                failed_local_sources.push(i);
            }
        }
    }
    let destination = if !poles.is_empty() && poles.len() == source.poles.len() {
        sort_poles(&mut poles);
        QnmPoleSet {
            request,
            poles,
            coverage: QnmPoleCoverage::TrackedOnly,
            winding_count: None,
            issues: Vec::new(),
            search: None,
        }
    } else {
        discover(problem, domain, request)?
    };
    let near: Vec<Vec<_>> = source
        .poles
        .iter()
        .map(|a| {
            destination
                .poles
                .iter()
                .enumerate()
                .filter_map(|(j, b)| {
                    ((a.signature.k0 - b.signature.k0).norm() <= radius).then_some(j)
                })
                .collect()
        })
        .collect();
    let counts: Vec<_> = (0..destination.poles.len())
        .map(|j| near.iter().filter(|indices| indices.contains(&j)).count())
        .collect();
    let mut matches = Vec::new();
    let mut ambiguous_source = Vec::new();
    let mut unmatched_source = Vec::new();
    for (i, indices) in near.iter().enumerate() {
        match indices.as_slice() {
            [] => unmatched_source.push(i),
            &[j] if counts[j] == 1 => matches.push((i, j)),
            _ => ambiguous_source.push(i),
        }
    }
    let unmatched_destination = counts
        .iter()
        .enumerate()
        .filter_map(|(j, &count)| (count == 0).then_some(j))
        .collect();
    Ok(QnmPoleTrack {
        destination,
        matches,
        ambiguous_source,
        unmatched_source,
        unmatched_destination,
        failed_local_sources,
    })
}

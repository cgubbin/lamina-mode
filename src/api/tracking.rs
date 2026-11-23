//! Prediction, counted correction and overlap matching for guided bases.

use super::*;
use lamina_core::{AnalyticalMaterialStack, Scatter2};

/// One parameter step at fixed vacuum wavelength and polarization.
pub struct GuidedTrackingStep {
    pub stack: AnalyticalMaterialStack<Complex>,
    pub window: GuidedWindow,
    pub perturbation: GuidedPerturbation,
    /// Signed increment of the parameter named by the perturbation.
    pub increment: f64,
    pub registration: GuidedOverlapRegistration,
    /// Minimum absolute power overlap in W/m for a candidate correspondence.
    pub minimum_overlap: f64,
    /// Score difference within which alternative assignments remain plausible.
    pub ambiguity_margin: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum GuidedTrackingError {
    #[error("invalid tracking step or incompatible source and destination bases")]
    InvalidStep,
    #[error(transparent)]
    Enumeration(#[from] EnumerationError),
    #[error(
        "overlap failed for source mode {source_index} and destination mode {destination_index}"
    )]
    Overlap {
        source_index: usize,
        destination_index: usize,
        #[source]
        error: GuidedOverlapError,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct GuidedModeMatch {
    pub source: usize,
    pub destination: usize,
    pub overlap_w_per_m: Complex,
}

#[derive(Debug)]
pub struct GuidedModeCluster {
    pub source: Vec<usize>,
    pub destination: Vec<usize>,
}

#[derive(Debug)]
pub struct GuidedTrackingResult {
    pub basis: GuidedBasis,
    /// True when independently counted local root brackets covered the new basis.
    pub used_local_correction: bool,
    /// Signed predictions in source order, with individual failure evidence.
    pub predictions: Vec<Result<f64, GuidedBetaError>>,
    pub matches: Vec<GuidedModeMatch>,
    pub ambiguous: Vec<GuidedModeCluster>,
    /// Destination indices without a plausible match.
    pub unmatched_destination: Vec<usize>,
    /// Source indices without a plausible match.
    pub unmatched_source: Vec<usize>,
}

pub fn track_guided_basis(
    previous: &GuidedBasis,
    step: GuidedTrackingStep,
) -> Result<GuidedTrackingResult, GuidedTrackingError> {
    let source_window = previous.request;
    if !step.increment.is_finite()
        || !step.minimum_overlap.is_finite()
        || step.minimum_overlap <= 0.0
        || !step.ambiguity_margin.is_finite()
        || step.ambiguity_margin < 0.0
        || step.registration.origins_cm.iter().any(|v| !v.is_finite())
        || step.perturbation.wavelength_rate() != 0.0
        || source_window.polarisation != step.window.polarisation
        || source_window.direction != step.window.direction
        || (source_window.wavelength.as_centimetres() - step.window.wavelength.as_centimetres())
            .abs()
            > 1e-12 * source_window.wavelength.as_centimetres()
    {
        return Err(GuidedTrackingError::InvalidStep);
    }
    let backend = Scatter2::new();
    let predictions: Vec<_> = previous
        .modes
        .iter()
        .map(|mode| {
            mode.guided_beta_sensitivity_with(
                &backend,
                &step.perturbation,
                Default::default(),
                Default::default(),
            )
            .and_then(|result| {
                let prediction =
                    mode.physical_coordinate().re + step.increment * result.beta_rate.re;
                if prediction.is_finite() {
                    Ok(prediction)
                } else {
                    Err(GuidedBetaError::NonFinite)
                }
            })
        })
        .collect();
    let seeds = previous
        .is_complete()
        .then(|| {
            predictions
                .iter()
                .map(|p| p.as_ref().ok().copied())
                .collect::<Option<Vec<_>>>()
        })
        .flatten();
    let (basis, used_local_correction) =
        super::guided::search_guided_modes(step.stack, step.window, seeds.as_deref())?;
    let n = previous.modes.len();
    let m = basis.modes.len();
    let mut row_best = vec![0.0_f64; n];
    let mut col_best = vec![0.0_f64; m];
    let mut candidates = Vec::new();
    for (i, a) in previous.modes.iter().enumerate() {
        let predicted = predictions[i]
            .as_ref()
            .copied()
            .unwrap_or(a.physical_coordinate().re);
        for (j, b) in basis.modes.iter().enumerate() {
            let pair =
                guided_power_overlap_with(&backend, [a, b], step.registration, Default::default())
                    .map_err(|error| GuidedTrackingError::Overlap {
                        source_index: i,
                        destination_index: j,
                        error,
                    })?;
            let overlap = pair.power_w_per_m.norm();
            if overlap < step.minimum_overlap {
                continue;
            }
            let beta = b.physical_coordinate().re;
            // Fractional propagation-constant discrepancy only breaks overlap ties.
            let score = overlap / (1.0 + (beta - predicted).abs() / beta.abs().max(1.0));
            candidates.push((
                GuidedModeMatch {
                    source: i,
                    destination: j,
                    overlap_w_per_m: pair.power_w_per_m,
                },
                score,
            ));
            row_best[i] = row_best[i].max(score);
            col_best[j] = col_best[j].max(score);
        }
    }
    let mut by_source = vec![Vec::new(); n];
    let mut by_destination = vec![Vec::new(); m];
    for (edge, score) in candidates {
        if score + step.ambiguity_margin >= row_best[edge.source]
            && score + step.ambiguity_margin >= col_best[edge.destination]
        {
            by_source[edge.source].push(edge);
            by_destination[edge.destination].push(edge.source);
        }
    }
    let mut visited_source = vec![false; n];
    let mut visited_destination = vec![false; m];
    let mut matches = Vec::new();
    let mut ambiguous = Vec::new();
    let mut unmatched_source = Vec::new();
    for (i, edges) in by_source.iter().enumerate() {
        if edges.is_empty() {
            unmatched_source.push(i);
            continue;
        }
        if visited_source[i] {
            continue;
        }
        let mut source = Vec::new();
        let mut destination = Vec::new();
        let mut pending = vec![i];
        visited_source[i] = true;
        while let Some(index) = pending.pop() {
            source.push(index);
            for edge in &by_source[index] {
                let j = edge.destination;
                if !visited_destination[j] {
                    visited_destination[j] = true;
                    destination.push(j);
                    for &next in &by_destination[j] {
                        if !visited_source[next] {
                            visited_source[next] = true;
                            pending.push(next);
                        }
                    }
                }
            }
        }
        if source.len() == 1
            && destination.len() == 1
            && previous.is_complete()
            && basis.is_complete()
        {
            matches.push(edges[0]);
        } else {
            source.sort_unstable();
            destination.sort_unstable();
            ambiguous.push(GuidedModeCluster {
                source,
                destination,
            });
        }
    }
    let unmatched_destination = by_destination
        .iter()
        .enumerate()
        .filter_map(|(j, edges)| edges.is_empty().then_some(j))
        .collect();
    Ok(GuidedTrackingResult {
        basis,
        used_local_correction,
        predictions,
        matches,
        ambiguous,
        unmatched_destination,
        unmatched_source,
    })
}

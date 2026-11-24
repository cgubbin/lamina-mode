//! Conditioned guided projection of supplied electromagnetic fields.

use super::*;
use lamina_core::Scatter2;
use nalgebra::{DMatrix, DVector};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionSide {
    Left,
    Right,
    Interior,
}

/// A registered input field which vanishes outside the supplied finite interval.
/// Samples use [Ex, Ey, Ez, Z0 Hx, Z0 Hy, Z0 Hz] in core conventions.
pub struct GuidedProjectionInput<'a> {
    pub wavelength_cm: f64,
    pub support_cm: [f64; 2],
    pub interfaces_cm: &'a [f64],
    pub sample: &'a dyn Fn(f64, ProjectionSide) -> [Complex; 6],
}

#[derive(Debug)]
pub struct GuidedFieldProjection {
    pub modes: Vec<ModePoint>,
    pub origin_cm: f64,
    /// Row i, column j is the power pairing of mode j with mode i.
    pub gram: DMatrix<Complex>,
    pub right_hand_side: DVector<Complex>,
    pub rank: usize,
    pub singular_values: Vec<f64>,
    pub quadrature_change_w_per_m: f64,
    /// Available for full numerical rank.
    pub solution: Option<GuidedProjectionSolution>,
}

#[derive(Debug)]
pub struct GuidedProjectionSolution {
    pub coefficients: DVector<Complex>,
    pub condition_number: f64,
    /// Norm of the power projection equation residual in W/m.
    pub residual_w_per_m: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum GuidedProjectionError {
    #[error("expected a complete nonempty basis and a valid registered input at its wavelength")]
    InvalidInput,
    #[error(transparent)]
    Overlap(#[from] GuidedOverlapError),
    #[error(transparent)]
    Fields(#[from] GuidedPowerError),
    #[error("nonfinite supplied field")]
    NonFiniteField,
    #[error("projection quadrature exceeded its accuracy or sampling budget")]
    Quadrature,
}

/// Project a finite-support input field onto an ordered guided basis.
/// `settings` applies to every mode-field recovery, in the Gram matrix and in
/// the right-hand side.
pub fn project_guided_field(
    basis: &GuidedBasis,
    origin_cm: f64,
    input: GuidedProjectionInput<'_>,
    rank_rtol: f64,
    settings: FieldRecoverySettings,
) -> Result<GuidedFieldProjection, GuidedProjectionError> {
    use super::overlap::{interface_positions, pairing, sample_grid_with_exterior};
    use GuidedProjectionError::*;
    let [lo, hi] = input.support_cm;
    let wavelength = basis.request.wavelength.as_centimetres();
    if !basis.is_complete()
        || basis.modes.is_empty()
        || !origin_cm.is_finite()
        || !lo.is_finite()
        || !hi.is_finite()
        || lo >= hi
        || !input.wavelength_cm.is_finite()
        || input.wavelength_cm <= 0.0
        || (input.wavelength_cm - wavelength).abs() > 1e-12 * wavelength
        || !(0.0 < rank_rtol && rank_rtol < 1.0)
        || input.interfaces_cm.iter().any(|&x| !(lo < x && x < hi))
    {
        return Err(InvalidInput);
    }
    let backend = Scatter2::new();
    let n = basis.modes.len();
    let mut gram = DMatrix::zeros(n, n);
    let mut rhs = DVector::zeros(n);
    let mut quadrature_change_w_per_m = 0.0_f64;
    for (i, mode) in basis.modes.iter().enumerate() {
        for j in i..n {
            let overlap = guided_power_overlap_with(
                &backend,
                [&basis.modes[j], mode],
                GuidedOverlapRegistration {
                    origins_cm: [origin_cm; 2],
                },
                settings,
            )?
            .power_w_per_m;
            gram[(i, j)] = overlap;
            gram[(j, i)] = overlap.conj();
        }
        let edges = interface_positions(mode, origin_cm)?;
        let widths: Vec<_> = mode
            .provenance()
            .stack()
            .iter()
            .map(|layer| layer.thickness().as_centimetres())
            .collect();
        let signature = mode.signature().map_err(|_| InvalidInput)?;
        let first_edge = edges[0];
        let last_edge = edges[edges.len() - 1];
        let mut partition = vec![lo, hi];
        partition.extend(edges.iter().copied().filter(|&x| lo < x && x < hi));
        partition.extend_from_slice(input.interfaces_cm);
        partition.sort_by(f64::total_cmp);
        partition.dedup();
        let mut previous: Option<Complex> = None;
        for level in 0..=11 {
            let panels = 8 << level;
            if (partition.len() - 1).saturating_mul(panels + 1) > 131070 {
                return Err(Quadrature);
            }
            // sample_grid includes two tail anchors. Pad the sampling partition
            // to cover the full modal stack, then integrate the input support only.
            let mut grid = partition.clone();
            grid.extend(edges.iter().copied().filter(|&x| x < lo || x > hi));
            grid.sort_by(f64::total_cmp);
            if (grid.len() - 1).saturating_mul(panels + 1) > 131070 {
                return Err(Quadrature);
            }
            let profile = mode.guided_power_profile_with(
                backend,
                &sample_grid_with_exterior(&edges, &widths, &grid, panels, true),
                settings,
            )?;
            let fields = profile.fields.quantity();
            let e = fields.electric();
            let h = fields.magnetic();
            let mut total = Complex::ZERO;
            for (interval, edge) in grid.windows(2).enumerate() {
                let [left, right] = [edge[0], edge[1]];
                if left < lo || right > hi {
                    continue;
                }
                let mut sum = Complex::ZERO;
                for j in 0..=panels {
                    let (coordinate, side, weight) = if j == 0 {
                        (left, ProjectionSide::Right, 1.0)
                    } else if j == panels {
                        (right, ProjectionSide::Left, 1.0)
                    } else {
                        (
                            left + (right - left) * (j as f64 / panels as f64),
                            ProjectionSide::Interior,
                            if j % 2 == 1 { 4.0 } else { 2.0 },
                        )
                    };
                    let supplied = (input.sample)(coordinate, side);
                    if supplied
                        .iter()
                        .any(|z| !z.re.is_finite() || !z.im.is_finite())
                    {
                        return Err(NonFiniteField);
                    }
                    let p = 1 + interval * (panels + 1) + j;
                    let phase = if right <= first_edge {
                        (Complex::I * signature.left * (first_edge - coordinate)).exp()
                    } else if left >= last_edge {
                        (Complex::I * signature.right * (coordinate - last_edge)).exp()
                    } else {
                        Complex::ONE
                    };
                    let modal = [e.x()[p], e.y()[p], e.z()[p], h.x()[p], h.y()[p], h.z()[p]]
                        .map(|v| phase * v);
                    sum += weight * pairing(supplied, modal);
                }
                total += sum * ((right - left) / (3.0 * panels as f64));
            }
            if let Some(prior) = previous {
                let change = (total - prior).norm();
                if change <= 1e-9 * total.norm().max(1.0) {
                    rhs[i] = total;
                    quadrature_change_w_per_m = quadrature_change_w_per_m.max(change);
                    break;
                }
            }
            if level == 11 {
                return Err(Quadrature);
            }
            previous = Some(total);
        }
    }
    let svd = gram.clone().svd(true, true);
    let largest = svd.singular_values[0];
    let threshold = rank_rtol * largest;
    let rank = svd
        .singular_values
        .iter()
        .filter(|&&s| s > threshold)
        .count();
    let solution = if rank == n {
        let coefficients = svd.solve(&rhs, threshold).map_err(|_| InvalidInput)?;
        Some(GuidedProjectionSolution {
            residual_w_per_m: (&gram * &coefficients - &rhs).norm(),
            condition_number: largest / svd.singular_values[n - 1],
            coefficients,
        })
    } else {
        None
    };
    Ok(GuidedFieldProjection {
        modes: basis.modes.clone(),
        origin_cm,
        gram,
        right_hand_side: rhs,
        rank,
        singular_values: svd.singular_values.as_slice().to_vec(),
        quadrature_change_w_per_m,
        solution,
    })
}

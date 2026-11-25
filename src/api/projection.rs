//! Conditioned projection of supplied electromagnetic fields onto guided
//! bases, under the power overlap, and onto modal bases, under the
//! unconjugated pairing.
//!
//! A modal basis's modes are bi-orthogonal under ⟨·,·⟩, so a field F has the
//! coefficients a solving G a = b, G_mn = ⟨f_m, f_n⟩ = δ_mn up to quadrature
//! and b_m = ⟨f_m, F⟩. Between walls and PMLs the part of F outside the span
//! is carried by modes outside the window. With an open side the spectrum
//! also has a radiation continuum, which a discrete basis never includes:
//! the residual then holds F's radiation content and is physical.

use super::basis::{BasisFieldError, BasisScope, BasisStatus, ModalBasis};
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
    use super::integration::{conjugated, samples};
    use super::overlap::interface_positions;
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
        let layout = interface_positions(mode, origin_cm)?;
        let edges = &layout.interfaces;
        let signature = mode.signature().map_err(|_| InvalidInput)?;
        let [left_root, right_root] = signature.open_exteriors().ok_or(InvalidInput)?;
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
                &layout.sampling(&grid, panels, true),
                settings,
            )?;
            let modal = samples(&profile.fields);
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
                        (Complex::I * left_root * (first_edge - coordinate)).exp()
                    } else if left >= last_edge {
                        (Complex::I * right_root * (coordinate - last_edge)).exp()
                    } else {
                        Complex::ONE
                    };
                    sum += weight * conjugated(&supplied, &modal[p].map(|v| phase * v));
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

/// The field beyond the support on an open side, one exterior wave of it:
/// `edge` · exp(iκζ) at outward distance ζ ≥ 0 from the support edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExteriorWave {
    /// κ in cm^-1, on either sheet.
    pub kappa_cm: Complex,
    /// [Ex, Ey, Ez, Z0 Hx, Z0 Hy, Z0 Hz] at the support edge.
    pub edge: [Complex; 6],
}

/// A supplied field for projection onto a modal basis. Inside a PML the
/// field is the exterior solution continued to the stretched coordinate,
/// as modal profiles give it. Beyond the support it vanishes, except on an
/// open side whose support edge lies in the exterior, where it continues as
/// the stated exterior waves.
pub struct BasisProjectionInput<'a> {
    pub field: GuidedProjectionInput<'a>,
    /// Left and right continuations; empty for none.
    pub exterior: [&'a [ExteriorWave]; 2],
}

/// A supplied field F expanded in the normalised modes f_m of a modal basis:
/// G a = b with G_mn = ⟨f_m, f_n⟩ and b_m = ⟨f_m, F⟩, unconjugated.
#[derive(Debug)]
pub struct BasisFieldProjection {
    pub origin_cm: f64,
    /// Basis indices of the normalised modes, in key order: the rows and
    /// columns of `gram`.
    pub modes: Vec<usize>,
    pub gram: DMatrix<Complex>,
    pub right_hand_side: DVector<Complex>,
    pub rank: usize,
    /// Of the Gram matrix, descending.
    pub singular_values: Vec<f64>,
    /// max |G − I|.
    pub gram_defect: f64,
    pub partition_cm: Vec<f64>,
    pub sample_count: usize,
    pub quadrature_change_w_per_m: f64,
    /// (1/4Z0) ∫ (|E|² + |Z0 H|²) dz of F over the bounded partition, W/m.
    pub field_scale_w_per_m: f64,
    /// True when the basis is discrete without continuum: the reconstruction
    /// residual then includes F's radiation content, which no discrete mode
    /// represents, and is physical rather than numerical.
    pub residual_includes_continuum: bool,
    /// Available at full numerical rank.
    pub solution: Option<BasisProjectionSolution>,
    pub status: BasisStatus,
    pub scope: BasisScope,
}

#[derive(Debug)]
pub struct BasisProjectionSolution {
    pub coefficients: DVector<Complex>,
    pub condition_number: f64,
    /// |G a − b|, W/m.
    pub equation_residual_w_per_m: f64,
    /// Field scale of F − Σ a_m f_m over the bounded partition, W/m.
    pub residual_scale_w_per_m: f64,
    /// √(residual scale / field scale); `None` for a zero field.
    pub relative_residual: Option<f64>,
}

#[derive(Debug, thiserror::Error)]
pub enum BasisProjectionError {
    #[error(
        "expected a basis with a normalised mode and a valid registered input at its wavelength, \
         within its walls, continued only beyond the stack on an open side"
    )]
    InvalidInput,
    #[error("nonfinite supplied field")]
    NonFiniteField,
    #[error("field recovery failed for mode {mode}")]
    Fields {
        mode: usize,
        #[source]
        source: Box<FieldRecoveryFailure>,
    },
    #[error("projection quadrature exceeded its accuracy or sampling budget")]
    Quadrature,
    #[error("the {side:?} open tail is singular: |κ_a + κ_b| is below its floor")]
    TailSingular { side: lamina_core::StackSide },
}

/// Expand a supplied field in the normalised modes of `basis`, whose first
/// interface lies at `origin_cm` in the field's coordinate. Both pairings
/// integrate on the basis's partition with the support ends and the field's
/// discontinuities, dz̃ in a PML and continued tails on open sides, all at one
/// refinement level. Ill-conditioned modes are left out; incomplete bases are
/// accepted.
pub fn basis_projection_with<B: lamina_core::ModalFieldBackend<Region> + Clone>(
    backend: &B,
    basis: &ModalBasis,
    origin_cm: f64,
    input: BasisProjectionInput<'_>,
    rank_rtol: f64,
) -> Result<BasisFieldProjection, BasisProjectionError> {
    use super::integration::{
        Layout, Sample, converge, equivalent, field_scale, integral, split_pmls, tails,
        unconjugated,
    };
    use super::pairing::{kappas, sampled};
    use BasisProjectionError::*;
    use lamina_core::StackSide;
    let field = &input.field;
    let [lo, hi] = field.support_cm;
    let wavelength = basis.request.wavelength.as_centimetres();
    let k0 = basis.request.k0_cm();
    let modes: Vec<usize> = (0..basis.modes.len())
        .filter(|&i| basis.modes[i].is_normalised())
        .collect();
    let layout = Layout::new(&basis.stack, origin_cm).ok_or(InvalidInput)?;
    let last = layout.interfaces.len() - 1;
    let faces = [layout.interfaces[0], layout.interfaces[last]];
    let finite = |v: &Complex| v.re.is_finite() && v.im.is_finite();
    // A continuation needs an open side and a support edge in its exterior.
    let continued = |side: usize| {
        let waves = input.exterior[side];
        waves.is_empty()
            || (layout.walls[side].is_none()
                && [lo <= faces[0], hi >= faces[1]][side]
                && waves
                    .iter()
                    .all(|w| finite(&w.kappa_cm) && w.edge.iter().all(finite)))
    };
    if modes.is_empty()
        || !lo.is_finite()
        || !hi.is_finite()
        || lo >= hi
        || !field.wavelength_cm.is_finite()
        || (field.wavelength_cm - wavelength).abs() > 1e-12 * wavelength
        || !(0.0 < rank_rtol && rank_rtol < 1.0)
        || field.interfaces_cm.iter().any(|&x| !(lo < x && x < hi))
        || !continued(0)
        || !continued(1)
    {
        return Err(InvalidInput);
    }
    // The support may reach a wall to round-off, never beyond it.
    let tolerance = 1e-12 * (hi - lo).max(faces[1] - faces[0]);
    let clamp = |side: usize, edge: f64| match layout.walls[side] {
        Some((_, wall)) if (edge - wall) * [-1.0, 1.0][side] > 0.0 => {
            ((edge - wall).abs() <= tolerance).then_some(wall)
        }
        _ => Some(edge),
    };
    let support = [
        clamp(0, lo).ok_or(InvalidInput)?,
        clamp(1, hi).ok_or(InvalidInput)?,
    ];
    let mut partition = Layout::partition(&[&layout]);
    partition.extend(support);
    partition.extend(field.interfaces_cm.iter().copied());
    partition.sort_by(f64::total_cmp);
    partition.dedup();
    let polarisation = basis
        .request
        .family
        .polarisation()
        .ok_or(BasisProjectionError::InvalidInput)?;
    let betas: Vec<_> = modes
        .iter()
        .map(|&i| basis.modes[i].point.physical_coordinate())
        .collect();
    let partition = split_pmls(
        partition,
        &[(&layout, &basis.stack, polarisation)],
        k0,
        &betas,
    );
    let kappa: Vec<_> = modes.iter().map(|&i| kappas(&basis.modes[i])).collect();
    let n = modes.len();
    let singular = |side| TailSingular { side };
    // The supplied field at the samples of `Layout::sampling`, zero outside
    // its support, from inside each interval at its ends.
    let supplied = |panels: usize| -> Result<Vec<Sample>, BasisProjectionError> {
        let zero = [Complex::ZERO; 6];
        let mut out = vec![zero];
        for edges in partition.windows(2) {
            let (left, right) = (edges[0], edges[1]);
            let inside = support[0] <= left && right <= support[1];
            for j in 0..=panels {
                if !inside {
                    out.push(zero);
                    continue;
                }
                let (z, side) = if j == 0 {
                    (left, ProjectionSide::Right)
                } else if j == panels {
                    (right, ProjectionSide::Left)
                } else {
                    let z = left + (right - left) * (j as f64 / panels as f64);
                    (z, ProjectionSide::Interior)
                };
                let value = (field.sample)(z, side);
                if !value.iter().all(finite) {
                    return Err(NonFiniteField);
                }
                out.push(value);
            }
        }
        out.push(zero);
        equivalent(&mut out, &layout.stretches(&partition, panels));
        Ok(out)
    };
    // F is scaled at the first level to about unit field scale, so that the
    // convergence test is relative for every amplitude.
    let mut scale: Option<f64> = None;
    let converged = converge(
        &partition,
        || Quadrature,
        |panels, _| {
            let f = modes
                .iter()
                .map(|&i| {
                    let mode = &basis.modes[i];
                    sampled(
                        backend,
                        mode,
                        &layout,
                        &partition,
                        panels,
                        basis.request.recovery,
                    )
                    .map_err(|e| match e {
                        BasisFieldError::IllConditioned => InvalidInput,
                        BasisFieldError::Recovery(source) => Fields { mode: i, source },
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut big_f = supplied(panels)?;
            let norm = |u: &[Sample]| integral(&partition, panels, |s| field_scale(&u[s])).re;
            let factor = *scale.get_or_insert_with(|| {
                let n = norm(&big_f);
                if n > 0.0 && n.is_finite() {
                    n.sqrt().recip()
                } else {
                    1.0
                }
            });
            big_f.iter_mut().flatten().for_each(|v| *v *= factor);
            let mut entries = Vec::with_capacity(n + n * (n + 1) / 2 + 1);
            for (m, fm) in f.iter().enumerate() {
                let mut b = integral(&partition, panels, |s| unconjugated(&fm[s], &big_f[s]));
                for (side, which, anchor) in
                    [(0, StackSide::Left, 0), (1, StackSide::Right, fm.len() - 1)]
                {
                    for wave in input.exterior[side] {
                        let km = kappa[m][side].ok_or(InvalidInput)?;
                        let sum = km + wave.kappa_cm;
                        if sum.norm().is_nan() || sum.norm() < super::open::TAIL_SINGULAR_FLOOR * k0
                        {
                            return Err(singular(which));
                        }
                        let edge = wave.edge.map(|v| v * factor);
                        b += Complex::I * unconjugated(&fm[anchor], &edge) / sum;
                    }
                }
                entries.push(b);
            }
            for m in 0..n {
                for k in m..n {
                    let tail = tails(&f[m], &f[k], [kappa[m], kappa[k]], k0, false).map_err(
                        |failure| match failure {
                            super::integration::TailFailure::Singular(side)
                            | super::integration::TailFailure::Unbounded(side) => singular(side),
                        },
                    )?;
                    entries.push(
                        integral(&partition, panels, |s| unconjugated(&f[m][s], &f[k][s])) + tail,
                    );
                }
            }
            entries.push(Complex::new(norm(&big_f), 0.0));
            Ok((entries, (f, big_f, panels)))
        },
    )?;
    let e = &converged.entries;
    let factor = scale.unwrap_or(1.0);
    let rhs = DVector::from_fn(n, |m, _| e[m]);
    let mut gram = DMatrix::zeros(n, n);
    let mut p = n;
    for m in 0..n {
        for k in m..n {
            gram[(m, k)] = e[p];
            gram[(k, m)] = e[p];
            p += 1;
        }
    }
    let field_scale_w_per_m = e[p].re / (factor * factor);
    let gram_defect = (0..n * n)
        .map(|q| {
            let (i, j) = (q / n, q % n);
            (gram[(i, j)] - if i == j { Complex::ONE } else { Complex::ZERO }).norm()
        })
        .fold(0.0, f64::max);
    let svd = gram.clone().svd(true, true);
    let mut singular_values = svd.singular_values.as_slice().to_vec();
    singular_values.sort_by(|a, b| b.total_cmp(a));
    let largest = singular_values[0];
    let threshold = rank_rtol * largest;
    let rank = singular_values.iter().filter(|&&s| s > threshold).count();
    let solution = if rank == n {
        let scaled = svd.solve(&rhs, threshold).map_err(|_| InvalidInput)?;
        let (f, big_f, panels) = &converged.extra;
        let residual: Vec<Sample> = (0..big_f.len())
            .map(|s| {
                std::array::from_fn(|c| {
                    f.iter()
                        .zip(scaled.iter())
                        .fold(big_f[s][c], |r, (fm, a)| r - a * fm[s][c])
                })
            })
            .collect();
        let residual_scale_w_per_m =
            integral(&partition, *panels, |s| field_scale(&residual[s])).re / (factor * factor);
        Some(BasisProjectionSolution {
            equation_residual_w_per_m: (&gram * &scaled - &rhs).norm() / factor,
            condition_number: largest / singular_values[n - 1],
            coefficients: scaled / Complex::new(factor, 0.0),
            relative_residual: (field_scale_w_per_m > 0.0)
                .then(|| (residual_scale_w_per_m / field_scale_w_per_m).sqrt()),
            residual_scale_w_per_m,
        })
    } else {
        None
    };
    Ok(BasisFieldProjection {
        origin_cm,
        modes,
        gram,
        right_hand_side: rhs / Complex::new(factor, 0.0),
        rank,
        singular_values,
        gram_defect,
        partition_cm: partition,
        sample_count: converged.sample_count,
        // Gram entries are unscaled and b is scaled by `factor`: a bound for both.
        quadrature_change_w_per_m: converged.change * factor.recip().max(1.0),
        field_scale_w_per_m,
        residual_includes_continuum: basis.scope == BasisScope::DiscreteWithoutContinuum,
        solution,
        status: basis.status,
        scope: basis.scope,
    })
}

//! Unconjugated and conjugated pairings between modes of two stacks at one
//! frequency in one registered frame, and pairing matrices between bases.
//!
//! ⟨a, b⟩ = ¼ ∫ (E_a × H_b + E_b × H_a) · x̂ dz is symmetric; modes of a stack
//! closed by walls are bi-orthogonal under it, because the Lorentz boundary
//! terms vanish on the walls. The conjugated product
//! (a, b) = ¼ ∫ (E_a × H_b* + E_b* × H_a) · x̂ dz is power bookkeeping. Both
//! integrate over the union of the stacks' interfaces, PML starts and wall
//! planes, up to the walls, which must coincide in kind and position, as must
//! PMLs in start, thickness and profile.
//!
//! A PML is a uniaxial medium εΛ, μΛ with Λ = diag(s, s, 1/s): complex
//! symmetric, hence reciprocal, and backed by a wall, so modes of a stack
//! closed by PMLs are bi-orthogonal too. Inside a PML the unconjugated pairing
//! is the contour integral ∫ (E_a × H_b + E_b × H_a) · x̂ dζ̃ of the continued
//! fields, which depends on the PML only through its complex thickness τ; the
//! conjugated product integrates the equivalent-medium fields in real
//! coordinates, s* E_y H_z* − s E_z H_y* for each term, and depends on the
//! profile.
use super::basis::{BasisFieldError, BasisMode, BasisScope, BasisStatus, ModalBasis};
use super::integration::{
    Layout, Sample, conjugated, converge, equivalent, integral, intervals, split_pmls, unconjugated,
};
use super::*;
use lamina_core::{ModalFieldBackend, StackSide};
use nalgebra::DMatrix;

#[derive(Clone, Debug)]
pub struct ModalPairing {
    pub registration: GuidedOverlapRegistration,
    /// ⟨a, b⟩, W/m.
    pub unconjugated_w_per_m: Complex,
    /// (a, b), W/m.
    pub conjugated_w_per_m: Complex,
    /// Per partition interval, left to right: [unconjugated, conjugated].
    pub regional_w_per_m: Vec<[Complex; 2]>,
    pub partition_cm: Vec<f64>,
    pub sample_count: usize,
    pub quadrature_change_w_per_m: f64,
}

#[derive(Clone, Debug)]
pub struct BasisPairing {
    pub registration: GuidedOverlapRegistration,
    /// Basis indices of the rows (first basis) and columns (second basis):
    /// the normalised modes of each, in key order.
    pub modes: [Vec<usize>; 2],
    /// ⟨f_i, g_j⟩, W/m.
    pub unconjugated: DMatrix<Complex>,
    /// (f_i, g_j), W/m.
    pub conjugated: DMatrix<Complex>,
    /// Of the unconjugated matrix, descending.
    pub singular_values: Vec<f64>,
    /// Condition number of the unconjugated matrix's leading (k + 1) × (k + 1) block.
    pub leading_condition_numbers: Vec<f64>,
    /// max |G − I| of each basis's own unconjugated Gram on the same grid.
    pub self_defects: [f64; 2],
    pub partition_cm: Vec<f64>,
    pub sample_count: usize,
    pub quadrature_change_w_per_m: f64,
    pub status: [BasisStatus; 2],
    pub scope: [BasisScope; 2],
}

#[derive(Debug, thiserror::Error)]
pub enum PairingError {
    #[error("expected modes at one common real vacuum wavelength")]
    IncompatibleWavelength,
    #[error("invalid transverse registration or unrepresentable interface spacing")]
    Registration,
    #[error("the {side:?} terminations differ in kind, registered position or PML profile")]
    Terminations { side: StackSide },
    #[error("mode {operand} is ill-conditioned and has no normalised fields")]
    IllConditioned { operand: usize },
    #[error("field recovery failed for operand {operand}")]
    Fields {
        operand: usize,
        #[source]
        source: Box<FieldRecoveryFailure>,
    },
    #[error("pairing quadrature exceeded its accuracy or sampling budget")]
    Quadrature,
}

/// Both stacks' layouts, when their wavelengths, walls and PMLs agree.
fn compatible(
    stacks: [&ModeStack; 2],
    k0: [f64; 2],
    registration: GuidedOverlapRegistration,
) -> Result<[Layout; 2], PairingError> {
    if k0.iter().any(|k| !k.is_finite() || *k <= 0.0)
        || (k0[0] - k0[1]).abs() > 1e-12 * k0[0].max(k0[1])
    {
        return Err(PairingError::IncompatibleWavelength);
    }
    let layout = |i: usize| {
        Layout::new(stacks[i], registration.origins_cm[i]).ok_or(PairingError::Registration)
    };
    let layouts = [layout(0)?, layout(1)?];
    let [a, b] = layouts.each_ref().map(|l| l.walls);
    let span = match a {
        [Some((_, left)), Some((_, right))] => right - left,
        _ => 0.0,
    };
    let [pa, pb] = layouts.each_ref().map(|l| l.pmls);
    for (side, which) in [StackSide::Left, StackSide::Right].into_iter().enumerate() {
        let walls = matches!(
            (a[side], b[side]),
            (Some((ka, za)), Some((kb, zb))) if ka == kb && (za - zb).abs() <= 1e-12 * span
        );
        let pmls = match (pa[side], pb[side]) {
            (None, None) => true,
            (Some(x), Some(y)) => x.matches(&y, 1e-12 * span),
            _ => false,
        };
        if !(walls && pmls) {
            return Err(PairingError::Terminations { side: which });
        }
    }
    Ok(layouts)
}

/// Both pairings of two normalised basis modes in a registered frame.
pub fn modal_pairing_with<B: ModalFieldBackend<Region> + Clone>(
    backend: &B,
    modes: [&BasisMode; 2],
    registration: GuidedOverlapRegistration,
    settings: FieldRecoverySettings,
) -> Result<ModalPairing, PairingError> {
    if let Some(operand) = modes.iter().position(|m| !m.is_normalised()) {
        return Err(PairingError::IllConditioned { operand });
    }
    let k0 = modes.map(|m| m.point.signature().map_or(f64::NAN, |s| s.k0.re));
    let stacks = modes.map(|m| m.point.provenance().stack().clone());
    let layouts = compatible([&stacks[0], &stacks[1]], k0, registration)?;
    let partition = split_pmls(
        Layout::partition(&[&layouts[0], &layouts[1]]),
        &[0, 1].map(|i| {
            (
                &layouts[i],
                &stacks[i],
                modes[i].point.provenance().polarisation(),
            )
        }),
        k0[0],
        &modes.map(|m| m.point.physical_coordinate()),
    );
    let result = converge(
        &partition,
        || PairingError::Quadrature,
        |panels, _| {
            let fields = |operand: usize| {
                let layout = &layouts[operand];
                sampled(
                    backend,
                    modes[operand],
                    operand,
                    layout,
                    &partition,
                    panels,
                    settings,
                )
            };
            let (a, b) = (fields(0)?, fields(1)?);
            let u = intervals(&partition, panels, |i| unconjugated(&a[i], &b[i]));
            let c = intervals(&partition, panels, |i| conjugated(&a[i], &b[i]));
            // The totals, then each interval's contributions, all converged.
            let mut entries = vec![u.iter().sum(), c.iter().sum()];
            entries.extend(u.iter().zip(&c).flat_map(|(u, c)| [*u, *c]));
            Ok((entries, ()))
        },
    )?;
    let e = &result.entries;
    Ok(ModalPairing {
        registration,
        unconjugated_w_per_m: e[0],
        conjugated_w_per_m: e[1],
        regional_w_per_m: e[2..].chunks(2).map(|r| [r[0], r[1]]).collect(),
        partition_cm: partition,
        sample_count: result.sample_count,
        quadrature_change_w_per_m: result.change,
    })
}

fn sampled<B: ModalFieldBackend<Region> + Clone>(
    backend: &B,
    mode: &BasisMode,
    operand: usize,
    layout: &Layout,
    partition: &[f64],
    panels: usize,
    settings: FieldRecoverySettings,
) -> Result<Vec<Sample>, PairingError> {
    let sampling = layout.sampling(partition, panels, false);
    let mut samples = mode
        .fields_with(backend.clone(), &sampling, settings)
        .map_err(|e| match e {
            BasisFieldError::IllConditioned => PairingError::IllConditioned { operand },
            BasisFieldError::Recovery(source) => PairingError::Fields { operand, source },
        })?;
    equivalent(&mut samples, &layout.stretches(partition, panels));
    Ok(samples)
}

/// Both pairing matrices between the normalised modes of two bases, with
/// conditioning and each basis's own Gram defect, all from one shared grid
/// at one refinement level. Incomplete bases are accepted.
pub fn basis_pairing_with<B: ModalFieldBackend<Region> + Clone>(
    backend: &B,
    bases: [&ModalBasis; 2],
    registration: GuidedOverlapRegistration,
) -> Result<BasisPairing, PairingError> {
    let layouts = compatible(
        [&bases[0].stack, &bases[1].stack],
        bases.map(|b| b.request.k0_cm()),
        registration,
    )?;
    let modes = bases.map(|basis| {
        (0..basis.modes.len())
            .filter(|&i| basis.modes[i].is_normalised())
            .collect::<Vec<_>>()
    });
    let betas: Vec<_> = (0..2)
        .flat_map(|b| {
            modes[b]
                .iter()
                .map(move |&i| bases[b].modes[i].point.physical_coordinate())
        })
        .collect();
    let partition = split_pmls(
        Layout::partition(&[&layouts[0], &layouts[1]]),
        &[0, 1].map(|i| {
            let ModeFamily::Polarised(polarisation) = bases[i].request.family;
            (&layouts[i], &bases[i].stack, polarisation)
        }),
        bases[0].request.k0_cm(),
        &betas,
    );
    let [n, m] = [modes[0].len(), modes[1].len()];
    let result = converge(
        &partition,
        || PairingError::Quadrature,
        |panels, _| {
            let fields = |operand: usize| {
                modes[operand]
                    .iter()
                    .map(|&i| {
                        let basis = bases[operand];
                        let mode = &basis.modes[i];
                        let layout = &layouts[operand];
                        let settings = basis.request.recovery;
                        sampled(backend, mode, operand, layout, &partition, panels, settings)
                    })
                    .collect::<Result<Vec<_>, _>>()
            };
            let (a, b) = (fields(0)?, fields(1)?);
            let pair = |x: &[Sample], y: &[Sample], density: fn(&Sample, &Sample) -> Complex| {
                integral(&partition, panels, |i| density(&x[i], &y[i]))
            };
            let mut entries = Vec::with_capacity(2 * n * m + n * n + m * m);
            for f in &a {
                for g in &b {
                    entries.push(pair(f, g, unconjugated));
                    entries.push(pair(f, g, conjugated));
                }
            }
            for set in [&a, &b] {
                for f in set.iter() {
                    for g in set.iter() {
                        entries.push(pair(f, g, unconjugated));
                    }
                }
            }
            Ok((entries, ()))
        },
    )?;
    let e = &result.entries;
    let unconjugated = DMatrix::from_fn(n, m, |i, j| e[2 * (i * m + j)]);
    let conjugated = DMatrix::from_fn(n, m, |i, j| e[2 * (i * m + j) + 1]);
    let defect = |offset: usize, k: usize| {
        (0..k * k)
            .map(|p| {
                let identity = if p / k == p % k { 1.0 } else { 0.0 };
                (e[offset + p] - identity).norm()
            })
            .fold(0.0, f64::max)
    };
    let condition = |block: DMatrix<Complex>| {
        let values = block.singular_values();
        values.max() / values.min()
    };
    let mut singular_values = if n == 0 || m == 0 {
        vec![]
    } else {
        unconjugated.clone().singular_values().as_slice().to_vec()
    };
    singular_values.sort_by(|a, b| b.total_cmp(a));
    Ok(BasisPairing {
        registration,
        leading_condition_numbers: (1..=n.min(m))
            .map(|k| condition(unconjugated.view((0, 0), (k, k)).into_owned()))
            .collect(),
        singular_values,
        unconjugated,
        conjugated,
        self_defects: [defect(2 * n * m, n), defect(2 * n * m + n * n, m)],
        modes,
        partition_cm: partition,
        sample_count: result.sample_count,
        quadrature_change_w_per_m: result.change,
        status: bases.map(|b| b.status),
        scope: bases.map(|b| b.scope),
    })
}

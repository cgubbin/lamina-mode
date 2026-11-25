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
//!
//! Beyond the partition, a side open in both stacks contributes a closed-form
//! tail: unconjugated i v₀/(κ_a + κ_b) by analytic continuation, the integral
//! along a rotated ray on which both fields decay, for guided and leaky modes
//! alike; conjugated i v₀/(κ_a − κ_b*) only where Im κ_a + Im κ_b > 0, and
//! otherwise absent, never a regularised number.
use super::basis::{BasisFieldError, BasisMode, BasisScope, BasisStatus, ModalBasis};
use super::integration::{
    Kappas, Layout, Sample, TailFailure, conjugated, converge, equivalent, integral, intervals,
    split_pmls, tails, unconjugated,
};
use super::*;
use lamina_core::{ModalFieldBackend, StackSide};
use nalgebra::DMatrix;

#[derive(Clone, Debug)]
pub struct ModalPairing {
    pub registration: GuidedOverlapRegistration,
    /// ⟨a, b⟩, W/m.
    pub unconjugated_w_per_m: Complex,
    /// (a, b), W/m; absent when an open tail diverges.
    pub conjugated_w_per_m: Result<Complex, Absent>,
    /// Per partition interval, left to right: [unconjugated, conjugated].
    /// Open tails are not included.
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
    /// (f_i, g_j), W/m; absent where an open tail diverges.
    pub conjugated: DMatrix<Result<Complex, Absent>>,
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

/// Why a conjugated pairing has no value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Absent {
    /// The conjugated tail on `side` diverges: Im κ_a + Im κ_b ≤ 0.
    Unbounded { side: StackSide },
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
    #[error("the {side:?} open tail is singular: |κ_a + κ_b| is below its floor")]
    TailSingular { side: StackSide },
}

/// Each side's exterior wavevector of a mode, `None` on closed sides.
pub(super) fn kappas(mode: &BasisMode) -> Kappas {
    mode.point
        .signature()
        .map_or([None; 2], |s| [s.left, s.right])
}

/// The open tails of a pair: unconjugated, and conjugated unless absent.
fn pair_tails(
    a: &[Sample],
    b: &[Sample],
    kappas: [Kappas; 2],
    k0: f64,
) -> Result<(Complex, Result<Complex, Absent>), PairingError> {
    let singular = |side| PairingError::TailSingular { side };
    let unconjugated = tails(a, b, kappas, k0, false).map_err(|e| match e {
        TailFailure::Singular(side) | TailFailure::Unbounded(side) => singular(side),
    })?;
    let conjugated = match tails(a, b, kappas, k0, true) {
        Ok(value) => Ok(value),
        Err(TailFailure::Unbounded(side)) => Err(Absent::Unbounded { side }),
        Err(TailFailure::Singular(side)) => return Err(singular(side)),
    };
    Ok((unconjugated, conjugated))
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
    let span = Layout::partition(&[&layouts[0], &layouts[1]])
        .iter()
        .fold([f64::INFINITY, f64::NEG_INFINITY], |[lo, hi], &z| {
            [lo.min(z), hi.max(z)]
        });
    let span = span[1] - span[0];
    let [pa, pb] = layouts.each_ref().map(|l| l.pmls);
    for (side, which) in [StackSide::Left, StackSide::Right].into_iter().enumerate() {
        // Open in both stacks, or walls of one kind at one registered z.
        let walls = match (a[side], b[side]) {
            (None, None) => true,
            (Some((ka, za)), Some((kb, zb))) => ka == kb && (za - zb).abs() <= 1e-12 * span,
            _ => false,
        };
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
                    layout,
                    &partition,
                    panels,
                    settings,
                )
                .map_err(|e| field_failure(e, operand))
            };
            let (a, b) = (fields(0)?, fields(1)?);
            let u = intervals(&partition, panels, |i| unconjugated(&a[i], &b[i]));
            let c = intervals(&partition, panels, |i| conjugated(&a[i], &b[i]));
            let (tail, conjugated_tail) = pair_tails(&a, &b, modes.map(kappas), k0[0])?;
            // The totals, then each interval's contributions, all converged.
            let mut entries = vec![
                u.iter().sum::<Complex>() + tail,
                c.iter().sum::<Complex>() + conjugated_tail.unwrap_or(Complex::ZERO),
            ];
            entries.extend(u.iter().zip(&c).flat_map(|(u, c)| [*u, *c]));
            Ok((entries, conjugated_tail.map(|_| ())))
        },
    )?;
    let e = &result.entries;
    Ok(ModalPairing {
        registration,
        unconjugated_w_per_m: e[0],
        conjugated_w_per_m: result.extra.map(|()| e[1]),
        regional_w_per_m: e[2..].chunks(2).map(|r| [r[0], r[1]]).collect(),
        partition_cm: partition,
        sample_count: result.sample_count,
        quadrature_change_w_per_m: result.change,
    })
}

fn field_failure(e: BasisFieldError, operand: usize) -> PairingError {
    match e {
        BasisFieldError::IllConditioned => PairingError::IllConditioned { operand },
        BasisFieldError::Recovery(source) => PairingError::Fields { operand, source },
    }
}

/// A normalised mode's equivalent-medium samples on `partition` at `panels`.
pub(super) fn sampled<B: ModalFieldBackend<Region> + Clone>(
    backend: &B,
    mode: &BasisMode,
    layout: &Layout,
    partition: &[f64],
    panels: usize,
    settings: FieldRecoverySettings,
) -> Result<Vec<Sample>, BasisFieldError> {
    let sampling = layout.sampling(partition, panels, false);
    let mut samples = mode.fields_with(backend.clone(), &sampling, settings)?;
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
    let k0 = bases[0].request.k0_cm();
    let kappa = [0, 1].map(|b| {
        modes[b]
            .iter()
            .map(|&i| kappas(&bases[b].modes[i]))
            .collect::<Vec<_>>()
    });
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
                        sampled(backend, mode, layout, &partition, panels, settings)
                            .map_err(|e| field_failure(e, operand))
                    })
                    .collect::<Result<Vec<_>, _>>()
            };
            let (a, b) = (fields(0)?, fields(1)?);
            let pair = |x: &[Sample], y: &[Sample], density: fn(&Sample, &Sample) -> Complex| {
                integral(&partition, panels, |i| density(&x[i], &y[i]))
            };
            let mut entries = Vec::with_capacity(2 * n * m + n * n + m * m);
            let mut absent = Vec::with_capacity(n * m);
            for (f, kf) in a.iter().zip(&kappa[0]) {
                for (g, kg) in b.iter().zip(&kappa[1]) {
                    let (tail, conjugated_tail) = pair_tails(f, g, [*kf, *kg], k0)?;
                    entries.push(pair(f, g, unconjugated) + tail);
                    entries.push(pair(f, g, conjugated) + conjugated_tail.unwrap_or(Complex::ZERO));
                    absent.push(conjugated_tail.err());
                }
            }
            for (set, k) in [(&a, &kappa[0]), (&b, &kappa[1])] {
                for (f, kf) in set.iter().zip(k) {
                    for (g, kg) in set.iter().zip(k) {
                        let (tail, _) = pair_tails(f, g, [*kf, *kg], k0)?;
                        entries.push(pair(f, g, unconjugated) + tail);
                    }
                }
            }
            Ok((entries, absent))
        },
    )?;
    let e = &result.entries;
    let unconjugated = DMatrix::from_fn(n, m, |i, j| e[2 * (i * m + j)]);
    let conjugated = DMatrix::from_fn(n, m, |i, j| match result.extra[i * m + j] {
        Some(cause) => Err(cause),
        None => Ok(e[2 * (i * m + j) + 1]),
    });
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

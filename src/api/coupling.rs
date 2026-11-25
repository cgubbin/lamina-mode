//! Local perturbation matrices for nonmagnetic guided planar modes, under the
//! power overlap, and for the modes of a modal basis, under the unconjugated
//! pairing. Regions are isotropic or diagonal in the stack frame; each field
//! component is weighted by its own permittivity entry.

use super::basis::{BasisFieldError, BasisScope, BasisStatus, ModalBasis, ModeFamily};
use super::integration::{Sample, converge};
use super::*;
use crate::problem::material::{self, Principal};
use lamina_core::ModalFieldBackend;
use nalgebra::DMatrix;

#[derive(Debug)]
pub struct GuidedLocalCoupling {
    pub modes: Vec<ModePoint>,
    pub direction: GuidedPerturbation,
    /// Hermitian perturbation matrix in inverse centimetres per parameter unit.
    /// Diagonal entries give signed-power-weighted propagation-constant rates.
    pub perturbation_matrix: DMatrix<Complex>,
    /// Row is the receiving mode, column is the differentiated mode.
    /// The diagonal and entries within unresolved groups are absent.
    pub coefficients: DMatrix<Option<Complex>>,
    pub unresolved_groups: Vec<Vec<usize>>,
    pub quadrature_change_per_cm: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum GuidedCouplingError {
    #[error(
        "expected a complete common nonmagnetic dielectric basis and a fixed-wavelength direction"
    )]
    Ineligible,
    #[error("invalid propagation-constant gap threshold")]
    Threshold,
    #[error("mode {mode} field recovery failed")]
    Fields {
        mode: usize,
        #[source]
        source: GuidedPowerError,
    },
    #[error("coupling quadrature exceeded its accuracy or sampling budget")]
    Quadrature,
}

/// Form local coupling coefficients with the positive-left-primary phase rule.
/// `gap_atol_cm` is an absolute separation in inverse centimetres.
pub fn guided_local_coupling_with<B: ModalFieldBackend<Region> + Clone>(
    backend: &B,
    basis: &GuidedBasis,
    direction: &GuidedPerturbation,
    gap_atol_cm: f64,
    settings: FieldRecoverySettings,
) -> Result<GuidedLocalCoupling, GuidedCouplingError> {
    use GuidedCouplingError::*;
    let modes = &basis.modes;
    if !gap_atol_cm.is_finite() || gap_atol_cm < 0.0 {
        return Err(Threshold);
    }
    let first = modes.first().ok_or(Ineligible)?.provenance();
    let stack = first.stack();
    let k = modes[0].signature().map_err(|_| Ineligible)?.k0;
    let polarisation = first.polarisation();
    let sign = if basis.request.direction == GuidedDirection::Forward {
        1.0
    } else {
        -1.0
    };
    if !basis.is_complete()
        || direction.wavelength_rate() != 0.0
        || direction.thickness_rates().len() != stack.len()
        || modes.iter().any(|mode| {
            let provenance = mode.provenance();
            provenance.stack() != stack
                || provenance.polarisation() != polarisation
                || mode.signature().map_or(true, |s| {
                    (s.k0 - k).norm() > 1e-12 * k.re || s.k_parallel.re * sign <= 0.0
                })
        })
    {
        return Err(Ineligible);
    }
    let rates: Vec<_> = direction
        .regional_rates()
        .iter()
        .map(|rate| rate.tensors()[0])
        .collect();
    let epsilon = material::regions(stack)
        .zip(direction.regional_rates())
        .map(|((region, material), rate)| {
            let coordinate = Complex::new(k.re, 0.0);
            let [e, m] = material::tensors(region, material, coordinate).map_err(|_| Ineligible)?;
            if m.entries() != [Complex::ONE; 3]
                || rate.tensors()[1].entries() != [0.0; 3]
                || e.entries()
                    .iter()
                    .any(|e| e.im != 0.0 || e.re <= 0.0 || !e.re.is_finite())
            {
                return Err(Ineligible);
            }
            Ok(e.map(|e| e.re))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let widths: Vec<_> = stack
        .iter()
        .map(|layer| layer.thickness().as_centimetres())
        .collect();
    let layout = super::overlap::interface_positions(&modes[0], 0.0).map_err(|_| Ineligible)?;
    let interfaces = &layout.interfaces;
    let betas: Vec<_> = modes
        .iter()
        .map(|mode| mode.physical_coordinate().re)
        .collect();
    let decay: Vec<_> = modes
        .iter()
        .map(|mode| {
            let s = mode.signature().expect("validated signature");
            s.open_exteriors().expect("an open stack").map(|r| r.im)
        })
        .collect();

    let n = modes.len();
    let prefactor = 0.25 * k.re * super::power::CM_TO_M_OVER_Z0;
    let mut previous: Option<DMatrix<Complex>> = None;
    let mut accepted = None;
    for level in 0..=11 {
        let panels = 8 << level;
        let count = widths.len().saturating_mul(panels + 1).saturating_add(2);
        if count.saturating_mul(n) > 262144 {
            return Err(Quadrature);
        }
        let sampling = layout.sampling(interfaces, panels, true);
        let fields = modes
            .iter()
            .enumerate()
            .map(|(mode, point)| {
                point
                    .guided_power_profile_with(backend.clone(), &sampling, settings)
                    .map_err(|source| Fields { mode, source })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut matrix = DMatrix::zeros(n, n);
        for i in 0..n {
            let a = fields[i].fields.quantity().electric();
            for j in i..n {
                let b = fields[j].fields.quantity().electric();
                let components = |p: usize| {
                    [
                        a.x()[p].conj() * b.x()[p],
                        a.y()[p].conj() * b.y()[p],
                        a.z()[p].conj() * b.z()[p],
                    ]
                };
                let tangential = |p| a.x()[p].conj() * b.x()[p] + a.y()[p].conj() * b.y()[p];
                let normal = |p| a.z()[p].conj() * b.z()[p];
                let mut value = rates[0].weigh(components(0)) / (decay[i][0] + decay[j][0])
                    + rates[rates.len() - 1].weigh(components(count - 1))
                        / (decay[i][1] + decay[j][1]);
                for (layer, width) in widths.iter().enumerate() {
                    let offset = 1 + layer * (panels + 1);
                    match rates[layer + 1] {
                        Principal::Isotropic(0.0) => {}
                        Principal::Isotropic(rate) => {
                            let product = |p| tangential(p) + normal(p);
                            let sum = super::integration::simpson_sum(offset, panels, product);
                            value += rate * width / (3.0 * panels as f64) * sum;
                        }
                        rate => {
                            let product = |p| rate.weigh(components(p));
                            let sum = super::integration::simpson_sum(offset, panels, product);
                            value += width / (3.0 * panels as f64) * sum;
                        }
                    }
                }
                for (interface, &velocity) in direction.interface_velocities().iter().enumerate() {
                    // Use the left-side limit. Tangential E and normal D are continuous.
                    let p = if interface == 0 {
                        0
                    } else {
                        interface * (panels + 1)
                    };
                    // Tangential E and normal D = ε_zz E_z are continuous.
                    let jump = match (epsilon[interface], epsilon[interface + 1]) {
                        (Principal::Isotropic(left), Principal::Isotropic(right)) => {
                            (left - right) * tangential(p)
                                - (left.recip() - right.recip()) * left * left * normal(p)
                        }
                        (left, right) => {
                            let (l, r) = (left.entries(), right.entries());
                            let [x, y, z] = components(p);
                            (l[0] - r[0]) * x + (l[1] - r[1]) * y
                                - (l[2].recip() - r[2].recip()) * l[2] * l[2] * z
                        }
                    };
                    value += velocity * jump;
                }
                let value = prefactor * value;
                matrix[(i, j)] = value;
                matrix[(j, i)] = value.conj();
            }
        }
        if let Some(previous) = previous {
            let change = (&matrix - previous).norm();
            if change <= 1e-9 * matrix.norm().max(1.0) {
                accepted = Some((matrix, change));
                break;
            }
        }
        previous = Some(matrix);
    }
    let (perturbation_matrix, quadrature_change_per_cm) = accepted.ok_or(Quadrature)?;
    let mut ordered: Vec<_> = (0..n).collect();
    ordered.sort_by(|&i, &j| betas[i].total_cmp(&betas[j]));
    let mut group_ids = vec![0; n];
    let mut groups = vec![vec![ordered[0]]];
    for pair in ordered.windows(2) {
        if (betas[pair[0]] - betas[pair[1]]).abs() > gap_atol_cm {
            groups.push(Vec::new());
        }
        group_ids[pair[1]] = groups.len() - 1;
        groups.last_mut().expect("at least one group").push(pair[1]);
    }
    let coefficients = DMatrix::from_fn(n, n, |i, j| {
        (group_ids[i] != group_ids[j])
            .then(|| sign * perturbation_matrix[(i, j)] / (betas[j] - betas[i]))
    });
    let unresolved_groups = groups
        .into_iter()
        .filter(|members| members.len() > 1)
        .collect();
    Ok(GuidedLocalCoupling {
        modes: modes.clone(),
        direction: direction.clone(),
        perturbation_matrix,
        coefficients,
        unresolved_groups,
        quadrature_change_per_cm,
    })
}

/// Local coupling between the normalised modes of a modal basis along a
/// fixed-wavelength direction:
///
/// ```text
/// K_mn = (ω ε0/4) [∫ Ẽ_m · ∂ε/∂p · E_n dz̃ + Σ_k v_k (jump terms at interface k)],
/// C_mn = K_mn / (β_n − β_m),
/// ```
///
/// with Ẽ = (−E_x, E_y, E_z) the electric field of the mirrored mode, dz̃ in a
/// PML and continued tails on open sides. Lorentz reciprocity between the
/// mirrored mode m and the perturbed mode n gives dβ_n/dp = K_nn and
/// C_mn = ⟨f_m, ∂f_n/∂p⟩ under unit unconjugated self-pairing.
#[derive(Debug)]
pub struct BasisLocalCoupling {
    /// Basis indices of the normalised modes, in key order: the rows and
    /// columns of both matrices.
    pub modes: Vec<usize>,
    pub direction: GuidedPerturbation,
    /// Complex symmetric, in inverse centimetres per parameter unit; the
    /// diagonal is each mode's dβ/dp.
    pub perturbation_matrix: DMatrix<Complex>,
    /// Row is the receiving mode, column the differentiated mode, per
    /// parameter unit. The diagonal and entries within a group are absent.
    pub coefficients: DMatrix<Option<Complex>>,
    /// Modes whose β differ by at most `gap_cm`, linked into groups of two
    /// or more, as basis indices.
    pub groups: Vec<Vec<usize>>,
    pub gap_cm: f64,
    pub partition_cm: Vec<f64>,
    pub sample_count: usize,
    pub quadrature_change_per_cm: f64,
    pub status: BasisStatus,
    pub scope: BasisScope,
}

#[derive(Debug, thiserror::Error)]
pub enum BasisCouplingError {
    #[error("the direction varies the vacuum wavelength; coupling is at fixed wavelength")]
    Wavelength,
    #[error(
        "the direction has a permeability rate, or moves an interface across a jump in permeability"
    )]
    Magnetic,
    #[error("the direction moves an interface against the {side:?} wall or PML")]
    Termination { side: lamina_core::StackSide },
    #[error("the direction does not match the stack's interfaces and regions")]
    Geometry,
    #[error("invalid propagation-constant gap threshold")]
    Threshold,
    #[error("the basis has no normalised mode")]
    Empty,
    #[error("a region's constitutive tensors cannot be evaluated at the basis's wavelength")]
    Materials,
    #[error("field recovery failed for mode {mode}")]
    Fields {
        mode: usize,
        #[source]
        source: Box<FieldRecoveryFailure>,
    },
    #[error("coupling quadrature exceeded its accuracy or sampling budget")]
    Quadrature,
    #[error("the {side:?} open tail is singular: |κ_a + κ_b| is below its floor")]
    TailSingular { side: lamina_core::StackSide },
}

/// Local coupling of the normalised modes of `basis` along `direction`.
/// `gap_cm` (default: the basis's degeneracy gap) links modes into groups.
pub fn basis_local_coupling_with<B: ModalFieldBackend<Region> + Clone>(
    backend: &B,
    basis: &ModalBasis,
    direction: &GuidedPerturbation,
    gap_cm: Option<f64>,
) -> Result<BasisLocalCoupling, BasisCouplingError> {
    use super::integration::{Layout, simpson_sum, split_pmls};
    use super::pairing::kappas;
    use BasisCouplingError::*;
    use lamina_core::StackSide;
    let stack = &basis.stack;
    let layers = stack.len();
    if direction.interface_velocities().len() != layers + 1 {
        return Err(Geometry);
    }
    if direction.wavelength_rate() != 0.0 {
        return Err(Wavelength);
    }
    let rates: Vec<_> = direction
        .regional_rates()
        .iter()
        .map(|rate| rate.tensors())
        .collect();
    if rates.iter().any(|[_, mu]| mu.entries() != [0.0; 3]) {
        return Err(Magnetic);
    }
    let gap = gap_cm.unwrap_or(basis.degeneracy_gap_cm);
    if !gap.is_finite() || gap < 0.0 {
        return Err(Threshold);
    }
    let k0 = basis.request.k0_cm();
    let at = Complex::new(k0, 0.0);
    let tensors = material::regions(stack)
        .map(|(region, m)| material::tensors(region, m, at))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Materials)?;
    let layout = Layout::new(stack, 0.0).ok_or(Geometry)?;
    let interfaces = &layout.interfaces;
    let span = interfaces[layers] - interfaces[0];
    let closed = [StackSide::Left, StackSide::Right]
        .into_iter()
        .enumerate()
        .find(|&(side, _)| layout.walls[side].is_some());
    for (k, &velocity) in direction.interface_velocities().iter().enumerate() {
        if velocity == 0.0 {
            continue;
        }
        // Walls are fixed in the stack frame, whose origin is the first interface.
        if let (0, Some((_, side))) = (k, closed) {
            return Err(Termination { side });
        }
        let z = interfaces[k];
        for (side, which) in [StackSide::Left, StackSide::Right].into_iter().enumerate() {
            let near = |plane: f64| (plane - z).abs() <= 1e-12 * span.max(f64::MIN_POSITIVE);
            if layout.walls[side].is_some_and(|(_, w)| near(w))
                || layout.pmls[side].is_some_and(|p| near(p.start))
            {
                return Err(Termination { side: which });
            }
        }
        if tensors[k][1].entries() != tensors[k + 1][1].entries() {
            return Err(Magnetic);
        }
    }
    let modes: Vec<usize> = (0..basis.modes.len())
        .filter(|&i| basis.modes[i].is_normalised())
        .collect();
    if modes.is_empty() {
        return Err(Empty);
    }
    let n = modes.len();
    let betas: Vec<_> = modes
        .iter()
        .map(|&i| basis.modes[i].point.physical_coordinate())
        .collect();
    let kappa: Vec<_> = modes.iter().map(|&i| kappas(&basis.modes[i])).collect();
    let ModeFamily::Polarised(polarisation) = basis.request.family;
    let partition = split_pmls(
        Layout::partition(&[&layout]),
        &[(&layout, stack, polarisation)],
        k0,
        &betas,
    );
    // Each interval's region: 0 the left exterior (gap and PML), 1..=layers
    // the layers, layers + 1 the right exterior.
    let regions: Vec<usize> = partition
        .windows(2)
        .map(|e| {
            let middle = 0.5 * (e[0] + e[1]);
            interfaces.partition_point(|&z| z <= middle)
        })
        .collect();
    // The left-side limit's sample index at each interface.
    let limits: Vec<Option<usize>> = interfaces
        .iter()
        .map(|&z| partition.iter().position(|&p| p == z))
        .collect();
    // Ẽ_a ∘ E_b componentwise: the mirrored field's electric part.
    let mirrored = |a: &Sample, b: &Sample| [-a[0] * b[0], a[1] * b[1], a[2] * b[2]];
    let prefactor = 0.25 * k0 * super::power::CM_TO_M_OVER_Z0;
    let converged = converge(
        &partition,
        || Quadrature,
        |panels, _| {
            let sampling = layout.sampling(&partition, panels, false);
            let stretches = layout.stretches(&partition, panels);
            let fields = modes
                .iter()
                .map(|&i| {
                    basis.modes[i]
                        .fields_with(backend.clone(), &sampling, basis.request.recovery)
                        .map_err(|e| match e {
                            BasisFieldError::Recovery(source) => Fields { mode: i, source },
                            BasisFieldError::IllConditioned => Empty,
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut entries = Vec::with_capacity(n * (n + 1) / 2);
            for m in 0..n {
                let a = &fields[m];
                for j in m..n {
                    let b = &fields[j];
                    let mut value = Complex::ZERO;
                    for (interval, edges) in partition.windows(2).enumerate() {
                        let rate = rates[regions[interval]][0];
                        if rate == Principal::Isotropic(0.0) {
                            continue;
                        }
                        let offset = 1 + interval * (panels + 1);
                        let density = |s: usize| rate.weigh(mirrored(&a[s], &b[s])) * stretches[s];
                        let width = edges[1] - edges[0];
                        value +=
                            simpson_sum(offset, panels, density) * (width / (3.0 * panels as f64));
                    }
                    for (side, which, anchor, region) in [
                        (0, StackSide::Left, 0, 0),
                        (1, StackSide::Right, a.len() - 1, layers + 1),
                    ] {
                        let rate = rates[region][0];
                        let (Some(ka), Some(kb)) = (kappa[m][side], kappa[j][side]) else {
                            continue;
                        };
                        if rate == Principal::Isotropic(0.0) {
                            continue;
                        }
                        let sum = ka + kb;
                        if sum.norm().is_nan() || sum.norm() < super::open::TAIL_SINGULAR_FLOOR * k0
                        {
                            return Err(TailSingular { side: which });
                        }
                        value += Complex::I * rate.weigh(mirrored(&a[anchor], &b[anchor])) / sum;
                    }
                    for (k, &velocity) in direction.interface_velocities().iter().enumerate() {
                        if velocity == 0.0 {
                            continue;
                        }
                        let p = match limits[k] {
                            Some(0) => 0,
                            Some(q) => q * (panels + 1),
                            None => return Err(Geometry),
                        };
                        // Tangential E and normal D = ε_zz E_z are continuous.
                        let (l, r) = (tensors[k][0].entries(), tensors[k + 1][0].entries());
                        let [x, y, z] = mirrored(&a[p], &b[p]);
                        value += velocity
                            * ((l[0] - r[0]) * x + (l[1] - r[1]) * y
                                - (l[2].inv() - r[2].inv()) * l[2] * l[2] * z);
                    }
                    entries.push(value * (prefactor / k0));
                }
            }
            Ok((entries, ()))
        },
    )?;
    let mut perturbation_matrix = DMatrix::zeros(n, n);
    let mut p = 0;
    for m in 0..n {
        for j in m..n {
            perturbation_matrix[(m, j)] = converged.entries[p] * k0;
            perturbation_matrix[(j, m)] = converged.entries[p] * k0;
            p += 1;
        }
    }
    // Groups: connected components of |Δβ| ≤ gap.
    let mut component: Vec<usize> = (0..n).collect();
    for i in 0..n {
        for j in 0..i {
            if (betas[i] - betas[j]).norm() <= gap {
                let (from, to) = (
                    component[i].max(component[j]),
                    component[i].min(component[j]),
                );
                component
                    .iter_mut()
                    .filter(|c| **c == from)
                    .for_each(|c| *c = to);
            }
        }
    }
    let groups = (0..n)
        .filter(|&i| component[i] == i)
        .map(|i| {
            (i..n)
                .filter(|&j| component[j] == i)
                .map(|j| modes[j])
                .collect::<Vec<_>>()
        })
        .filter(|g| g.len() > 1)
        .collect();
    let coefficients = DMatrix::from_fn(n, n, |m, j| {
        (component[m] != component[j]).then(|| perturbation_matrix[(m, j)] / (betas[j] - betas[m]))
    });
    Ok(BasisLocalCoupling {
        modes,
        direction: direction.clone(),
        perturbation_matrix,
        coefficients,
        groups,
        gap_cm: gap,
        partition_cm: partition,
        sample_count: converged.sample_count,
        quadrature_change_per_cm: converged.change * k0,
        status: basis.status,
        scope: basis.scope,
    })
}

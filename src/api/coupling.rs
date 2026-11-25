//! Local perturbation matrix for nonmagnetic guided planar modes. Regions are
//! isotropic or diagonal in the stack frame; each field component is weighted
//! by its own permittivity entry.

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

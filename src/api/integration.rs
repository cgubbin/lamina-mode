//! Transverse quadrature shared by the pairings of planar modes.
//!
//! Stacks are placed in one registered frame. The partition is the union of
//! their interfaces, of the starts of their PMLs and of the planes of the
//! walls closing them, a PML's backing wall included; every partition point is
//! sampled from both sides, each interval by composite Simpson, and
//! refinement doubles the panels of every interval at once. An open exterior
//! beyond the partition enters as a closed-form tail.
//!
//! Inside a PML the sampled fields are the exterior solution continued to the
//! stretched coordinate ζ̃(ζ). The equivalent medium in real coordinates has
//! the same tangential fields and normal components s E_z and s H_z, so the
//! densities of the equivalent-medium fields, integrated in real ζ, give the
//! conjugated product of that medium and the unconjugated pairing as the
//! contour integral ∫ (E × H) · x̂ s dζ = ∫ (E × H) · x̂ dζ̃.
use super::*;
use lamina_core::{
    AnalyticalMaterialStack, EvaluateMeromorphicMaterial, ExteriorSampling, FieldSampling,
    LayerSampling, OwnedModeFields, Scalar, StretchProfile, WallKind,
};
use lamina_units::Length;

/// Most samples per mode at one refinement level.
const SAMPLE_BUDGET: usize = 131072;

/// Fields at one sample: [Ex, Ey, Ez, Z0 Hx, Z0 Hy, Z0 Hz].
pub(super) type Sample = [Complex; 6];

/// A PML in a registered frame, in cm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Stretch {
    /// Registered position of the inner face.
    pub start: f64,
    pub thickness: f64,
    pub order: f64,
    pub strength: Complex,
}

impl Stretch {
    /// s(ζ) = 1 + c (ζ/d)^m at outward depth ζ, clamped to [0, d].
    pub fn at(&self, depth: f64) -> Complex {
        let fraction = (depth / self.thickness).clamp(0.0, 1.0);
        1.0 + self.strength * fraction.powf(self.order)
    }

    /// ζ̃(ζ) = ζ + c d/(m+1) (ζ/d)^(m+1) at outward depth ζ, clamped to [0, d].
    pub fn stretched(&self, depth: f64) -> Complex {
        let depth = depth.clamp(0.0, self.thickness);
        let fraction = depth / self.thickness;
        depth
            + self.strength
                * (self.thickness / (self.order + 1.0))
                * fraction.powf(self.order + 1.0)
    }

    /// Equal start, thickness and profile to `tolerance` cm and 1e-12.
    pub fn matches(&self, other: &Self, tolerance: f64) -> bool {
        (self.start - other.start).abs() <= tolerance
            && (self.thickness - other.thickness).abs() <= tolerance
            && (self.order - other.order).abs() <= 1e-12 * self.order.max(1.0)
            && (self.strength - other.strength).norm() <= 1e-12 * self.strength.norm().max(1.0)
    }
}

/// A stack's interfaces, walls and PMLs in a registered frame, in cm.
#[derive(Clone, Debug)]
pub(super) struct Layout {
    pub interfaces: Vec<f64>,
    widths: Vec<f64>,
    /// Each side's wall kind and registered position, a PML's backing wall
    /// included, `None` for an open side.
    pub walls: [Option<(WallKind, f64)>; 2],
    /// Each PML side's stretch.
    pub pmls: [Option<Stretch>; 2],
}

impl Layout {
    /// `None` for a non-finite origin, an unrepresentable interface spacing
    /// or a termination other than a wall or a PML.
    pub fn new(stack: &AnalyticalMaterialStack<Complex>, origin: f64) -> Option<Self> {
        if !origin.is_finite() {
            return None;
        }
        let mut interfaces = vec![origin];
        let mut widths = vec![];
        let mut right = origin;
        for layer in stack.iter() {
            let width = layer.thickness().as_centimetres();
            let left = right;
            right = left + width;
            if !right.is_finite() || right <= left {
                return None;
            }
            widths.push(width);
            interfaces.push(right);
        }
        let closures = crate::problem::closures(stack).ok()?;
        let outward = [-1.0, 1.0];
        let faces = [origin, right];
        let mut walls = [None; 2];
        let mut pmls = [None; 2];
        for side in 0..2 {
            let Some((kind, gap, pml)) = closures[side] else {
                continue;
            };
            let start = faces[side] + outward[side] * gap;
            let thickness = pml.map_or(0.0, |pml| pml.thickness().as_centimetres());
            walls[side] = Some((kind, start + outward[side] * thickness));
            pmls[side] = pml.map(|pml| {
                let StretchProfile::Polynomial { order, strength } = *pml.profile();
                Stretch {
                    start,
                    thickness,
                    order,
                    strength,
                }
            });
        }
        Some(Self {
            interfaces,
            widths,
            walls,
            pmls,
        })
    }

    /// The union of the interfaces of `layouts`, their PML starts and their
    /// wall planes, ascending. Walls and PMLs of compatible layouts coincide
    /// to round-off; the outermost of each side is used, whatever the order
    /// of `layouts`.
    pub fn partition(layouts: &[&Layout]) -> Vec<f64> {
        let mut partition: Vec<_> = layouts
            .iter()
            .flat_map(|layout| layout.interfaces.iter().copied())
            .collect();
        let walls = |side: usize| {
            layouts
                .iter()
                .filter_map(move |l| l.walls[side])
                .map(|(_, z)| z)
        };
        let starts = |side: usize| {
            layouts
                .iter()
                .filter_map(move |l| l.pmls[side])
                .map(|pml| pml.start)
        };
        partition.extend(walls(0).reduce(f64::min));
        partition.extend(walls(1).reduce(f64::max));
        partition.extend(starts(0).reduce(f64::min));
        partition.extend(starts(1).reduce(f64::max));
        partition.sort_by(f64::total_cmp);
        partition.dedup();
        partition
    }

    /// Samples at `panels` Simpson panels per partition interval, each
    /// interval's ends from inside it, between one anchor sample on each
    /// side. An anchor lies on the outermost partition point, or on the
    /// interface when `exterior_at_interface`. Distances into a gap stop at
    /// its wall.
    pub fn sampling(
        &self,
        partition: &[f64],
        panels: usize,
        exterior_at_interface: bool,
    ) -> FieldSampling<f64> {
        let interfaces = &self.interfaces;
        let last = interfaces.len() - 1;
        let reach = |side: usize, d: f64| {
            let d = if exterior_at_interface { 0.0 } else { d };
            let d = match self.walls[side] {
                Some((_, wall)) => d.min((wall - interfaces[side * last]).abs()),
                None => d,
            };
            Length::centimetres(d)
        };
        let mut sampling = FieldSampling::new().left_exterior(ExteriorSampling::point(reach(
            0,
            interfaces[0] - partition[0],
        )));
        for interval in partition.windows(2) {
            let (left, right) = (interval[0], interval[1]);
            // The left boundary belongs to the region immediately to its right.
            let region = interfaces.partition_point(|&edge| edge <= left);
            let coordinates = (0..=panels).map(|i| {
                if i == panels {
                    right
                } else {
                    left + (right - left) * (i as f64 / panels as f64)
                }
            });
            sampling = if region == 0 {
                sampling.left_exterior(ExteriorSampling::distances(
                    coordinates.map(|z| reach(0, interfaces[0] - z)),
                ))
            } else if region > last {
                sampling.right_exterior(ExteriorSampling::distances(
                    coordinates.map(|z| reach(1, z - interfaces[last])),
                ))
            } else {
                let width = self.widths[region - 1];
                sampling.layer(
                    region - 1,
                    LayerSampling::offsets(coordinates.map(|z| {
                        let offset = if z == interfaces[region] {
                            width
                        } else {
                            (z - interfaces[region - 1]).clamp(0.0, width)
                        };
                        Length::centimetres(offset)
                    })),
                )
            };
        }
        sampling.right_exterior(ExteriorSampling::point(reach(
            1,
            partition[partition.len() - 1] - interfaces[last],
        )))
    }
}

/// Most pieces a PML interval is split into.
const MAX_PML_PIECES: usize = 4096;

/// `partition` with each PML interval split into equal pieces whose
/// transverse phase, the largest |κ| over `betas` times the interval's
/// complex thickness ζ̃(ζ₂) − ζ̃(ζ₁), shared out among them, is at most the
/// largest phase |κ| width of an interval outside the PMLs, and at least one
/// radian: panels then resolve a PML as they resolve the layers and gaps.
/// Fields there grow or decay as exp(±iκζ̃).
pub(super) fn split_pmls(
    partition: Vec<f64>,
    stacks: &[(&Layout, &AnalyticalMaterialStack<Complex>)],
    k0: f64,
    betas: &[Complex],
) -> Vec<f64> {
    let at = Scalar(Complex::new(k0, 0.0));
    let kappa = |material: &lamina_core::AnalyticalMaterialHandle<Complex>| {
        let chi = material.evaluate_relative_permittivity_complex(at)
            * material.evaluate_relative_permeability_complex(at)
            * (k0 * k0);
        betas
            .iter()
            .map(|beta| (chi - beta * beta).sqrt().norm())
            .fold(0.0, f64::max)
    };
    // Each interval's phase and whether it lies in a PML.
    let phases: Vec<(f64, bool)> = partition
        .windows(2)
        .map(|edges| {
            let (left, right) = (edges[0], edges[1]);
            let middle = 0.5 * (left + right);
            stacks
                .iter()
                .fold((0.0_f64, false), |(phase, pml), &(layout, stack)| {
                    let last = layout.interfaces.len() - 1;
                    let side = if middle < layout.interfaces[0] {
                        Some(0)
                    } else if middle > layout.interfaces[last] {
                        Some(1)
                    } else {
                        None
                    };
                    let exterior = [stack.left_exterior(), stack.right_exterior()];
                    let (here, inside) = match side {
                        Some(side) => match layout.pmls[side] {
                            Some(stretch) if (middle - stretch.start) * [-1.0, 1.0][side] > 0.0 => {
                                let depth = |z: f64| (z - stretch.start).abs();
                                let length = stretch.stretched(depth(right))
                                    - stretch.stretched(depth(left));
                                (kappa(exterior[side]) * length.norm(), true)
                            }
                            _ => (kappa(exterior[side]) * (right - left), false),
                        },
                        None => {
                            let index = layout.interfaces.partition_point(|&z| z <= middle);
                            let phase = stack
                                .layers_left_to_right()
                                .get(index.saturating_sub(1))
                                .map_or(0.0, |layer| kappa(layer.material()) * (right - left));
                            (phase, false)
                        }
                    };
                    (phase.max(here), pml || inside)
                })
        })
        .collect();
    let reference = phases
        .iter()
        .filter(|(_, pml)| !pml)
        .map(|(phase, _)| *phase)
        .fold(1.0, f64::max);
    let mut split: Vec<f64> = partition.first().copied().into_iter().collect();
    for (edges, &(phase, pml)) in partition.windows(2).zip(&phases) {
        let pieces = if pml && phase.is_finite() {
            ((phase / reference).ceil() as usize).clamp(1, MAX_PML_PIECES)
        } else {
            1
        };
        let (left, right) = (edges[0], edges[1]);
        split.extend((1..pieces).map(|i| left + (right - left) * (i as f64 / pieces as f64)));
        split.push(right);
    }
    split
}

impl Layout {
    /// The stretch s at each sample of [`Layout::sampling`] for `partition`
    /// at `panels`: s(ζ) inside a PML, by the interval holding the sample,
    /// and one elsewhere.
    pub fn stretches(&self, partition: &[f64], panels: usize) -> Vec<Complex> {
        let mut stretches = vec![Complex::ONE];
        for interval in partition.windows(2) {
            let (left, right) = (interval[0], interval[1]);
            let middle = 0.5 * (left + right);
            let pml = match self.pmls {
                [Some(pml), _] if middle < pml.start => Some((pml, -1.0)),
                [_, Some(pml)] if middle > pml.start => Some((pml, 1.0)),
                _ => None,
            };
            stretches.extend((0..=panels).map(|i| {
                let z = left + (right - left) * (i as f64 / panels as f64);
                pml.map_or(Complex::ONE, |(pml, outward)| {
                    pml.at(outward * (z - pml.start))
                })
            }));
        }
        stretches.push(Complex::ONE);
        stretches
    }
}

/// The equivalent-medium fields in real coordinates: normal components
/// E_z and Z0 H_z multiplied by the stretch, tangential ones unchanged.
pub(super) fn equivalent(samples: &mut [Sample], stretches: &[Complex]) {
    for (sample, s) in samples.iter_mut().zip(stretches) {
        sample[2] *= s;
        sample[5] *= s;
    }
}

/// Samples of [`Layout::sampling`] for `partition` at `panels`, within budget.
pub(super) fn sample_count(partition: &[f64], panels: usize) -> Option<usize> {
    (partition.len() - 1)
        .checked_mul(panels + 1)
        .and_then(|n| n.checked_add(2))
        .filter(|&n| n <= SAMPLE_BUDGET)
}

/// The six components at every sample.
pub(super) fn samples(fields: &OwnedModeFields) -> Vec<Sample> {
    let e = fields.quantity().electric();
    let h = fields.quantity().magnetic();
    (0..e.x().len())
        .map(|i| [e.x()[i], e.y()[i], e.z()[i], h.x()[i], h.y()[i], h.z()[i]])
        .collect()
}

/// Simpson-weighted sum of `value` over one interval's samples, before the
/// width factor.
pub(super) fn simpson_sum(
    offset: usize,
    panels: usize,
    value: impl Fn(usize) -> Complex,
) -> Complex {
    let mut sum = value(offset) + value(offset + panels);
    for j in 1..panels {
        sum += value(offset + j) * if j % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum
}

/// Composite Simpson of `value` over each partition interval.
pub(super) fn intervals(
    partition: &[f64],
    panels: usize,
    value: impl Fn(usize) -> Complex,
) -> Vec<Complex> {
    partition
        .windows(2)
        .enumerate()
        .map(|(i, edges)| {
            simpson_sum(1 + i * (panels + 1), panels, &value)
                * ((edges[1] - edges[0]) / (3.0 * panels as f64))
        })
        .collect()
}

/// Composite Simpson of `value` over the whole partition.
pub(super) fn integral(
    partition: &[f64],
    panels: usize,
    value: impl Fn(usize) -> Complex,
) -> Complex {
    intervals(partition, panels, value).into_iter().sum()
}

/// A converged set of quadrature estimates.
pub(super) struct Converged<T> {
    pub entries: Vec<Complex>,
    pub extra: T,
    pub sample_count: usize,
    /// Largest change of an entry between the last two levels.
    pub change: f64,
}

/// Evaluate `level` at 8, 16, … panels per interval until every entry
/// changes by at most 1e-9 of max(|entry|, 1) between two levels; `budget`
/// when the samples or levels run out.
pub(super) fn converge<T, E>(
    partition: &[f64],
    budget: impl Fn() -> E,
    mut level: impl FnMut(usize, usize) -> Result<(Vec<Complex>, T), E>,
) -> Result<Converged<T>, E> {
    let mut previous: Option<Vec<Complex>> = None;
    for level_index in 0..11 {
        let panels = 8 << level_index;
        let sample_count = sample_count(partition, panels).ok_or_else(&budget)?;
        let (entries, extra) = level(panels, sample_count)?;
        if let Some(previous) = previous {
            let mut change = 0.0_f64;
            let mut converged = true;
            for (value, prior) in entries.iter().zip(&previous) {
                let delta = (value - prior).norm();
                converged &= delta <= 1e-9 * value.norm().max(1.0);
                change = change.max(delta);
            }
            if converged {
                return Ok(Converged {
                    entries,
                    extra,
                    sample_count,
                    change,
                });
            }
        }
        previous = Some(entries);
    }
    Err(budget())
}

/// Conjugated pairing density ¼ (E_a × H_b* + E_b* × H_a) · x̂, W/m per cm.
pub(super) fn conjugated(a: &Sample, b: &Sample) -> Complex {
    (a[1] * b[5].conj() - a[2] * b[4].conj() + b[1].conj() * a[5] - b[2].conj() * a[4])
        * (0.25 * super::power::CM_TO_M_OVER_Z0)
}

/// Unconjugated symmetric pairing density ¼ (E_a × H_b + E_b × H_a) · x̂, W/m per cm.
/// Exchanging the modes exchanges two terms of one sum, so the result is
/// bit for bit symmetric.
pub(super) fn unconjugated(a: &Sample, b: &Sample) -> Complex {
    let (ab, ba) = (a[1] * b[5] - a[2] * b[4], b[1] * a[5] - b[2] * a[4]);
    (ab + ba) * (0.25 * super::power::CM_TO_M_OVER_Z0)
}

/// Field-scale density ¼ (|E|² + |Z0 H|²) / Z0, W/m per cm.
pub(super) fn field_scale(a: &Sample) -> Complex {
    Complex::new(
        a.iter().map(|v| v.norm_sqr()).sum::<f64>() * (0.25 * super::power::CM_TO_M_OVER_Z0),
        0.0,
    )
}

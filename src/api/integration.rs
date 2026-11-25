//! Transverse quadrature shared by the pairings of planar modes.
//!
//! Stacks are placed in one registered frame. The partition is the union of
//! their interfaces and of the planes of the walls closing them; every
//! partition point is sampled from both sides, each interval by composite
//! Simpson, and refinement doubles the panels of every interval at once. An
//! open exterior beyond the partition enters as a closed-form tail.
use super::*;
use lamina_core::{
    AnalyticalMaterialStack, ExteriorSampling, FieldSampling, LayerSampling, OwnedModeFields,
    WallKind,
};
use lamina_units::Length;

/// Most samples per mode at one refinement level.
const SAMPLE_BUDGET: usize = 131072;

/// Fields at one sample: [Ex, Ey, Ez, Z0 Hx, Z0 Hy, Z0 Hz].
pub(super) type Sample = [Complex; 6];

/// A stack's interfaces and walls in a registered frame, in cm.
#[derive(Clone, Debug)]
pub(super) struct Layout {
    pub interfaces: Vec<f64>,
    widths: Vec<f64>,
    /// Each side's wall kind and registered position, `None` for an open side.
    pub walls: [Option<(WallKind, f64)>; 2],
}

impl Layout {
    /// `None` for a non-finite origin, an unrepresentable interface spacing
    /// or a termination other than a wall.
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
        let [left, right_wall] = crate::problem::walls(stack).ok()?;
        Some(Self {
            interfaces,
            widths,
            walls: [
                left.map(|(kind, gap)| (kind, origin - gap)),
                right_wall.map(|(kind, gap)| (kind, right + gap)),
            ],
        })
    }

    /// The union of the interfaces of `layouts` and their wall planes,
    /// ascending. Walls of compatible layouts coincide; the first one's are used.
    pub fn partition(layouts: &[&Layout]) -> Vec<f64> {
        let mut partition: Vec<_> = layouts
            .iter()
            .flat_map(|layout| layout.interfaces.iter().copied())
            .collect();
        if let Some(first) = layouts.first() {
            partition.extend(first.walls.iter().flatten().map(|&(_, z)| z));
        }
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

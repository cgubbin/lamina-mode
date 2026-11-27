//! Counted guided bases of stacks with full-tensor regions.
//!
//! A full-tensor region couples TE and TM unless its stack-frame tensors have
//! no `xy`, `yx`, `yz` or `zy` entries. Every mode is then found from the
//! coupled characteristic of a 4×4 backend. Its zeros in the eligible
//! interval are counted by the argument principle on a rectangle
//! `[lo, hi] × [−h, h]` of the β chart (mirrored for backward modes), on
//! which every open exterior keeps its outgoing pair continuous, away from
//! the incoming pair and from the real axis, and keeps the exterior
//! reference basis conditioned (sampled evidence, recorded). Cells are then
//! localised and corrected in β as the modal-basis search does.
//!
//! Eligibility: Hermitian, positive-definite ε and μ in every region, so a
//! guided mode lies below `k0 max √(λ_max(ε) λ_max(μ))` and above each open
//! exterior's light line.
use super::basis::{BasisGap, BasisLimit, Cells, Search};
use super::*;
use crate::problem as p;
use crate::problem::J;
use crate::problem::guided::{GuidedProblem, GuidedRegionEvidence};
use crate::problem::material::{self as m, Tensor};
use crate::search::ModalSearchResult;
use argloc::ArgumentConfig;
use lamina_core::{
    Backend, CanonicalCoordinates, Channel, ComplexPlane, ExteriorSampling, ExteriorWavevectors,
    FieldSampling, ModalCharacteristicDirection, ModalDerivativeBackend, PlaneWaveDeterminant,
    ProjectPlaneWaveModeDeterminant, Scatter2, Scatter4, SeedJet, StackRegion, Transfer2,
    Transfer4,
};
use lamina_units::Length;
use nalgebra::Matrix3;
use ndarray::arr0;

/// A backend for guided search: its characteristic, its mode fields, and
/// the form of it whose characteristic is holomorphic about a point.
pub trait GuidedBackend: ModalDerivativeBackend + Clone {
    /// The backend for a counting region about `(k0, β)`: the scattering
    /// backend continues its layer splits from that point.
    fn about(&self, k0: f64, beta: f64) -> Self;
}

/// The characteristic projection of a backend's entries.
pub trait Characteristic:
    ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<J>>
{
}
impl<E: ProjectPlaneWaveModeDeterminant<Determinant = PlaneWaveDeterminant<J>>> Characteristic
    for E
{
}

macro_rules! same {
    ($($backend:ty),*) => {$(
        impl GuidedBackend for $backend {
            fn about(&self, _: f64, _: f64) -> Self {
                *self
            }
        }
    )*};
}
same!(Scatter2, Transfer2, Transfer4);

impl GuidedBackend for Scatter4 {
    fn about(&self, k0: f64, beta: f64) -> Self {
        Scatter4::continued_from(Complex::new(k0, 0.0), Complex::new(beta, 0.0))
    }
}

/// Why a region's tensors cannot carry a guided count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorCause {
    /// ε or μ is not Hermitian: a lossy or active region.
    NotHermitian,
    /// ε or μ has an eigenvalue at or below zero: a hyperbolic or negative
    /// region, whose modes are not bounded by the count's window.
    NotPositiveDefinite,
}

/// Evidence of an argument-principle count.
#[derive(Clone, Debug)]
pub struct HybridCountEvidence {
    pub channel: Channel,
    /// Per rectangle, the backend whose characteristic was counted and
    /// whose roots were corrected: the selected one, or Transfer4 where the
    /// selected one cannot count the rectangle (the scattering backend's
    /// continued layer split meets a layer branch point).
    pub count_backends: Vec<&'static str>,
    /// The counted rectangles in β, cm⁻¹, left to right in |β|.
    pub rectangles: Vec<Rect<f64>>,
    /// The region's conditions; `None` when it was refused.
    pub region: Option<GuidedRegionEvidence>,
    /// How far each end of the interval was moved inside to clear a root on
    /// it, cm⁻¹.
    pub endpoint_shift: [f64; 2],
    /// Per mode, the norm of the field outside its label's family relative
    /// to the whole, at the left interface; one for a hybrid mode.
    pub off_family: Vec<f64>,
    /// On a separable stack, the TE and TM counts of the same interval.
    pub separable_counts: Option<[Option<usize>; 2]>,
}

/// The cells of a hybrid count: rectangles of the guided β chart.
struct GuidedCells<'a, B> {
    problem: &'a GuidedProblem<B>,
    assumption: AnalyticAssumption,
}

type Domain<B> = p::guided::GuidedDomain<p::Compiled<B>>;

impl<B, E> Cells for GuidedCells<'_, B>
where
    B: GuidedBackend + Backend<J, ComplexPlane, Entries = E>,
    E: Characteristic,
{
    type Domain = Domain<B>;
    fn domain(&self, cell: Rect<f64>) -> Result<Self::Domain, BasisLimit> {
        let bounds = ConformalRegion::new(cell.x_min, cell.x_max, cell.y_min, cell.y_max)
            .map_err(|_| BasisLimit::NonIntegerCount)?;
        self.problem
            .guided_domain(bounds, self.assumption)
            .map_err(|_| BasisLimit::NonIntegerCount)
    }
    fn evaluate(&self, domain: &Self::Domain, z: Complex) -> Option<(Complex, Complex)> {
        self.problem
            .evaluate(domain, z)
            .ok()
            .map(|b| (*b.value(), *b.derivative()))
    }
    fn search(
        &self,
        domain: &Self::Domain,
        config: ArgumentConfig<f64>,
    ) -> Option<ModalSearchResult<Complex>> {
        Some(self.problem.search(domain, config).ok()?.into_parts().0)
    }
    fn corrected(
        &self,
        domain: &Self::Domain,
        seed: Complex,
        config: crate::refinement::ValidatedConfig,
    ) -> Result<(crate::refinement::AcceptedRoot, BranchSignature), BranchFailure> {
        self.problem
            .corrected_point(domain, seed, config)
            .map(|(root, signature, _)| (root, signature))
    }
    fn beta(&self, z: Complex) -> Option<Complex> {
        Some(z)
    }
    fn backend(&self) -> &'static str {
        std::any::type_name::<B>()
    }
    /// |∂F/∂β| |β| at the seed: a root's residual is judged as a relative
    /// change of β, whatever the gauge of the coupled characteristic.
    fn residual_scale(&self, domain: &Self::Domain, seed: Complex) -> Option<f64> {
        let (_, derivative) = self.evaluate(domain, seed)?;
        let scale = derivative.norm() * seed.norm().max(1.0);
        (scale.is_finite() && scale > 0.0).then_some(scale)
    }
}

fn hermitian(t: &Tensor) -> Matrix3<Complex> {
    Matrix3::from_fn(|r, c| t[r][c])
}

/// Refuse a region whose ε or μ is not Hermitian positive definite, naming
/// it; otherwise `k0 √(λ_max(ε) λ_max(μ))` of the region.
fn bound(region: StackRegion, tensors: &[Tensor; 2], k0: f64) -> Result<f64, EnumerationError> {
    let mut product = 1.0;
    for t in tensors {
        let m = hermitian(t);
        let defect = (m - m.adjoint()).norm();
        if !defect.is_finite() || defect > 1e-12 * m.norm() {
            return Err(EnumerationError::Tensor {
                region,
                cause: TensorCause::NotHermitian,
            });
        }
        let values = ((m + m.adjoint()) * Complex::new(0.5, 0.0)).symmetric_eigenvalues();
        if values.iter().any(|v| v.is_nan() || *v <= 0.0) {
            return Err(EnumerationError::Tensor {
                region,
                cause: TensorCause::NotPositiveDefinite,
            });
        }
        product *= values.max();
    }
    Ok(k0 * product.sqrt())
}

/// A refused open exterior keeps its constitutive cause.
fn exterior_error(e: p::ProblemError) -> EnumerationError {
    match e {
        p::ProblemError::Constitutive(e) => EnumerationError::Constitutive(e),
        e => EnumerationError::Problem(e.to_string()),
    }
}

/// Counted guided modes of `channel` in the request's interval, through the
/// selected backend's characteristic, or Transfer4's when it cannot count.
pub(super) fn hybrid_guided_modes<B, E>(
    backend: B,
    stack: ModeStack,
    req: GuidedWindow,
    channel: Channel,
) -> Result<GuidedBasis, EnumerationError>
where
    B: GuidedBackend + Backend<J, ComplexPlane, Entries = E>,
    E: Characteristic,
{
    let (k, [lo, hi], _) = super::guided::validate(&req)?;
    let walls = p::walls(&stack).map_err(|side| EnumerationError::Termination { side })?;
    let mut upper_bound: f64 = 0.0;
    for (region, material) in m::regions(&stack) {
        let [values, _] = m::full_tensors(region, material, k.into())?;
        upper_bound = upper_bound.max(bound(region, &values, k)?);
    }
    let exteriors = p::guided::GuidedExteriors::new(&stack, k, channel).map_err(exterior_error)?;
    let open = walls.iter().any(Option::is_none);
    let light_line = exteriors.light_line(upper_bound);
    let margin = if open { req.cutoff_margin } else { 0.0 };
    let (lo, hi) = match light_line {
        Some(line) => (lo.max(line + margin), hi.min(upper_bound)),
        None => (hi, hi),
    };
    let mut report = GuidedBasis {
        request: req,
        light_line: light_line.unwrap_or(upper_bound),
        upper_bound,
        eligible: (lo < hi).then_some([lo, hi]),
        expected: Some(0),
        modes: vec![],
        labels: vec![],
        unresolved: vec![],
        hybrid: None,
    };
    if lo >= hi {
        return Ok(report);
    }
    let outcome = count(
        &backend,
        &stack,
        &req,
        channel,
        k,
        [lo, hi],
        light_line.filter(|_| open),
    )?;
    report.expected = outcome.expected;
    report.unresolved = outcome.unresolved;
    let mut evidence = outcome.evidence;
    // Labels: TE or TM by the family of the fields at the left interface on
    // a separable stack, hybrid otherwise.
    let separable = m::regions(&stack).all(|(_, r)| r.is_separable());
    for point in outcome.modes {
        let (label, off) = match channel.polarisation() {
            Some(polarisation) => (polarisation.into(), 0.0),
            None if separable => family(&backend, &stack, &point),
            None => (ModeLabel::Hybrid, 1.0),
        };
        report.modes.push(point);
        report.labels.push(label);
        evidence.off_family.push(off);
    }
    if separable && channel == Channel::Coupled && report.expected.is_some() {
        let counts = [
            lamina_core::Polarisation::TransverseElectric,
            lamina_core::Polarisation::TransverseMagnetic,
        ]
        .map(|polarisation| {
            let window = GuidedWindow {
                family: polarisation.into(),
                abs_beta: [lo, hi],
                ..req
            };
            super::guided::count_family(backend.clone(), stack.clone(), window)
                .ok()
                .flatten()
        });
        if counts.iter().flatten().sum::<usize>() != report.expected.unwrap_or(0)
            || counts.iter().any(Option::is_none)
        {
            report.unresolved.push(GuidedGap {
                abs_beta: [lo, hi],
                expected: report.expected,
                cause: GuidedLimit::SeparableCount { counts },
            });
        }
        evidence.separable_counts = Some(counts);
    }
    report.hybrid = Some(evidence);
    Ok(report)
}

struct Count {
    expected: Option<usize>,
    modes: Vec<ModePoint>,
    unresolved: Vec<GuidedGap>,
    evidence: HybridCountEvidence,
}

/// Rectangles in a plain interval: the interval is cut into this many.
const PLAIN_CELLS: usize = 8;

/// The counting rectangles over `[lo, hi]`, left to right: of width equal
/// to their left end's distance from the light line and half-height
/// `0.45` of it, so each keeps the branch point at the light line at a
/// fixed fraction of its size; equal cells with no light line.
fn rectangles(lo: f64, hi: f64, light_line: Option<f64>) -> Vec<[f64; 3]> {
    let mut cells = vec![];
    match light_line {
        Some(line) => {
            let mut x = lo;
            while x < hi {
                let width = (x - line).max(f64::MIN_POSITIVE);
                // Merge a sliver into the last cell.
                let end = if x + 1.5 * width >= hi { hi } else { x + width };
                cells.push([x, end, 0.45 * (x - line).min(end - x)]);
                x = end;
            }
        }
        None => {
            let width = (hi - lo) / PLAIN_CELLS as f64;
            for i in 0..PLAIN_CELLS {
                let a = lo + width * i as f64;
                let b = if i + 1 == PLAIN_CELLS { hi } else { a + width };
                cells.push([a, b, 0.45 * width]);
            }
        }
    }
    cells
}

/// The outcome of one rectangle.
enum Cell {
    Counted {
        count: usize,
        modes: Vec<ModePoint>,
        gaps: Vec<GuidedGap>,
    },
    /// A root on a side, at |β|.
    BoundaryRoot(f64),
    Failed,
}

fn gap(bounds: Rect<f64>, expected: Option<usize>, cause: GuidedLimit) -> GuidedGap {
    GuidedGap {
        abs_beta: [
            bounds.x_min.abs().min(bounds.x_max.abs()),
            bounds.x_min.abs().max(bounds.x_max.abs()),
        ],
        expected,
        cause,
    }
}

/// Count, localise and correct in one rectangle with `backend` continued
/// from its centre.
fn cell<B, E>(
    backend: &B,
    stack: &ModeStack,
    req: &GuidedWindow,
    channel: Channel,
    k: f64,
    bounds: Rect<f64>,
) -> Result<Cell, EnumerationError>
where
    B: GuidedBackend + Backend<J, ComplexPlane, Entries = E>,
    E: Characteristic,
{
    let centre = 0.5 * (bounds.x_min + bounds.x_max);
    let problem = GuidedProblem::guided(stack.clone(), k, channel, backend.about(k, centre))
        .map_err(|e| EnumerationError::Problem(e.to_string()))?;
    // The unscaled near-zero threshold, as the guided counts were tuned with.
    let argument = DiscoveryConfig {
        zero_tolerance: Some(DEFAULT_ZERO_TOLERANCE),
        ..DiscoveryConfig::default()
    }
    .argument_config(k)
    .map_err(|_| EnumerationError::InvalidRequest)?;
    let engine = Search {
        cells: GuidedCells {
            problem: &problem,
            assumption: req.assumption,
        },
        argument,
        refinement: req.refinement,
        pml_sides: vec![],
        k0: k,
        mirror: None,
        resolution: 1e-9 * k,
        reverse: false,
        window: bounds,
        open: None,
    };
    let (counted, gaps, _) = engine.count(bounds);
    let Some(counted) = counted else {
        return Ok(match gaps.first() {
            Some(BasisGap {
                bounds: at,
                cause: BasisLimit::BoundaryRoot,
                ..
            }) => Cell::BoundaryRoot((0.5 * (at.x_min + at.x_max)).abs()),
            _ => Cell::Failed,
        });
    };
    let (points, gaps) = engine.roots(bounds, Some(counted), req.max_subdivisions);
    let mut gaps: Vec<GuidedGap> = gaps
        .into_iter()
        .map(|g| {
            let cause = match g.cause {
                BasisLimit::Multiplicity { count } => GuidedLimit::NearDegenerate {
                    count,
                    cell: g.bounds,
                },
                cause => GuidedLimit::Cell(cause),
            };
            gap(g.bounds, g.count, cause)
        })
        .collect();
    let mut modes = vec![];
    for point in points {
        let beta = point.physical_coordinate();
        if beta.im.abs() > 1e-9 * beta.norm() {
            gaps.push(gap(
                point.enclosure(),
                Some(1),
                GuidedLimit::NonRealRoot { beta },
            ));
        } else {
            modes.push(point);
        }
    }
    Ok(Cell::Counted {
        count: counted.0,
        modes,
        gaps,
    })
}

/// Count, localise and correct on rectangles covering `[lo, hi]`.
fn count<B, E>(
    backend: &B,
    stack: &ModeStack,
    req: &GuidedWindow,
    channel: Channel,
    k: f64,
    [lo, hi]: [f64; 2],
    light_line: Option<f64>,
) -> Result<Count, EnumerationError>
where
    B: GuidedBackend + Backend<J, ComplexPlane, Entries = E>,
    E: Characteristic,
{
    let sign = match req.direction {
        GuidedDirection::Forward => 1.0,
        GuidedDirection::Backward => -1.0,
    };
    let rectangle = |lo: f64, hi: f64, half: f64| {
        let [a, b] = if sign > 0.0 { [lo, hi] } else { [-hi, -lo] };
        Rect {
            x_min: a,
            x_max: b,
            y_min: -half,
            y_max: half,
        }
    };
    let exteriors = p::guided::GuidedExteriors::new(stack, k, channel).map_err(exterior_error)?;
    let mut cells = rectangles(lo, hi, light_line);
    let mut evidence = HybridCountEvidence {
        channel,
        count_backends: vec![],
        rectangles: vec![],
        region: None,
        endpoint_shift: [0.0; 2],
        off_family: vec![],
        separable_counts: None,
    };
    let unknown = |evidence, bounds, cause| Count {
        expected: None,
        modes: vec![],
        unresolved: vec![gap(bounds, None, cause)],
        evidence,
    };
    let (mut total, mut modes, mut unresolved) = (0, vec![], vec![]);
    let (mut index, mut attempts) = (0, 0);
    while index < cells.len() {
        let [a, b, h] = cells[index];
        let bounds = rectangle(a, b, h);
        let region = match exteriors.region(bounds) {
            Ok(region) => region,
            Err(cause) => return Ok(unknown(evidence, bounds, GuidedLimit::Region(cause))),
        };
        let merged = evidence.region.get_or_insert(region);
        merged.half_height = merged.half_height.min(region.half_height);
        merged.exterior_separation = merged.exterior_separation.min(region.exterior_separation);
        merged.pair_margin = merged.pair_margin.min(region.pair_margin);
        merged.gauge_conditioning = merged.gauge_conditioning.min(region.gauge_conditioning);
        let mut used = std::any::type_name::<B>();
        let mut outcome = cell(backend, stack, req, channel, k, bounds)?;
        // Where the selected backend cannot count or correct, Transfer4.
        let unfinished = |outcome: &Cell| match outcome {
            Cell::Failed => usize::MAX,
            Cell::Counted { gaps, .. } => gaps.len(),
            Cell::BoundaryRoot(_) => 0,
        };
        if unfinished(&outcome) > 0 && used != std::any::type_name::<Transfer4>() {
            let other = cell(&Transfer4::new(), stack, req, channel, k, bounds)?;
            if unfinished(&other) < unfinished(&outcome) {
                (used, outcome) = (std::any::type_name::<Transfer4>(), other);
            }
        }
        match outcome {
            Cell::Counted {
                count,
                modes: found,
                gaps,
            } => {
                total += count;
                modes.extend(found);
                unresolved.extend(gaps);
                evidence.count_backends.push(used);
                index += 1;
            }
            Cell::Failed => return Ok(unknown(evidence, bounds, GuidedLimit::CountUnknown)),
            Cell::BoundaryRoot(at) => {
                // Move an end of the interval inside by at most the cutoff
                // margin, or an inner side by a hundredth of the cell, and
                // count every cell again.
                let step = 0.01 * (b - a);
                let moved = match ((at - a).abs() <= (at - b).abs(), index) {
                    (true, 0) => {
                        let shift = req.cutoff_margin.min(step);
                        evidence.endpoint_shift[0] += shift;
                        cells[0][0] += shift;
                        evidence.endpoint_shift[0] <= req.cutoff_margin
                    }
                    (false, i) if i + 1 == cells.len() => {
                        let shift = req.cutoff_margin.min(step);
                        evidence.endpoint_shift[1] += shift;
                        cells[i][1] -= shift;
                        evidence.endpoint_shift[1] <= req.cutoff_margin
                    }
                    (true, i) => {
                        cells[i][0] += step;
                        cells[i - 1][1] += step;
                        true
                    }
                    (false, i) => {
                        cells[i][1] -= step;
                        cells[i + 1][0] -= step;
                        true
                    }
                };
                attempts += 1;
                if !moved || attempts > 16 {
                    return Ok(unknown(evidence, bounds, GuidedLimit::CountUnknown));
                }
                (total, index) = (0, 0);
                modes.clear();
                unresolved.clear();
                evidence.count_backends.clear();
            }
        }
    }
    evidence.rectangles = cells.iter().map(|&[a, b, h]| rectangle(a, b, h)).collect();
    modes.sort_by(|a, b| {
        b.physical_coordinate()
            .re
            .abs()
            .total_cmp(&a.physical_coordinate().re.abs())
    });
    Ok(Count {
        expected: Some(total),
        modes,
        unresolved,
        evidence,
    })
}

/// The exterior wavevectors of an accepted point, as constants.
pub(super) fn exterior_constants(signature: &BranchSignature) -> ExteriorWavevectors<J> {
    let constant = |v: Complex| <J as SeedJet>::constant(arr0(v));
    let side = |scalar: Option<Complex>, pair: Option<[Complex; 2]>| match (pair, scalar) {
        (Some(pair), _) => Some(lamina_core::ExteriorBranch::Pair(pair.map(constant))),
        (None, Some(kappa)) => Some(lamina_core::ExteriorBranch::Scalar(constant(kappa))),
        (None, None) => None,
    };
    ExteriorWavevectors::from_branches(
        side(signature.left, signature.pairs[0]),
        side(signature.right, signature.pairs[1]),
    )
}

/// TE or TM by the larger of the two families' field norms at the left
/// interface, with the relative norm of the other.
fn family<B: GuidedBackend>(backend: &B, stack: &ModeStack, point: &ModePoint) -> (ModeLabel, f64) {
    let fields = (|| {
        let signature = point.signature().ok()?;
        let coordinates = CanonicalCoordinates::new(
            <J as SeedJet>::constant(arr0(signature.k0)),
            <J as SeedJet>::constant(arr0(signature.k_parallel)),
        );
        let exterior = exterior_constants(signature);
        let rates = vec![(Complex::ZERO, Complex::ZERO); stack.len() + 2];
        let fields = backend
            .directional_fields(
                ModalCharacteristicDirection {
                    stack,
                    coordinates: &coordinates,
                    exterior: &exterior,
                    polarisation: Channel::Coupled,
                    thickness_rates: &vec![0.0; stack.len()],
                    constitutive_rates: &rates,
                },
                &FieldSampling::new().left_exterior(ExteriorSampling::point(Length::zero())),
                Complex::ONE,
            )
            .ok()?;
        let q = fields.fields.quantity();
        let at = |c: &ndarray::Array1<Complex>| c[0].norm_sqr();
        let (e, h) = (q.electric(), q.magnetic());
        Some([
            at(e.y()) + at(h.x()) + at(h.z()),
            at(e.x()) + at(e.z()) + at(h.y()),
        ])
    })();
    let Some([te, tm]) = fields else {
        return (ModeLabel::Hybrid, 1.0);
    };
    let total = te + tm;
    let (label, off) = if te >= tm {
        (ModeLabel::TransverseElectric, tm)
    } else {
        (ModeLabel::TransverseMagnetic, te)
    };
    let off = (off / total).sqrt();
    if off <= 1e-6 {
        (label, off)
    } else {
        (ModeLabel::Hybrid, off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lamina_core::{AnalyticalTensorMaterialHandle, Constant, Rotation, Stack};
    use lamina_materials::PrincipalAxes;
    use std::f64::consts::TAU;

    fn window(abs_beta: [f64; 2]) -> GuidedWindow {
        GuidedWindow {
            wavelength: Length::centimetres(1.0),
            family: ModeFamily::All,
            direction: GuidedDirection::Forward,
            abs_beta,
            cutoff_margin: 1e-3 * TAU,
            phase_tolerance: 1e-11,
            max_subdivisions: 1000,
            refinement: Default::default(),
            assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        }
    }

    fn oriented(principal: [f64; 3], rotation: Rotation<f64>) -> Region {
        let axis = |i: usize| Constant::dielectric(principal[i]);
        Region::oriented(
            AnalyticalTensorMaterialHandle::new(PrincipalAxes::biaxial(axis(0), axis(1), axis(2))),
            rotation,
        )
    }

    #[test]
    fn the_guided_upper_bound_encloses_every_mode() {
        let k = TAU;
        let clad = || -> Region {
            lamina_core::AnalyticalMaterialHandle::new(Constant::dielectric(2.1)).into()
        };
        let fixtures = [
            // Biaxial, principal axes rotated off every stack axis.
            oriented([3.0, 4.5, 6.0], Rotation::from_euler(0.4, 0.7, 0.2)),
            // Rotated uniaxial.
            oriented([4.0, 4.0, 6.25], Rotation::from_optic_axis(0.9, 0.5)),
        ];
        for core in fixtures {
            let stack: ModeStack = Stack::builder(clad(), clad())
                .layer(core, Length::centimetres(1.1))
                .finalise();
            let req = window([0.0, 4.0 * k]);
            let basis = hybrid_guided_modes(Transfer4::new(), stack.clone(), req, Channel::Coupled)
                .unwrap();
            assert!(basis.is_complete(), "{:?}", basis.unresolved);
            assert!(!basis.modes.is_empty());
            let bound = basis.upper_bound;
            // The window was widened past the bound and clipped to it.
            assert_eq!(basis.eligible.unwrap()[1], bound);
            let top = basis.modes[0].physical_coordinate().re;
            assert!(top < bound, "{top} {bound}");
            // No zero of the coupled characteristic above the bound.
            let above = count(
                &Transfer4::new(),
                &stack,
                &req,
                Channel::Coupled,
                k,
                [bound, 1.5 * bound],
                Some(basis.light_line),
            )
            .unwrap();
            assert_eq!(above.expected, Some(0), "{:?}", above.unresolved);
            println!(
                "upper bound {bound:.6}: highest mode {top:.6} ({:.3e} below), none in [bound, 1.5 bound]",
                (bound - top) / bound
            );
        }
    }
}

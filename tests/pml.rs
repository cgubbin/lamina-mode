//! Stacks closed by wall-backed PMLs: the complex-thickness box on the direct
//! chart, the scale of the characteristic on a rectangle, classification and
//! refusals, fields on the stretched coordinate, and convergence of the
//! closed stack's modes to the open stack's.
use lamina_core::{
    AnalyticalMaterialStack, ExteriorSampling, FieldSampling, LayerSampling, Polarisation,
    Scatter2, StackSide, Termination, Wall,
};
use lamina_mode::*;
use lamina_units::LengthUnit;
use std::f64::consts::PI;
mod support;
use support::resolved_roots;
mod pml_support;
use pml_support::*;

fn region(r: [f64; 4]) -> ConformalRegion<f64> {
    ConformalRegion::new(r[0], r[1], r[2], r[3]).unwrap()
}

/// Every accepted root of a closed stack on the direct chart over `r`, each
/// found once.
fn closed_roots(
    stack: AnalyticalMaterialStack<Complex>,
    pol: Polarisation,
    r: [f64; 4],
) -> Vec<ModePoint> {
    let problem = PropagationProblem::new(stack, K, pol).unwrap();
    let direct = |bounds| {
        let domain = problem.direct_domain(bounds, ASSUME).unwrap();
        assert_eq!(
            domain.provenance().coordinate_kind(),
            CoordinateKind::Wavenumber
        );
        domain
    };
    // A PML scales the characteristic by down to exp(−Im κ_ref τ) per side,
    // e^−33 for the thickest pair here, where |D̂| reaches 9e-13 on a contour
    // that passes no root; the near-zero threshold sits below that scale
    // rather than at the default 1e-12.
    let discovery = DiscoveryConfig {
        zero_tolerance: Some(1e-30),
        ..Default::default()
    };
    let (points, count) = resolved_roots(&problem, region(r), &direct, discovery);
    assert_eq!(points.len(), count);
    for point in &points {
        assert_eq!(point.native_coordinate(), point.physical_coordinate());
        assert_eq!(point.signature().unwrap().exteriors(), [None, None]);
        closed_without_gain(point);
    }
    points
}

/// Check that a mode of a PML-closed stack is closed on both sides with no
/// gain or loss inferred from Im β, and return its PML truncation estimate.
fn closed_without_gain(point: &ModePoint) -> f64 {
    let interpretation = point.describe(Default::default()).unwrap().interpretation;
    assert_eq!(interpretation.exterior, [ExteriorBehaviour::Closed; 2]);
    assert_eq!(interpretation.flux, [Flux::Closed; 2]);
    assert_eq!(
        interpretation.classification,
        Classification::Classified(ModeType::Closed)
    );
    let Evolution::PmlTruncation { truncation } = interpretation.evolution else {
        panic!("{:?}", interpretation.evolution);
    };
    assert!(truncation > 0. && truncation <= 1., "{truncation}");
    truncation
}

fn betas(points: &[ModePoint]) -> Vec<Complex> {
    points.iter().map(|p| p.physical_coordinate()).collect()
}

#[test]
fn a_wall_and_a_pml_close_a_box_with_the_complex_thickness_ladder() {
    let (gap_left, gap_right, d) = (0.25, 0.15, 0.6);
    let tau = d * PROFILE.mean();
    let r = [1.0 * K, 1.99 * K, -0.1 * K, 0.6 * K];
    for pol in [TE, TM] {
        for (wall, backing) in [(PMC, PEC), (PEC, PMC)] {
            let stack = slab(
                Wall::new(wall, cm(-gap_left)),
                pml(StackSide::Right, gap_right, d, PROFILE, backing),
            );
            let regions = [
                (CLAD, c(gap_left, 0.)),
                (CORE, c(2. * A, 0.)),
                (CLAD, gap_right + tau),
            ];
            let reference = |beta| box_value(pol, [wall, backing], &regions, beta);
            let expected = winding(&reference, r);
            let points = closed_roots(stack.clone(), pol, r);
            assert!(expected >= 4, "{expected}");
            assert_eq!(points.len(), expected);
            // A single search of the rectangle is complete exactly when it
            // accepts every root; what it leaves are clusters of the ladder.
            let problem = PropagationProblem::new(stack, K, pol).unwrap();
            let domain = problem.direct_domain(region(r), ASSUME).unwrap();
            let report = problem
                .plan(vec![domain], vec![])
                .unwrap()
                .solve(Default::default(), Default::default(), Default::default())
                .unwrap();
            assert_eq!(
                report.is_complete(),
                report.observations().len() == expected
            );
            for unresolved in report.unresolved() {
                assert!(matches!(
                    unresolved.cause,
                    UnresolvedCause::Candidate {
                        status: CandidateStatus::UnresolvedCluster,
                        ..
                    }
                ));
                let b = unresolved.bounds;
                let inside = |z: &Complex| {
                    z.re >= b.x_min && z.re <= b.x_max && z.im >= b.y_min && z.im <= b.y_max
                };
                assert!(betas(&points).iter().filter(|z| inside(z)).count() >= 2);
            }
            let mut worst: f64 = 0.;
            for point in &points {
                let beta = point.physical_coordinate();
                let polished = newton(&reference, beta).unwrap();
                worst = worst.max((polished - beta).norm() / beta.norm());
            }
            assert!(worst < 1e-10, "{worst:e}");
            eprintln!(
                "complex-thickness box {pol:?} wall {wall:?} backing {backing:?}: {} modes, \
                 worst relative error {worst:.1e}",
                points.len()
            );
        }
    }
}

#[test]
fn the_pml_scale_is_checked_on_the_requested_rectangle() {
    let stack = long_pml_and_wall();
    let accepted = [1.85 * K, 3. * K, -0.031 * K, 0.047 * K];
    let tau = LONG_D * LONG.mean();
    let edge = guided_kappa(c(accepted[1], 0.)) * tau;
    assert!(edge.im > 300., "{edge}");
    let regions = [(CLAD, 0.2 + tau), (CORE, c(2. * A, 0.)), (CLAD, c(0.3, 0.))];
    let reference = |beta| box_value(TE, [PEC, PMC], &regions, beta);
    let expected = winding(&reference, accepted);
    let points = closed_roots(stack.clone(), TE, accepted);
    assert!(expected >= 1);
    assert_eq!(points.len(), expected);
    let problem = PropagationProblem::new(stack.clone(), K, TE).unwrap();
    let domain = problem.direct_domain(region(accepted), ASSUME).unwrap();
    let report = problem
        .plan(vec![domain], vec![])
        .unwrap()
        .solve(Default::default(), Default::default(), Default::default())
        .unwrap();
    assert!(report.is_complete(), "{:?}", report.unresolved());
    assert_eq!(report.observations().len(), expected);
    for beta in betas(&points) {
        let polished = newton(&reference, beta).unwrap();
        assert!((polished - beta).norm() < 1e-10 * beta.norm(), "{beta}");
        eprintln!(
            "thick PML: β = {beta:.12}, |Im κτ| = {:.0} at the edge, worst relative error {:.1e}",
            edge.im,
            (polished - beta).norm() / beta.norm()
        );
    }
    let refusal = |stack, r| {
        let problem = PropagationProblem::new(stack, K, TE).unwrap();
        let error = problem.direct_domain(region(r), ASSUME).unwrap_err();
        *error.find_source::<DirectChartError>().unwrap()
    };
    // Out to 4 k0 the scale reaches about exp(315).
    let DirectChartError::PmlScale {
        side,
        log_magnitude,
    } = refusal(stack, [1.85 * K, 4. * K, -0.031 * K, 0.047 * K])
    else {
        panic!("not a scale refusal");
    };
    assert_eq!(side, StackSide::Left);
    assert!((300..330).contains(&log_magnitude), "{log_magnitude}");
    eprintln!("refused to 4 k0: {side:?} side, exp({log_magnitude})");
    // Two such PMLs out to 3.3 k0: each side's share, about 219, is in range,
    // their sum is not.
    let DirectChartError::PmlScale { log_magnitude, .. } = refusal(
        symmetric(0.2, LONG_D, LONG),
        [1.85 * K, 3.3 * K, -0.031 * K, 0.047 * K],
    ) else {
        panic!("not a scale refusal");
    };
    assert!((425..450).contains(&log_magnitude), "{log_magnitude}");
    eprintln!("refused, two PMLs to 3.3 k0: exp({log_magnitude})");
    // A strongly absorbing PML underflows on any rectangle: −Im(κ_ref τ) ≈ −1885.
    let lossy = Profile {
        m: 2.,
        strength: c(0., 6.),
    };
    let stack = slab(
        Wall::perfect_electric(cm(-0.2)),
        pml(StackSide::Right, 0.2, 100., lossy, PEC),
    );
    assert_eq!(
        refusal(stack, accepted),
        DirectChartError::PmlScale {
            side: StackSide::Right,
            log_magnitude: -1885,
        }
    );
}

#[test]
fn pml_sides_refuse_enumeration_integrals_and_open_strategies() {
    let stack = long_pml_and_wall();
    let window = GuidedWindow {
        wavelength: cm(1.),
        family: TE.into(),
        direction: GuidedDirection::Forward,
        abs_beta: [1.55 * K, 1.999 * K],
        cutoff_margin: 1e-8,
        phase_tolerance: 1e-11,
        max_subdivisions: 1000,
        refinement: Default::default(),
        assumption: ASSUME,
    };
    assert!(matches!(
        enumerate_guided_modes(stack.clone(), window),
        Err(EnumerationError::Termination {
            side: StackSide::Left
        })
    ));
    let frequency = FrequencyProblem::new(stack.clone(), 1.7 * K, TE).unwrap_err();
    assert!(
        frequency
            .find_source::<ConstructionError>()
            .is_some_and(|e| matches!(e, ConstructionError::ClosedSide))
    );
    // The PML has decayed the guided mode to round-off: β is real to the
    // eligibility tolerance, so the refusals come from the PML side.
    let points = closed_roots(stack, TE, [1.85 * K, 2.05 * K, -0.031 * K, 0.047 * K]);
    let point = &points[0];
    let refused = PmlIntegral {
        side: StackSide::Left,
    };
    let backend = Scatter2::new();
    let sampling = FieldSampling::new().layer(0, LayerSampling::uniform(8));
    assert!(matches!(
        point.guided_power_profile_with(backend, &sampling, Default::default()),
        Err(GuidedPowerError::Termination(e)) if e == refused
    ));
    assert!(matches!(
        point.guided_group_index_with(&backend, Default::default(), 1e-8),
        Err(GroupIndexError::Termination(e)) if e == refused
    ));
    let tangent = point
        .guided_beta_sensitivity_with(
            &backend,
            &GuidedPerturbation::thickness(1, 0).unwrap(),
            Default::default(),
            Default::default(),
        )
        .unwrap()
        .power_field_tangent_with(
            &backend,
            &sampling,
            GuidedObservationFrame::MaterialAttached,
            Complex::ONE,
        );
    assert!(matches!(
        tangent,
        Err(GuidedPowerSensitivityError::Termination(e)) if e == refused
    ));
    assert!(matches!(
        guided_power_overlap_with(
            &backend,
            [point, point],
            GuidedOverlapRegistration {
                origins_cm: [0.; 2]
            },
            Default::default(),
        ),
        Err(GuidedOverlapError::ClosedStack)
    ));
}

#[test]
fn fields_continue_into_the_pml_on_the_stretched_coordinate() {
    // ψ beyond the right interface is the exterior solution of the closed
    // side on the complex distance Z̃: sin κ(L − Z̃) / sin κL behind a Dirichlet
    // wall and cos κ(L − Z̃) / cos κL behind a Neumann one, L = g + τ.
    let (gap, d) = (0.3, 1.);
    let stack = symmetric(gap, d, PROFILE);
    let length = gap + d * PROFILE.mean();
    let n = 64;
    let distances: Vec<f64> = (0..=n).map(|i| (gap + d) * i as f64 / n as f64).collect();
    let complex_distance = |z: f64| {
        if z <= gap {
            c(z, 0.)
        } else {
            gap + PROFILE.stretched(z - gap, d)
        }
    };
    for pol in [TE, TM] {
        let points = closed_roots(
            stack.clone(),
            pol,
            [1.55 * K, 2.05 * K, -0.031 * K, 0.047 * K],
        );
        assert!(!points.is_empty());
        for point in &points {
            let beta = point.physical_coordinate();
            let kappa = leaky_kappa(beta);
            let sampling = FieldSampling::new().right_exterior(ExteriorSampling::distances_in(
                distances.iter().copied(),
                LengthUnit::Centimetre,
            ));
            let fields = point
                .recover_fields_with(Scatter2::new(), &sampling, Default::default())
                .unwrap();
            let quantity = fields.fields.quantity();
            let primary = if pol == TE {
                quantity.electric().y()
            } else {
                quantity.magnetic().y()
            };
            let shape = |z: Complex| {
                if dirichlet(PEC, pol) {
                    (kappa * (length - z)).sin()
                } else {
                    (kappa * (length - z)).cos()
                }
            };
            let mut worst: f64 = 0.;
            for (i, &z) in distances.iter().enumerate() {
                let expected = primary[0] * shape(complex_distance(z)) / shape(Complex::ZERO);
                worst = worst.max((primary[i] - expected).norm() / primary[0].norm());
            }
            assert!(worst < 1e-9, "{pol:?} {beta}: {worst:e}");
            eprintln!("fields through the PML {pol:?} β = {beta}: worst {worst:.1e}");
        }
    }
}

/// Least-squares slope of ln y against x.
fn log_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let (mx, my) = points
        .iter()
        .fold((0., 0.), |(a, b), &(x, y)| (a + x / n, b + y.ln() / n));
    let (sxy, sxx) = points.iter().fold((0., 0.), |(a, b), &(x, y)| {
        (a + (x - mx) * (y.ln() - my), b + (x - mx) * (x - mx))
    });
    sxy / sxx
}

/// The distance from `open` to the nearest of `betas`.
fn nearest(betas: &[Complex], open: Complex) -> (usize, f64) {
    betas
        .iter()
        .map(|b| (b - open).norm())
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

/// Distances below this are at the root accuracy and left out of the fits.
const DISTANCE_FLOOR: f64 = 1e-9;

/// Check the decay of `distances` along `x` against `rate` within 10 % and
/// report it.
fn check_rate(label: &str, x: &[f64], distances: &[f64], rate: f64) {
    let fitted: Vec<_> = x
        .iter()
        .zip(distances)
        .filter(|(_, d)| **d > DISTANCE_FLOOR)
        .map(|(&x, &d)| (x, d))
        .collect();
    assert!(fitted.len() >= 3, "{label}: {distances:?}");
    let slope = -log_slope(&fitted);
    let ratio = slope / rate;
    eprintln!(
        "{label}: |Δβ| {:?}, fitted rate {slope:.3} cm⁻¹, predicted {rate:.3} cm⁻¹, ratio {ratio:.4}",
        distances
            .iter()
            .map(|d| format!("{d:.2e}"))
            .collect::<Vec<_>>()
    );
    assert!((ratio - 1.).abs() < 0.1, "{label}: {slope} vs {rate}");
}

#[test]
fn guided_modes_converge_as_the_pml_moves_out_and_as_it_thickens() {
    let pol = TE;
    let guided = open_guided(pol, 1.55 * K);
    assert_eq!(guided.len(), 2);
    let window = [1.55 * K, 2.05 * K, -0.031 * K, 0.047 * K];
    let sequence = |stacks: Vec<AnalyticalMaterialStack<Complex>>| {
        stacks
            .into_iter()
            .map(|stack| {
                let betas = betas(&closed_roots(stack, pol, window));
                assert_eq!(betas.len(), guided.len());
                guided
                    .iter()
                    .map(|&(_, open)| nearest(&betas, c(open, 0.)).1)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    // Outward at fixed thickness: rate 2α per unit gap.
    let d = 0.5;
    let gaps = [0.1, 0.2, 0.3, 0.4, 0.5];
    let moved = sequence(gaps.iter().map(|&g| symmetric(g, d, PROFILE)).collect());
    // Thicker at a fixed start: rate 2α Re s̄ per unit thickness.
    let g = 0.1;
    let thicknesses = [0.3, 0.45, 0.6, 0.75, 0.9];
    let thickened = sequence(
        thicknesses
            .iter()
            .map(|&d| symmetric(g, d, PROFILE))
            .collect(),
    );
    for (j, &(odd, open)) in guided.iter().enumerate() {
        let alpha = guided_kappa(c(open, 0.)).im;
        let parity = if odd { "odd" } else { "even" };
        let column = |rows: &[Vec<f64>]| rows.iter().map(|row| row[j]).collect::<Vec<_>>();
        check_rate(
            &format!("guided {parity} β = {open:.6} cm⁻¹, gap at d = {d}"),
            &gaps,
            &column(&moved),
            2. * alpha,
        );
        check_rate(
            &format!("guided {parity} β = {open:.6} cm⁻¹, thickness at g = {g}"),
            &thicknesses,
            &column(&thickened),
            2. * alpha * PROFILE.mean().re,
        );
    }
}

#[test]
fn exposed_leaky_modes_converge_as_the_pml_thickens_and_the_rest_stay_off() {
    let pol = TE;
    let g = 0.1;
    let thicknesses = [0.75, 1., 1.25, 1.5, 1.75];
    let window = [0.77 * K, 2.03 * K, -0.031 * K, 2.07 * K];
    let (open, leaky) = open_ladder(pol, window);
    // Exposure, Im(κ (g + d s̄)) > 0, holds or fails over the whole sequence.
    let exposed: Vec<bool> = open
        .iter()
        .map(|(label, beta, kappa)| {
            let at = |d: f64| (kappa * (g + d * PROFILE.mean())).im > 0.;
            let first = at(thicknesses[0]);
            assert!(
                thicknesses.iter().all(|&d| at(d) == first),
                "{label} {beta}"
            );
            first
        })
        .collect();
    let leaky_exposed = (0..open.len())
        .filter(|&i| exposed[i] && open[i].0.starts_with("leaky"))
        .count();
    assert!(leaky >= 2 && leaky_exposed >= 2, "{leaky} {leaky_exposed}");
    for (i, (label, beta, kappa)) in open.iter().enumerate() {
        eprintln!(
            "open {label} β = {beta:.6} cm⁻¹, arg κ = {:.3}, {}",
            kappa.arg(),
            if exposed[i] { "exposed" } else { "not exposed" }
        );
    }
    let tolerance = 1e-4;
    let floor = 0.05 * K;
    let mut distances = vec![vec![]; open.len()];
    let mut previous: Option<Vec<Complex>> = None;
    for &d in &thicknesses {
        let length = g + d * PROFILE.mean();
        let points = closed_roots(symmetric(g, d, PROFILE), pol, window);
        let betas = betas(&points);
        // The closed mode nearest each exposed open mode, one each.
        let mut converging = vec![];
        for (i, (_, beta, _)) in open.iter().enumerate() {
            let (index, distance) = nearest(&betas, *beta);
            distances[i].push(distance);
            if exposed[i] {
                assert!(!converging.contains(&index));
                converging.push(index);
            }
        }
        let others: Vec<_> = (0..betas.len())
            .filter(|i| !converging.contains(i))
            .map(|i| betas[i])
            .collect();
        let mut least_distance = f64::INFINITY;
        let mut worst_angle: f64 = 0.;
        for beta in &others {
            for (_, open, _) in &open {
                least_distance = least_distance.min((beta - open).norm());
            }
            // Near the rotated continuum κ (g + d s̄) ∈ ℝ: arg κ ≈ −arg(g + d s̄),
            // with either root.
            let mut angle = (CLAD * K * K - beta * beta).sqrt().arg() + length.arg();
            angle -= PI * (angle / PI).round();
            worst_angle = worst_angle.max(angle.abs());
            if let Some(previous) = &previous {
                assert!(
                    previous.iter().all(|p| (p - beta).norm() > tolerance),
                    "{beta} did not move from d = {d}"
                );
            }
        }
        assert!(least_distance > floor, "d = {d}: {least_distance}");
        assert!(worst_angle < 0.25, "d = {d}: {worst_angle}");
        // Every mode is closed with no gain or loss inferred (`closed_roots`).
        // A guided mode's β moves off the real axis by the PML's perturbation
        // exp(2iκ(g + d s̄)), whose phase turns with d, so it may lie on either
        // side of the axis, by no more than its distance from the open mode.
        for (i, (label, _, _)) in open.iter().enumerate() {
            if !(exposed[i] && label.starts_with("guided")) {
                continue;
            }
            let (index, distance) = nearest(&betas, open[i].1);
            let beta = betas[index];
            assert!(beta.im.abs() <= distance, "{beta}");
            if beta.im < 0. {
                eprintln!(
                    "d = {d}: {label} β = {beta:.12} lies below the real axis, truncation {:.2e}",
                    closed_without_gain(&points[index])
                );
            }
        }
        eprintln!(
            "d = {d}: {} modes, {} off the open modes, least distance {:.3} k0, \
             largest |arg κ + arg(g + d s̄)| {worst_angle:.3}",
            betas.len(),
            others.len(),
            least_distance / K
        );
        previous = Some(others);
    }
    // Exactly one closed mode within tolerance of each open mode at the
    // thickest PML; the leaky ones converge at the predicted rate (the guided
    // ones are fitted against gap and thickness above).
    for (i, (label, beta, kappa)) in open.iter().enumerate() {
        if !exposed[i] {
            continue;
        }
        assert!(distances[i].last().unwrap() < &tolerance, "{label} {beta}");
        if label.starts_with("guided") {
            continue;
        }
        let rate = 2. * (kappa * PROFILE.mean()).im;
        check_rate(
            &format!("{label} β = {beta:.6} cm⁻¹, thickness at g = {g}"),
            &thicknesses,
            &distances[i],
            rate,
        );
    }
}

#[test]
fn no_pml_closed_mode_is_classified_as_gaining() {
    // At d = 0.75 cm the even guided mode lies below the real axis, where the
    // sign of Im β alone would read as gain.
    let (g, d) = (0.1, 0.75);
    let points = closed_roots(
        symmetric(g, d, PROFILE),
        TE,
        [1.55 * K, 2.05 * K, -0.031 * K, 0.047 * K],
    );
    let below: Vec<_> = points
        .iter()
        .filter(|p| p.physical_coordinate().im < 0.)
        .collect();
    assert!(!below.is_empty());
    for point in below {
        let truncation = closed_without_gain(point);
        let beta = point.physical_coordinate();
        eprintln!("β = {beta:.12}: closed, PML truncation {truncation:.2e}");
    }
}

#[test]
fn a_pml_beside_an_open_side_classifies_by_the_open_side() {
    // A PML on the left of the slab and the cladding open on the right. The
    // evolution carries the PML's truncation; the type follows from the open
    // side's exterior alone: decaying is guided, growing outward is leaky.
    let stack = slab(
        pml(StackSide::Left, 0.1, 1.75, PROFILE, PEC),
        Termination::Open,
    );
    let problem = PropagationProblem::new(stack, K, TE).unwrap();
    let domain = |bounds| {
        let sheet = ReferenceSelection::Sheet(CoordinateSheet::Principal);
        problem
            .domain(bounds, sheet, Default::default(), ASSUME)
            .unwrap()
    };
    let classified = |bounds| {
        let (points, count) = resolved_roots(&problem, bounds, &domain, Default::default());
        assert_eq!(points.len(), count);
        points
            .iter()
            .map(|point| {
                let interpretation = point.describe(Default::default()).unwrap().interpretation;
                assert!(matches!(
                    interpretation.evolution,
                    Evolution::PmlTruncation { .. }
                ));
                assert_eq!(interpretation.exterior[0], ExteriorBehaviour::Closed);
                (point.physical_coordinate(), interpretation.classification)
            })
            .collect::<Vec<_>>()
    };
    // Guided: the open slab's guided modes, the right exterior decaying.
    let guided = classified(ConformalRegion::new(-1., 0.8, -0.4, 0.37).unwrap());
    let open = open_guided(TE, 1.55 * K);
    assert_eq!(guided.len(), open.len());
    for (beta, classification) in &guided {
        assert_eq!(
            *classification,
            Classification::Classified(ModeType::Guided)
        );
        let nearest = open
            .iter()
            .map(|&(_, b)| (beta - b).norm())
            .fold(f64::INFINITY, f64::min);
        assert!(nearest < 1e-6, "{beta}");
    }
    // Leaky: on the open side κ = i k0 e^z, so an outgoing, growing κ lies
    // at −π < Im z < −π/2. The odd leaky mode of the open slab, exposed to
    // the PML, has κ ≈ 9.8 − 3.8i cm⁻¹: Re z ≈ 0.51, Im z ≈ −1.94.
    let leaky = classified(ConformalRegion::new(0.2, 0.8, -2.3, -1.65).unwrap());
    let reference = newton(
        &|beta| open_value(TE, true, beta, leaky_kappa(beta)),
        c(6.4, 5.8),
    )
    .unwrap();
    let (beta, classification) = leaky
        .iter()
        .min_by(|a, b| {
            (a.0 - reference)
                .norm()
                .total_cmp(&(b.0 - reference).norm())
        })
        .unwrap();
    assert!((beta - reference).norm() < 1e-4, "{beta} vs {reference}");
    assert_eq!(
        *classification,
        Classification::Classified(ModeType::RightLeaky)
    );
    eprintln!(
        "open beside a PML: {} guided, leaky β = {beta:.6} (open slab {reference:.6}); all {:?}",
        guided.len(),
        leaky.iter().map(|l| l.1).collect::<Vec<_>>()
    );
}

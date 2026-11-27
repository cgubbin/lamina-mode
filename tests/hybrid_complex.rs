//! Leaky and quasi-normal searches of stacks with full-tensor layers and
//! isotropic exteriors, through the coupled channel.
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalTensorMaterialHandle, Channel, Constant, ConstitutiveError,
    Layer, Polarisation, Rotation, Scatter2, Stack, StackRegion, Transfer4,
};
use lamina_materials::PrincipalAxes;
use lamina_mode::*;
use lamina_units::Length;

const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
const TE: Polarisation = Polarisation::TransverseElectric;
const TM: Polarisation = Polarisation::TransverseMagnetic;

fn cm(x: f64) -> Length<f64> {
    Length::centimetres(x)
}

fn isotropic(epsilon: f64) -> Region {
    AnalyticalMaterialHandle::new(Constant::dielectric(epsilon)).into()
}

fn uniaxial(ordinary: f64, extraordinary: f64, theta: f64, phi: f64) -> Region {
    Region::oriented(
        AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(
            Constant::dielectric(ordinary),
            Constant::dielectric(extraordinary),
        )),
        Rotation::from_optic_axis(theta, phi),
    )
}

/// A film over a substrate denser than its effective index: leaky modes.
fn leaky(film: Region) -> ModeStack {
    Stack::builder(isotropic(1.0), isotropic(2.25))
        .layer(film, cm(0.3))
        .finalise()
}

/// The accepted roots `(native, physical)` of a plan.
macro_rules! roots {
    ($plan:expr) => {{
        let report = $plan
            .solve(
                DiscoveryConfig::default(),
                RefinementConfig::default(),
                MatchConfig::default(),
            )
            .unwrap();
        let mut roots: Vec<(Complex, Complex)> = report
            .groups()
            .iter()
            .map(|g| {
                let m = &report.observations()[g.members()[0]];
                (m.native_coordinate(), m.physical_coordinate())
            })
            .collect();
        roots.sort_by(|a, b| a.1.re.total_cmp(&b.1.re).then(a.1.im.total_cmp(&b.1.im)));
        roots
    }};
}

/// The roots of a canonical in-plane plan.
macro_rules! propagation {
    ($problem:expr) => {
        roots!(
            $problem
                .canonical_plan(
                    InPlaneSearchExtent {
                        max_parallel_wavenumber: 24.0,
                        imaginary_margin: 0.05,
                    },
                    ReferenceSelection::Sheet(CoordinateSheet::Principal),
                    ConformalChartConfig::default(),
                    ASSUME,
                )
                .unwrap()
        )
    };
}

/// The roots of a canonical spectral plan.
macro_rules! spectral {
    ($problem:expr) => {
        roots!(
            $problem
                .canonical_plan(
                    WavenumberSearchExtent {
                        max_vacuum_wavenumber: 8.0,
                        imaginary_margin: 0.05,
                    },
                    [ReferenceSelection::Sheet(CoordinateSheet::Principal); 2],
                    ConformalChartConfig::default(),
                    ASSUME,
                )
                .unwrap()
        )
    };
}

/// The roots found by `$problem` on a small chart box about each of
/// `$roots`, on the sheet that root selects.
macro_rules! near {
    ($problem:expr, $roots:expr) => {{
        let problem = $problem;
        let mut found = vec![];
        for &(z, beta) in $roots {
            let r = 0.02;
            let domain = problem
                .domain(
                    ConformalRegion::new(z.re - r, z.re + r, z.im - r, z.im + r).unwrap(),
                    ReferenceSelection::Coordinate {
                        reference: z,
                        coordinate: beta,
                    },
                    ConformalChartConfig::default(),
                    ASSUME,
                )
                .unwrap();
            found.extend(roots!(problem.plan(vec![domain], vec![]).unwrap()));
        }
        found
    }};
}

fn same(a: &[Complex], b: &[Complex], tol: f64) -> f64 {
    assert_eq!(a.len(), b.len(), "{a:?} {b:?}");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm() / y.norm())
        .inspect(|d| assert!(*d < tol, "{a:?} {b:?}"))
        .fold(0.0, f64::max)
}

#[test]
fn leaky_modes_of_full_class_isotropic_films_are_the_te_and_tm_modes() {
    // On chart boxes about the scalar roots, clear of the exterior branch
    // points κ = 0, where the 4×4 exterior projector is not evaluated.
    let k0 = 6.0;
    let full = leaky(uniaxial(4.0, 4.0, 0.4, 0.3));
    let scalar = leaky(isotropic(4.0));
    let (mut worst, mut count, mut complex) = (0.0_f64, 0, 0);
    for pol in [TE, TM] {
        let reference = propagation!(
            PropagationProblem::with_backend(scalar.clone(), k0, pol, Scatter2::new()).unwrap()
        );
        let coupled = near!(
            PropagationProblem::with_backend(full.clone(), k0, Channel::Coupled, Transfer4)
                .unwrap(),
            &reference
        );
        let r: Vec<Complex> = reference.iter().map(|x| x.1).collect();
        let c: Vec<Complex> = coupled.iter().map(|x| x.1).collect();
        worst = worst.max(same(&c, &r, 1e-9));
        count += r.len();
        complex += r.iter().filter(|b| b.im.abs() > 1e-6).count();
    }
    assert!(complex >= 2);
    println!(
        "leaky: {count} TE and TM roots ({complex} complex) found by the coupled channel to {worst:e} relative"
    );
}

#[test]
fn quasi_normal_modes_of_full_class_isotropic_slabs_are_the_te_and_tm_modes() {
    let exterior = |v: f64| ConstantExterior::new(Complex::new(v, 0.0), Complex::ONE).unwrap();
    let layers = |layer: Region| [Layer::new(layer, cm(0.5))];
    let (mut worst, mut count) = (0.0_f64, 0);
    for pol in [TE, TM] {
        let reference = spectral!(
            ConformalSpectralProblem::with_backend(
                exterior(1.0),
                exterior(2.25),
                layers(isotropic(9.0)),
                3.0,
                pol,
                Scatter2::new(),
            )
            .unwrap()
        );
        let coupled = near!(
            ConformalSpectralProblem::with_backend(
                exterior(1.0),
                exterior(2.25),
                layers(uniaxial(9.0, 9.0, 0.4, 0.3)),
                3.0,
                Channel::Coupled,
                Transfer4,
            )
            .unwrap(),
            &reference
        );
        let r: Vec<Complex> = reference.iter().map(|x| x.1).collect();
        let c: Vec<Complex> = coupled.iter().map(|x| x.1).collect();
        assert!(r.iter().any(|k| k.im.abs() > 1e-6), "{r:?}");
        worst = worst.max(same(&c, &r, 1e-9));
        count += r.len();
    }
    println!(
        "quasi-normal: {count} TE and TM poles found by the coupled channel to {worst:e} relative"
    );
}

#[test]
fn real_hybrid_leaky_search_roots_are_the_guided_modes() {
    // A whole canonical window, its sides placed off the exterior branch
    // points: the real roots of the search are the guided count's modes.
    let stack = leaky(uniaxial(4.0, 6.25, 0.6, 0.7));
    let k0 = 6.0;
    let problem =
        PropagationProblem::with_backend(stack.clone(), k0, Channel::Coupled, Transfer4).unwrap();
    let report = problem
        .canonical_plan(
            InPlaneSearchExtent {
                max_parallel_wavenumber: 23.7,
                imaginary_margin: 0.0517,
            },
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            ConformalChartConfig::default(),
            ASSUME,
        )
        .unwrap()
        .solve(
            DiscoveryConfig::default(),
            RefinementConfig::default(),
            MatchConfig::default(),
        )
        .unwrap();
    // Every characteristic evaluation succeeded: what is left unresolved
    // are clusters of complex roots the general plan does not split.
    let clusters = report
        .unresolved()
        .iter()
        .filter(|u| {
            assert!(
                matches!(u.cause, UnresolvedCause::Candidate { .. }),
                "{u:?}"
            );
            true
        })
        .count();
    let mut roots: Vec<Complex> = report
        .groups()
        .iter()
        .map(|g| report.observations()[g.members()[0]].physical_coordinate())
        .collect();
    roots.sort_by(|a, b| a.re.total_cmp(&b.re));
    let guided = enumerate_guided_modes_with(
        Transfer4,
        stack,
        GuidedWindow {
            wavelength: cm(std::f64::consts::TAU / k0),
            family: ModeFamily::All,
            direction: GuidedDirection::Forward,
            abs_beta: [0.0, 24.0],
            cutoff_margin: 1e-3 * k0,
            phase_tolerance: 1e-11,
            max_subdivisions: 1000,
            refinement: Default::default(),
            assumption: ASSUME,
        },
    )
    .unwrap();
    assert!(guided.is_complete());
    let line = guided.light_line;
    let real: Vec<Complex> = roots
        .iter()
        .copied()
        .filter(|b| b.im.abs() <= 1e-9 * b.norm() && b.re > line)
        .collect();
    let mut expected: Vec<Complex> = guided
        .modes
        .iter()
        .map(|m| m.physical_coordinate())
        .collect();
    expected.sort_by(|a, b| a.re.total_cmp(&b.re));
    let worst = same(&real, &expected, 1e-9);
    let leaky = roots.iter().filter(|b| b.im.abs() > 1e-6).count();
    println!(
        "hybrid leaky search: {} guided roots agree with the guided count to {worst:e}; {leaky} complex roots and {clusters} unresolved clusters of them",
        real.len()
    );
}

#[test]
fn an_anisotropic_exterior_is_refused_off_the_guided_sheet() {
    let stack: ModeStack = Stack::builder(isotropic(1.0), uniaxial(2.25, 3.1, 0.6, 0.7))
        .layer(uniaxial(4.0, 6.25, 0.6, 0.7), cm(0.3))
        .finalise();
    let refused = |error: &Diagnostic| {
        matches!(
            error.find_source::<ConstructionError>(),
            Some(ConstructionError::Constitutive(
                ConstitutiveError::Unsupported {
                    region: StackRegion::RightExterior,
                    ..
                }
            ))
        )
    };
    let error = PropagationProblem::with_backend(stack.clone(), 2.0, Channel::Coupled, Transfer4)
        .unwrap_err();
    assert!(refused(&error), "{error:?}");
    let error = FrequencyProblem::with_backend(stack.clone(), 1.0, Channel::Coupled, Transfer4)
        .unwrap_err();
    assert!(refused(&error), "{error:?}");
    // The guided sheet refuses it too: its incoming waves can make the 4×4
    // reference basis lose rank above the light line.
    let w = GuidedWindow {
        wavelength: cm(1.0),
        family: ModeFamily::All,
        direction: GuidedDirection::Forward,
        abs_beta: [0.0, 20.0],
        cutoff_margin: 1e-2,
        phase_tolerance: 1e-11,
        max_subdivisions: 1000,
        refinement: Default::default(),
        assumption: ASSUME,
    };
    let error = enumerate_guided_modes(stack, w).unwrap_err();
    assert!(
        matches!(
            error,
            EnumerationError::Constitutive(ConstitutiveError::Unsupported {
                region: StackRegion::RightExterior,
                ..
            })
        ),
        "{error:?}"
    );
}

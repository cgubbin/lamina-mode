//! Searches on polar films that used to end silently incomplete.
use lamina_core::{
    AnalyticalMaterialHandle, AnalyticalTensorMaterialHandle, ComplexScalar, Constant,
    DerivativeOrder, DifferentiableMaterial, DifferentiableMeromorphicMaterial, Material,
    MeromorphicMaterial, Polarisation, Rotation, Sampled, Stack,
};
use lamina_materials::{Lorentz, LorentzOscillator, PrincipalAxes};
use lamina_mode::*;
use lamina_units::{Length, VacuumWavenumber};
use std::f64::consts::TAU;

const TM: Polarisation = Polarisation::TransverseMagnetic;
const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;

/// ε(ν) = ε∞ (1 + (ν_LO² − ν_TO²)/(ν_TO² − ν² − iγν)) at k0 = 2πν.
fn phonon(eps_inf: f64, to: f64, lo: f64, gamma: f64) -> Lorentz<f64> {
    let oscillator = LorentzOscillator::from_delta_epsilon(
        eps_inf * (lo * lo / (to * to) - 1.0),
        VacuumWavenumber::inverse_centimetres(to),
        VacuumWavenumber::inverse_centimetres(gamma),
    )
    .unwrap();
    Lorentz::new(eps_inf, vec![oscillator]).unwrap()
}
/// The c-axis (ε_zz) oscillator of hBN: TO 780, LO 830, γ 4 cm⁻¹.
fn hbn_par() -> Lorentz<f64> {
    phonon(2.95, 780.0, 830.0, 4.0)
}
/// The in-plane oscillator of hBN: TO 1370, LO 1610, γ 5 cm⁻¹.
fn hbn_perp() -> Lorentz<f64> {
    phonon(4.87, 1370.0, 1610.0, 5.0)
}
/// A material that declares no singularities, as a user model may not.
struct Undeclared<M>(M);
impl<M: Material<Real = f64>> Material for Undeclared<M> {
    type Real = f64;
    fn relative_permittivity<I, X>(&self, k: I) -> I::Mapped<X>
    where
        X: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        self.0.relative_permittivity(k)
    }
}
impl<M: DifferentiableMaterial<Real = f64>> DifferentiableMaterial for Undeclared<M> {
    fn relative_permittivity_derivative<I, X>(&self, k: I, order: DerivativeOrder) -> I::Mapped<X>
    where
        X: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = f64>,
    {
        self.0.relative_permittivity_derivative(k, order)
    }
}
impl<M: MeromorphicMaterial<Real = f64>> MeromorphicMaterial for Undeclared<M> {
    fn relative_permittivity_complex<I, X>(&self, k: I) -> I::Mapped<X>
    where
        X: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = X>,
    {
        self.0.relative_permittivity_complex(k)
    }
    fn relative_permeability_complex<I, X>(&self, k: I) -> I::Mapped<X>
    where
        X: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = X>,
    {
        self.0.relative_permeability_complex(k)
    }
}
impl<M: DifferentiableMeromorphicMaterial<Real = f64>> DifferentiableMeromorphicMaterial
    for Undeclared<M>
{
    fn relative_permittivity_complex_derivative<I, X>(
        &self,
        k: I,
        order: DerivativeOrder,
    ) -> I::Mapped<X>
    where
        X: ComplexScalar<RealField = f64>,
        I: Sampled<Elem = X>,
    {
        self.0.relative_permittivity_complex_derivative(k, order)
    }
}
fn air() -> Region {
    AnalyticalMaterialHandle::new(Constant::dielectric(1.0)).into()
}
/// A film in air: isotropic with the c-axis oscillator, or uniaxial hBN.
fn film(uniaxial: bool, nm: f64) -> ModeStack {
    let region: Region = if uniaxial {
        Region::oriented(
            AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(hbn_perp(), hbn_par())),
            Rotation::identity(),
        )
    } else {
        AnalyticalMaterialHandle::new(hbn_par()).into()
    };
    Stack::builder(air(), air())
        .layer(region, Length::nanometres(nm))
        .finalise()
}
/// A ν box (cm⁻¹) in k0, referenced at its centre on the decaying sheet.
fn frequency_input(beta: f64, nu: Complex, half: f64) -> FrequencyDomainInput {
    let centre = nu * TAU;
    let kappa = {
        let r = (centre * centre - beta * beta).sqrt();
        if r.im < 0.0 { -r } else { r }
    };
    FrequencyDomainInput {
        bounds: FrequencyRegion::new(
            (nu.re - half) * TAU,
            (nu.re + half) * TAU,
            (nu.im - half) * TAU,
            (nu.im + half) * TAU,
        )
        .unwrap(),
        reference: FrequencyReference::Explicit(ExteriorReferenceValues {
            frequency: centre,
            kappa: [kappa; 2],
        }),
        continuation: Default::default(),
        exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
        exclusions: vec![],
    }
}

/// ν* = √(ν_LO² − γ²/4) − iγ/2, where the c-axis permittivity vanishes.
fn enz_zero() -> Complex {
    Complex::new((830.0f64 * 830.0 - 4.0).sqrt(), -2.0)
}
fn enz_beta() -> f64 {
    1.4 * TAU * 830.0
}

/// An isotropic polar film searched in complex k0 around ε = 0, where its TM
/// characteristic has a pole: the negative winding there is a violated
/// assumption, never an empty, complete box. The model declares nothing, so
/// only the winding can tell.
#[test]
fn a_negative_winding_violates_the_pole_free_assumption() {
    let region: Region = AnalyticalMaterialHandle::new(Undeclared(hbn_par())).into();
    let stack = Stack::builder(air(), air())
        .layer(region, Length::nanometres(50.0))
        .finalise();
    let problem = FrequencyProblem::new(stack, enz_beta(), TM).unwrap();
    assert_eq!(problem.known_singularities().undeclared.len(), 1);
    let domain = problem
        .domain(frequency_input(enz_beta(), enz_zero(), 0.2), ASSUME)
        .unwrap();
    let report = problem
        .plan(vec![domain], vec![])
        .unwrap()
        .solve(Default::default(), Default::default(), Default::default())
        .unwrap();
    let entry = report.discovery().entries().next().unwrap().unwrap();
    assert_eq!(*entry.termination(), Termination::Converged);
    assert!(report.observations().is_empty());
    let negative: Vec<_> = entry
        .leaves()
        .iter()
        .filter(|leaf| leaf.data.root_count < 0)
        .collect();
    assert!(!negative.is_empty());
    let violations: Vec<_> = report
        .unresolved()
        .into_iter()
        .filter_map(|region| match region.cause {
            UnresolvedCause::AssumptionViolated(violation) => Some((region.bounds, violation)),
            _ => None,
        })
        .collect();
    assert_eq!(violations.len(), negative.len());
    for ((bounds, violation), leaf) in violations.iter().zip(&negative) {
        assert_eq!(*bounds, leaf.bounds);
        assert_eq!(
            *violation,
            WindingViolation::Negative {
                count: leaf.data.root_count
            }
        );
    }
    assert!(!report.is_complete());
}

/// The bound ENZ mode near ν = 829.565 − 2.00i cm⁻¹ at β = 1.4 k_LO.
const ENZ_ROOT: Complex = Complex::new(829.565, -2.0);

fn solve(problem: &FrequencyProblem, domains: Vec<Domain>) -> SolveReport {
    problem
        .plan(domains, vec![])
        .unwrap()
        .solve(Default::default(), Default::default(), Default::default())
        .unwrap()
}

#[test]
fn a_uniaxial_film_lists_the_singularities_its_tm_channel_sees() {
    let problem = FrequencyProblem::new(film(true, 50.0), enz_beta(), TM).unwrap();
    let singularities = problem.known_singularities();
    assert!(singularities.undeclared.is_empty());
    let layer = lamina_core::StackRegion::Layer(lamina_core::FiniteLayerIndex::new(0));
    assert!(singularities.known.iter().all(|s| s.region == layer));
    let count = |kind, axis| {
        singularities
            .known
            .iter()
            .filter(|s| s.kind == kind && s.axis == Some(axis))
            .count()
    };
    // Poles of every entry; zeros of ε_xx and ε_zz but not ε_yy.
    assert_eq!(
        [0, 1, 2].map(|axis| count(SingularityKind::Pole, axis)),
        [2, 2, 2]
    );
    assert_eq!(
        [0, 1, 2].map(|axis| count(SingularityKind::Zero, axis)),
        [2, 0, 2]
    );
    let zero = singularities
        .known
        .iter()
        .find(|s| s.kind == SingularityKind::Zero && s.axis == Some(2) && s.k0.re > 0.0)
        .unwrap();
    assert!((zero.k0 - enz_zero() * TAU).norm() < 1e-9 * zero.k0.norm());
}

#[test]
fn a_box_holding_a_zero_of_the_permittivity_is_refused() {
    for uniaxial in [false, true] {
        let problem = FrequencyProblem::new(film(uniaxial, 50.0), enz_beta(), TM).unwrap();
        let error = problem
            .domain(frequency_input(enz_beta(), ENZ_ROOT, 2.0), ASSUME)
            .unwrap_err();
        let Some(FrequencyError::KnownSingularity(point)) = error.frequency_error() else {
            panic!("{error}")
        };
        assert_eq!(point.kind, SingularityKind::Zero);
        assert_eq!(point.quantity, ConstitutiveQuantity::Permittivity);
        assert_eq!(point.axis, uniaxial.then_some(2));
        assert!((point.k0 - enz_zero() * TAU).norm() < 1e-9 * point.k0.norm());
        let message = error.to_string();
        assert!(
            message.contains(if uniaxial {
                "zero of ε_zz of layer 0"
            } else {
                "zero of ε of layer 0"
            }),
            "{message}"
        );
    }
}

#[test]
fn holes_around_known_singularities_recover_the_enz_root() {
    for uniaxial in [false, true] {
        let problem = FrequencyProblem::new(film(uniaxial, 50.0), enz_beta(), TM).unwrap();
        // The root from a small box clear of the singularity.
        let small = problem
            .domain(frequency_input(enz_beta(), ENZ_ROOT, 0.3), ASSUME)
            .unwrap();
        let small = solve(&problem, vec![small]);
        assert!(small.is_complete());
        let [expected] = small.observations() else {
            panic!("{:?}", small.observations())
        };
        let mut input = frequency_input(enz_beta(), ENZ_ROOT, 2.0);
        let holes = problem
            .singularity_exclusions(input.bounds, 0.1 * TAU)
            .unwrap();
        assert_eq!(holes.len(), 1, "{holes:?}");
        input.exclusions.extend(holes.clone());
        let domains = problem.domains(input, ASSUME).unwrap();
        assert_eq!(domains.len(), 4);
        let report = solve(&problem, domains);
        assert!(report.is_complete(), "{:?}", report.unresolved());
        let [found] = report.observations() else {
            panic!("{:?}", report.observations())
        };
        let (k, e) = (found.physical_coordinate(), expected.physical_coordinate());
        assert!((k - e).norm() < 1e-9 * e.norm(), "{k} vs {e}");
        let pairs = found
            .signature()
            .unwrap()
            .open_exteriors()
            .unwrap()
            .into_iter()
            .zip(expected.signature().unwrap().open_exteriors().unwrap());
        for (x, y) in pairs {
            assert!((x - y).norm() < 1e-9 * y.norm(), "{x} vs {y}");
        }
        assert!(
            report
                .exclusions()
                .contains(&CoverageExclusion::Frequency(holes[0].clone()))
        );
    }
}

#[test]
fn tiles_around_a_hole_find_the_roots_of_the_whole_box() {
    let problem = FrequencyProblem::new(film(false, 50.0), enz_beta(), TM).unwrap();
    let whole = problem
        .domain(frequency_input(enz_beta(), ENZ_ROOT, 0.3), ASSUME)
        .unwrap();
    let whole = solve(&problem, vec![whole]);
    let mut input = frequency_input(enz_beta(), ENZ_ROOT, 0.3);
    // A declared hole across the top edge, away from the root.
    let k = |nu: f64| nu * TAU;
    input.exclusions.push(FrequencyExclusion {
        region: FrequencyRegion::new(k(829.5), k(829.7), k(-1.8), k(-1.6)).unwrap(),
        reason: ExclusionReason::CallerDeclared("test".into()),
    });
    let domains = problem.domains(input, ASSUME).unwrap();
    assert_eq!(domains.len(), 3);
    let tiled = solve(&problem, domains);
    assert!(whole.is_complete() && tiled.is_complete());
    let [a] = whole.observations() else { panic!() };
    let [b] = tiled.observations() else { panic!() };
    assert!(
        (a.physical_coordinate() - b.physical_coordinate()).norm()
            < 1e-9 * a.physical_coordinate().norm()
    );
    let (sa, sb) = (a.signature().unwrap(), b.signature().unwrap());
    for (x, y) in sa
        .open_exteriors()
        .unwrap()
        .iter()
        .zip(sb.open_exteriors().unwrap())
    {
        assert!((x - y).norm() < 1e-9 * x.norm(), "{x} vs {y}");
    }
}

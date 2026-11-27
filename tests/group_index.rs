//! Certified group index against finite differences of dβ/dk0 on dispersive slabs.
use lamina_core::{AnalyticalMaterialStack, Constant, Polarisation, Scatter2, Stack};
use lamina_materials::{Sellmeier, SellmeierTerm};
use lamina_mode::*;
use lamina_units::{Length, VacuumAngularWavenumber};

mod hbn_support;
use hbn_support::*;

/// A strongly dispersive core (pole close to the operating wavenumber) and a
/// mildly dispersive substrate, around a 1 cm wavelength.
fn stack() -> AnalyticalMaterialStack<Complex> {
    let pole = |b: f64, q: f64| {
        SellmeierTerm::new(b, VacuumAngularWavenumber::inverse_centimetres(q)).unwrap()
    };
    let core = Sellmeier::new(1.5, vec![pole(1.2, 9.0)]).unwrap();
    let substrate = Sellmeier::new(1.6, vec![pole(0.3, 15.0)]).unwrap();
    Stack::from_analytical_materials(Constant::vacuum(), substrate)
        .analytical_layer(core, Length::centimetres(1.2))
        .finalise()
}

fn modes(wavelength_cm: f64, polarisation: Polarisation) -> GuidedBasis {
    let k0 = std::f64::consts::TAU / wavelength_cm;
    let basis = enumerate_guided_modes(
        stack(),
        GuidedWindow {
            wavelength: Length::centimetres(wavelength_cm),
            family: polarisation.into(),
            direction: GuidedDirection::Forward,
            abs_beta: [0., 3. * k0],
            cutoff_margin: 1e-8,
            phase_tolerance: 1e-11,
            max_subdivisions: 1000,
            refinement: Default::default(),
            assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        },
    )
    .unwrap();
    assert!(basis.is_complete(), "{:?}", basis.unresolved);
    basis
}

#[test]
fn group_index_matches_finite_difference_and_energy_velocity() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let wavelength = 1.0;
        let k0 = std::f64::consts::TAU / wavelength;
        let base = modes(wavelength, polarisation);
        assert!(base.modes.len() >= 2);
        let beta_at = |k: f64| {
            modes(std::f64::consts::TAU / k, polarisation)
                .modes
                .iter()
                .map(|p| p.physical_coordinate().re)
                .collect::<Vec<_>>()
        };
        let mut errors = Vec::new();
        for h in [1e-3, 1e-4, 1e-5] {
            let (plus, minus) = (beta_at(k0 + h * k0), beta_at(k0 - h * k0));
            let mut worst: f64 = 0.;
            for (i, point) in base.modes.iter().enumerate() {
                let group = point
                    .guided_group_index_with(&Scatter2::new(), Default::default(), 1e-8)
                    .unwrap();
                let fd = (plus[i] - minus[i]) / (2. * h * k0);
                // Dispersion makes the group index exceed the phase index.
                assert!(group.value > point.physical_coordinate().re / k0);
                assert!(group.derivative.im.abs() < 1e-10 * group.value);
                assert!(group.relative_discrepancy < 1e-8);
                worst = worst.max((group.value - fd).abs() / group.value);
            }
            errors.push(worst);
        }
        // Central differences converge at second order: each tenfold smaller
        // step cuts the error about a hundredfold, down to the root-correction
        // floor.
        let order = (errors[0] / errors[1]).log10();
        assert!((1.7..2.3).contains(&order), "{errors:?}");
        assert!(errors[2] < 1e-7, "{errors:?}");
    }
}

#[test]
fn group_index_rejects_invalid_tolerance() {
    let basis = modes(1.0, Polarisation::TransverseElectric);
    assert!(matches!(
        basis.modes[0].guided_group_index_with(&Scatter2::new(), Default::default(), 0.0),
        Err(GroupIndexError::InvalidTolerance)
    ));
}

/// A lossy hBN film in its type II band, where ε_xx is complex: the group
/// index is refused naming the film's layer and the entry.
#[test]
fn an_ineligible_entry_is_named() {
    let stack = hbn_film(hbn(5.0, 4.0), 100.0);
    let nu = 1500.0;
    let point = root(&stack, nu, 31.4 * k0_of(nu), 0.3);
    let error = point
        .guided_group_index_with(&Scatter2::new(), FieldRecoverySettings::default(), 1e-6)
        .unwrap_err();
    let GroupIndexError::IneligibleEntry {
        region,
        entry,
        value,
    } = &error
    else {
        panic!("{error}")
    };
    assert_eq!(
        *region,
        lamina_core::StackRegion::Layer(lamina_core::FiniteLayerIndex::new(0))
    );
    assert_eq!(*entry, TensorEntry::EpsilonXx);
    assert!(value.re < 0.0 && value.im != 0.0);
    assert!(error.to_string().contains("ε_xx of layer 0"), "{error}");
}

/// dβ/dk0 from the implicit slope −(∂F/∂k0)/(∂F/∂β) at the root.
fn implicit_slope(point: &ModePoint) -> Complex {
    point
        .local_derivatives_with(
            Scatter2::new(),
            &[PhysicalParameter::K0, PhysicalParameter::Beta],
            LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
            Default::default(),
        )
        .unwrap()
        .implicit_slope(ImplicitSlopeRequest::BetaPerK0, Default::default())
        .unwrap()
        .value
}

/// The certified group index of a lossless mode at ν̃ against the implicit
/// slope and central differences of β re-found at ν̃(1 ± h); also its
/// unit-power profile, whose power sign is the sign of the group velocity.
fn check_certified(stack: &ModeStack, nu: f64, beta: f64, radius: f64) -> f64 {
    let point = root(stack, nu, beta, radius);
    let beta = point.physical_coordinate().re;
    let group = point
        .guided_group_index_with(&Scatter2::new(), FieldRecoverySettings::default(), 1e-8)
        .unwrap_or_else(|e| panic!("ν̃ {nu}, β {beta}: {e}"));
    assert!(group.relative_discrepancy < 1e-8, "{group:?}");
    let slope = implicit_slope(&point);
    assert!(
        (slope - group.derivative).norm() < 1e-10 * slope.norm(),
        "{slope} {group:?}"
    );
    // β moves fast with ν̃ near the phonon edges and the fold: small
    // relative steps, Richardson-extrapolated.
    let central = |h: f64| {
        let at = |nu: f64| root(stack, nu, beta, radius).physical_coordinate().re;
        (at(nu * (1. + h)) - at(nu * (1. - h))) / (2. * h * k0_of(nu))
    };
    let (coarse, fine) = (central(1e-6), central(5e-7));
    let fd = (4. * fine - coarse) / 3.;
    assert!(
        (fd - group.value).abs() < 1e-6 * group.value.abs(),
        "ν̃ {nu}, β {beta}: n_g {} vs differences {fd}",
        group.value
    );
    let sampling =
        lamina_core::FieldSampling::new().layer(0, lamina_core::LayerSampling::uniform(5));
    let profile = point
        .guided_power_profile_with(Scatter2::new(), &sampling, FieldRecoverySettings::default())
        .unwrap_or_else(|e| panic!("ν̃ {nu}, β {beta}: {e}"));
    let ProfileNormalization::GuidedPower(power) = profile.normalization else {
        panic!()
    };
    assert_eq!(power.signed_power_w_per_m, group.value.signum());
    group.value
}

/// Lossless hBN film (100 nm in air): hyperbolic TM modes in both
/// Reststrahlen bands are certified, forward in type II (ε_xx < 0) and
/// backward in type I (ε_zz < 0), as is the forward piece of the type I
/// fundamental below its fold.
#[test]
fn lossless_hyperbolic_modes_are_certified() {
    let stack = hbn_film(hbn(0.0, 0.0), 100.0);
    // (ν̃, effective indices, sign of the group velocity)
    for (nu, indices, sign) in [
        (1501.0, &[31.373, 57.983][..], 1.0),
        (801.0, &[40.973, 87.153][..], -1.0),
        (829.0, &[1.193][..], 1.0),
        (829.0, &[1.741, 7.444][..], -1.0),
    ] {
        for &n in indices {
            let radius = if n < 2.0 { 0.05 } else { 0.01 };
            let n_g = check_certified(&stack, nu, n * k0_of(nu), radius);
            assert_eq!(n_g.signum(), sign, "ν̃ {nu}, n_eff {n}: n_g {n_g}");
        }
    }
}

/// The surface phonon polariton of a lossless polar half-space under air,
/// β = k0 √(ε/(1 + ε)), between ν̃_TO and the surface frequency (ε < −1): its
/// certified group index against the closed form
/// dβ/dk0 = √(ε/(1 + ε)) + k0 ε'/(2 (1 + ε)² √(ε/(1 + ε))), ε' = dε/dk0.
#[test]
fn surface_phonon_polariton_group_index_is_analytic() {
    let material = phonon(2.95, 780.0, 830.0, 0.0);
    let substrate: Region = lamina_core::AnalyticalMaterialHandle::new(material).into();
    let stack: ModeStack = Stack::builder(air(), substrate).finalise();
    for nu in [800.0, 810.0, 815.0] {
        let k0 = k0_of(nu);
        let (to, lo) = (780.0_f64, 830.0_f64);
        let e = 2.95 * (1.0 + (lo * lo - to * to) / (to * to - nu * nu));
        assert!(e < -1.0);
        let de_dk0 =
            2.95 * (lo * lo - to * to) * 2.0 * nu / (to * to - nu * nu).powi(2) / k0_of(1.0);
        let n = (e / (1.0 + e)).sqrt();
        let analytic = n + k0 * de_dk0 / (2.0 * (1.0 + e).powi(2) * n);
        let point = root_on(&stack, nu, n * k0, 0.05, e);
        assert!((point.physical_coordinate().re - n * k0).abs() < 1e-10 * n * k0);
        let group = point
            .guided_group_index_with(&Scatter2::new(), FieldRecoverySettings::default(), 1e-8)
            .unwrap_or_else(|e| panic!("ν̃ {nu}: {e}"));
        assert!(
            (group.value - analytic).abs() < 1e-7 * analytic,
            "ν̃ {nu}: {} vs {analytic}",
            group.value
        );
    }
}

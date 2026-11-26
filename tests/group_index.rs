//! Certified group index against finite differences of dβ/dk0 on dispersive slabs.
use lamina_core::{AnalyticalMaterialStack, Constant, Polarisation, Scatter2, Stack};
use lamina_materials::{Sellmeier, SellmeierTerm};
use lamina_mode::*;
use lamina_units::{Length, VacuumAngularWavenumber};

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

/// A lossless hBN film in its type II band, where ε_xx < 0: the group index
/// is refused naming the film's layer and the entry.
#[test]
fn an_ineligible_entry_is_named() {
    use lamina_core::{AnalyticalMaterialHandle, AnalyticalTensorMaterialHandle, Rotation};
    use lamina_materials::{Lorentz, LorentzOscillator, PrincipalAxes};
    use lamina_units::VacuumWavenumber;
    let phonon = |eps_inf: f64, to: f64, lo: f64| {
        let oscillator = LorentzOscillator::from_delta_epsilon(
            eps_inf * (lo * lo / (to * to) - 1.0),
            VacuumWavenumber::inverse_centimetres(to),
            VacuumWavenumber::inverse_centimetres(0.0),
        )
        .unwrap();
        Lorentz::new(eps_inf, vec![oscillator]).unwrap()
    };
    let air = || -> Region { AnalyticalMaterialHandle::new(Constant::dielectric(1.0)).into() };
    let film = Region::oriented(
        AnalyticalTensorMaterialHandle::new(PrincipalAxes::uniaxial(
            phonon(4.87, 1370.0, 1610.0),
            phonon(2.95, 780.0, 830.0),
        )),
        Rotation::identity(),
    );
    let stack = Stack::builder(air(), air())
        .layer(film, Length::nanometres(100.0))
        .finalise();
    let k0 = std::f64::consts::TAU * 1500.0;
    let problem = PropagationProblem::new(stack, k0, Polarisation::TransverseMagnetic).unwrap();
    let domain = problem
        .domain(
            ConformalRegion::new(3.0, 3.5, -0.6, 0.6).unwrap(),
            ReferenceSelection::Sheet(CoordinateSheet::Principal),
            ConformalChartConfig::default(),
            AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        )
        .unwrap();
    let report = problem
        .plan(vec![domain], vec![])
        .unwrap()
        .solve(Default::default(), Default::default(), Default::default())
        .unwrap();
    let [point] = report.observations() else {
        panic!("{:?}", report.unresolved())
    };
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
    assert!(value.re < 0.0 && value.im == 0.0);
    assert!(error.to_string().contains("ε_xx of layer 0"), "{error}");
}

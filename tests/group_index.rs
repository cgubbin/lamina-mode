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
            polarisation,
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
                    .guided_group_index_with(&Scatter2::new(), 1e-8)
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
        basis.modes[0].guided_group_index_with(&Scatter2::new(), 0.0),
        Err(GroupIndexError::InvalidTolerance)
    ));
}

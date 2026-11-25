//! Registered overlap of two shifted asymmetric slab modes.
use lamina_core::{Constant, Polarisation, Scatter2, Stack};
use lamina_mode::*;
use lamina_units::Length;

fn main() -> Result<(), String> {
    let mode = |width| {
        let stack =
            Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(1.44))
                .analytical_layer(Constant::dielectric(4.0), Length::centimetres(width))
                .finalise();
        let result = enumerate_guided_modes(
            stack,
            GuidedWindow {
                wavelength: Length::centimetres(1.0),
                family: Polarisation::TransverseElectric.into(),
                direction: GuidedDirection::Forward,
                abs_beta: [0.0, 15.0],
                cutoff_margin: 1e-8,
                phase_tolerance: 1e-11,
                max_subdivisions: 1000,
                refinement: Default::default(),
                assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
            },
        )
        .map_err(|e| format!("{e:?}"))?;
        if !result.is_complete() {
            return Err(format!("unresolved basis: {:?}", result.unresolved));
        }
        result
            .modes
            .into_iter()
            .next()
            .ok_or_else(|| "missing fundamental mode".to_owned())
    };
    let a = mode(0.25)?;
    let b = mode(0.30)?;
    let backend = Scatter2::new();
    let registration = GuidedOverlapRegistration {
        origins_cm: [0.0, 0.07],
    };
    let stationary = GuidedPerturbation::new(
        "second slab thickness",
        "cm",
        vec![0.0; 2],
        0.0,
        vec![ConstitutiveRate::default(); 3],
    )
    .map_err(|e| format!("{e:?}"))?;
    let thickness = GuidedPerturbation::thickness(1, 0).map_err(|e| format!("{e:?}"))?;
    let result = guided_power_overlap_sensitivity_with(
        &backend,
        [&a, &b],
        [&stationary, &thickness],
        registration,
        Default::default(),
    )
    .map_err(|e| format!("{e:?}"))?;
    let tangent = result.tangent.map_err(|e| format!("{e:?}"))?;
    println!(
        "Power overlap: {:.9} W/m from {} samples",
        result.base.power_w_per_m, result.base.sample_count
    );
    println!(
        "Thickness rate: {:.9} W/m per cm from {} samples",
        tangent.power_rate_w_per_m, tangent.sample_count
    );
    println!(
        "Last quadrature changes: {:.3e} W/m and {:.3e} W/m per cm",
        result.base.quadrature_change_w_per_m, tangent.quadrature_change_w_per_m
    );
    for step in [1e-3, 1e-4, 1e-5] {
        let evaluate = |d| {
            guided_power_overlap_with(&backend, [&a, &mode(d)?], registration, Default::default())
                .map(|overlap| overlap.power_w_per_m)
                .map_err(|e| format!("{e:?}"))
        };
        let fd = (evaluate(0.30 + step)? - evaluate(0.30 - step)?) / (2.0 * step);
        println!(
            "Step {step:.0e} cm: central difference {fd:.9}, error {:.3e}",
            (fd - tangent.power_rate_w_per_m).norm()
        );
    }
    Ok(())
}

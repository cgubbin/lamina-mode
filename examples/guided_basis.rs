//! Enumerate and sample a power-normalized TE slab basis at one vacuum wavelength.
use lamina_core::{Constant, FieldSampling, Polarisation, Scatter2, Stack};
use lamina_mode::*;
use lamina_units::Length;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(2.25))
        .analytical_layer(Constant::dielectric(4.), Length::centimetres(2.))
        .finalise();
    let request = GuidedWindow {
        wavelength: Length::centimetres(1.),
        polarisation: Polarisation::TransverseElectric,
        direction: GuidedDirection::Forward,
        abs_beta: [0., 20.],
        cutoff_margin: 1e-7,
        phase_tolerance: 1e-10,
        max_subdivisions: 1000,
        refinement: Default::default(),
        assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
    };
    let basis = enumerate_guided_modes(stack, request)?;
    println!(
        "eligible |β| (cm^-1): {:?}, expected {:?}, accepted {}",
        basis.eligible,
        basis.expected,
        basis.modes.len()
    );
    if !basis.is_complete() {
        return Err(format!("unresolved: {:?}", basis.unresolved).into());
    }
    let backend = Scatter2::new();
    let direction =
        GuidedPerturbation::thickness(1, 0).map_err(|e| format!("thickness direction: {e:?}"))?;
    let sampling = FieldSampling::new().layer_centres();
    for (i, point) in basis.modes.iter().enumerate() {
        let sensitivity = point
            .guided_beta_sensitivity_with(
                &backend,
                &direction,
                Default::default(),
                Default::default(),
            )
            .map_err(|e| format!("mode {i} propagation sensitivity: {e:?}"))?
            .power_field_tangent_with(
                &backend,
                &sampling,
                GuidedObservationFrame::MaterialAttached,
                Complex::ONE,
            )
            .map_err(|e| format!("mode {i} field sensitivity: {e:?}"))?;
        let beta = sensitivity.eigenvalue.point().physical_coordinate();
        let beta_rate = sensitivity.eigenvalue.beta_rate;
        let centre = &sensitivity.samples[0];
        let ey_rate = centre
            .tangent
            .map_err(|e| format!("mode {i} centre: {e:?}"))?[1];
        // A slab propagator with unit input amplitude and a fixed length.
        let length_cm = 0.5;
        let phase = (Complex::I * beta * length_cm).exp();
        let output = phase * centre.field[1];
        let output_rate = phase * (ey_rate + Complex::I * beta_rate * length_cm * centre.field[1]);
        println!(
            "{i}: beta={beta:.9}, d beta/d thickness={beta_rate:.6}, output Ey={output:.6}, d Ey/d thickness={output_rate:.6}"
        );
    }
    Ok(())
}

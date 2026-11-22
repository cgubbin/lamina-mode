#[path = "support/pole_fixtures.rs"]
mod fixtures;
#[path = "support/pole_material.rs"]
mod material;
#[path = "support/pole_green.rs"]
mod reference;

use lamina_core::{FiniteLayerIndex, Polarisation, Scatter2};
use lamina_mode::{
    ImplicitSlopeRequest, ImplicitSlopeSettings, LocalAnalyticityAssumption,
    LocalDerivativeSettings, PhysicalParameter,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let point = fixtures::cavity(Polarisation::TransverseMagnetic, true)?;
    let request = [
        PhysicalParameter::K0,
        PhysicalParameter::Beta,
        PhysicalParameter::LayerThickness(FiniteLayerIndex::new(0)),
        // Deliberately out of bounds to show that one selector failure does
        // not discard the successful outcomes above.
        PhysicalParameter::LayerThickness(FiniteLayerIndex::new(1)),
    ];
    let report = point.local_derivatives_with(
        Scatter2::new(),
        &request,
        LocalAnalyticityAssumption::CallerAssumedHolomorphicInK0AndBetaAndRegularInThickness,
        LocalDerivativeSettings::default(),
    )?;

    println!("accepted point: k0={}", point.physical_coordinate());
    println!(
        "source backend: {}; derivative backend: {}",
        report.source_backend, report.evaluation_backend
    );
    println!("characteristic: {:?}", report.characteristic);
    for outcome in &report.outcomes {
        match &outcome.result {
            Ok(partial) => println!(
                "{:?}: {} {:?}",
                partial.parameter, partial.value, partial.units
            ),
            Err(error) => println!("{:?}: unavailable ({error})", outcome.parameter),
        }
    }
    match report.implicit_slope(
        ImplicitSlopeRequest::K0PerBeta,
        ImplicitSlopeSettings::default(),
    ) {
        Ok(slope) => println!("dk0/dbeta: {} {:?}", slope.value, slope.units),
        Err(error) => println!("dk0/dbeta: unavailable ({error})"),
    }
    Ok(())
}

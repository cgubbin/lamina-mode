//! Coherent propagation from two supplied, fixed modal amplitudes.
use lamina_core::{Constant, FieldSampling, LayerSampling, Polarisation, Scatter2, Stack};
use lamina_mode::*;
use lamina_units::Length;
use std::f64::consts::{PI, TAU};

const THICKNESS: f64 = 0.5;
const WAVELENGTH: f64 = 1.0;
const OUTPUT: f64 = 0.37;
const INPUT: [Complex; 2] = [Complex::ONE, Complex::new(0.3, 0.6)];

fn basis(thickness: f64, wavelength: f64) -> Result<Vec<ModePoint>, String> {
    let stack = Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
        .analytical_layer(Constant::dielectric(4.0), Length::centimetres(thickness))
        .finalise();
    let result = enumerate_guided_modes(
        stack,
        GuidedWindow {
            wavelength: Length::centimetres(wavelength),
            family: Polarisation::TransverseElectric.into(),
            direction: GuidedDirection::Forward,
            abs_beta: [0.0, 20.0],
            cutoff_margin: 1e-8,
            phase_tolerance: 1e-11,
            max_subdivisions: 1000,
            refinement: Default::default(),
            assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        },
    )
    .map_err(|e| format!("{e:?}"))?;
    if !result.is_complete() || result.modes.len() < 2 {
        return Err(format!(
            "incomplete two-mode basis: {:?}",
            result.unresolved
        ));
    }
    Ok(result.modes.into_iter().take(2).collect())
}

fn sampling() -> FieldSampling<f64> {
    FieldSampling::new().layer(0, LayerSampling::uniform_with_boundaries(5, true, true))
}

struct Contribution {
    beta: Complex,
    beta_rate: Complex,
    ey: Complex,
    ey_rate: Complex,
}

fn contributions(direction: &GuidedPerturbation) -> Result<Vec<Contribution>, String> {
    let backend = Scatter2::new();
    basis(THICKNESS, WAVELENGTH)?
        .iter()
        .map(|point| {
            let result = point
                .guided_beta_sensitivity_with(
                    &backend,
                    direction,
                    Default::default(),
                    Default::default(),
                )
                .map_err(|e| format!("{e:?}"))?
                .power_field_tangent_with(
                    &backend,
                    &sampling(),
                    GuidedObservationFrame::MaterialAttached,
                    Complex::ONE,
                )
                .map_err(|e| format!("{e:?}"))?;
            let sample = &result.samples[1]; // one quarter of the layer thickness
            Ok(Contribution {
                beta: point.physical_coordinate(),
                beta_rate: result.eigenvalue.beta_rate,
                ey: sample.field[1],
                ey_rate: sample.tangent.map_err(|e| format!("{e:?}"))?[1],
            })
        })
        .collect()
}

fn propagate(modes: &[Contribution], x: f64) -> (Complex, Complex) {
    modes.iter().zip(INPUT).fold(
        (Complex::ZERO, Complex::ZERO),
        |(field, rate), (mode, amplitude)| {
            let weight = amplitude * (Complex::I * mode.beta * x).exp();
            (
                field + weight * mode.ey,
                rate + weight * (mode.ey_rate + Complex::I * x * mode.beta_rate * mode.ey),
            )
        },
    )
}

fn analytic_beta(order: usize) -> f64 {
    let k = TAU / WAVELENGTH;
    let (mut lo, mut hi) = (k, 2.0 * k);
    for _ in 0..60 {
        let beta = (lo + hi) / 2.0;
        let core = (4.0 * k * k - beta * beta).max(0.0).sqrt();
        let decay = (beta * beta - k * k).max(0.0).sqrt();
        let phase = core * THICKNESS / 2.0 - decay.atan2(core);
        if phase > order as f64 * PI / 2.0 {
            lo = beta;
        } else {
            hi = beta;
        }
    }
    (lo + hi) / 2.0
}

fn main() -> Result<(), String> {
    let thickness = GuidedPerturbation::thickness(1, 0).map_err(|e| format!("{e:?}"))?;
    let wavelength = GuidedPerturbation::new(
        "wavelength",
        "cm",
        vec![0.0; 2],
        1.0,
        vec![ConstitutiveRate::default(); 3],
    )
    .map_err(|e| format!("{e:?}"))?;
    for direction in [thickness, wavelength] {
        let modes = contributions(&direction)?;
        let (field, rate) = propagate(&modes, OUTPUT);
        let beat = TAU / (analytic_beta(0) - analytic_beta(1));
        println!(
            "{}: beat={beat:.9} cm, Ey={field:.9}, dEy={rate:.9}, d|Ey|²={:.9}",
            direction.parameter(),
            2.0 * (field.conj() * rate).re
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn independently_propagated(thickness: f64, wavelength: f64) -> Complex {
        basis(thickness, wavelength)
            .unwrap()
            .iter()
            .zip(INPUT)
            .map(|(point, input)| {
                let profile = point
                    .guided_power_profile_with(Scatter2::new(), &sampling(), Default::default())
                    .unwrap();
                input
                    * profile.fields.quantity().electric().y()[1]
                    * (Complex::I * point.physical_coordinate() * OUTPUT).exp()
            })
            .sum()
    }

    #[test]
    fn beat_period_and_output_sensitivities() {
        let wavelength = GuidedPerturbation::new(
            "wavelength",
            "cm",
            vec![0.0; 2],
            1.0,
            vec![ConstitutiveRate::default(); 3],
        )
        .unwrap();
        for (direction, dd, dl) in [
            (GuidedPerturbation::thickness(1, 0).unwrap(), 1.0, 0.0),
            (wavelength, 0.0, 1.0),
        ] {
            let modes = contributions(&direction).unwrap();
            for (order, mode) in modes.iter().enumerate() {
                assert!((mode.beta - analytic_beta(order)).norm() < 1e-8);
            }
            let beat = TAU / (analytic_beta(0) - analytic_beta(1));
            let intensity = |z| propagate(&modes, z).0.norm_sqr();
            let initial = intensity(OUTPUT);
            assert!((intensity(OUTPUT + beat) - initial).abs() < 1e-8 * initial);
            assert!((intensity(OUTPUT + beat / 2.0) - initial).abs() > 0.01 * initial);
            let (field, rate) = propagate(&modes, OUTPUT);
            let intensity_rate = 2.0 * (field.conj() * rate).re;
            for step in [1e-3, 1e-4, 1e-5] {
                let plus = independently_propagated(THICKNESS + step * dd, WAVELENGTH + step * dl);
                let minus = independently_propagated(THICKNESS - step * dd, WAVELENGTH - step * dl);
                let field_fd = (plus - minus) / (2.0 * step);
                let intensity_fd = (plus.norm_sqr() - minus.norm_sqr()) / (2.0 * step);
                let field_error = (field_fd - rate).norm() / rate.norm().max(1.0);
                let intensity_error =
                    (intensity_fd - intensity_rate).abs() / intensity_rate.abs().max(1.0);
                println!(
                    "{} step={step:e} field={field_error:e} intensity={intensity_error:e}",
                    direction.parameter()
                );
                if step == 1e-5 {
                    assert!(field_error < 2e-5 && intensity_error < 2e-5);
                }
            }
        }
    }
}

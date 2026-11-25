//! Shared quick/full numerical validation; reports retain every attempted case.
#![allow(dead_code)]
#[path = "pole_fixtures.rs"]
mod fixtures;
#[path = "pole_material.rs"]
mod material;
#[path = "pole_green.rs"]
mod reference;
use lamina_core::{
    EvaluateMeromorphicMaterial, ExteriorSampling, FieldSampling, LayerSampling,
    ModalNormalizationBackend, Scalar, Scatter2, Stack, Transfer2,
};
use lamina_mode::*;
use lamina_units::Length;
use std::{io::Write, path::Path};
type Error = Box<dyn std::error::Error>;
fn c(v: f64) -> Complex {
    Complex::new(v, 0.)
}
fn positions(point: &ModePoint, xs: &[f64]) -> FieldSampling<f64> {
    let provenance = point.provenance();
    let stack = provenance.stack();
    let width: f64 = stack
        .layers_left_to_right()
        .iter()
        .map(|l| l.thickness().as_centimetres())
        .sum();
    let mut result = FieldSampling::new();
    for &x in xs {
        if x <= 0. {
            result = result.left_exterior(ExteriorSampling::point(Length::centimetres(-x)));
        } else if x >= width {
            result = result.right_exterior(ExteriorSampling::point(Length::centimetres(x - width)));
        } else {
            let mut left = 0.;
            for (i, l) in stack.layers_left_to_right().iter().enumerate() {
                let d = l.thickness().as_centimetres();
                if x <= left + d {
                    result = result.layer(i, LayerSampling::point(Length::centimetres(x - left)));
                    break;
                }
                left += d;
            }
        }
    }
    result
}
fn primary(f: &ModeFields) -> &ndarray::Array1<Complex> {
    if f.point.provenance().polarisation() == Some(lamina_core::Polarisation::TransverseElectric) {
        f.fields.quantity().electric().y()
    } else {
        f.fields.quantity().magnetic().y()
    }
}
fn require(ok: bool, message: &str) -> Result<(), Error> {
    if ok {
        Ok(())
    } else {
        Err(message.to_string().into())
    }
}
fn error(a: Complex, b: Complex) -> f64 {
    (a - b).norm() / b.norm()
}
pub fn run(out: &Path, full: bool) -> Result<(), Error> {
    std::fs::create_dir(out)?;
    let outcome = run_inner(out, full);
    let status = if outcome.is_ok() {
        "complete"
    } else {
        "failed"
    };
    let failure = outcome
        .as_ref()
        .err()
        .map(|e| format!("{e}"))
        .unwrap_or_default();
    std::fs::write(
        out.join("summary.json"),
        format!(
            "{{\"schema\":1,\"status\":{status:?},\"suite\":{:?},\"convention\":\"scalar outgoing unit-delta Fourier component; lambda in cm^-1\",\"residue_tolerance\":1e-7,\"failure\":{failure:?}}}\n",
            if full { "full" } else { "quick" }
        ),
    )?;
    outcome
}
fn run_inner(out: &Path, full: bool) -> Result<(), Error> {
    let mut rows = std::fs::File::create(out.join("cases.csv"))?;
    writeln!(
        rows,
        "case,status,residue_error,contour_error,backend_error,gauge_error,partition_error,omitted_boundary_error,hermitian_product_error,wrong_variable_error,omitted_dispersion_error"
    )?;
    let mut convergence = std::fs::File::create(out.join("convergence.csv"))?;
    writeln!(
        convergence,
        "case,position,direction,relative_offset,relative_error"
    )?;
    let mut contributions = std::fs::File::create(out.join("contributions.csv"))?;
    writeln!(contributions, "case,item,re,im")?;
    let mut coordinates = std::fs::File::create(out.join("coordinates.csv"))?;
    writeln!(
        coordinates,
        "case,variable,polarisation,k0_re,k0_im,beta_re,beta_im,kappa_left_re,kappa_left_im,kappa_right_re,kappa_right_im"
    )?;
    std::fs::write(
        out.join("settings.json"),
        r#"{"factor_atol":1e-12,"factor_rtol":1e-12,"factor_reference_scale":1,"max_condition":1e10,"kappa_cutoff_inverse_cm":1e-10,"constitutive_cutoff":1e-12,"characteristic_limit":1e-5,"relative_boundary_limit":1e-8,"contour_radius_over_pole":1e-4,"seed_amplitude":[1,0],"local_assumption":"caller-assumed holomorphic pole-free characteristic; no causality certification","models":{"cavity":"vacuum | epsilon=4 d=.3 cm | epsilon=2.25","interface":"vacuum | Drude epsilon_inf=1 kp=1e5 cm^-1 eta=.02","film":"vacuum | same Drude d=40 nm | vacuum, k0=.5 kp","dispersive_electric":"finite epsilon=3.6+.03 k0 mu=1 d=.3 cm; right epsilon=2.25+.02 k0 mu=1","dispersive_magnetic":"same epsilon; finite mu=1+.015 k0, right mu=1+.005 k0","guided":"TE k0=2 beta=3.5 or TM k0=.5 kp beta=.8 kp, epsilon left/core/right=1/4/2.25; independent phase thickness"}}"#,
    )?;
    let mut failures = Vec::new();
    for (name, point) in fixtures::cases() {
        let attempt = (|| -> Result<(), Error> {
            let point = point?;
            let sig = *point.signature().map_err(Clone::clone)?;
            let provenance = point.provenance();
            let stack = &reference::scalar_stack(provenance.stack());
            let pol = provenance.polarisation().expect("a scalar channel");
            let spectral = matches!(provenance.direction(), SearchCoordinate::Spectral { .. });
            let variable = if spectral {
                PoleVariable::K0AtFixedBeta
            } else {
                PoleVariable::BetaAtFixedK0
            };
            let recovery = FieldRecoverySettings {
                characteristic_limit: 1e-5,
                ..Default::default()
            };
            let mode = point.normalize_with(
                Transfer2::new(),
                variable,
                AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
                Default::default(),
                recovery,
            )?;
            std::fs::write(
                out.join(format!("{name}_evidence.txt")),
                format!("{mode:#?}"),
            )?;
            let distance = 1. / sig.k0.norm().max(sig.k_parallel.norm());
            let width: f64 = stack
                .layers_left_to_right()
                .iter()
                .map(|l| l.thickness().as_centimetres())
                .sum();
            let z = [-0.17 * distance, width * 0.23, width + 0.13 * distance];
            let sampling = positions(&point, &z);
            let field = mode.fields(&sampling)?;
            let psi = primary(&field);
            let center = if spectral { sig.k0 } else { sig.k_parallel };
            let radius = center.norm() * 1e-4;
            let pairs = if full {
                vec![(0, 1), (1, 2), (0, 2)]
            } else {
                vec![(0, 2), (0, 1)]
            };
            let mut residue_error: f64 = 0.;
            let mut contour_error: f64 = 0.;
            let mut last_direct = c(0.);
            for (pair, (a, b)) in pairs.into_iter().enumerate() {
                let direct = reference::contour_residue(
                    stack,
                    sig,
                    spectral,
                    pol,
                    [z[a], z[b]],
                    radius,
                    if full { 256 } else { 128 },
                );
                let refined = reference::contour_residue(
                    stack,
                    sig,
                    spectral,
                    pol,
                    [z[a], z[b]],
                    radius,
                    if full { 512 } else { 256 },
                );
                residue_error = residue_error.max(error(psi[a] * psi[b], direct));
                contour_error = contour_error.max(error(direct, refined));
                if a == 0 && b == 2 {
                    last_direct = direct;
                }
                for (direction, value) in [Complex::new(1., 0.4), Complex::new(-0.3, 1.)]
                    .into_iter()
                    .enumerate()
                {
                    let mut errors = Vec::new();
                    for fraction in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10] {
                        let offset = value * center.norm() * fraction;
                        let g = reference::Green::new(stack, sig, spectral, center + offset, pol);
                        let e = error(offset * g.value(z[a], z[b]), direct);
                        writeln!(convergence, "{name},{pair},{direction},{fraction},{e}")?;
                        errors.push(e);
                    }
                    let best = errors.iter().copied().fold(f64::INFINITY, f64::min);
                    require(
                        best < 1e-5 && best < errors[0] * 0.05,
                        "no residue convergence window",
                    )?;
                }
            }
            require(
                residue_error < 1e-7 && contour_error < 1e-8,
                "independent residue mismatch",
            )?;
            let other = point.normalize_with(
                Scatter2::new(),
                variable,
                AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
                Default::default(),
                recovery,
            )?;
            let f = other.fields(&sampling)?;
            let p = primary(&f);
            let predicted = psi[0] * psi[2];
            let backend_error = error(p[0] * p[2], predicted);
            require(backend_error < 1e-7, "backend residue mismatch")?;
            let mut gauge_error: f64 = 0.;
            for amplitude in [Complex::new(0.7, -0.8), Complex::new(-0.4, 0.9)] {
                let n = point.normalize_with(
                    Transfer2::new(),
                    variable,
                    AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
                    PoleNormalizationSettings {
                        amplitude,
                        ..Default::default()
                    },
                    recovery,
                )?;
                let f = n.fields(&sampling)?;
                let p = primary(&f);
                gauge_error = gauge_error.max(error(p[0] * p[2], predicted));
            }
            require(gauge_error < 1e-8, "complex gauge mismatch")?;
            let mut partition_error: f64 = 0.;
            for offsets in [
                [0.11 * distance, 0.],
                [0., 0.23 * distance],
                [0.31 * distance, 0.47 * distance],
            ] {
                let n = point.normalize_with(
                    Transfer2::new(),
                    variable,
                    AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
                    PoleNormalizationSettings {
                        exterior_offsets_cm: offsets,
                        ..Default::default()
                    },
                    recovery,
                )?;
                partition_error =
                    partition_error.max(error(n.normalization.total, mode.normalization.total));
            }
            require(partition_error < 1e-8, "exterior partition mismatch")?;
            let n = &mode.normalization;
            let [left, right] = sig.open_exteriors().ok_or("an open stack")?;
            writeln!(
                coordinates,
                "{name},{variable:?},{pol:?},{},{},{},{},{},{},{},{}",
                sig.k0.re,
                sig.k0.im,
                sig.k_parallel.re,
                sig.k_parallel.im,
                left.re,
                left.im,
                right.re,
                right.im
            )?;
            for (i, value) in n.layers.iter().enumerate() {
                writeln!(contributions, "{name},layer_{i},{},{}", value.re, value.im)?;
            }
            for (i, value) in n.outgoing_boundaries.iter().enumerate() {
                writeln!(
                    contributions,
                    "{name},boundary_{i},{},{}",
                    value.re, value.im
                )?;
            }
            for (i, value) in n.exterior_segments.iter().enumerate() {
                writeln!(
                    contributions,
                    "{name},segment_{i},{},{}",
                    value.re, value.im
                )?;
            }
            writeln!(contributions, "{name},total,{},{}", n.total.re, n.total.im)?;
            writeln!(contributions, "{name},condition,{},0", n.condition)?;
            if let Some(m) = n.legacy_finite_maxwell {
                writeln!(contributions, "{name},finite_maxwell,{},{}", m.re, m.im)?;
            }

            let boundary = n.outgoing_boundaries.iter().sum::<Complex>();
            let missing = n.total - boundary;
            let omitted_boundary_error = if missing.norm() == 0. {
                None
            } else {
                Some(error(predicted * n.total / missing, last_direct))
            };
            let hermitian_product_error = error(psi[0].conj() * psi[2], last_direct);
            let opposite = if spectral {
                PoleVariable::BetaAtFixedK0
            } else {
                PoleVariable::K0AtFixedBeta
            };
            let wrong = Transfer2::new().pole_normalization(
                stack,
                lamina_core::PoleNormalizationInput {
                    variable: opposite,
                    ..n.input
                },
                Default::default(),
            );
            let wrong_variable_error = match wrong {
                Ok(w) => Some(error(predicted * n.total / w.total, last_direct)),
                Err(_) => None,
            };
            require(
                omitted_boundary_error.is_none_or(|e| e > 1e-4),
                "omitted scalar boundary control failed",
            )?;
            require(
                hermitian_product_error > 1e-4,
                "Hermitian product control failed",
            )?;
            require(
                wrong_variable_error.is_none_or(|e| e > 1e-4),
                "wrong-variable control failed",
            )?;
            let mut omitted_dispersion_error = None;
            if name.starts_with("dispersive") || name == "interface_temporal" {
                let frozen =
                    |m: &lamina_core::AnalyticalMaterialHandle<Complex>| material::Frozen {
                        epsilon: m.evaluate_relative_permittivity_complex(Scalar(sig.k0)),
                        mu: m.evaluate_relative_permeability_complex(Scalar(sig.k0)),
                    };
                let mut builder = Stack::from_analytical_materials(
                    frozen(stack.left_exterior()),
                    frozen(stack.right_exterior()),
                );
                for l in stack.layers_left_to_right() {
                    builder = builder.analytical_layer(frozen(l.material()), l.thickness());
                }
                let frozen = builder.finalise();
                let wrong =
                    Transfer2::new().pole_normalization(&frozen, n.input, Default::default())?;
                let e = error(predicted * n.total / wrong.total, last_direct);
                require(e > 1e-4, "omitted dispersion control failed")?;
                omitted_dispersion_error = Some(e);
            }
            let omitted_boundary_error = omitted_boundary_error
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unavailable".into());
            let wrong_variable_error = wrong_variable_error
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unavailable".into());
            writeln!(
                rows,
                "{name},complete,{residue_error},{contour_error},{backend_error},{gauge_error},{partition_error},{omitted_boundary_error},{hermitian_product_error},{wrong_variable_error},{}",
                omitted_dispersion_error
                    .map(|v| v.to_string())
                    .unwrap_or_default()
            )?;
            Ok(())
        })();
        if let Err(e) = attempt {
            writeln!(rows, "{name},failed,,,,,,,,,,")?;
            failures.push(format!("{name}: {e}"));
        }
    }
    std::fs::write(out.join("failures.txt"), failures.join("\n"))?;
    if !failures.is_empty() {
        return Err(failures.join("; ").into());
    }
    Ok(())
}

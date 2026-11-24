//! Run a finite-window QNM thickness sweep and write a standalone report.
use lamina_core::{
    AnalyticalMaterialStack, CanonicalCoordinates, ComplexPlaneEvaluator, Constant,
    ExteriorWavevectors, IncidentSide, ModeJet1, Polarisation, Scatter2, SeedJet, Stack,
};
use lamina_mode::*;
use lamina_units::Length;
use ndarray::{Ix0, arr0};
use std::{fmt::Write, hint::black_box, path::PathBuf, time::Instant};

#[path = "support/pole_material.rs"]
mod material;

type Matrix = [[Complex; 2]; 2];
type J = ModeJet1<Complex, Ix0>;
const POINTS: usize = 65;
const DESIGNS: usize = 41;
const AMPLITUDE_LIMIT: f64 = 1e-4;
const PHASE_LIMIT: f64 = 3e-3;

struct Input {
    output: PathBuf,
    window: [f64; 2],
    depth: f64,
    exteriors: [f64; 2],
    /// Relative permittivity and width of each layer.
    layers: Vec<(Complex, f64)>,
    step: f64,
    span: f64,
}

impl Default for Input {
    fn default() -> Self {
        Self {
            output: "qnm-design-sweep.html".into(),
            window: [3.2, 11.3],
            depth: 4.2,
            exteriors: [1.0, 1.44],
            layers: vec![(Complex::new(4.0, 0.12), 0.30), (Complex::from(2.3), 0.07)],
            step: 2e-4,
            span: 3e-4,
        }
    }
}

impl Input {
    fn parse() -> Result<Option<Self>, String> {
        let mut input = Self::default();
        let numbers = |s: &str| {
            s.split(',')
                .map(|v| v.parse::<f64>().map_err(|_| format!("invalid number {v}")))
                .collect::<Result<Vec<_>, _>>()
        };
        let pair = |s: &str| {
            numbers(s)?
                .try_into()
                .map_err(|_| "expected two comma-separated numbers".to_owned())
        };
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            if flag == "--help" {
                println!(
                    "qnm_design_sweep [--output FILE] [--window LO,HI] [--depth D] [--exteriors EPS_L,EPS_R] [--layers 'EPS_RE,EPS_IM,WIDTH;...'] [--step H] [--span H]"
                );
                return Ok(None);
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing argument for {flag}"))?;
            match flag.as_str() {
                "--output" => input.output = value.into(),
                "--window" => input.window = pair(&value)?,
                "--exteriors" => input.exteriors = pair(&value)?,
                "--depth" => input.depth = value.parse().map_err(|_| "invalid depth")?,
                "--step" => input.step = value.parse().map_err(|_| "invalid step")?,
                "--span" => input.span = value.parse().map_err(|_| "invalid span")?,
                "--layers" => {
                    input.layers = value
                        .split(';')
                        .map(|entry| {
                            let [re, im, width] = numbers(entry)?
                                .try_into()
                                .map_err(|_| "expected three numbers per layer")?;
                            Ok((Complex::new(re, im), width))
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                }
                _ => return Err(format!("unknown flag {flag}")),
            }
        }
        let [lo, hi] = input.window;
        if !(lo.is_finite() && hi.is_finite() && 0.0 < lo && lo < hi)
            || [input.depth, input.step, input.span]
                .into_iter()
                .chain(input.exteriors)
                .any(|x| !x.is_finite() || x <= 0.0)
            || input.layers.is_empty()
            || input.layers.iter().any(|(e, d)| {
                !e.re.is_finite()
                    || !e.im.is_finite()
                    || e.re <= 0.0
                    || e.im < 0.0
                    || !d.is_finite()
                    || *d <= 0.0
            })
            || input.layers[0].1 <= input.step.max(8.0 * input.span)
        {
            return Err("expected a positive spectral window, positive passive layers, and a first layer thicker than the sweep excursions".to_owned());
        }
        Ok(Some(input))
    }

    fn stack(&self, offset: f64) -> AnalyticalMaterialStack<Complex> {
        let mut stack = Stack::from_analytical_materials(
            Constant::dielectric(self.exteriors[0]),
            Constant::dielectric(self.exteriors[1]),
        );
        for (i, &(epsilon, width)) in self.layers.iter().enumerate() {
            let width = if i == 0 { width + offset } else { width };
            stack = stack.analytical_layer(
                material::Frozen {
                    epsilon,
                    mu: Complex::ONE,
                },
                Length::centimetres(width),
            );
        }
        stack.finalise()
    }
}

fn direct(input: &Input, k: &[f64], offset: f64) -> Result<Vec<Matrix>, String> {
    let stack = input.stack(offset);
    let solver = ComplexPlaneEvaluator::<J, _, _>::compile(&stack, Scatter2::new())
        .map_err(|e| e.to_string())?;
    let n = input.exteriors.map(f64::sqrt);
    let seed = |value| <J as SeedJet>::constant(arr0(value));
    k.iter()
        .map(|&k| {
            let result = solver
                .solve(
                    &CanonicalCoordinates::new(seed(k.into()), seed(Complex::ZERO)),
                    &ExteriorWavevectors::new(seed((k * n[0]).into()), seed((k * n[1]).into())),
                    Polarisation::TransverseElectric,
                )
                .map_err(|e| format!("{e:?}"))?;
            let l = result.amplitudes(IncidentSide::Left);
            let r = result.amplitudes(IncidentSide::Right);
            let matrix = [
                [l.reflection().value()[()], r.transmission().value()[()]],
                [l.transmission().value()[()], r.reflection().value()[()]],
            ];
            if matrix
                .iter()
                .flatten()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
            {
                return Err("nonfinite direct scattering amplitude".into());
            }
            Ok(matrix)
        })
        .collect()
}

struct Comparison {
    amplitude: f64,
    phase: f64,
    passivity_excess: f64,
    direct_seconds: f64,
    model_seconds: f64,
}

impl Comparison {
    fn passes(&self) -> bool {
        self.amplitude <= AMPLITUDE_LIMIT && self.phase <= PHASE_LIMIT
    }
}

fn compare(
    input: &Input,
    grid: &[f64],
    base: &[Matrix],
    rates: &[Matrix],
    span: f64,
) -> Result<Comparison, String> {
    let offsets: Vec<_> = (0..DESIGNS)
        .map(|i| span * (2.0 * i as f64 / (DESIGNS - 1) as f64 - 1.0))
        .collect();
    let start = Instant::now();
    let direct = offsets
        .iter()
        .map(|&p| direct(input, grid, p))
        .collect::<Result<Vec<_>, _>>()?;
    let direct_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let affine: Vec<Vec<Matrix>> = offsets
        .iter()
        .map(|&p| {
            base.iter()
                .zip(rates)
                .map(|(s, ds)| {
                    std::array::from_fn(|i| std::array::from_fn(|j| s[i][j] + p * ds[i][j]))
                })
                .collect()
        })
        .collect();
    let model_seconds = start.elapsed().as_secs_f64();
    let mut amplitude = 0.0_f64;
    let mut phase = 0.0_f64;
    let mut passivity_excess = 0.0_f64;
    let admittance = input.exteriors.map(f64::sqrt);
    for (target, prediction) in black_box(&direct)
        .iter()
        .flatten()
        .zip(black_box(&affine).iter().flatten())
    {
        let scale = (admittance[1] / admittance[0]).sqrt();
        let [[a, b], [c, d]] = *prediction;
        let (b, c) = (b / scale, c * scale);
        let a0 = a.norm_sqr() + c.norm_sqr();
        let a1 = b.norm_sqr() + d.norm_sqr();
        let off_diagonal = a.conj() * b + c.conj() * d;
        let gain = 0.5 * (a0 + a1 + (a0 - a1).hypot(2.0 * off_diagonal.norm()));
        passivity_excess = passivity_excess.max((gain - 1.0).max(0.0));
        for (&a, &b) in target.iter().flatten().zip(prediction.iter().flatten()) {
            amplitude = amplitude.max((a - b).norm());
            if a.norm() >= 1e-3 && b.norm() >= 1e-3 {
                phase = phase.max((b / a).arg().abs());
            }
        }
    }
    Ok(Comparison {
        amplitude,
        phase,
        passivity_excess,
        direct_seconds,
        model_seconds,
    })
}

fn main() -> Result<(), String> {
    let Some(input) = Input::parse()? else {
        return Ok(());
    };
    let begin = Instant::now();
    let [lo, hi] = input.window;
    let centre = lo + (hi - lo) / 2.0;
    let request = QnmPoleRequest {
        beta_per_cm: 0.0,
        polarisation: Polarisation::TransverseElectric,
        domain: FrequencyDomainInput {
            bounds: FrequencyRegion::new(lo, hi, -input.depth, 0.35).map_err(|e| e.to_string())?,
            reference: FrequencyReference::RealRadiation {
                frequency: centre,
                choices: [RadiationChoice::OutgoingOrDecaying; 2],
                tolerance: Default::default(),
            },
            continuation: Default::default(),
            exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
            exclusions: vec![],
        },
        assumption: AnalyticAssumption::CallerAssumedHolomorphicPoleFree,
        discovery: Default::default(),
        refinement: Default::default(),
        matching: Default::default(),
        normalization: Default::default(),
        recovery: Default::default(),
    };
    let source = select_qnm_poles(input.stack(0.0), request).map_err(|e| format!("{e:?}"))?;
    if source.coverage != QnmPoleCoverage::Reconciled || source.poles.is_empty() {
        return Err(format!(
            "array of usable poles is incomplete: coverage {:?}, winding {:?}, accepted {}, issues {:?}",
            source.coverage,
            source.winding_count,
            source.poles.len(),
            source.issues
        ));
    }
    let discovery_seconds = begin.elapsed().as_secs_f64();
    let derivative = differentiate_qnm_design(
        &source,
        |p| input.stack(p),
        QnmDesignSettings {
            step: input.step,
            match_radius_per_cm: 0.2,
            channels: QnmConstantChannels {
                epsilon: input.exteriors,
                mu: [1.0; 2],
            },
            window: QnmWindowSettings {
                k0_per_cm: input.window,
                regular_degree: 18,
                validation_points: POINTS,
                amplitude_atol: 2e-8,
            },
        },
    )
    .map_err(|e| format!("{e:?}"))?;
    let grid: Vec<_> = (0..POINTS)
        .map(|i| {
            if i == POINTS - 1 {
                hi
            } else {
                lo + (hi - lo) * i as f64 / (POINTS - 1) as f64
            }
        })
        .collect();
    let base = grid
        .iter()
        .map(|&k| {
            derivative
                .base()
                .amplitudes(k)
                .ok_or("base spectrum evaluation failed")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rates = grid
        .iter()
        .map(|&k| {
            derivative
                .amplitudes_rate(k)
                .map(|r| r.total)
                .ok_or("scattering rate evaluation failed")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let setup_seconds = begin.elapsed().as_secs_f64();
    let near = compare(&input, &grid, &base, &rates, input.span)?;
    let wider = compare(&input, &grid, &base, &rates, 8.0 * input.span)?;
    let construction = derivative.base().evidence.direct_construction_samples
        + derivative
            .stations
            .iter()
            .map(|s| s.construction_samples)
            .sum::<usize>();
    let validation = derivative.base().evidence.held_out_samples
        + derivative
            .stations
            .iter()
            .map(|s| s.held_out_samples)
            .sum::<usize>();
    let refreshes = derivative
        .stations
        .iter()
        .filter(|s| s.coverage != QnmPoleCoverage::TrackedOnly)
        .count();
    let mut html = String::from(
        r#"<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QNM design sweep</title>
<style>body{margin:auto;max-width:68rem;padding:2rem;color:#172d37;background:#f5f8f7;font:1rem/1.55 system-ui,sans-serif}h1{font-size:clamp(2rem,5vw,3.4rem);line-height:1.06}section{background:white;padding:1rem 1.4rem;margin-block:1rem;border:1px solid #d2dfdc;border-radius:.7rem;overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;white-space:nowrap}th,td{text-align:right;padding:.45rem;border-bottom:1px solid #d2dfdc}th:first-child,td:first-child{text-align:left}code{background:#e7efed}</style>
<main><h1>QNM design sweep</h1><p>Normal-incidence TE scattering. The design coordinate is the thickness of the first finite layer.</p>"#,
    );
    writeln!(html,"<p>Real window {lo:.6}–{hi:.6} cm⁻¹. Exterior permittivities {:.6} and {:.6}. Centered sensitivity step {:.3e} cm.</p>",input.exteriors[0],input.exteriors[1],input.step).unwrap();
    writeln!(html,"<section><h2>Layers and poles</h2><p>Numerical pole coverage {:?}. Winding count {:?}.</p><table><thead><tr><th>Layer</th><th>ε</th><th>Width (cm)</th></tr></thead><tbody>",source.coverage,source.winding_count).unwrap();
    for (i, (e, w)) in input.layers.iter().enumerate() {
        writeln!(html, "<tr><td>{i}</td><td>{e:.6}</td><td>{w:.6}</td></tr>").unwrap();
    }
    html.push_str("</tbody></table><table><thead><tr><th>Pole k₀ (cm⁻¹)</th><th>dk₀/dd</th><th>d wavelength/dd</th><th>d linewidth/dd</th></tr></thead><tbody>");
    for (pole, rate) in source.poles.iter().zip(&derivative.pole_rates) {
        writeln!(
            html,
            "<tr><td>{:.9}</td><td>{:.9}</td><td>{:.7}</td><td>{:.7}</td></tr>",
            pole.signature.k0, rate.pole_per_cm, rate.wavelength_cm, rate.linewidth_per_cm
        )
        .unwrap();
    }
    html.push_str("</tbody></table></section>");
    writeln!(html,"<section><h2>Direct sweep comparison</h2><p>Each sweep covers {DESIGNS} designs at {POINTS} frequencies. Tolerances are {AMPLITUDE_LIMIT:.0e} in complex amplitude and {PHASE_LIMIT:.0e} rad in phase. Phase uses a magnitude floor of 10⁻³.</p><p>Setup took {setup_seconds:.4} s, including {discovery_seconds:.4} s for base pole selection. One base search plus {refreshes} discovery refreshes. {construction} direct calibration samples and {validation} direct validation samples.</p>").unwrap();
    html.push_str("<table><thead><tr><th>Thickness half-span (cm)</th><th>Amplitude error</th><th>Phase error (rad)</th><th>Passivity excess</th><th>Direct time (s)</th><th>Cached model time (s)</th><th>Tolerance</th></tr></thead><tbody>");
    for (span, result) in [(input.span, &near), (8.0 * input.span, &wider)] {
        writeln!(html,"<tr><td>{span:.3e}</td><td>{:.3e}</td><td>{:.3e}</td><td>{:.3e}</td><td>{:.6}</td><td>{:.6}</td><td>{}</td></tr>",result.amplitude,result.phase,result.passivity_excess,result.direct_seconds,result.model_seconds,if result.passes() {"Passed"} else {"Exceeded"}).unwrap();
    }
    html.push_str("</tbody></table>");
    writeln!(html, "<p>Base held-out amplitude error {:.3e}. The base maximum passivity excess was {:.3e}. The sweep passivity column checks the largest excess of the flux-normalized S†S eigenvalues above one.</p>",
        derivative.base().evidence.held_out_max_amplitude_error, derivative.base().evidence.held_out_max_passivity_excess).unwrap();
    if near.passes() && near.direct_seconds > near.model_seconds {
        let break_even =
            setup_seconds * DESIGNS as f64 / (near.direct_seconds - near.model_seconds);
        writeln!(html,"<p>Estimated break-even on this machine is about {:.0} design spectra at the same grid and within the tested near-base range. This estimate assumes reuse of the cached model and excludes refreshes beyond this range.</p>",break_even.ceil()).unwrap();
    }
    html.push_str("</section><section><h2>Base spectrum and thickness derivatives</h2><p>Left incidence. R, T and A are power fractions. The pole residues and the fitted regular part both contribute to the complex rates.</p><table><thead><tr><th>k₀ (cm⁻¹)</th><th>r</th><th>t</th><th>R</th><th>T</th><th>A</th><th>dr/dd</th><th>dt/dd</th></tr></thead><tbody>");
    for ((&k, s), ds) in grid.iter().zip(&base).zip(&rates) {
        let power = derivative
            .base()
            .power_ratios(k)
            .ok_or("power ratio failed")?;
        let r = power[0][0];
        let t = power[1][0];
        writeln!(html,"<tr><td>{k:.6}</td><td>{:.7}</td><td>{:.7}</td><td>{r:.7}</td><td>{t:.7}</td><td>{:.7}</td><td>{:.7}</td><td>{:.7}</td></tr>",s[0][0],s[1][0],1.0-r-t,ds[0][0],ds[1][0]).unwrap();
    }
    html.push_str("</tbody></table></section></main></html>");
    std::fs::write(&input.output, html).map_err(|e| e.to_string())?;
    println!(
        "{} · poles={} · near={} ({:.3e}, {:.3e}) · wider={} ({:.3e}, {:.3e})",
        input.output.display(),
        source.poles.len(),
        near.passes(),
        near.amplitude,
        near.phase,
        wider.passes(),
        wider.amplitude,
        wider.phase
    );
    Ok(())
}

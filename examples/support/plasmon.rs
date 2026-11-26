//! Repeated public point searches for a two-half-space TM surface plasmon.
//! The analytical answer locates small benchmark charts, but is not returned as
//! a numerical root: every marker is an accepted APM discovery/refinement result.
use lamina_core::{AnalyticalMaterialStack, Constant, Polarisation, Stack, Transfer2};
use lamina_materials::Drude;
use lamina_mode::*;
use lamina_units::VacuumAngularWavenumber as K;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};
pub const KP: f64 = 1e5;
pub const ETA: f64 = 0.02;
const ASSUME: AnalyticAssumption = AnalyticAssumption::CallerAssumedHolomorphicPoleFree;
fn c(x: f64) -> Complex {
    Complex::new(x, 0.)
}
pub fn metal(u: Complex) -> Complex {
    c(1.) - c(1.) / (u * (u + Complex::new(0., ETA)))
}
fn stack() -> AnalyticalMaterialStack<Complex> {
    let m = Drude::new(
        1.,
        K::inverse_centimetres(KP),
        K::inverse_centimetres(ETA * KP),
    )
    .unwrap();
    Stack::from_analytical_materials(Constant::vacuum(), m).finalise()
}
fn decay(q: Complex) -> Complex {
    let s = q.sqrt();
    if s.im < 0. { -s } else { s }
}
pub fn waves(u: Complex, v: Complex) -> [Complex; 2] {
    [
        decay(u * u - v * v) * KP,
        decay(metal(u) * u * u - v * v) * KP,
    ]
}
fn z_at(u: Complex, v: Complex) -> Complex {
    let q = waves(u, v);
    ((q[0] + q[1]) / (Complex::i() * 2. * u * KP)).ln()
}
pub fn beta_exact(u: f64) -> Complex {
    let e = metal(c(u));
    c(u) * (e / (c(1.) + e)).sqrt()
}
pub fn omega_exact(v: f64) -> Complex {
    let mut u = Complex::new(
        (2. * v * v / (1. + 2. * v * v + (1. + 4. * v.powi(4)).sqrt())).sqrt(),
        -ETA / 4.,
    );
    for _ in 0..30 {
        let f = u.powi(4) + Complex::i() * ETA * u.powi(3)
            - (1. + 2. * v * v) * u * u
            - Complex::i() * 2. * ETA * v * v * u
            + c(v * v);
        let df = 4. * u.powi(3) + Complex::i() * 3. * ETA * u * u
            - 2. * (1. + 2. * v * v) * u
            - Complex::i() * 2. * ETA * v * v;
        u -= f / df;
    }
    u
}
fn refinement(frequency: bool) -> RefinementConfig {
    RefinementConfig {
        residual_atol: Some(1e-8),
        residual_rtol: 1e-11,
        residual_scale: KP,
        native_atol: if frequency { 1e-5 } else { 1e-10 },
        native_rtol: 1e-11,
        ..Default::default()
    }
}
fn solve(plan: Plan, frequency: bool) -> Result<SolveReport, Box<dyn std::error::Error>> {
    let r = plan.solve(
        DiscoveryConfig::default(),
        refinement(frequency),
        MatchConfig::default(),
    )?;
    if r.observations().len() != 1 {
        return Err(format!(
            "expected one accepted root, got {}: {r:?}",
            r.observations().len()
        )
        .into());
    }
    let d = r
        .discovery()
        .entries()
        .next()
        .unwrap()
        .map_err(Clone::clone)?;
    if !matches!(d.termination(), Termination::Converged) {
        eprintln!("Discovery termination: {:?}", d.termination());
    }
    Ok(r)
}
fn freq_input(v: f64, u: Complex) -> FrequencyDomainInput {
    // Re upper bound stays below dielectric q=0 at u=v. Drude poles at
    // u=0,-i*ETA and the metallic q zeros are outside these local rectangles.
    let h = (0.24 * (v - u.re)).min(0.012);
    let hi = h.min(0.004);
    let anchor = u + Complex::new(0.25 * h, 0.2 * hi);
    FrequencyDomainInput {
        bounds: FrequencyRegion::new(
            (u.re - h) * KP,
            (u.re + h) * KP,
            (u.im - hi) * KP,
            (u.im + hi) * KP,
        )
        .unwrap(),
        reference: FrequencyReference::Explicit(ExteriorReferenceValues {
            frequency: anchor * KP,
            kappa: waves(anchor, c(v)),
        }),
        continuation: ContinuationConfig::default(),
        exterior_assumption: ExteriorAnalyticAssumption::CallerAssumedHolomorphicNonzero,
        exclusions: vec![],
    }
}
pub type Error = Box<dyn std::error::Error>;
/// Dimensionless physical tolerances, independent of native refinement stopping.
pub const ROOT_TOL: f64 = 1e-8;
pub const BOUNDARY_TOL: f64 = 1e-8;
pub fn physical_checks(
    case: &str,
    root: Complex,
    exact: Complex,
    k: [Complex; 2],
    boundary: f64,
    backend: f64,
) -> Result<(), Error> {
    if ![
        root.re, root.im, exact.re, exact.im, k[0].re, k[0].im, k[1].re, k[1].im, boundary, backend,
    ]
    .iter()
    .all(|x| x.is_finite())
    {
        return Err("nonfinite physical evidence".into());
    }
    if k.iter().any(|v| v.im <= 0.) {
        return Err("exterior decay sign".into());
    }
    if (case == "beta" && root.im <= 0.) || (case == "omega" && root.im >= 0.) {
        return Err("passive attenuation sign".into());
    }
    if (root - exact).norm() / exact.norm() > ROOT_TOL {
        return Err("relative root error".into());
    }
    if boundary > BOUNDARY_TOL {
        return Err("unsquared TM boundary mismatch".into());
    }
    if backend > BOUNDARY_TOL {
        return Err("backend disagreement".into());
    }
    Ok(())
}
/// New output directories prevent accidental reuse of earlier successful data.
pub fn run(out: &Path, full: bool) -> Result<(), Error> {
    std::fs::create_dir(out)?;
    let result = run_inner(out, full);
    let status = if result.is_ok() { "complete" } else { "failed" };
    let failure = result
        .as_ref()
        .err()
        .map(|e| format!("{e:?}"))
        .unwrap_or_default();
    // Debug string escaping is JSON-compatible for these ASCII diagnostic strings.
    std::fs::write(
        out.join("summary.json"),
        format!(
            "{{\"schema\":1,\"status\":{status:?},\"suite\":{:?},\"samples_per_case\":{},\"map_shape\":[{},{}],\"kp_inverse_cm\":{KP},\"eta\":{ETA},\"root_tolerance\":{ROOT_TOL},\"boundary_tolerance\":{BOUNDARY_TOL},\"failure\":{failure:?},\"scope\":\"independent locally guided searches; not global coverage or QNM normalization\"}}\n",
            if full { "full" } else { "quick" },
            if full { 33 } else { 3 },
            if full { 61 } else { 0 },
            if full { 81 } else { 0 }
        ),
    )?;
    result
}
fn run_inner(out: &Path, full: bool) -> Result<(), Error> {
    std::fs::write(
        out.join("settings.txt"),
        format!(
            "Model: vacuum | Drude epsilon_inf=1, kp={KP} cm^-1, eta={ETA}\nAssumptions: {ASSUME:?}; explicit decaying complex anchors; exterior caller-assumed holomorphic nonzero; no exclusions.\nDiscovery: {:#?}\nSpatial refinement: {:#?}\nTemporal refinement: {:#?}\nMatching: {:#?}\nChart config: {:#?}\nContinuation: {:#?}\n",
            DiscoveryConfig::default(),
            refinement(false),
            refinement(true),
            MatchConfig::default(),
            ConformalChartConfig::<f64>::default(),
            ContinuationConfig::default()
        ),
    )?;
    // Numeric controls are also machine-readable without a runtime serialization dependency.
    struct Number<T>(T);
    impl std::fmt::Display for Number<f64> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::fmt::Display for Number<usize> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::fmt::Display for Number<u32> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::fmt::Display for Number<Option<f64>> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self.0 {
                Some(value) => write!(f, "{value}"),
                None => f.write_str("null"),
            }
        }
    }
    macro_rules! numeric_json {
        ($value:expr, $($field:ident),+ $(,)?) => {{
            let value = $value;
            format!("{{{}}}", vec![$(format!("\"{}\":{}", stringify!($field), Number(value.$field))),+].join(","))
        }};
    }
    let discovery = numeric_json!(
        DiscoveryConfig::default(),
        score_tolerance,
        zero_tolerance,
        residual_tolerance,
        boundary_shift_fraction,
        relative_tolerance,
        absolute_tolerance,
        max_iterations,
        max_depth,
        max_leaves,
        max_function_evaluations
    );
    let spatial = numeric_json!(
        refinement(false),
        residual_atol,
        residual_rtol,
        residual_scale,
        native_atol,
        native_rtol,
        max_updates,
        max_backtracks,
        max_evaluations
    );
    let temporal = numeric_json!(
        refinement(true),
        residual_atol,
        residual_rtol,
        residual_scale,
        native_atol,
        native_rtol,
        max_updates,
        max_backtracks,
        max_evaluations
    );
    let chart = numeric_json!(
        ConformalChartConfig::<f64>::default(),
        integration_tolerance,
        max_depth
    );
    let continuation = numeric_json!(
        ContinuationConfig::default(),
        integral_tolerance,
        consistency_tolerance,
        cutoff,
        max_depth,
        max_evaluations
    );
    let matching = MatchConfig::default();
    let native = numeric_json!(matching.native, absolute, relative);
    let physical = numeric_json!(matching.physical, absolute, relative);
    let exterior = numeric_json!(matching.exterior, absolute, relative);
    std::fs::write(
        out.join("settings.json"),
        format!(
            "{{\"discovery\":{discovery},\"spatial_refinement\":{spatial},\"temporal_refinement\":{temporal},\"chart\":{chart},\"continuation\":{continuation},\"matching\":{{\"native\":{native},\"physical\":{physical},\"exterior\":{exterior}}}}}"
        ),
    )?;
    let mut failures = Vec::new();
    let mut file = BufWriter::new(File::create(out.join("roots.csv"))?);
    writeln!(
        file,
        "case,parameter,re,im,exact_re,exact_im,residual,dispersion_error,kappa_left_re,kappa_left_im,kappa_right_re,kappa_right_im,backend_relative_error,te_checked,te_accepted,classification,radiation"
    )?;
    for i in 0..if full { 33 } else { 3 } {
        let u = 0.15 + (0.695 - 0.15) * (i as f64) / if full { 32. } else { 2. };
        let outcome = (|| -> Result<(), Error> {
            let target = beta_exact(u);
            let reference = target + Complex::new(0.004, 0.002);
            let z = z_at(c(u), target);
            let zr = z_at(c(u), reference);
            let bounds =
                ConformalRegion::new(z.re - 0.045, z.re + 0.045, z.im - 0.045, z.im + 0.045)?;
            let selection = ReferenceSelection::Coordinate {
                reference: zr,
                coordinate: reference * KP,
            };
            let p = PropagationProblem::new(stack(), u * KP, Polarisation::TransverseMagnetic)?;
            let d = p.domain(
                bounds,
                selection,
                ConformalChartConfig::<f64>::default(),
                ASSUME,
            )?;
            let report = solve(p.plan(vec![d.clone()], vec![])?, false)?;
            std::fs::write(
                out.join(format!("evidence_beta_{i}.txt")),
                format!("{report:#?}"),
            )?;
            let point = &report.observations()[0];
            let v = point.physical_coordinate() / KP;
            let detail = point.describe(InterpretationConfig::default())?;
            let k = detail.data.kappa.map(|k| k.expect("an open stack"));
            let boundary = (k[0] + k[1] / metal(c(u))) / KP;
            let t = PropagationProblem::with_backend(
                stack(),
                u * KP,
                Polarisation::TransverseMagnetic,
                Transfer2::new(),
            )?;
            let td = t.domain(
                bounds,
                selection,
                ConformalChartConfig::<f64>::default(),
                ASSUME,
            )?;
            let s = p.evaluate(&d, point.native_coordinate())?;
            let tv = t.evaluate(&td, point.native_coordinate())?;
            let off = point.native_coordinate() + Complex::new(0.001, 0.);
            let so = p.evaluate(&d, off)?;
            let to = t.evaluate(&td, off)?;
            let backend = ((s.value - tv.value).norm() / KP).max((so.value - to.value).norm() / KP);

            let mut te = 0;
            let te_checked = i % 8 == 0;
            if te_checked {
                let p = PropagationProblem::new(stack(), u * KP, Polarisation::TransverseElectric)?;
                let d = p.domain(
                    bounds,
                    selection,
                    ConformalChartConfig::<f64>::default(),
                    ASSUME,
                )?;
                te = p
                    .plan(vec![d], vec![])?
                    .solve(
                        DiscoveryConfig::default(),
                        refinement(false),
                        MatchConfig::default(),
                    )?
                    .observations()
                    .len();
            }
            physical_checks("beta", v, target, k, boundary.norm(), backend)?;
            if te != 0 {
                return Err(format!("beta parameter {}: TE control accepted {te} roots", u).into());
            }
            if !matches!(
                detail.interpretation.classification,
                Classification::Unsupported(_)
            ) {
                return Err("metal classification unexpectedly supported".into());
            }
            writeln!(
                file,
                "beta,{u},{},{},{},{},{},{},{},{},{},{},{},{},{},\"{:?}\",\"{:?}\"",
                v.re,
                v.im,
                target.re,
                target.im,
                point.evidence().value.norm() / KP,
                boundary.norm(),
                k[0].re / KP,
                k[0].im / KP,
                k[1].re / KP,
                k[1].im / KP,
                backend,
                te_checked,
                if te_checked {
                    te.to_string()
                } else {
                    String::new()
                },
                detail.interpretation.classification,
                detail.interpretation.radiation
            )?;
            eprintln!("beta {} u={u:.4} error={:.2e}", i + 1, (v - target).norm());
            Ok(())
        })();
        if let Err(error) = outcome {
            failures.push(format!("beta index={i} parameter={u}: {error}"));
        }
    }
    for i in 0..if full { 33 } else { 3 } {
        let v = 0.2 + (3.0 - 0.2) * (i as f64) / if full { 32. } else { 2. };
        let outcome = (|| -> Result<(), Error> {
            let target = omega_exact(v);
            let p = FrequencyProblem::new(stack(), v * KP, Polarisation::TransverseMagnetic)?;
            let input = freq_input(v, target);
            let d = p.domain(input.clone(), ASSUME)?;
            let report = solve(p.plan(vec![d.clone()], vec![])?, true)?;
            std::fs::write(
                out.join(format!("evidence_omega_{i}.txt")),
                format!("{report:#?}"),
            )?;
            let point = &report.observations()[0];
            let u = point.physical_coordinate() / KP;
            let detail = point.describe(InterpretationConfig::default())?;
            let k = detail.data.kappa.map(|k| k.expect("an open stack"));
            let boundary = (k[0] + k[1] / metal(u)) / KP;
            let t = FrequencyProblem::with_backend(
                stack(),
                v * KP,
                Polarisation::TransverseMagnetic,
                Transfer2::new(),
            )?;
            let td = t.domain(input.clone(), ASSUME)?;
            let s = p.evaluate(&d, point.native_coordinate())?;
            let tv = t.evaluate(&td, point.native_coordinate())?;
            let off = point.native_coordinate() + Complex::new(0.1, 0.);
            let so = p.evaluate(&d, off)?;
            let to = t.evaluate(&td, off)?;
            let backend = ((s.value - tv.value).norm() / KP).max((so.value - to.value).norm() / KP);
            let mut te = 0;
            let te_checked = i % 8 == 0;
            if te_checked {
                let p = FrequencyProblem::new(stack(), v * KP, Polarisation::TransverseElectric)?;
                let d = p.domain(input, ASSUME)?;
                te = p
                    .plan(vec![d], vec![])?
                    .solve(
                        DiscoveryConfig::default(),
                        refinement(true),
                        MatchConfig::default(),
                    )?
                    .observations()
                    .len();
            }
            physical_checks("omega", u, target, k, boundary.norm(), backend)?;
            if te != 0 {
                return Err(
                    format!("omega parameter {}: TE control accepted {te} roots", v).into(),
                );
            }
            if !matches!(
                detail.interpretation.classification,
                Classification::Unsupported(_)
            ) {
                return Err("metal classification unexpectedly supported".into());
            }
            writeln!(
                file,
                "omega,{v},{},{},{},{},{},{},{},{},{},{},{},{},{},\"{:?}\",\"{:?}\"",
                u.re,
                u.im,
                target.re,
                target.im,
                point.evidence().value.norm() / KP,
                boundary.norm(),
                k[0].re / KP,
                k[0].im / KP,
                k[1].re / KP,
                k[1].im / KP,
                backend,
                te_checked,
                if te_checked {
                    te.to_string()
                } else {
                    String::new()
                },
                detail.interpretation.classification,
                detail.interpretation.radiation
            )?;
            eprintln!("omega {} v={v:.4} error={:.2e}", i + 1, (u - target).norm());
            Ok(())
        })();
        if let Err(error) = outcome {
            failures.push(format!("omega index={i} parameter={v}: {error}"));
        }
    }
    file.flush()?;
    std::fs::write(out.join("failures.txt"), failures.join("\n"))?;
    if !failures.is_empty() {
        return Err(failures.join("; ").into());
    }
    if !full {
        return Ok(());
    }
    // Characteristic landscapes from public checked evaluations. Complex-beta
    // grid is parameterised by physical beta, with z obtained from the same
    // decay branches; no principal-root sheet jumps occur in this local window.
    let u = 0.6;
    let center = beta_exact(u);
    let p = PropagationProblem::new(stack(), u * KP, Polarisation::TransverseMagnetic)?;
    let z = z_at(c(u), center);
    let anchor = center + Complex::new(0.003, 0.002);
    let zr = z_at(c(u), anchor);
    let d = p.domain(
        ConformalRegion::new(z.re - 0.3, z.re + 0.3, z.im - 0.3, z.im + 0.3)?,
        ReferenceSelection::Coordinate {
            reference: zr,
            coordinate: anchor * KP,
        },
        ConformalChartConfig::<f64>::default(),
        ASSUME,
    )?;
    let mut map_roots = BufWriter::new(File::create(out.join("map_roots.csv"))?);
    writeln!(map_roots, "case,re,im,exact_re,exact_im")?;
    let result = solve(p.plan(vec![d.clone()], vec![])?, false)?;
    let point = result.observations()[0].physical_coordinate() / KP;
    writeln!(
        map_roots,
        "beta,{},{},{},{}",
        point.re, point.im, center.re, center.im
    )?;
    let mut landscape = BufWriter::new(File::create(out.join("characteristics.csv"))?);
    writeln!(landscape, "case,x,y,real,imag")?;
    for j in 0..61 {
        for i in 0..81 {
            let v = Complex::new(
                center.re - 0.07 + 0.14 * i as f64 / 80.,
                center.im - 0.03 + 0.06 * j as f64 / 60.,
            );
            let value = p.evaluate(&d, z_at(c(u), v))?.value / KP;
            writeln!(
                landscape,
                "beta,{},{},{},{}",
                v.re, v.im, value.re, value.im
            )?;
        }
    }
    let v = 1.0;
    let center = omega_exact(v);
    let p = FrequencyProblem::new(stack(), v * KP, Polarisation::TransverseMagnetic)?;
    let mut input = freq_input(v, center);
    input.bounds = FrequencyRegion::new(
        (center.re - 0.025) * KP,
        (center.re + 0.025) * KP,
        (center.im - 0.012) * KP,
        (center.im + 0.012) * KP,
    )?;
    let d = p.domain(input, ASSUME)?;
    let result = solve(p.plan(vec![d.clone()], vec![])?, true)?;
    let point = result.observations()[0].physical_coordinate() / KP;
    writeln!(
        map_roots,
        "omega,{},{},{},{}",
        point.re, point.im, center.re, center.im
    )?;
    map_roots.flush()?;
    for j in 0..61 {
        for i in 0..81 {
            let u = center
                + Complex::new(
                    -0.025 + 0.05 * i as f64 / 80.,
                    -0.012 + 0.024 * j as f64 / 60.,
                );
            let value = p.evaluate(&d, u * KP)?.value / KP;
            writeln!(
                landscape,
                "omega,{},{},{},{}",
                u.re, u.im, value.re, value.im
            )?;
        }
    }
    landscape.flush()?;
    Ok(())
}

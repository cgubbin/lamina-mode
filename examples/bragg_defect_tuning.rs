//! Compare a Bragg defect pole target with direct transmission optimization.
use lamina_core::{
    AnalyticalMaterialStack, CanonicalCoordinates, ComplexPlaneEvaluator, Constant,
    ExteriorWavevectors, IncidentSide, ModeJet1, Polarisation, Scatter2, SeedJet, Stack,
};
use lamina_mode::*;
use lamina_units::Length;
use ndarray::{Ix0, arr0};

#[path = "support/pole_material.rs"]
mod material;

type J = ModeJet1<Complex, Ix0>;
const TARGET: f64 = 6.36;
const BASE_WIDTH: f64 = 1.0 / 3.0;
const BOUNDS: [f64; 2] = [0.30, 0.36];
const POL: Polarisation = Polarisation::TransverseElectric;

fn stack(width: f64) -> AnalyticalMaterialStack<Complex> {
    let h = 2.1;
    let l = 1.45;
    let mut builder = Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum());
    for n in [h, l].into_iter().cycle().take(8) {
        builder =
            builder.analytical_layer(Constant::dielectric(n * n), Length::centimetres(0.25 / n));
    }
    builder = builder.analytical_layer(
        material::Frozen {
            epsilon: Complex::new(2.25, 0.035),
            mu: Complex::ONE,
        },
        Length::centimetres(width),
    );
    for n in [l, h].into_iter().cycle().take(10) {
        builder =
            builder.analytical_layer(Constant::dielectric(n * n), Length::centimetres(0.25 / n));
    }
    builder.finalise()
}

fn flux(width: f64) -> [f64; 3] {
    let solver = ComplexPlaneEvaluator::<J, _, _>::compile(&stack(width), Scatter2::new()).unwrap();
    let make_k = || <J as SeedJet>::constant(arr0(Complex::from(TARGET)));
    let result = solver
        .solve(
            &CanonicalCoordinates::new(make_k(), <J as SeedJet>::constant(arr0(Complex::ZERO))),
            &ExteriorWavevectors::new(make_k(), make_k()),
            POL,
        )
        .unwrap()
        .amplitudes(IncidentSide::Left);
    // Both exterior channels are vacuum at normal incidence.
    let r = result.reflection().value()[()].norm_sqr();
    let t = result.transmission().value()[()].norm_sqr();
    [r, t, 1.0 - r - t]
}

fn main() -> Result<(), String> {
    let request = QnmPoleRequest {
        beta_per_cm: 0.0,
        polarisation: POL,
        domain: FrequencyDomainInput {
            bounds: FrequencyRegion::new(5.8, 6.85, -0.4, 0.08).map_err(|e| e.to_string())?,
            reference: FrequencyReference::RealRadiation {
                frequency: TARGET,
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
    let base =
        select_qnm_poles(stack(BASE_WIDTH), request.clone()).map_err(|e| format!("{e:?}"))?;
    if base.coverage != QnmPoleCoverage::Reconciled || base.poles.len() != 1 {
        return Err(format!(
            "expected one reconciled defect pole, found {:?} with {} accepted",
            base.coverage,
            base.poles.len()
        ));
    }
    let mut local_steps = 0;
    let mut refreshes = 0;
    let mut track = |width| {
        let result = track_qnm_poles(
            &base,
            stack(width),
            request.clone(),
            QnmPoleTrackingSettings {
                match_radius_per_cm: 0.50,
                audit_coverage: false,
            },
        )
        .map_err(|e| format!("{e:?}"))?;
        if result.matches != [(0, 0)] || result.destination.poles.len() != 1 {
            return Err(format!("unresolved matching at width {width}"));
        }
        if result.destination.coverage == QnmPoleCoverage::TrackedOnly {
            local_steps += 1;
        } else if result.destination.coverage == QnmPoleCoverage::Reconciled {
            refreshes += 1;
        } else {
            return Err(format!("unresolved discovery at width {width}"));
        }
        Ok(result.destination.poles[0].signature.k0)
    };

    let [mut lo, mut hi] = BOUNDS;
    let flo = track(lo)?.re - TARGET;
    let fhi = track(hi)?.re - TARGET;
    if flo * fhi >= 0.0 {
        return Err("target is outside the bracketed pole excursion".into());
    }
    for _ in 0..28 {
        let mid = (lo + hi) / 2.0;
        if (track(mid)?.re - TARGET).signum() == flo.signum() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let pole_width = (lo + hi) / 2.0;
    let tuned = track(pole_width)?;
    assert!((tuned.re - TARGET).abs() < 1e-8);

    let [mut lo, mut hi] = BOUNDS;
    for _ in 0..48 {
        let a = lo + (hi - lo) / 3.0;
        let b = hi - (hi - lo) / 3.0;
        if flux(a)[1] < flux(b)[1] {
            lo = a;
        } else {
            hi = b;
        }
    }
    let direct_width = (lo + hi) / 2.0;
    let optimized = track(direct_width)?;
    let direct_flux = flux(direct_width);
    for i in 0..=100 {
        let width = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * i as f64 / 100.0;
        assert!(flux(width)[1] <= direct_flux[1] + 1e-10);
    }
    assert!(direct_flux[1] >= flux(pole_width)[1] - 1e-12);
    for (label, width, pole) in [
        ("baseline", BASE_WIDTH, base.poles[0].signature.k0),
        ("pole-tuned", pole_width, tuned),
        ("direct optimum", direct_width, optimized),
    ] {
        let [r, t, a] = flux(width);
        assert!(a >= -1e-12);
        println!("{label}: width={width:.9}, k0={pole:.9}, R={r:.9}, T={t:.9}, A={a:.9}");
    }
    println!("Local pole corrections {local_steps}, discovery refreshes {refreshes}");
    Ok(())
}

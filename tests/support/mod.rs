//! Helpers shared by the closed-stack tests.
use lamina_mode::*;

/// Accepted roots over `bounds` with argloc's count, under `discovery`.
/// argloc may leave well-conditioned roots together as one cluster; each
/// cluster's enclosure is searched again in halves until it resolves.
pub fn resolved_roots(
    problem: &PropagationProblem,
    bounds: ConformalRegion<f64>,
    domain: &dyn Fn(ConformalRegion<f64>) -> Domain,
    discovery: DiscoveryConfig,
) -> (Vec<ModePoint>, usize) {
    let report = problem
        .plan(vec![domain(bounds)], vec![])
        .unwrap()
        .solve(discovery, Default::default(), Default::default())
        .unwrap();
    let mut points = report.observations().to_vec();
    let mut count = 0;
    for chart in report.discovery().entries() {
        for estimate in chart.unwrap().estimates() {
            let raw = estimate.raw();
            if raw.multiplicity == 1 {
                count += 1;
                continue;
            }
            let r = raw.enclosure;
            let mid = 0.5 * (r.x_min + r.x_max);
            for (lo, hi) in [(r.x_min, mid), (mid, r.x_max)] {
                let half = ConformalRegion::new(lo, hi, r.y_min, r.y_max).unwrap();
                let (more, n) = resolved_roots(problem, half, domain, discovery);
                points.extend(more);
                count += n;
            }
        }
    }
    (points, count)
}

#[path = "../examples/support/plasmon.rs"]
mod plasmon;
use lamina_mode::Complex;
#[test]
fn independent_references_and_physical_rejections() {
    let b = plasmon::beta_exact(0.6);
    assert!((b.re - 0.907).abs() < 0.01 && b.im > 0.);
    let u = plasmon::omega_exact(1.);
    assert!((u.re - 0.618).abs() < 0.001 && u.im < 0.);
    let k = plasmon::waves(Complex::new(0.6, 0.), b);
    let check = |root, waves| plasmon::physical_checks("beta", root, b, waves, 0., 0.);
    assert!(check(b, k).is_ok());
    assert!(
        check(b, [-k[0], k[1]])
            .unwrap_err()
            .to_string()
            .contains("decay")
    );
    assert!(
        check(b.conj(), k)
            .unwrap_err()
            .to_string()
            .contains("attenuation")
    );
}
#[test]
fn quick_public_physical_regressions() {
    let out = std::env::temp_dir().join(format!("lamina-plasmon-test-{}", std::process::id()));
    assert!(!out.exists(), "test output must be fresh");
    let result = plasmon::run(&out, false);
    let summary = std::fs::read_to_string(out.join("summary.json")).unwrap();
    assert!(summary.contains("complete"), "{summary}");
    let rows = std::fs::read_to_string(out.join("roots.csv")).unwrap();
    assert_eq!(rows.lines().count(), 7);
    assert!(rows.contains(",false,,"));
    assert!(plasmon::run(&out, false).is_err(), "reject stale output");
    std::fs::remove_dir_all(out).unwrap();
    result.unwrap();
}

# Plasmon physical validation

From lamina-mode, run the bounded numerical regression without Python:

```sh
cargo test --locked --test plasmon_benchmarks
cargo run --release --locked --example plasmon_validation -- --suite quick --output /tmp/plasmon-quick
```

The quick suite searches three regular points in each complex case. The full
suite searches 33 points per case, two additional map roots, ten local TE controls
and two 61×81 characteristic landscapes:

```sh
cargo run --release --locked --example plasmon_validation -- --suite full --output /tmp/plasmon-full
python -m venv /tmp/plasmon-plot-env
/tmp/plasmon-plot-env/bin/pip install -r scripts/plasmon-plot-requirements.txt
/tmp/plasmon-plot-env/bin/python scripts/plot_plasmon_validation.py /tmp/plasmon-full
```

Output directories must be new; create their parent first. Open `report.html`
in the output directory. Figures are written as PNG, PDF and SVG. Plotting is
optional and never part of Rust tests. The script requires a complete full run,
checks its schema, settings, physical evidence and grids, and never substitutes
old data or analytical values for missing numerical observations.

## Model and independent checks

Two half-spaces, vacuum | passive Drude metal, with no finite layers. This is
an illustrative material rather than a fit to gold or silver. Define
u=k0/kp=ω/ωp, v=β/kp, kp=10⁵ cm⁻¹, η=0.02 and εm=1−1/[u(u+iη)].
At real u, v=u√[εm/(1+εm)] on the passive forward branch. At real v, the
independent reference solves

```text
u⁴+iηu³−(1+2v²)u²−2iηv²u+v²=0
```

starting from the positive lossless bound branch. These references place local
search charts and explicit decaying exterior anchors. Every numerical marker
comes from public discovery and refinement, not the reference formula. The
frequency windows avoid the Drude poles and exterior zeros for these fixtures;
the caller's analytic assumptions do not certify arbitrary devices or windows.

The independent unsquared TM condition is κd/εd+κm/εm=0. Checks require
Im κd,m>0, Im β>0 with exp(+iβz), and Im ω<0 with exp(−iωt).
Relative root error, boundary mismatch divided by kp and the Scatter2/Transfer2
characteristic difference divided by kp must be ≤10⁻⁸. Backends use the same
core characteristic gauge and are compared at each root and a nearby regular
point, not just at two near-zero values. Negative-control tests reject reversed
exterior and passive signs.

Spatial samples span u=0.15…0.695; temporal samples span v=0.2…3.
Quick samples use the two endpoints and midpoint. TE controls run at indices
0,8,16,24,32 in the full suite and index 0 in the quick suite: `te_checked=false`
with empty `te_accepted` means **untested**, not zero modes. These are local
controls, not an exhaustive absence proof. Real-frequency and real-wavevector
curves are different slices at finite loss.

## Recorded evidence and limits

`summary.json` records suite, model, tolerances, expected shape and completion
or failure. `settings.json` and `settings.txt` record discovery, refinement, continuation, chart
and matching budgets and analytic declarations. `roots.csv` retains coordinates,
independent references, residuals, decay factors, backend discrepancy, TE status,
classification and radiation interpretation. `evidence_*.txt` retains the public
solve reports including termination, candidates and provenance. Per-point errors
are collected in `failures.txt`; a failed run returns a nonzero exit status and
is refused by the plotter. Characteristic data and independently solved map roots
are in `characteristics.csv` and `map_roots.csv`. `plot_summary.json` reports
measured maxima without overwriting numerical completion metadata.

Negative-permittivity exteriors remain unsupported by the named dielectric
classifier. Explicit complex anchors retain unspecified built-in radiation
qualification. Decay and boundary checks establish the physical branch
independently; they do not extend those public interpretation contracts.
No blind global discovery, branch continuation, fields or QNM normalization is
claimed. The original core `artifacts/plasmon-validation` remains historical
evidence; maintained source now lives in `examples/support/plasmon.rs`.

"""Validate the completed Rust run and render its physical report with gnuplot."""
import csv
import html
import json
import math
import re
from pathlib import Path
import subprocess
import sys

out = Path(sys.argv[1]).resolve()
# Publish the report index only after every case and field signature passes.
for previous in ("report.html", "summary.json"):
    (out / previous).unlink(missing_ok=True)
if (out / "schema.txt").read_text().strip() != "modal-branch-report-v1":
    raise ValueError("Expected a completed branch report")
cases = {"plasmon-frequency": 81, "plasmon-propagation": 81, "lower": 101, "upper": 101, "page": 331}
data = {}
for case, count in cases.items():
    with (out / f"{case}.csv").open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != count:
        raise ValueError(f"Incomplete or unexpected station count for {case}")
    for row in rows:
        for key, value in row.items():
            if key != "backend" and value and not math.isfinite(float(value)):
                raise ValueError(f"Nonfinite {case} {key}")
        if float(row["residual"]) > float(row["residual_limit"]) or float(row["correction"]) > float(row["correction_limit"]):
            raise ValueError(f"Unaccepted station in {case}")
    data[case] = rows

def figure(name, layout, panels):
    commands = ["set terminal svg size 1050,760 font 'sans,13'", f"set output '{name}.svg'",
                "set linetype 1 lc rgb '#205e97'", "set linetype 2 lc rgb '#cf541d'", "set linetype 3 lc rgb '#29836a'", "set linetype 4 lc rgb '#805492'", "set linetype 5 lc rgb '#a86a08'", "set linetype 6 lc rgb '#373e50'", "set datafile separator ','", "set grid", f"set multiplot layout {layout}"]
    for title, xlabel, ylabel, curves in panels:
        commands += [f"set title '{title}'", f"set xlabel '{xlabel}'", f"set ylabel '{ylabel}'",
                     "plot " + ", ".join(curves)]
    subprocess.run(["gnuplot"], input="\n".join(commands + ["unset multiplot"]), text=True, cwd=out, check=True)

def line(case, x, y, label, style="lines lw 2"):
    return f"'{case}.csv' using ({x}):({y}) with {style} title '{label}'"

panels = []
for case, x, ys in [("plasmon-propagation",3,[1,2]), ("plasmon-frequency",1,[3,4])]:
    variable = "β" if x == 3 else "k₀"
    control = "k₀" if x == 3 else "β"
    for component, y, ref in zip(["Re","Im"],ys,[13,14]):
        panels.append((case, f"{control} / kp", f"{component} {variable} / kp", [
            line(case,f"column({x})/1e5",f"column({ref})/1e5","Reference"),
            line(case,f"column({x})/1e5",f"column({y})/1e5","Solver","points pt 6 ps 0.45")]))
figure("plasmons","2,2",panels)
gap = [float(b["k0_re_cm"])-float(a["k0_re_cm"]) for a,b in zip(data["lower"],data["upper"])]
if min(gap)<=0 or abs(gap[50]/1e4-0.04577579915)>2e-7:
    raise ValueError("Coupled-slab gap disagrees with the independent boundary equation")
for name, ref in [("lower",4.9776353638),("upper",5.02341116295)]:
    if abs(float(data[name][50]["k0_re_cm"])/1e4-ref)>1e-7:
        raise ValueError(f"Incorrect {name} branch at the design crossing")
base = [line(c,"column(1)/1e4","column(3)/1e4",c) for c in ("lower","upper")]
# Isolated TE0 modes provide the uncoupled dispersion on the same axes.
def isolated(beta, n):
    q0=math.sqrt((5*n)**2-49)
    width=2*math.atan(math.sqrt(24)/q0)/q0
    lo,hi=beta/n,beta
    for _ in range(55):
        mid=(lo+hi)/2
        q=math.sqrt((n*mid)**2-beta**2)
        if q*width>2*math.atan(math.sqrt(beta**2-mid**2)/q):
            hi=mid
        else:
            lo=mid
    return (lo+hi)/2
with (out/"isolated.csv").open("w") as f:
    f.write("beta_per_um,core1,core2\n")
    for row in data["lower"]:
        beta=float(row["beta_re_cm"])/1e4
        f.write(f"{beta},{isolated(beta,2.0)},{isolated(beta,1.7)}\n")
base += [line("isolated","column(1)",f"column({i+2})",f"Isolated core {i+1}","lines dt 2") for i in range(2)]
page = data["page"]
turning = []
for a,b in zip(page,page[1:]):
    u,v = float(a["slope_re"]),float(b["slope_re"])
    if u*v<0:
        t=u/(u-v)
        turning.append([(float(a[k])+t*(float(b[k])-float(a[k])))*factor for k,factor in
                        [("beta_re_cm",0.0001),("k0_re_cm",0.0299792458),("k0_im_cm",0.0299792458)]])
references = [(7.41,805,-4.14),(30.3,778,-5.06)]
if len(turning)!=2 or any(abs(x-y)>tol for point,ref in zip(turning,references) for x,y,tol in zip(point,ref,(0.05,0.5,0.01))):
    raise ValueError(f"Page stationary points disagree with Table 3.2: {turning}")
figure("branches","2,2",[
    ("Coupled slabs","β (1/µm)","Re k₀ (1/µm)",base),
    ("Page TM branch","β (1/µm)","Re ω (rad/ps)",[line("page","column(1)/1e4","column(3)*0.0299792458","Solver")]),
    ("Group velocity","β (1/µm)","d Re ω / dβ (units of c)",[line("page","column(1)/1e4","column(6)","Implicit slope")]),
    ("Temporal damping","β (1/µm)","Im ω (rad/ps)",[line("page","column(1)/1e4","column(4)*0.0299792458","Solver")])])
gauges = {}
for case, ids in [("lower",[0,50,100]),("upper",[0,50,100]),("page",[20,34,50,220,263,300])]:
    curves = [[],[]]
    for i in ids:
        path = out / f"{case}-{i}-fields.csv"
        with path.open() as f:
            header=f.readline()
            gauges[path.stem]=header.split(";")[0].lstrip("# ").strip()
            coordinates=re.findall(r"Complex \{ re: ([^,]+), im: ([^ }]+) \}",header)
            expected=[(float(data[case][i][prefix+"_re"+unit]),float(data[case][i][prefix+"_im"+unit])) for prefix,unit in [("k0","_cm"),("beta","_cm"),("left",""),("right","")]]
            if [(float(x),float(y)) for x,y in coordinates]!=expected:
                raise ValueError(f"Field provenance differs from branch point for {path}")
            field=list(csv.DictReader(f))
        if len(field)!=(324 if case=="page" else 405) or any(not math.isfinite(float(v)) for r in field for v in r.values()):
            raise ValueError(f"Incomplete field profile {path}")
        for j,start in enumerate((2,8)):
            magnitude="sqrt("+"+".join(f"column({k})**2" for k in range(start,start+6))+")"
            curves[j].append(line(path.stem,"column(1)",magnitude,f"β={float(data[case][i]['beta_re_cm'])/1e4:.2f}"))
    figure(case+"-fields","2,1",[(case+" profiles","Depth (µm)",label,c) for label,c in zip(["|E|","|H|"],curves)])
summary={"page_stationary_beta_per_um_and_omega_per_ps":turning,"design_crossing_gap_per_um":gap[50]/1e4,
         "smallest_sampled_gap_per_um":min(gap)/1e4,"field_gauges":gauges,
         "max_plasmon_relative_error":max(float(r["relative_error"]) for c in cases if c.startswith("plasmon") for r in data[c])}
(out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
sections = [
    ("Interface plasmon","plasmons","Circles are accepted continuation points. Lines are independent interface solutions. The real-frequency slice gives spatial attenuation and the real-wavevector slice gives temporal decay."),
    ("Coupled slabs and stopped light","branches",f"At the uncoupled crossing the finite gap gives a splitting of {gap[50]/1e4:.5f} µm⁻¹. On the Page branch the real group velocity changes sign at β ≈ {turning[0][0]:.3f} and {turning[1][0]:.3f} µm⁻¹."),
    ("Lower coupled branch","lower-fields","The depth axis starts at the first core. Each trace uses the accepted point at the indicated wavevector."),
    ("Upper coupled branch","upper-fields","The two slab fields redistribute as the branches approach and separate."),
    ("Page waveguide","page-fields","The 110 nm semiconductor core spans 0 to 0.11 µm. The 50 nm ITO cap ends at 0.16 µm.")]
page_html = ["<!doctype html><meta charset=utf-8><title>Physical mode continuation</title><style>body{max-width:64rem;margin:3rem auto;padding:0 1rem;font:17px/1.5 system-ui}img{width:100%}</style><h1>Physical mode continuation</h1>"]
for title,img,caption in sections:
    page_html.append(f"<h2>{title}</h2><p>{caption}</p><img src='{img}.svg' alt='{title}'>")
page_html += ["<p>Field amplitudes use "+html.escape(", ".join(sorted(set(gauges.values()))))+". The CSV files retain all six complex components and the exterior wavevectors.</p>",
              "<p>Stopped-light reference: <a href='https://doi.org/10.25560/30760'>Page, Chapter 3, Tables 3.1–3.2</a>. <a href='summary.json'>Numerical comparisons and profile gauges</a>.</p>"]
(out/"report.html").write_text("\n".join(page_html))

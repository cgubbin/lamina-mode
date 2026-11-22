"""Plot measured public-solver CSVs; analytic lines are independent references."""
from pathlib import Path
import argparse, csv, json, math, html
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path, help="fresh complete full-suite output directory")
args = parser.parse_args()
ROOT = args.run.resolve()
metadata = json.loads((ROOT / "summary.json").read_text())
if metadata.get("schema") != 1 or metadata.get("status") != "complete" or metadata.get("suite") != "full":
    raise ValueError("Expected a complete schema-1 full validation run")
if metadata.get("samples_per_case") != 33 or metadata.get("map_shape") != [61,81] or metadata.get("kp_inverse_cm") != 1e5 or metadata.get("eta") != .02:
    raise ValueError("Run settings do not match this benchmark schema")
def load(name):
    with (ROOT / name).open() as f:
        return list(csv.DictReader(f))
settings = json.loads((ROOT / "settings.json").read_text())
for key in ("discovery","spatial_refinement","temporal_refinement","chart","continuation","matching"):
    if key not in settings:
        raise ValueError(f"Missing numerical settings: {key}")
for case in ("beta","omega"):
    for i in range(33):
        if not (ROOT/f"evidence_{case}_{i}.txt").is_file():
            raise ValueError("Missing discovery/refinement evidence")
rows = load("roots.csv")
maps = load("characteristics.csv")
map_rows = load("map_roots.csv")
for data, count in [(rows,33),(maps,61*81),(map_rows,1)]:
    if len(data) != 2*count or any(sum(r["case"]==case for r in data)!=count for case in ("beta","omega")):
        raise ValueError("Incomplete or mismatched case data")
    for row in data:
        for key,value in row.items():
            if key not in ("case","classification","radiation","te_checked","te_accepted") and not math.isfinite(float(value)):
                raise ValueError(f"Nonfinite {key}")
for case in ("beta","omega"):
    subset=[r for r in rows if r["case"]==case]
    if len({r["parameter"] for r in subset})!=33:
        raise ValueError("Duplicate or missing dispersion parameters")
    for i,r in enumerate(subset):
        if r["te_checked"] != str(i%8==0).lower() or r["te_accepted"] != ("0" if i%8==0 else ""):
            raise ValueError("Missing or inconsistent TE controls")
        root=complex(float(r["re"]),float(r["im"]))
        exact=complex(float(r["exact_re"]),float(r["exact_im"]))
        if abs(root-exact)/abs(exact)>metadata["root_tolerance"] or float(r["dispersion_error"])>metadata["boundary_tolerance"] or float(r["backend_relative_error"])>metadata["boundary_tolerance"]:
            raise ValueError("Failed physical tolerance")
        if min(float(r["kappa_left_im"]),float(r["kappa_right_im"]))<=0 or (root.imag<=0 if case=="beta" else root.imag>=0):
            raise ValueError("Failed physical sign")
    grid=[r for r in maps if r["case"]==case]
    if len({(r["x"],r["y"]) for r in grid})!=61*81 or len({r["x"] for r in grid})!=81 or len({r["y"] for r in grid})!=61:
        raise ValueError("Incomplete characteristic grid")
if (ROOT/"failures.txt").read_text().strip():
    raise ValueError("Run records failed cases")
centers={r["case"]:r for r in map_rows}
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def group(case):
    rs=[r for r in rows if r['case']==case]
    return {k:np.array([float(r[k]) for r in rs]) for k in rs[0] if k not in ['case','classification','radiation','te_checked','te_accepted']}
b=group('beta');w=group('omega');eta=.02
blue='#126f86';orange='#c66b23';gray='#737b85';purple='#9562ac'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','axes.grid':True,'grid.alpha':.17,'legend.fontsize':9,'legend.frameon':False})
def eps(u):return 1-1/(u*(u+1j*eta))
def exact_b(u):return u*np.sqrt(eps(u)/(1+eps(u)))
def exact_w(v):
    u=np.sqrt(2*v*v/(1+2*v*v+np.sqrt(1+4*v**4))).astype(complex)-1j*eta/4
    for _ in range(25):
        u-=(u**4+1j*eta*u**3-(1+2*v*v)*u*u-2j*eta*v*v*u+v*v)/(4*u**3+3j*eta*u*u-2*(1+2*v*v)*u-2j*eta*v*v)
    return u
def save(fig,name):
    fig.savefig(ROOT/(name+'.png'),dpi=190,bbox_inches='tight')
    fig.savefig(ROOT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(ROOT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)
model=r'Vacuum | Drude metal: $\varepsilon_m=1-[u(u+0.02i)]^{-1}$, $u=\omega/\omega_p$, $k_p=10^5\,\mathrm{cm}^{-1}$'
u=np.linspace(.15,.695,750); ba=exact_b(u)
v=np.linspace(.2,3,750); wa=exact_w(v)
vl=np.linspace(.0,3.,750); ul=np.sqrt(2*vl*vl/(1+2*vl*vl+np.sqrt(1+4*vl**4)))
fig,ax=plt.subplots(2,2,figsize=(12,8.3),layout='constrained')
fig.suptitle('A lossy interface plasmon, solved in both complex cases',fontsize=18,fontweight='bold')
fig.supxlabel(model,fontsize=10)
a=ax[0,0];a.plot(ba.real,u,color=orange,lw=2,label='Independent TM relation');a.scatter(b['re'],b['parameter'],s=25,facecolors='white',edgecolors=orange,lw=1.3,zorder=4,label='lamina-mode accepted roots');a.plot(vl,ul,'--',c=gray,lw=1,label='Lossless reference');a.plot([0,.75],[0,.75],':',c=gray,lw=1,label='Vacuum light line');a.set(xlabel=r'$\mathrm{Re}(\beta)/k_p$',ylabel=r'$\omega/\omega_p$ (real)',title='A   Real frequency → complex wavevector',xlim=(0,3),ylim=(0,.76));a.legend(loc='lower right')
a=ax[0,1];a.plot(v,wa.real,color=blue,lw=2,label='Independent quartic root');a.scatter(w['parameter'],w['re'],s=25,facecolors='white',edgecolors=blue,lw=1.3,zorder=4,label='lamina-mode accepted roots');a.plot(vl,ul,'--',c=gray,lw=1,label='Lossless reference');a.axhline(1/np.sqrt(2),c=gray,ls=':',lw=1);a.text(1.2,.716,r'$\omega_{sp}/\omega_p=1/\sqrt{2}$',fontsize=9,color=gray);a.set(xlabel=r'$\beta/k_p$ (real)',ylabel=r'$\mathrm{Re}(\omega)/\omega_p$',title='B   Real wavevector → complex frequency',xlim=(0,3),ylim=(0,.76));a.legend(loc='lower right')
a=ax[1,0];a.plot(u,ba.imag,color=orange,lw=2);a.scatter(b['parameter'],b['im'],s=25,facecolors='white',edgecolors=orange,lw=1.3,zorder=4);a.set(xlabel=r'$\omega/\omega_p$ (real)',ylabel=r'$\mathrm{Im}(\beta)/k_p$',title='C   Positive spatial attenuation');a.text(.03,.93,r'$\exp(i\beta z)\;\Rightarrow\;\mathrm{Im}(\beta)>0$',transform=a.transAxes,va='top',color=orange)
a=ax[1,1];a.plot(v,-wa.imag,color=blue,lw=2);a.scatter(w['parameter'],-w['im'],s=25,facecolors='white',edgecolors=blue,lw=1.3,zorder=4);a.axhline(eta/2,c=gray,ls='--',lw=1,label=r'High-$\beta$ limit: $\gamma/2$');a.set(xlabel=r'$\beta/k_p$ (real)',ylabel=r'$-\mathrm{Im}(\omega)/\omega_p$',title='D   Positive temporal decay rate',ylim=(0,.011));a.text(.03,.9,r'$\exp(-i\omega t)\;\Rightarrow\;\mathrm{Im}(\omega)<0$',transform=a.transAxes,va='top',color=blue);a.legend(loc='lower right')
save(fig,'dispersion')
fig,axes=plt.subplots(1,2,figsize=(12.3,5.5),layout='constrained')
fig.suptitle('The solver characteristic actually vanishes',fontsize=18,fontweight='bold')
fig.supxlabel('Checked characteristic F from the public API; contours cross at the accepted numerical root.',fontsize=10)
for a,case,title,xlabel,ylabel in zip(axes,['beta','omega'],[r'Real $\omega=0.60\,\omega_p$',r'Real $\beta=1.00\,k_p$'],[r'$\mathrm{Re}(\beta)/k_p$',r'$\mathrm{Re}(\omega)/\omega_p$'],[r'$\mathrm{Im}(\beta)/k_p$',r'$\mathrm{Im}(\omega)/\omega_p$']):
    rs=[r for r in maps if r['case']==case]
    x=np.array([float(r['x']) for r in rs]).reshape(61,81);y=np.array([float(r['y']) for r in rs]).reshape(61,81)
    z=np.array([complex(float(r['real']),float(r['imag'])) for r in rs]).reshape(61,81)
    im=a.pcolormesh(x,y,np.log10(np.maximum(abs(z),1e-15)),cmap='magma',vmin=-4,vmax=-.25,shading='auto',rasterized=True)
    a.contour(x,y,z.real,levels=[0],colors=['#38dfdd'],linewidths=1.7)
    a.contour(x,y,z.imag,levels=[0],colors=['#9ade56'],linewidths=1.7)
    r=centers[case]
    a.scatter([float(r['re'])],[float(r['im'])],s=130,facecolors='none',edgecolors='white',lw=2,zorder=8)
    a.scatter([float(r['exact_re'])],[float(r['exact_im'])],marker='+',s=60,c='black',lw=1.5,zorder=9)
    a.set(xlabel=xlabel,ylabel=ylabel,title=title);a.grid(False)
    fig.colorbar(im,ax=a,label=r'$\log_{10}|F/k_p|$',shrink=.85)
handles=[Line2D([],[],color='#38bdbb',lw=2,label=r'Re $F=0$'),Line2D([],[],color='#7eaf37',lw=2,label=r'Im $F=0$'),Line2D([],[],color='black',marker='o',markerfacecolor='none',lw=0,label='Accepted numerical root'),Line2D([],[],color='black',marker='+',lw=0,label='Independent analytic root')]
fig.legend(handles=handles,loc='outside lower center',ncol=4,bbox_to_anchor=(.5,-.055))
save(fig,'characteristic_maps')
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
fig.suptitle('Physicality checks beyond finding a small determinant',fontsize=18,fontweight='bold')
fig.supxlabel('Explicit decay-branch references; these checks do not constitute global mode coverage or QNM normalization.',fontsize=10)
for a,data,label,color in zip(axes[0],[b,w],[r'Real $\omega$; complex $\beta$',r'Real $\beta$; complex $\omega$'],[orange,blue]):
    x=data['parameter'];err=np.abs((data['re']+1j*data['im'])-(data['exact_re']+1j*data['exact_im']))/np.abs(data['exact_re']+1j*data['exact_im'])
    a.semilogy(x,np.maximum(err,1e-17),'o-',c=color,ms=3,label='Relative root error');a.semilogy(x,np.maximum(data['dispersion_error'],1e-17),'s--',c=gray,ms=3,label='TM boundary mismatch / kp');a.set(title=label,ylabel='Dimensionless error',xlabel=r'Prescribed real $\omega/\omega_p$' if data is b else r'Prescribed real $\beta/k_p$');a.legend()
for a,data,label,color in zip(axes[1],[b,w],[r'Real $\omega$: exterior decay',r'Real $\beta$: exterior decay'],[orange,blue]):
    x=data['parameter'];a.plot(x,data['kappa_left_im'],c=color,lw=2,label='Vacuum side');a.plot(x,data['kappa_right_im'],c=gray,lw=2,label='Metal side');a.set(title=label,ylabel=r'$\mathrm{Im}(\kappa)/k_p$  (must be positive)',xlabel=r'Prescribed real $\omega/\omega_p$' if data is b else r'Prescribed real $\beta/k_p$',ylim=(0,None));a.legend()
save(fig,'physicality_checks')
summary={}
for case,data in [('complex_beta',b),('complex_frequency',w)]:
    root=data['re']+1j*data['im'];exact=data['exact_re']+1j*data['exact_im']
    summary[case]={'accepted_points':len(root),'max_relative_root_error':float(np.max(abs(root-exact)/abs(exact))),'max_boundary_mismatch_over_kp':float(max(data['dispersion_error'])),'max_characteristic_residual_over_kp':float(max(data['residual'])),'max_scatter_transfer_value_difference_over_kp':float(max(data['backend_relative_error'])),'min_dielectric_decay_over_kp':float(min(data['kappa_left_im'])),'min_metal_decay_over_kp':float(min(data['kappa_right_im'])),'te_modes_at_sampled_checks':0}
summary['model']={'epsilon_infinity':1,'epsilon_dielectric':1,'gamma_over_omega_p':eta,'kp_inverse_cm':1e5,'fitted_material':False,'te_checks_per_case':5,'frequency_reference':'explicit complex; built-in radiation qualification remains unspecified','branch_tracking':'independent local point searches; analytic solution locates valid benchmark charts'}
(ROOT/'plot_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
report = """<!doctype html><meta charset="utf-8"><title>Plasmon validation</title>
<style>body{font:17px system-ui;max-width:1150px;margin:40px auto;padding:20px;line-height:1.5}img{width:100%}pre{overflow:auto}</style>
<h1>Interface plasmon validation</h1>
<p>Vacuum | illustrative passive Drude metal, not a fitted material. Time exp(−iωt), propagation exp(+iβz).
Real-frequency and real-wavevector curves are different slices at finite loss.</p>
<p>33 independent public discovery/refinement solves per case, two map solves, five local TE controls per case.
Analytic references place local charts; numerical markers come from accepted solver observations.
This does not demonstrate blind global discovery.</p>
<img src="dispersion.png"><img src="characteristic_maps.png"><img src="physicality_checks.png">
<h2>Independent physical checks</h2><p>u=k0/kp=ω/ωp, v=β/kp, εm=1−1/[u(u+iη)].
v=u√[εm/(1+εm)]. Temporal roots solve u⁴+iηu³−(1+2v²)u²−2iηv²u+v²=0 on the positive lossless bound branch.
The unsquared TM condition κd/εd+κm/εm=0, exterior decay and passive signs are checked.
Scatter2/Transfer2 comparisons include roots and nearby off-root evaluations.</p>
<h2>Interpretation limits</h2><p>Metallic named classification remains unsupported. Explicit complex anchors retain unspecified built-in radiation qualification.
The physical bound branch is checked independently. No fields, QNM normalization or global coverage are claimed.
TE controls apply only to the recorded local windows.</p>
<h2>Measured results</h2><pre>"""+html.escape(json.dumps(summary,indent=2))+"""</pre>
<h2>Reproduce</h2><pre>cargo run --release --locked --example plasmon_validation -- --suite full --output NEW_DIRECTORY
python scripts/plot_plasmon_validation.py NEW_DIRECTORY</pre>
<p>See docs/plasmon-validation.md for optional plotting dependencies, conventions and numerical settings.
Raw numerical summary.json, settings.txt, roots.csv, evidence files and failure records belong to this run.
PNG, PDF and SVG versions of each figure are available beside this page.</p>"""
(ROOT/'report.html').write_text(report)

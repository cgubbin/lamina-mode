"""Plot and check one complete scalar pole-normalization validation run."""
import argparse,csv,json,math,html
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path)
a=p.parse_args();root=a.run
metadata=json.loads((root/'summary.json').read_text())
if metadata.get('schema')!=1 or metadata.get('status')!='complete' or metadata.get('suite') not in ('quick','full'):
    raise ValueError('Expected a complete schema-1 normalization run')
settings=json.loads((root/'settings.json').read_text())
if settings.get('factor_atol')!=1e-12 or settings.get('contour_radius_over_pole')!=1e-4:
    raise ValueError('Mismatched normalization settings')
def load(name):
    with (root/name).open() as f:return list(csv.DictReader(f))
rows=load('cases.csv');convergence=load('convergence.csv');contributions=load('contributions.csv');coordinates=load('coordinates.csv')
names=['guided_te','guided_tm','cavity_tm_direct','cavity_te_conformal','cavity_te_direct','cavity_tm_conformal','interface_spatial','interface_temporal','film_even_H','film_odd_H','dispersive_electric','dispersive_magnetic']
if len(rows)!=12 or {r['case'] for r in rows}!=set(names) or len(coordinates)!=12 or {r['case'] for r in coordinates}!=set(names):
    raise ValueError('Missing or duplicate cases')
for r in rows:
    if r['status']!='complete' or not (root/(r['case']+'_evidence.txt')).is_file():raise ValueError('Missing successful case evidence')
    for k in ['residue_error','contour_error','backend_error','gauge_error','partition_error']:
        x=float(r[k])
        if not math.isfinite(x) or x<0 or x>(1e-7 if k in ['residue_error','backend_error'] else 1e-8):raise ValueError('Failed physical validation')
    for k in ['omitted_boundary_error','hermitian_product_error','wrong_variable_error','omitted_dispersion_error']:
        if r[k] in ('','unavailable'):continue
        x=float(r[k])
        if not math.isfinite(x) or x<=1e-4:raise ValueError('Negative control failed')
for r in contributions:
    if not all(math.isfinite(float(r[k])) for k in ('re','im')):raise ValueError('Nonfinite contribution')
for name in names:
    values=[r for r in contributions if r['case']==name]
    if sum(r['item']=='total' for r in values)!=1 or sum(r['item'].startswith('boundary_') for r in values)!=2:raise ValueError('Incomplete contributions')
    for position in range(3 if metadata['suite']=='full' else 2):
        for direction in range(2):
            points=[r for r in convergence if r['case']==name and int(r['position'])==position and int(r['direction'])==direction]
            if len(points)!=7 or len({r['relative_offset'] for r in points})!=7:raise ValueError('Incomplete convergence sequence')
            if any(not math.isfinite(float(r['relative_error'])) for r in points):raise ValueError('Nonfinite convergence data')
if (root/'failures.txt').read_text().strip():raise ValueError('Run records failures')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,1,figsize=(12,12),layout='constrained')
colors=plt.get_cmap('tab20')
for i,name in enumerate(names):
    data=[r for r in convergence if r['case']==name and r['position']=='0' and r['direction']=='0']
    axes[0].loglog([float(r['relative_offset']) for r in data],[max(float(r['relative_error']),1e-17) for r in data],'.-',color=colors(i),label=name.replace('_',' '))
axes[0].invert_xaxis();axes[0].set(title='Approaching each pole: convergence, then roundoff',xlabel='Complex offset / |pole| (decreasing →)',ylabel='Independent residue relative error');axes[0].legend(ncol=3,fontsize=8)
x=list(range(len(rows)))
axes[1].bar(x,[max(float(r['residue_error']),1e-17) for r in rows],color='#176f8a');axes[1].set_yscale('log');axes[1].axhline(1e-7,color='gray',ls='--',label='Acceptance tolerance');axes[1].set(title='Normalized-mode prediction versus independent contour residue',ylabel='Relative error');axes[1].legend()
for key,marker in [('backend_error','o'),('gauge_error','s'),('partition_error','^')]:axes[2].semilogy(x,[max(float(r[key]),1e-17) for r in rows],marker+'-',label=key.replace('_',' '))
axes[2].set(title='Independent invariance checks',ylabel='Relative discrepancy');axes[2].legend(ncol=3)
for ax in axes:ax.grid(alpha=.2)
for ax in axes[1:]:ax.set_xticks(x,[r['case'].replace('_','\n') for r in rows],fontsize=8)
fig.suptitle('Planar mode normalization: physical validation',fontsize=18)
for ext in ('png','pdf','svg'):fig.savefig(root/('normalization_validation.'+ext),dpi=170)
maxima={k:max(float(r[k]) for r in rows) for k in ['residue_error','contour_error','backend_error','gauge_error','partition_error']}
(root/'plot_summary.json').write_text(json.dumps(maxima,indent=2)+'\n')
report='''<!doctype html><meta charset="utf-8"><title>Pole normalization validation</title>
<style>body{font:17px system-ui;max-width:1200px;margin:32px auto;padding:20px;line-height:1.5}img{width:100%}pre{overflow:auto}</style>
<h1>Planar pole normalization validated by independent residues</h1>
<p>The scalar Green convention is L G=delta, with a positive unit flux jump and physical k0 or beta in cm⁻¹. This is one planar Fourier component, not a finite 3D mode volume.</p>
<h2>No infinite exterior integral is needed</h2><p>At normal incidence with nondispersive exteriors, the outgoing electric and magnetic contributions cancel. Core's finite-layer Maxwell overlap already contains the equivalent scalar boundary contribution. TE/TM analytic cavity regressions check this identity and reject double counting. The scalar formula uses analytic endpoint derivatives; no infinite exterior quadrature is performed.</p>
<img src="normalization_validation.png">
<h2>Independent checks</h2><p>Twelve accepted modal points include guided TE/TM, both spectral formulations and polarizations, both complex interface-plasmon cases, both film modes and electric/magnetic dispersion. A separately constructed matched-wave Green function provides unit-jump and contour residues. Complex gauge changes and independent left/right boundary displacements preserve predictions. Negative controls omit scalar boundary or dispersion terms, conjugate a bilinear residue factor or use the wrong pole variable; each is detected. An unavailable control factor is recorded as such, not assigned a fabricated error.</p>
<pre>'''+html.escape(json.dumps(maxima,indent=2))+'''</pre>
<h2>Scope</h2><p>Normalized fields use the declared scalar pole convention, which differs by explicit factors from the legacy Maxwell-integral convention. Exterior branches and radiation provenance are retained. No exceptional-point certification, mode volume, Purcell factor or complete modal expansion is claimed.</p>
<p>Raw files: <a href="cases.csv">case errors and controls</a>, <a href="convergence.csv">convergence</a>, <a href="contributions.csv">bilinear contributions</a>, <a href="coordinates.csv">physical coordinates</a>, <a href="settings.json">models and settings</a>.</p>
<p>Reproduce from lamina-mode: <code>cargo run --release --locked --example pole_normalization -- --suite full --output NEW_DIRECTORY</code>, then <code>python scripts/plot_pole_normalization.py NEW_DIRECTORY</code>.</p>'''
(root/'report.html').write_text(report)
print(json.dumps(maxima,indent=2))

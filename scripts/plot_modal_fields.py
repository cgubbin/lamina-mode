"""Plot a fresh modal_fields example run (requires Matplotlib)."""
import argparse, csv, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run',type=Path)
a=parser.parse_args()
with (a.run/'fields.csv').open() as f:
    rows=list(csv.DictReader(f))
names=['interface_spatial','interface_temporal','film_even_H','film_odd_H','leaky_cavity']
if set(r['mode'] for r in rows)!=set(names):
    raise ValueError('Incomplete field profile run')
for r in rows:
    if any(not math.isfinite(float(v)) for k,v in r.items() if k!='mode'):
        raise ValueError('Nonfinite profile data')
fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained')
for ax,name in zip(axes.flat,names):
    data=[r for r in rows if r['mode']==name]
    expected=243 if name.startswith('film') or name=='leaky_cavity' else 162
    if len(data)!=expected:raise ValueError(f'Incomplete {name}')
    x=[float(r['z']) for r in data]
    p='ey' if name=='leaky_cavity' else 'hy'
    for suffix,style in [('re','-'),('im','--')]:
        ax.plot(x,[float(r[p+'_'+suffix]) for r in data],style,label=f'{p} {suffix}')
    ax.axvline(0,color='gray',lw=.7)
    if name.startswith('film'):ax.axvline(.4,color='gray',lw=.7)
    if name=='leaky_cavity':ax.axvline(.3,color='gray',lw=.7)
    ax.set(title=name.replace('_',' '),xlabel='z (cm)' if name=='leaky_cavity' else r'$k_p z$',ylabel='Primary field, left boundary = 1')
    ax.grid(alpha=.2);ax.legend()
axes.flat[-1].axis('off')
axes.flat[-1].text(0,.9,'Arbitrary complex amplitude: E and H share one scale.\nFilm parity refers to H_y about the film centre.\nLeaky exterior growth is retained at finite distance.\nNo QNM normalization or mode volume is claimed.',va='top',linespacing=1.6)
fig.suptitle('Recovered homogeneous mode fields',fontsize=18)
for extension in ['png','pdf','svg']:fig.savefig(a.run/('modal_fields.'+extension),dpi=170)

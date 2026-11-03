import pstats, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
PROF = sys.argv[1] if len(sys.argv) > 1 else 'prof_197755658.out'
OC = sys.argv[2] if len(sys.argv) > 2 else 'rerun_v2/TIC_197755658_out/TIC197755658_secondary_oc.csv'

# ---------- (a) runtime breakdown from cProfile ----------
st = pstats.Stats(PROF).stats
def cum(name, file_hint=None):
    return sum(v[3] for (f, _, fn), v in st.items() if fn == name and (file_hint is None or file_hint in f))
total = cum('main', 'apsidal_pipeline')
groups = [('Period search (BLS)', cum('bls_search', 'apsidal_pipeline')),
          ('Eclipse template fits', cum('fit_local_eclipses') + cum('median_bin_template')
           + cum('measure_eclipse_half_width', 'apsidal_pipeline')),
          ('Ephemeris refinement', cum('refine_ephemeris_dip_times', 'apsidal_pipeline')),
          ('Data loading', cum('load_and_stitch', 'apsidal_pipeline'))]
other = max(total - sum(g[1] for g in groups), 0)
groups.append(('Other (checks, plots)', other))
for name, t in groups:
    print(f"{name:26s} {t:8.1f} s  {100 * t / total:5.1f}%")
print(f"{'total':26s} {total:8.1f} s")

fig, ax = plt.subplots(figsize=(3.3, 1.5))
names = [g[0] for g in groups][::-1]
fr = np.array([g[1] for g in groups][::-1]) / total * 100
cols = ['0.7'] * (len(groups) - 1) + ['#76b900']
ax.barh(names, fr, color=cols, height=0.6)
for i, v in enumerate(fr):
    ax.text(v + 1, i, f'{v:.0f}%', va='center', fontsize=7)
ax.set_xlim(0, 110)
ax.set_xlabel('share of CPU runtime [%]')
tot_txt = f'{total / 60:.1f} min' if total >= 60 else f'{total:.0f} s'
ax.set_title(f'Where the CPU pipeline spends its time (one TESS star, {tot_txt})', fontsize=7.5)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.savefig('fig_runtime.pdf'); fig.savefig('fig_runtime.png', dpi=300)

# ---------- (b) validated secondary O-C ----------
a = np.loadtxt(OC, delimiter=',', skiprows=1, ndmin=2)
t, oc = a[:, 0], a[:, 1]
keep = np.abs(oc - np.median(oc)) < 10 * np.median(np.abs(oc - np.median(oc))) * 1.4826 + 5
sl, ic = np.polyfit(t[keep], oc[keep], 1)
fig, ax = plt.subplots(figsize=(3.3, 1.8))
ax.plot(t[keep], oc[keep], 'o', ms=2, color='#1f77b4')
tt = np.linspace(t.min(), t.max(), 50)
ax.plot(tt, sl * tt + ic, '-', color='k', lw=0.8)
ax.set_xlabel('time [BTJD]')
ax.set_ylabel('secondary O$-$C [min]')
ax.text(0.97, 0.95, 'drift [min/1000 d]\nthis pipeline  $-47.4$\nmirror timing  $-47.8$\nmodel-based ETVs (Penev)  $-47.3$',
        transform=ax.transAxes, ha='right', va='top', fontsize=6.5,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', lw=0.5))
ax.set_title('TIC 197755658: apsidal motion, three independent methods', fontsize=7.5)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.savefig('fig_oc.pdf'); fig.savefig('fig_oc.png', dpi=300)
print(f"fitted slope on plotted points: {sl * 1000:+.2f} min/1000 d")
print("saved fig_runtime.pdf/.png and fig_oc.pdf/.png")

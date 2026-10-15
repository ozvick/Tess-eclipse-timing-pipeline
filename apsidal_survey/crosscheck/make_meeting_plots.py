import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

os.makedirs('meeting_plots', exist_ok=True)
plt.rcParams.update({'font.size': 12})

def load(f):
    return list(csv.DictReader(open(f)))

# 1. Victor vs Penev drift
rows = load('etv_comparison.csv')
S = np.array([float(r['S']) for r in rows]); Se = np.array([float(r['Se']) for r in rows])
Pn = np.array([float(r['pen']) for r in rows]); Pe = np.array([float(r['pen_err']) for r in rows])
z = np.abs(np.array([float(r['z']) for r in rows]))
fig, ax = plt.subplots(figsize=(7, 7))
for m, c, lab in ((z < 2, 'tab:blue', '|z| < 2'), ((z >= 2) & (z < 3), 'tab:orange', '2 ≤ |z| < 3'),
                  (z >= 3, 'tab:red', '|z| ≥ 3')):
    ax.errorbar(S[m], Pn[m], xerr=Se[m], yerr=Pe[m], fmt='o', ms=5, lw=0.8, color=c,
                label=f'{lab} ({m.sum()})')
for r, zz in zip(rows, z):
    if zz >= 3:
        ax.annotate(r['tic'], (float(r['S']), float(r['pen'])), fontsize=8,
                    xytext=(4, 4), textcoords='offset points')
lim = max(np.abs(S).max(), np.abs(Pn).max()) * 1.5
ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
for f in (ax.set_xscale, ax.set_yscale):
    f('symlog', linthresh=0.3)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('Victor: secondary − primary drift [min/1000 d]')
ax.set_ylabel('Penev ETVs, sign flipped [min/1000 d]')
ax.set_title(f'Drift comparison, {len(rows)} systems')
ax.legend(loc='upper left')
fig.tight_layout(); fig.savefig('meeting_plots/1_drift_comparison.png', dpi=150); plt.close(fig)

# 2. z distribution, before and after sqrt(2)
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(-8, 8, 33)
for f, c, lab in (('etv_comparison.csv', 'tab:red', 'Penev errors as stored'),
                  ('etv_comparison_sqrt2.csv', 'tab:blue', 'Penev errors × √2')):
    if os.path.exists(f):
        zz = np.clip([float(r['z']) for r in load(f)], -7.9, 7.9)
        ax.hist(zz, bins=bins, histtype='step', lw=2, color=c, label=lab)
x = np.linspace(-8, 8, 400)
ax.plot(x, len(rows) * (bins[1] - bins[0]) * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi),
        'k--', label='expected from chance')
ax.set_xlabel('z = (Victor − Penev) / combined error')
ax.set_ylabel('number of systems')
ax.set_title('How well do the two methods agree?')
ax.legend()
fig.tight_layout(); fig.savefig('meeting_plots/2_agreement_z.png', dpi=150); plt.close(fig)

# 3. Penev reduced chi2, before and after sqrt(2)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
lb = np.logspace(-1.5, 1.7, 40)
for ax, key, name in zip(axes, ('chi2_p', 'chi2_s'), ('primaries', 'secondaries')):
    for f, c, lab in (('penev_etv_drifts.csv', 'tab:red', 'as stored'),
                      ('penev_etv_drifts_sqrt2.csv', 'tab:blue', 'errors × √2')):
        v = np.array([float(r[key]) for r in load(f) if r['variant'] == 'none' and r[key]])
        ax.hist(v, bins=lb, histtype='step', lw=2, color=c,
                label=f'{lab} (median {np.median(v):.2f})')
    ax.axvline(1, color='k', ls='--')
    ax.set_xscale('log')
    ax.set_xlabel('reduced χ² of straight-line fit to ETVs')
    ax.set_title(f"Penev's ETVs: {name}")
    ax.legend(fontsize=10)
axes[0].set_ylabel('number of systems')
fig.tight_layout(); fig.savefig('meeting_plots/3_penev_chi2.png', dpi=150); plt.close(fig)

# 4. pipeline period vs confirmed period
arb = [r for r in load('period_arbitration.csv') if r['verdict'] in ('all agree', 'Penev') and r['P_prsa']]
def cat(x):
    for name, v in (('1', 1), ('2', 2), ('1/2', 0.5), ('3', 3), ('4', 4)):
        if abs(x / v - 1) < 0.01:
            return name
    return 'other'
cnt = Counter(cat(float(r['P_v']) / float(r['P_prsa'])) for r in arb)
order = ['1', '1/2', '2', '3', '4', 'other']
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(order, [cnt.get(k, 0) for k in order],
              color=['tab:green'] + ['tab:red'] * 5)
for b in bars:
    ax.annotate(str(int(b.get_height())), (b.get_x() + b.get_width() / 2, b.get_height()),
                ha='center', va='bottom')
ax.set_xlabel('pipeline period / confirmed period')
ax.set_ylabel('number of systems')
ax.set_title(f'Pipeline periods vs. confirmed periods ({len(arb)} systems)')
fig.tight_layout(); fig.savefig('meeting_plots/4_period_ratio.png', dpi=150); plt.close(fig)

print("saved:", sorted(os.listdir('meeting_plots')))

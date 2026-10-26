import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

# 1. implied k2 per detection
rows = sorted(csv.DictReader(open('apsidal_consistency_v3.csv')), key=lambda r: float(r['k2_med']))
y = np.arange(len(rows))
k2 = np.array([float(r['k2_med']) for r in rows])
lo = np.array([float(r['k2_lo']) for r in rows]); hi = np.array([float(r['k2_hi']) for r in rows])
fig, ax = plt.subplots(figsize=(8, 9))
ax.axvspan(0.002, 0.01, color='tab:green', alpha=0.15, label='typical stars (0.002–0.01)')
ax.axvline(0.75, color='tab:red', ls='--', label='physical maximum (0.75)')
col = ['tab:red' if float(r['k2_med']) > 0.75 else 'k' for r in rows]
for i in range(len(rows)):
    ax.errorbar(k2[i], y[i], xerr=[[k2[i] - lo[i]], [hi[i] - k2[i]]], fmt='o', color=col[i], ms=5)
ax.set_yticks(y); ax.set_yticklabels([f"{r['tic']}  (e={float(r['e']):.2f})" for r in rows], fontsize=9)
ax.set_xscale('symlog', linthresh=1e-4); ax.set_xlim(-0.01, 2)
ax.set_xlabel('implied apsidal-motion constant k₂')
ax.set_title('Are the drifts consistent with tides + GR?')
ax.legend(loc='lower right', fontsize=10)
fig.tight_layout(); fig.savefig('meeting_plots/5_implied_k2.png', dpi=150); plt.close(fig)

# 2. pipeline vs mirror drift
rows = list(csv.DictReader(open('retime_all.csv')))
S = np.array([float(r['pipeline']) for r in rows]); M = np.array([float(r['mirror']) for r in rows])
Me = np.array([float(r['mirror_err']) for r in rows]); bad = np.array([bool(r['flag']) for r in rows])
ok = Me < 20
fig, ax = plt.subplots(figsize=(7, 7))
ax.errorbar(S[ok & ~bad], M[ok & ~bad], yerr=Me[ok & ~bad], fmt='o', color='tab:blue', label=f'agree ({(ok & ~bad).sum()})')
ax.errorbar(S[ok & bad], M[ok & bad], yerr=Me[ok & bad], fmt='o', color='tab:red', label=f'disagree ({(ok & bad).sum()})')
for r, s, m, b, o in zip(rows, S, M, bad, ok):
    if b and o:
        ax.annotate(r['tic'], (s, m), fontsize=8, xytext=(4, 4), textcoords='offset points')
lim = 100
ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
for f in (ax.set_xscale, ax.set_yscale):
    f('symlog', linthresh=0.3)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('pipeline drift [min/1000 d]'); ax.set_ylabel('mirror-method drift [min/1000 d]')
ax.set_title('Pipeline vs independent re-timing')
ax.legend(loc='upper left')
fig.tight_layout(); fig.savefig('meeting_plots/6_pipeline_vs_mirror.png', dpi=150); plt.close(fig)
print("saved meeting_plots/5_implied_k2.png and 6_pipeline_vs_mirror.png")

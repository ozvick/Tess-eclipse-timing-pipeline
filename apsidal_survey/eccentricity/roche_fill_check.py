"""Does each system's e sin(omega) excess track how full its Roche lobes are?

For every system in ecc_components.csv with fitted parameters in binary_radii_ecc.csv:
  filling = largest of R_i / (a * r_L,i)   (Eggleton Roche-lobe radius)
  excess  = |e sin w| (Penev fit) - |e sin w| (pipeline, from eclipse durations)
  predicted = spurious |e sin w| at that filling, from the ellc distortion simulations (radii fixed)
"Reliable" systems have similar eclipse depths (depth ratio > 0.5), where the pipeline's duration
ratio is trustworthy. Writes roche_fill_check.csv and roche_fill_check.png.
"""
import csv
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# simulation calibration: Roche-lobe filling -> spurious |e sin w| (radii fixed), both runs combined
CAL = np.array([(0.23, 0.0002), (0.30, 0.0005), (0.31, 0.00057), (0.36, 0.0011), (0.41, 0.0016),
                (0.48, 0.0021), (0.50, 0.0033), (0.58, 0.0040), (0.65, 0.0068), (0.79, 0.0169), (0.89, 0.0388)])


def predicted(fill):
    return float(np.exp(np.interp(fill, CAL[:, 0], np.log(CAL[:, 1]), left=np.log(CAL[0, 1]), right=np.nan)))


def eggleton(q):
    return 0.49 * q ** (2 / 3) / (0.6 * q ** (2 / 3) + np.log(1 + q ** (1 / 3)))


ecc = {int(r['tic']): r for r in csv.DictReader(open('ecc_components.csv'))}
par = {int(float(r['tic'])): r for r in csv.DictReader(open('binary_radii_ecc.csv'))}
rows = []
for t, r in ecc.items():
    if t not in par:
        continue
    p = par[t]
    M1, M2, R1, R2, a = (float(p[k]) for k in ('M1', 'M2', 'R1', 'R2', 'a_Rsun'))
    fill = max(R1 / (a * eggleton(M1 / M2)), R2 / (a * eggleton(M2 / M1)))
    pen, pipe = float(r['abs_esinw_penev']), float(r['abs_esinw_pipe'])
    err = float(r['abs_esinw_pipe_err'])
    rows.append(dict(tic=t, P=float(r['P']), fill=fill, esinw_penev=pen, esinw_pipe=pipe, esinw_pipe_err=err,
                     excess=pen - pipe, predicted=predicted(fill), e_penev=float(r['e_penev']),
                     abs_ecosw_pipe=float(r['abs_ecosw_pipe']), depth_ratio=float(r['depth_ratio']),
                     reliable=float(r['depth_ratio']) > 0.5))
rows.sort(key=lambda x: -x['fill'])
with open('roche_fill_check.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} systems with both e sin w comparison and fitted parameters")

A = lambda k, sel=None: np.array([x[k] for x in rows if sel is None or sel(x)], dtype=float)
for label, sel in (('all', None), ('reliable (depth ratio > 0.5)', lambda x: x['reliable'])):
    f, ex = A('fill', sel), A('excess', sel)
    if len(f) > 4:
        rho, pv = spearmanr(f, ex)
        print(f"  {label:28s} n={len(f):3d}  Spearman(filling, excess) = {rho:+.2f}  (p = {pv:.3g})")

print("\nBy filling (reliable systems):")
for lo, hi in ((0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)):
    m = [x for x in rows if x['reliable'] and lo <= x['fill'] < hi]
    if m:
        print(f"  filling {lo:.1f}-{min(hi, 1):.1f}: n={len(m):2d}  median excess {np.median([x['excess'] for x in m]):+.4f}  "
              f"median predicted {np.nanmedian([x['predicted'] for x in m]):.4f}")

print(f"\n{'TIC':>10} {'P':>6} {'fill':>5} {'Penev esinw':>11} {'pipe esinw':>10} {'excess':>8} {'predicted':>9} {'rel':>4}")
for x in rows[:30]:
    pr = '   n/a' if np.isnan(x['predicted']) else f"{x['predicted']:9.4f}"
    print(f"{x['tic']:>10} {x['P']:6.3f} {x['fill']:5.2f} {x['esinw_penev']:11.4f} {x['esinw_pipe']:10.4f} "
          f"{x['excess']:+8.4f} {pr} {'yes' if x['reliable'] else 'no':>4}")

fig, ax = plt.subplots(figsize=(7, 5))
for rel, mk, lab in ((True, 'o', 'similar eclipse depths (reliable)'), (False, 'x', 'unequal depths')):
    f, ex = A('fill', lambda x: x['reliable'] == rel), A('excess', lambda x: x['reliable'] == rel)
    ax.plot(f, ex, mk, ms=5, alpha=0.8, label=lab)
grid = np.linspace(0.2, 0.89, 100)
ax.plot(grid, [predicted(g) for g in grid], 'k-', lw=1.5, label='simulation: spherical fit of distorted stars')
ax.axhline(0, color='0.6', lw=0.7)
ax.set_yscale('symlog', linthresh=0.005)
ax.set_xlabel('Roche-lobe filling (larger star)')
ax.set_ylabel('e sin w excess: Penev fit - eclipse durations')
ax.legend(fontsize=8)
ax.set_title('Does the e sin w excess track tidal distortion?', fontsize=10)
fig.tight_layout(); fig.savefig('roche_fill_check.png', dpi=150)
print('\nsaved roche_fill_check.png')

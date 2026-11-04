"""Is the eccentricity in Penev's catalog timing-based (e cos w) or duration-based (e sin w)?

For every system in rerun_v2/ with a secondary eclipse, estimates without any light-curve model:
  |e cos w| from the secondary-eclipse phase:   phi_sec - 0.5 ~ (2/pi) e cos w
  |e sin w| from the eclipse-duration ratio r = secondary / primary half-width:  e sin w ~ (r - 1)/(r + 1)
and compares them with Penev's fitted values (tic_etrigwquants.csv) as a function of period.
Absolute values are compared, since the two methods may label the eclipses differently.
Both formulas are first-order in e (edge-on, small e); expect them to degrade above e ~ 0.2.

Writes ecc_components.csv and ecc_components.png; prints a summary and the systems where
Penev's eccentricity rests mainly on e sin w but the durations do not support it.
"""
import csv, glob, json
from math import pi, sqrt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

trig = {int(float(r['tic_id'])): r for r in csv.DictReader(open('tic_etrigwquants.csv'))}
summ = {}
for p in glob.glob('rerun_v2/TIC_*_out/TIC*_summary.json'):
    s = json.load(open(p))
    summ[int(s['tic'])] = s

# width quantization step, estimated from the spacing of all measured half-widths
ws = np.array(sorted({round(v, 8) for s in summ.values()
                      for v in (s.get('primary_half_width_phase'), s.get('secondary_half_width_phase')) if v}))
d = np.diff(ws); d = d[d > 1e-6]
q = float(np.min(d)) if len(d) else 0.0
print(f"half-width step (quantization): {q:.5f} in phase")

rows = []
for t, s in summ.items():
    phs, wp, wsec, P = (s.get(k) for k in ('secondary_phase', 'primary_half_width_phase',
                                          'secondary_half_width_phase', 'period_days'))
    if None in (phs, wp, wsec, P) or t not in trig or not wp or not wsec:
        continue
    Q = {k: float(v) for k, v in trig[t].items() if k.startswith('e')}
    pc, ps = Q['ecosw_Q0.5'], Q['esinw_Q0.5']
    pce = (Q['ecosw_Q0.841'] - Q['ecosw_Q0.159']) / 2
    pse = (Q['esinw_Q0.841'] - Q['esinw_Q0.159']) / 2
    ecw = abs(pi / 2 * (phs - 0.5))
    r = wsec / wp
    esw = abs((r - 1) / (r + 1))
    sig_r = r * sqrt((q / sqrt(12) / wsec) ** 2 + (q / sqrt(12) / wp) ** 2)
    esw_err = 2 / (r + 1) ** 2 * sig_r
    e_pen = sqrt(pc ** 2 + ps ** 2)
    rows.append(dict(tic=t, P=P, sec_phase=phs, width_ratio=r,
                     abs_ecosw_pipe=ecw, abs_esinw_pipe=esw, abs_esinw_pipe_err=esw_err,
                     abs_ecosw_penev=abs(pc), ecosw_penev_err=pce, abs_esinw_penev=abs(ps), esinw_penev_err=pse,
                     e_penev=e_pen, esinw_share=(ps ** 2 / e_pen ** 2 if e_pen > 0 else np.nan),
                     depth_ratio=(s.get('secondary_depth') or np.nan) / (s.get('primary_depth') or np.nan)))

rows.sort(key=lambda r: r['P'])
with open('ecc_components.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} systems with both pipeline and Penev values -> ecc_components.csv")

A = lambda k: np.array([r[k] for r in rows], dtype=float)
P, ecp, esp, ese = A('P'), A('abs_ecosw_pipe'), A('abs_esinw_pipe'), A('abs_esinw_pipe_err')
ecn, esn, en, share = A('abs_ecosw_penev'), A('abs_esinw_penev'), A('e_penev'), A('esinw_share')
small = en < 0.2                                  # where the first-order formulas are reliable


def agreement(x, y, err, mask):
    z = (x[mask] - y[mask]) / np.maximum(err[mask], 1e-4)
    return np.median(np.abs(x[mask] - y[mask])), np.mean(np.abs(z) < 2)


md_c, f_c = agreement(ecp, ecn, np.full(len(rows), 0.003), small)
md_s, f_s = agreement(esp, esn, np.hypot(ese, A('esinw_penev_err')), small)
print(f"\nSystems with Penev e < 0.2 (n={small.sum()}):")
print(f"  |e cos w|: median |pipeline - Penev| = {md_c:.4f}")
print(f"  |e sin w|: median |pipeline - Penev| = {md_s:.4f}, within 2 sigma for {100 * f_s:.0f}%")

print("\nBy period (Penev e > 0.02):")
for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 1e9)):
    m = (P >= lo) & (P < hi) & (en > 0.02)
    if m.sum():
        print(f"  P {lo:>2}-{'inf' if hi > 1e8 else int(hi):>3} d: n={m.sum():3d}  median Penev e={np.median(en[m]):.3f}  "
              f"median e sin w share of e^2={np.median(share[m]):.2f}  "
              f"median |e sin w| pipeline/Penev={np.median(esp[m] / np.maximum(esn[m], 1e-4)):.2f}")

print("\nPenev's e rests mainly on e sin w (share > 0.7), but the durations disagree by > 2 sigma:")
flag = (en > 0.02) & (share > 0.7) & (np.abs(esp - esn) > 2 * np.hypot(ese, A('esinw_penev_err')))
for r in [r for r, f in zip(rows, flag) if f]:
    print(f"  TIC {r['tic']:>10}  P={r['P']:6.3f}  Penev e={r['e_penev']:.3f} (|esinw| {r['abs_esinw_penev']:.3f})  "
          f"pipeline |esinw| {r['abs_esinw_pipe']:.3f} +/- {r['abs_esinw_pipe_err']:.3f}  "
          f"|ecosw| pipe {r['abs_ecosw_pipe']:.3f} / Penev {r['abs_ecosw_penev']:.3f}  depth ratio {r['depth_ratio']:.2f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
lim = [1e-4, 1]
for a, x, y, lab in ((ax[0], ecn, ecp, '|e cos w|'), (ax[1], esn, esp, '|e sin w|')):
    a.loglog(np.maximum(x, 1e-4), np.maximum(y, 1e-4), 'o', ms=4, alpha=0.7)
    a.plot(lim, lim, 'k--', lw=0.8)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel(f'Penev fit {lab}'); a.set_ylabel(f'pipeline (model-free) {lab}')
ax[0].set_title('timing-based: secondary-eclipse phase'); ax[1].set_title('duration-based: eclipse-width ratio')
ax[1].axhspan(1e-4, 0.02, color='0.9', zorder=0); ax[1].text(1.2e-4, 0.012, 'below width resolution', fontsize=8)
ax[2].semilogx(P, ecn, 'o', ms=4, label='|e cos w| (Penev)')
ax[2].semilogx(P, esn, 's', ms=4, mfc='none', label='|e sin w| (Penev)')
ax[2].set_xlabel('orbital period (d)'); ax[2].set_ylabel('component'); ax[2].legend(fontsize=8)
ax[2].set_title("Penev's eccentricity components vs period")
fig.tight_layout(); fig.savefig('ecc_components.png', dpi=150)
print('\nsaved ecc_components.png')

"""Do Penev's fitted parameters reproduce the equal eclipse durations seen in the data?

For each system, simulates a noise-free light curve with ellc from Penev's best fit (period, e, omega,
inclination, radii, masses, temperature ratio; from binary_radii_ecc.csv), cuts out the primary and
secondary eclipses, and measures their duration ratio the same way as plot_eclipse_overlay.py
(stretching the primary profile to match the secondary) plus the full width at 15% depth.
If the model's ratio is ~1 like the data, the fit is self-consistent: the two eclipses' different
impact parameters cancel the e sin w effect (a degeneracy). If the model's ratio is far from 1,
the fitted parameters do not describe the observed eclipses.

Usage: python penev_model_eclipses.py [binary_radii_ecc.csv]      Writes penev_model_eclipses.png/.csv
Surface-brightness ratio is approximated as teff_ratio**4 and limb darkening as quadratic (0.3, 0.2);
both affect depths and shapes slightly but hardly the durations.
"""
import csv, sys
from math import sqrt, cos, sin, radians
import numpy as np
import ellc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAR = sys.argv[1] if len(sys.argv) > 1 else 'binary_radii_ecc.csv'
OBSERVED = {158728853: 0.990, 269852699: 1.000, 99028319: 0.980, 281498280: 0.982, 436592342: 0.996,
            199420503: 1.026, 162585265: 0.996, 84342725: 0.994, 130158361: 1.022, 329770815: 1.016,
            121568625: 1.030}
par = {int(float(r['tic'])): r for r in csv.DictReader(open(PAR))}


def eclipse_profile(t, f, tc, half):
    m = np.abs(t - tc) < half
    x, y = t[m] - tc, f[m]
    base = np.median(y[np.abs(x) > 0.85 * half])
    depth = base - y.min()
    return x * 24, (base - y) / depth, depth


def width_at(x, y, level=0.15):
    """Full width (h) where the normalized profile crosses `level`, interpolated on both sides."""
    inside = np.where(y > level)[0]
    if len(inside) == 0:
        return np.nan
    i0, i1 = inside[0], inside[-1]
    left = np.interp(level, [y[i0 - 1], y[i0]], [x[i0 - 1], x[i0]]) if i0 > 0 else x[i0]
    right = np.interp(level, [y[i1 + 1], y[i1]], [x[i1 + 1], x[i1]]) if i1 < len(x) - 1 else x[i1]
    return right - left


def stretch_ratio(xp, yp, xs, ys):
    best = (np.inf, 1.0)
    for k in np.linspace(0.6, 1.6, 1001):
        chi = np.sum((ys - np.interp(xs / k, xp, yp, left=0, right=0)) ** 2)
        if chi < best[0]:
            best = (chi, k)
    return best[1]


rows = []
tics = [t for t in OBSERVED if t in par]
n = len(tics); ncol = min(3, n); nrow = int(np.ceil(n / ncol)) if n else 1
fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)
for ax, t in zip(axes.flat, tics):
    p = par[t]
    P, e, w, inc = float(p['per']), float(p['ecc']), radians(float(p['w_deg'])), float(p['inc'])
    M1, M2, R1, R2, a = (float(p[k]) for k in ('M1', 'M2', 'R1', 'R2', 'a_Rsun'))
    sb = float(np.clip(float(p['teff_ratio']) ** 4, 0.01, 10))
    kw = dict(radius_1=R1 / a, radius_2=R2 / a, sbratio=sb, incl=inc, q=M2 / M1, period=P, t_zero=0.0,
              f_c=sqrt(e) * cos(w), f_s=sqrt(e) * sin(w), ld_1='quad', ld_2='quad', ldc_1=[0.3, 0.2], ldc_2=[0.3, 0.2])
    tt = np.linspace(-0.25 * P, 0.75 * P, 400001)
    ff = ellc.lc(tt, **kw)
    t_pri = tt[np.argmin(np.where(np.abs(tt) < 0.1 * P, ff, np.inf))]
    t_sec = tt[np.argmin(np.where(np.abs(tt) > 0.15 * P, ff, np.inf))]
    half = 2.0 * (R1 + R2) / a * P / (2 * np.pi) / max(sqrt(1 - e * e), 0.3)   # generous window (days)
    xp, yp, dp = eclipse_profile(tt, ff, t_pri, half)
    xs, ys, ds = eclipse_profile(tt, ff, t_sec, half)
    k = stretch_ratio(xp, yp, xs, ys)
    wr = width_at(xs, ys) / width_at(xp, yp)
    b_p = a * cos(radians(inc)) / R1 * (1 - e * e) / (1 + e * sin(w))
    b_s = a * cos(radians(inc)) / R1 * (1 - e * e) / (1 - e * sin(w))
    es = e * sin(w)
    r_naive = (1 + abs(es)) / (1 - abs(es))
    obs = OBSERVED[t]
    verdict = ('model has equal durations like the data: degeneracy' if abs(k - 1) < 0.02
               else 'model durations differ: fit does not match the data' if abs(k - obs) > 0.03
               else 'model close to the data')
    rows.append(dict(tic=t, P=P, e=e, omega_deg=float(p['w_deg']), incl=inc, esinw=es, b_primary=b_p, b_secondary=b_s,
                     ratio_edge_on_formula=r_naive, ratio_model=k, width15_ratio_model=wr, ratio_observed=obs,
                     depth_ratio_model=ds / dp, verdict=verdict))
    print(f"TIC {t:>10} P={P:6.3f} e={e:.3f} w={float(p['w_deg']):6.1f} i={inc:5.2f}  b_p={b_p:.2f} b_s={b_s:.2f}  "
          f"ratio: edge-on formula {r_naive:.3f} | Penev model {k:.3f} (15% width {wr:.3f}) | data {obs:.3f}  -> {verdict}",
          flush=True)
    ax.plot(xp, yp, '-', color='tab:blue', lw=2, label='model primary')
    ax.plot(xs, ys, '-', color='tab:orange', lw=2, label=f'model secondary (ratio {k:.3f})')
    ax.invert_yaxis(); ax.set_xlim(-1.1 * half * 24, 1.1 * half * 24)
    ax.set_title(f'TIC {t}: Penev fit, e={e:.3f}, i={inc:.1f}°\n(data ratio {obs:.3f})', fontsize=9)
    ax.set_xlabel('time from mid-eclipse (h)'); ax.set_ylabel('normalized depth'); ax.legend(fontsize=7, loc='lower center')
for ax in list(axes.flat)[len(tics):]:
    ax.axis('off')
fig.suptitle("Eclipses predicted by Penev's fitted parameters (ellc), scaled to the same depth", fontsize=11)
fig.tight_layout(); fig.savefig('penev_model_eclipses.png', dpi=140)
if rows:
    with open('penev_model_eclipses.csv', 'w', newline='') as fh:
        wr_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr_.writeheader(); wr_.writerows(rows)
missing = [t for t in OBSERVED if t not in par]
if missing:
    print('not in the parameter file:', missing)
print('saved penev_model_eclipses.png and penev_model_eclipses.csv')

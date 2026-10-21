import sys, glob, json, csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

t = int(sys.argv[1])
jp = f'rerun_fixedP/TIC_{t}_out/TIC{t}_summary.json'
jp = jp if os.path.exists(jp) else f'TIC_{t}_out/TIC{t}_summary.json'
s = json.load(open(jp))
P, T0, ph = s['period_days'], s['t0_btjd'], s['secondary_phase']

files = sorted(set(glob.glob(f'TIC_{t}_fits/**/*_lc.fits', recursive=True) +
                   glob.glob(f'TIC{t}_fits/**/*_lc.fits', recursive=True)))
T, F = [], []
for f in files:
    with fits.open(f) as h:
        d = h[1].data
        ok = np.isfinite(d['TIME']) & np.isfinite(d['PDCSAP_FLUX']) & (d['QUALITY'] == 0)
        T.append(d['TIME'][ok]); F.append(d['PDCSAP_FLUX'][ok] / np.nanmedian(d['PDCSAP_FLUX'][ok]))
T, F = map(np.concatenate, (T, F))

def mirror_time(tc_guess, halfwin):
    m = np.abs(T - tc_guess) < halfwin
    if m.sum() < 50:
        return None
    x, y = T[m] - tc_guess, F[m]
    base, ymin = np.percentile(y, 90), np.percentile(y, 1)
    depth = base - ymin
    if depth < 0.02:
        return None
    xin = x[y < base - 0.1 * depth]
    lo, hi = xin.min(), xin.max()
    pad = max(0.05, 0.1 * (hi - lo))
    pre = (x > lo - 5 * pad) & (x < lo - pad)
    post = (x > hi + pad) & (x < hi + 5 * pad)
    if pre.sum() < 10 or post.sum() < 10:
        return None
    x1, y1, x2, y2 = x[pre].mean(), np.median(y[pre]), x[post].mean(), np.median(y[post])
    yn = y / (y1 + (y2 - y1) * (x - x1) / (x2 - x1))
    sig = np.std(yn[pre | post])
    ecl = yn < 1 - 0.1 * depth
    xe, ye = x[ecl], yn[ecl]
    xc0 = 0.5 * (lo + hi)
    if (xe < xc0).sum() < 15 or (xe > xc0).sum() < 15:
        return None
    o = np.argsort(xe); xs, ys = xe[o], ye[o]
    span = 0.25 * (hi - lo)
    grid = xc0 + np.linspace(-span, span, 401)
    chi = []
    for tc in grid:
        xr = 2 * tc - xs
        ok = (xr > xs[0]) & (xr < xs[-1])
        chi.append(np.sum((ys[ok] - np.interp(xr[ok], xs, ys)) ** 2) / (2 * sig**2) / max(ok.sum(), 1) * len(xs))
    chi = np.array(chi); i = int(np.argmin(chi))
    j = slice(max(i - 10, 0), min(i + 11, len(grid)))
    a, b, _ = np.polyfit(grid[j], chi[j], 2)
    if a <= 0:
        return None
    return tc_guess - b / (2 * a), np.sqrt(2 / a)

hw = min(1.0, 0.2 * P)
nmin, nmax = int((T.min() - T0) / P) - 1, int((T.max() - T0) / P) + 1
pri = {n: r for n in range(nmin, nmax + 1) if (r := mirror_time(T0 + n * P, hw))}
sec = {n: r for n in range(nmin, nmax + 1) if (r := mirror_time(T0 + (n + ph) * P, hw))}

def wlin(x, y, e):
    A = np.vstack([x, np.ones_like(x)]).T; w = 1 / e**2
    cov = np.linalg.inv(A.T @ (A * w[:, None])); b = cov @ (A.T @ (w * y))
    return b, np.sqrt(np.diag(cov))

n_p = np.array(sorted(pri), float); tp = np.array([pri[int(n)][0] for n in n_p]); ep = np.array([pri[int(n)][1] for n in n_p])
keep = np.ones(len(n_p), bool)
for _ in range(3):
    (Pf, T0f), _ = wlin(n_p[keep], tp[keep], ep[keep])
    res = (tp - (T0f + n_p * Pf)) * 1440
    mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826 + 1e-3
    keep = np.abs(res - np.median(res[keep])) < max(10, 5 * mad)
(Pf, T0f), _ = wlin(n_p[keep], tp[keep], ep[keep])
p_oc = (tp - (T0f + n_p * Pf)) * 1440

n_s = np.array(sorted(sec), float); ts = np.array([sec[int(n)][0] for n in n_s]); es = np.array([sec[int(n)][1] for n in n_s]) * 1440
d = (ts - (T0f + n_s * Pf)) * 1440
s_oc = d - np.median(d)
(sl, _), (sle, _) = wlin(ts, s_oc, es)

pen = {}
if os.path.exists('binary_radii.csv'):
    pen = {int(r['tic']): float(r['per']) for r in csv.DictReader(open('binary_radii.csv'))}
print(f"TIC {t}: timed {len(n_p)} primaries ({(~keep).sum()} rejected as outliers), {len(n_s)} secondaries")
print(f"  period: pipeline JSON {P:.6f}  |  mirror primary fit {Pf:.6f}  |  Penev {pen.get(t, float('nan')):.6f}")
print(f"  primary O-C scatter (kept): {np.std(p_oc[keep]):.2f} min")
for n, tt, o, e in zip(n_s, ts, s_oc, es):
    print(f"   sec epoch {int(n):4d}  BTJD {tt:7.1f}  O-C {o:+7.2f} +/- {e:4.2f} min")
print(f"  secondary-minus-primary slope: mirror {sl*1000:+.2f} +/- {sle*1000:.2f}  |  "
      f"pipeline {s['apsidal_slope_min_per_1000d']:+.2f} min/1000d")

fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax[0].errorbar(tp[keep], p_oc[keep], ep[keep] * 1440, fmt='o', color='tab:blue', label='kept')
if (~keep).any():
    ax[0].plot(tp[~keep], np.clip(p_oc[~keep], -20, 20), 'x', color='k', ms=10, label='rejected (clipped to plot)')
ax[0].axhline(0, color='0.5'); ax[0].set_ylabel('primary O-C [min]'); ax[0].legend()
ax[1].errorbar(ts, s_oc, es, fmt='o', color='tab:red')
ax[1].axhline(0, color='0.5'); ax[1].set_ylabel('secondary - primary [min]'); ax[1].set_xlabel('BTJD')
fig.suptitle(f'TIC {t}: mirror-method timing (slope {sl*1000:+.2f}; pipeline {s["apsidal_slope_min_per_1000d"]:+.2f} min/1000d)')
fig.tight_layout(); fig.savefig(f'retime_{t}.png', dpi=120)

import glob, json, csv, os
import numpy as np
from astropy.io import fits

TICS = [125857387, 289044378, 295885511, 347506869, 12915124, 55497281, 71531463, 89522181,
        124282575, 160328766, 161988269, 184246521, 197755658, 220420534, 220430912, 257737073,
        272532173, 272822330, 278683641, 319936228, 322262620, 329770815, 331183881, 341029467,
        358232450, 372758722, 381857817, 396903513, 427312571]

def load(t):
    files = sorted(set(glob.glob(f'TIC_{t}_fits/**/*_lc.fits', recursive=True) +
                       glob.glob(f'TIC{t}_fits/**/*_lc.fits', recursive=True)))
    T, F = [], []
    for f in files:
        with fits.open(f) as h:
            d = h[1].data
            ok = np.isfinite(d['TIME']) & np.isfinite(d['PDCSAP_FLUX']) & (d['QUALITY'] == 0)
            if ok.sum() > 100:
                T.append(d['TIME'][ok]); F.append(d['PDCSAP_FLUX'][ok] / np.nanmedian(d['PDCSAP_FLUX'][ok]))
    return (np.concatenate(T), np.concatenate(F)) if T else (None, None)

def mirror_time(T, F, tc_guess, halfwin):
    m = np.abs(T - tc_guess) < halfwin
    if m.sum() < 50:
        return None
    x, y = T[m] - tc_guess, F[m]
    base, ymin = np.percentile(y, 90), np.percentile(y, 1)
    depth = base - ymin
    if depth < 0.005:
        return None
    xin = x[y < base - 0.1 * depth]
    lo, hi = xin.min(), xin.max()
    pad = max(0.02, 0.1 * (hi - lo))
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
    if (xe < xc0).sum() < 10 or (xe > xc0).sum() < 10:
        return None
    o = np.argsort(xe); xs, ys = xe[o], ye[o]
    grid = xc0 + np.linspace(-0.25 * (hi - lo), 0.25 * (hi - lo), 401)
    chi = []
    for tc in grid:
        xr = 2 * tc - xs
        ok = (xr > xs[0]) & (xr < xs[-1])
        chi.append(np.sum((ys[ok] - np.interp(xr[ok], xs, ys)) ** 2) / (2 * sig**2) / max(ok.sum(), 1) * len(xs))
    chi = np.array(chi); i = int(np.argmin(chi))
    j = slice(max(i - 10, 0), min(i + 11, len(grid)))
    a, b, _ = np.polyfit(grid[j], chi[j], 2)
    return (tc_guess - b / (2 * a), np.sqrt(2 / a)) if a > 0 else None

def wlin(x, y, e):
    A = np.vstack([x, np.ones_like(x)]).T; w = 1 / e**2
    cov = np.linalg.inv(A.T @ (A * w[:, None])); b = cov @ (A.T @ (w * y))
    chi2 = np.sum(w * (y - A @ b) ** 2) / max(len(x) - 2, 1)
    return b, np.sqrt(np.diag(cov)), chi2

def clipped_fit(x, y, e, floor):
    keep = np.ones(len(x), bool)
    for _ in range(4):
        b, _, _ = wlin(x[keep], y[keep], e[keep])
        res = y - (b[0] * x + b[1])
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826 + 1e-6
        new = np.abs(res - np.median(res[keep])) < max(floor, 5 * mad)
        if (new == keep).all():
            break
        keep = new
    b, be, chi2 = wlin(x[keep], y[keep], e[keep])
    return b, be * np.sqrt(max(chi2, 1)), keep

rows = []
for t in TICS:
    jp = f'rerun_fixedP/TIC_{t}_out/TIC{t}_summary.json'
    jp = jp if os.path.exists(jp) else f'TIC_{t}_out/TIC{t}_summary.json'
    if not os.path.exists(jp):
        print(f"{t}: no summary"); continue
    s = json.load(open(jp))
    P, T0, ph, S = s['period_days'], s['t0_btjd'], s['secondary_phase'], s['apsidal_slope_min_per_1000d']
    T, F = load(t)
    if T is None:
        print(f"{t}: no light curves"); continue
    hw = min(1.0, 0.2 * P)
    ns = range(int((T.min() - T0) / P) - 1, int((T.max() - T0) / P) + 2)
    pri = {n: r for n in ns if (r := mirror_time(T, F, T0 + n * P, hw))}
    sec = {n: r for n in ns if (r := mirror_time(T, F, T0 + (n + ph) * P, hw))}
    if len(pri) < 3 or len(sec) < 3:
        print(f"{t}: too few timed eclipses ({len(pri)} pri, {len(sec)} sec)"); continue
    n_p = np.array(sorted(pri), float); tp = np.array([pri[int(n)][0] for n in n_p]); ep = np.array([pri[int(n)][1] for n in n_p])
    (Pf, T0f), _, kp = clipped_fit(n_p, tp, ep, 10 / 1440)
    n_s = np.array(sorted(sec), float); ts = np.array([sec[int(n)][0] for n in n_s]); es = np.array([sec[int(n)][1] for n in n_s]) * 1440
    d = (ts - (T0f + n_s * Pf)) * 1440
    (sl, _), (sle, _), ks = clipped_fit(ts, d, es, 10)
    sl, sle = sl * 1000, sle * 1000
    z = (S - sl) / np.hypot(sle, s.get('apsidal_slope_err_min_per_1000d') or 0)
    flag = 'DISAGREE' if abs(z) > 3 and abs(S - sl) > 0.3 * max(abs(S), abs(sl)) else ''
    rows.append(dict(tic=t, P=P, n_pri=int(kp.sum()), n_sec=int(ks.sum()), pipeline=S, mirror=sl,
                     mirror_err=sle, z=z, flag=flag))
    print(f"{t:>10}  P {P:7.3f}  pri {kp.sum():3d}/{len(kp):<3d} sec {ks.sum():3d}/{len(ks):<3d}  "
          f"pipeline {S:+8.2f}  mirror {sl:+8.2f} +/- {sle:5.2f}  z {z:+6.1f}  {flag}", flush=True)

with open('retime_all.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nDISAGREE: {[r['tic'] for r in rows if r['flag']]}")
print("wrote retime_all.csv")

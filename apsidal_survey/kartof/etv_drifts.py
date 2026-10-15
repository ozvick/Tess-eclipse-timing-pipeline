import pickle, glob, re, csv
import numpy as np
from scipy.optimize import brentq

def pct(t, q):
    lo, hi = t['below_integral'], t['above_integral']
    mlo = -float(np.ravel(lo(lo.t_min))[0])
    tot = mlo + float(np.ravel(hi(hi.t_max))[0])
    target = q * tot
    if target <= mlo:
        return brentq(lambda x: float(np.ravel(lo(x))[0]) + mlo - target, lo.t_min, lo.t_max)
    return brentq(lambda x: float(np.ravel(hi(x))[0]) + mlo - target, hi.t_min, hi.t_max)

def wfit(x, y, s):
    x, y, s = map(np.asarray, (x, y, s))
    if len(x) < 4:
        return None
    w = 1 / s**2
    A = np.vstack([x, np.ones_like(x)]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    beta = cov @ (A.T @ (w * y))
    chi2 = np.sum(w * (y - A @ beta)**2) / max(len(x) - 2, 1)
    return beta[0], np.sqrt(cov[0, 0] * max(chi2, 1)), chi2

out = []
files = sorted(glob.glob('/mnt/md2/TESS_EBs/ETV/tess*_etv_distros_none.pkl') +
               glob.glob('/mnt/md2/TESS_EBs/ETV/tess*_etv_distros_spline.pkl'))
for k, f in enumerate(files):
    m = re.search(r'tess(\d+)_etv_distros_(none|spline)\.pkl', f)
    if not m:
        continue
    tic, variant = int(m.group(1)), m.group(2)
    try:
        d = pickle.load(open(f, 'rb'))
    except Exception as e:
        print(f"load failed {f}: {e}")
        continue
    n1 = d['num_primary_tasks']
    pts = {'p': ([], [], []), 's': ([], [], [])}
    for i, (t, typ) in enumerate(zip(d['distributions'], d['task_types'])):
        if typ != 'eclipse':
            continue
        try:
            p16, p50, p84 = (pct(t, q) for q in (0.16, 0.5, 0.84))
        except Exception:
            continue
        sig = (p84 - p16) / 2
        if not sig > 0:
            continue
        key = 'p' if i < n1 else 's'
        pts[key][0].append(abs(int(t['eclipse_indices'][0])))
        pts[key][1].append(p50)
        pts[key][2].append(sig)
    fp, fs = wfit(*pts['p']), wfit(*pts['s'])
    row = dict(tic=tic, variant=variant, n_p=len(pts['p'][0]), n_s=len(pts['s'][0]))
    if fp:
        row.update(slope_p=fp[0], err_p=fp[1], chi2_p=fp[2])
    if fs:
        row.update(slope_s=fs[0], err_s=fs[1], chi2_s=fs[2])
    if fp and fs:
        row.update(sp=fs[0] - fp[0], sp_err=np.hypot(fp[1], fs[1]))
    out.append(row)
    if (k + 1) % 50 == 0:
        print(f"{k + 1}/{len(files)} files done")

keys = ['tic', 'variant', 'n_p', 'n_s', 'slope_p', 'err_p', 'chi2_p',
        'slope_s', 'err_s', 'chi2_s', 'sp', 'sp_err']
with open('/home/victor/penev_etv_drifts.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(out)
print(f"wrote {len(out)} rows")

P = 3.15882
for r in out:
    if r['tic'] == 197755658 and 'sp' in r:
        conv = 1440 / P * 1000
        print(f"CHECK 197755658 [{r['variant']}]: s-p = {r['sp'] * conv:+.2f} +/- "
              f"{r['sp_err'] * conv:.2f} min/1000d  (yours: -47.98 +/- 0.22)")

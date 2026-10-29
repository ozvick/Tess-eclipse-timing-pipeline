"""On kartof: secondary-minus-primary drifts from Penev's ETV pickles (new or old run).

Usage:
  ~/venv_etv/bin/python ~/etv_drifts_v2.py [ETV_DIR] [OUT_CSV] [PATTERN]
Defaults: /mnt/md2/TESS_EBs/ETV   ~/penev_etv_drifts_v2.csv   tess*_etv_distros_none.pkl

Handles pickles that store either plain dicts (mode / a / b / above_integral / below_integral)
or TimeShiftDistribution objects (anything with .ppf). Uses single-eclipse tasks only.
Per system it records: n primary/secondary, drift per epoch (days/epoch, Penev's sign),
errors scaled by sqrt(reduced chi2) when > 1, the reduced chi2 of both straight-line fits
(for the likelihood-width check), and the file's modification date (to tell new from old).
"""
import sys, glob, re, csv, os, pickle, datetime, logging
import numpy as np
from scipy.optimize import brentq

logging.disable(logging.CRITICAL)
sys.path.insert(0, '/home/victor/code')
sys.path.insert(0, '/home/kpenev/projects/git/TESS_EB_Bayesian/scripts')

etv_dir = sys.argv[1] if len(sys.argv) > 1 else '/mnt/md2/TESS_EBs/ETV'
out_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/penev_etv_drifts_v2.csv')
pattern = sys.argv[3] if len(sys.argv) > 3 else 'tess*_etv_distros_none.pkl'


def quantiles(t):
    if hasattr(t, 'ppf'):
        return tuple(float(t.ppf(q)) for q in (0.16, 0.5, 0.84))
    lo, hi = t['below_integral'], t['above_integral']
    mlo = -float(np.ravel(lo(lo.t_min))[0])
    tot = mlo + float(np.ravel(hi(hi.t_max))[0])
    out = []
    for q in (0.16, 0.5, 0.84):
        target = q * tot
        if target <= mlo:
            out.append(brentq(lambda x: float(np.ravel(lo(x))[0]) + mlo - target, lo.t_min, lo.t_max))
        else:
            out.append(brentq(lambda x: float(np.ravel(hi(x))[0]) + mlo - target, hi.t_min, hi.t_max))
    return tuple(out)


def wfit(x, y, s):
    x, y, s = map(np.asarray, (x, y, s))
    if len(x) < 4:
        return None
    w = 1 / s**2
    A = np.vstack([x, np.ones_like(x)]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    beta = cov @ (A.T @ (w * y))
    chi2 = np.sum(w * (y - A @ beta) ** 2) / max(len(x) - 2, 1)
    return beta[0], np.sqrt(cov[0, 0] * max(chi2, 1)), chi2


files = sorted(glob.glob(os.path.join(etv_dir, pattern)))
print(f"{len(files)} files matching {pattern} in {etv_dir}")
rows, failed, formats = [], 0, set()
for k, f in enumerate(files):
    m = re.search(r'tess(\d+)_', os.path.basename(f))
    if not m:
        continue
    tic = int(m.group(1))
    try:
        d = pickle.load(open(f, 'rb'))
    except Exception as e:
        failed += 1
        if failed <= 3:
            print(f"load failed {os.path.basename(f)}: {e!r}")
        continue
    formats.add(type(d['distributions'][0]).__name__ if d.get('distributions') else 'empty')
    n1 = d['num_primary_tasks']
    pts = {'p': ([], [], []), 's': ([], [], [])}
    for i, (t, typ) in enumerate(zip(d['distributions'], d['task_types'])):
        if typ != 'eclipse':
            continue
        try:
            p16, p50, p84 = quantiles(t)
        except Exception:
            continue
        sig = (p84 - p16) / 2
        if not sig > 0:
            continue
        idx = t['eclipse_indices'][0] if isinstance(t, dict) else getattr(t, 'eclipse_indices', [None])[0]
        if idx is None:
            continue
        key = 'p' if i < n1 else 's'
        pts[key][0].append(abs(int(idx))); pts[key][1].append(p50); pts[key][2].append(sig)
    fp, fs = wfit(*pts['p']), wfit(*pts['s'])
    row = dict(tic=tic, n_p=len(pts['p'][0]), n_s=len(pts['s'][0]),
               modified=datetime.date.fromtimestamp(os.path.getmtime(f)).isoformat())
    if fp:
        row.update(slope_p=fp[0], err_p=fp[1], chi2_p=fp[2])
    if fs:
        row.update(slope_s=fs[0], err_s=fs[1], chi2_s=fs[2])
    if fp and fs:
        row.update(sp=fs[0] - fp[0], sp_err=np.hypot(fp[1], fs[1]))
    rows.append(row)
    if (k + 1) % 50 == 0:
        print(f"{k + 1}/{len(files)} files done", flush=True)

keys = ['tic', 'n_p', 'n_s', 'slope_p', 'err_p', 'chi2_p', 'slope_s', 'err_s', 'chi2_s', 'sp', 'sp_err', 'modified']
with open(out_csv, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)

cp = [r['chi2_p'] for r in rows if 'chi2_p' in r]
cs = [r['chi2_s'] for r in rows if 'chi2_s' in r]
print(f"\nwrote {len(rows)} systems to {out_csv}  (load failures: {failed})")
print(f"distribution format(s) found: {sorted(formats)}")
if cp and cs:
    print(f"median reduced chi2: primary {np.median(cp):.2f}, secondary {np.median(cs):.2f}  "
          f"(about 2 = likelihood still exp(-chi2); about 1 = factor 1/2 fixed)")
dates = sorted({r['modified'] for r in rows})
print(f"file dates: {dates[0]} ... {dates[-1]}" if dates else "no files")

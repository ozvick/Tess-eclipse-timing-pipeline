"""Time every eclipse of a short-baseline system with a phase-folded template (no minimum number of
eclipses), and save primary/secondary O-C files in the pipeline's format for make_etv_pdfs.py.

Usage: python3 time_eclipses_simple.py TIC PERIOD T0 OUTDIR
   e.g. python3 time_eclipses_simple.py 72782654 6.63113319 1477.871181 paul_v2
"""
import glob, os, sys
import numpy as np
from astropy.io import fits

tic, P, T0, OUT = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]

# ---- light curve: SPOC PDCSAP, good quality, each sector normalized to its median
T, F = [], []
for f in sorted(glob.glob(f'TIC_{tic}_fits/**/*_lc.fits', recursive=True)):
    d = fits.getdata(f, 1)
    ok = np.isfinite(d['TIME']) & np.isfinite(d['PDCSAP_FLUX']) & (d['QUALITY'] == 0)
    T.append(d['TIME'][ok]); F.append(d['PDCSAP_FLUX'][ok] / np.nanmedian(d['PDCSAP_FLUX'][ok]))
T, F = np.concatenate(T), np.concatenate(F)
ph = ((T - T0) / P) % 1.0

# ---- eclipse phases and half-widths from a binned fold
nb = 2000
edges = np.linspace(0, 1, nb + 1)
idx = np.clip(np.digitize(ph, edges) - 1, 0, nb - 1)
binned = np.array([np.median(F[idx == i]) if np.any(idx == i) else np.nan for i in range(nb)])
centers = 0.5 * (edges[1:] + edges[:-1])
base = np.nanmedian(binned)
mask = (centers > 0.1) & (centers < 0.9)
phi_sec = centers[mask][np.nanargmin(binned[mask])]


def half_width(phi0):
    dphi = ((centers - phi0 + 0.5) % 1.0) - 0.5
    depth = base - np.nanmin(binned[np.abs(dphi) < 0.02])
    inside = np.abs(dphi[(binned < base - 0.03 * depth)])
    inside = inside[inside < 0.2]
    return inside.max() if len(inside) else 0.02


eclipses = {'primary': (0.0, half_width(0.0)), 'secondary': (phi_sec, half_width(phi_sec))}
print(f"secondary at phase {phi_sec:.4f}; half-widths: primary {eclipses['primary'][1]:.4f}, "
      f"secondary {eclipses['secondary'][1]:.4f}")
os.makedirs(f'{OUT}/TIC_{tic}_out', exist_ok=True)

for kind, (phi0, hw) in eclipses.items():
    win = 1.5 * hw * P                                   # half-window in days, includes baseline
    dt_all = ((ph - phi0 + 0.5) % 1.0 - 0.5) * P
    sel = np.abs(dt_all) < win
    order = np.argsort(dt_all[sel])
    xs, ys = dt_all[sel][order], F[sel][order]
    tb = np.linspace(-win, win, 301)                     # template: running median on a fine grid
    tmpl = np.array([np.median(ys[np.abs(xs - x) < win / 60]) if np.any(np.abs(xs - x) < win / 60) else np.nan
                     for x in tb])
    good = np.isfinite(tmpl)
    template = lambda x: np.interp(x, tb[good], tmpl[good])
    rows = []
    n0, n1 = int(np.floor((T.min() - T0) / P)) - 1, int(np.ceil((T.max() - T0) / P)) + 1
    for n in range(n0, n1 + 1):
        tc = T0 + (n + phi0) * P
        m = np.abs(T - tc) < win
        if m.sum() < 30 or np.abs(T[m] - tc).min() > 0.2 * hw * P:
            continue                                     # eclipse not covered by data
        tt, ff = T[m] - tc, F[m]
        if (tt < -0.5 * hw * P).sum() < 5 or (tt > 0.5 * hw * P).sum() < 5:
            continue                                     # need both ingress and egress
        shifts = np.linspace(-0.3 * hw * P, 0.3 * hw * P, 601)
        chi = np.array([np.sum((ff - template(tt - s)) ** 2) for s in shifts])
        k = int(np.clip(np.argmin(chi), 1, len(chi) - 2))
        a, b, c = np.polyfit(shifts[k - 1:k + 2], chi[k - 1:k + 2], 2)
        best = -b / (2 * a)
        sig2 = chi[k] / max(m.sum() - 1, 1)              # flux noise variance from residuals
        err = np.sqrt(2 * sig2 / (2 * a)) if a > 0 else np.nan
        rows.append((tc + best, best * 1440, err * 1440))
    rows = np.array(rows)
    path = f'{OUT}/TIC_{tic}_out/TIC{tic}_{kind}_oc.csv'
    np.savetxt(path, rows, delimiter=',', header='time_btjd,oc_min,oc_err_min', comments='', fmt='%.6f')
    print(f"{kind}: timed {len(rows)} eclipses, O-C scatter {np.std(rows[:, 1]) if len(rows) else float('nan'):.3f} min "
          f"-> {path}")

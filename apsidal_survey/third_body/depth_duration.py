"""Do eclipse depths and durations change over time? (Task 1)

For each eclipse type, builds the average eclipse shape from all eclipses, then fits every
individual eclipse with that shape scaled in depth (D) and stretched in duration (W), allowing a
small timing shift. Reports per-season medians and linear trends (percent per 1000 d).
Apsidal motion: durations of primary and secondary change in OPPOSITE directions, depths ~constant.
Orbit tilting by a third body: depths change, both eclipses in the SAME direction.

Usage: python3 depth_duration.py OUTDIR TIC      e.g. python3 depth_duration.py paul_v2 468792279
Needs TIC_<tic>_fits/ light curves and OUTDIR/TIC_<tic>_out/ (summary JSON + O-C files).
"""
import glob, json, sys
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT, tic = sys.argv[1], int(sys.argv[2])
d = f'{OUT}/TIC_{tic}_out'
s = json.load(open(f'{d}/TIC{tic}_summary.json'))
P = s['period_days']

T, F = [], []
for f in sorted(glob.glob(f'TIC_{tic}_fits/**/*_lc.fits', recursive=True)):
    x = fits.getdata(f, 1)
    ok = np.isfinite(x['TIME']) & np.isfinite(x['PDCSAP_FLUX']) & (x['QUALITY'] == 0)
    T.append(x['TIME'][ok]); F.append(x['PDCSAP_FLUX'][ok] / np.nanmedian(x['PDCSAP_FLUX'][ok]))
T, F = np.concatenate(T), np.concatenate(F)


def eclipse_segments(times, hw_days):
    """Locally normalized flux around each eclipse, in units of the half-width."""
    segs = []
    for tc in times:
        m = np.abs(T - tc) < 2.0 * hw_days
        if m.sum() < 30:
            continue
        x, y = (T[m] - tc) / hw_days, F[m]
        out = np.abs(x) > 1.2
        if out.sum() < 8 or (x < -0.6).sum() < 5 or (x > 0.6).sum() < 5:
            continue
        c = np.polyfit(x[out], y[out], 1)            # linear baseline from out-of-eclipse points
        segs.append((tc, x, y / np.polyval(c, x)))
    return segs


rows_all, results = {}, {}
for kind in ('primary', 'secondary'):
    hw_phase = s.get(f'{kind}_half_width_phase')
    oc_file = f'{d}/TIC{tic}_{kind}_oc.csv'
    if hw_phase is None or not glob.glob(oc_file):
        print(f'{kind}: missing half-width or O-C file, skipped'); continue
    hw = hw_phase * P
    tc_list = np.atleast_2d(np.genfromtxt(oc_file, delimiter=',', skip_header=1))[:, 0]
    segs = eclipse_segments(tc_list, hw)
    if len(segs) < 3:
        print(f'{kind}: only {len(segs)} usable eclipses, skipped'); continue
    xs = np.concatenate([g[1] for g in segs]); ys = np.concatenate([g[2] for g in segs])
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    grid = np.linspace(-1.6, 1.6, 321)
    shape = np.array([np.median(1 - ys[np.abs(xs - g) < 0.02]) if np.any(np.abs(xs - g) < 0.02) else np.nan
                      for g in grid])
    ok = np.isfinite(shape); shape = np.clip(shape, 0, None)
    depth0 = shape[ok].max(); shape = shape / depth0
    S = lambda x: np.interp(x, grid[ok], shape[ok], left=0, right=0)
    rows = []
    for tc, x, y in segs:
        best = None
        for W in np.linspace(0.85, 1.15, 241):
            for dt in np.linspace(-0.08, 0.08, 33):
                m = S((x - dt) / W)
                Dfit = np.sum(m * (1 - y)) / max(np.sum(m * m), 1e-12)
                chi = np.sum((1 - y - Dfit * m) ** 2)
                if best is None or chi < best[0]:
                    best = (chi, W, Dfit)
        rows.append((tc, best[2], best[1]))
    rows = np.array(rows)
    rows_all[kind] = rows
    tt, D, W = rows[:, 0], rows[:, 1], rows[:, 2]
    trend = {}
    for name, v in (('depth', D), ('duration', W)):
        c, cov = np.polyfit(tt - tt.mean(), v, 1, cov=True) if len(tt) > 3 else (np.polyfit(tt - tt.mean(), v, 1), None)
        err = np.sqrt(cov[0, 0]) if cov is not None else np.nan
        trend[name] = (c[0] / np.mean(v) * 1000 * 100, err / np.mean(v) * 1000 * 100)
    results[kind] = trend
    print(f"\n{kind}: {len(rows)} eclipses, mean depth {np.mean(D):.4f}")
    cuts = np.where(np.diff(tt) > 10)[0] + 1
    for g_t, g_D, g_W in zip(np.split(tt, cuts), np.split(D, cuts), np.split(W, cuts)):
        eD = np.std(g_D) / np.sqrt(len(g_D)) if len(g_D) > 1 else np.nan
        eW = np.std(g_W) / np.sqrt(len(g_W)) if len(g_W) > 1 else np.nan
        print(f"  season BTJD {g_t.mean():7.0f} (n={len(g_t):2d}): depth {np.median(g_D):.4f} +/- {eD:.4f}   "
              f"duration x{np.median(g_W):.4f} +/- {eW:.4f}")
    for name, (v, e) in trend.items():
        print(f"  {name} trend: {v:+.2f} +/- {e:.2f} % per 1000 d")

if results:
    print('\nSummary (% per 1000 d):')
    for kind, tr in results.items():
        print(f"  {kind:9s} depth {tr['depth'][0]:+6.2f} +/- {tr['depth'][1]:.2f}   "
              f"duration {tr['duration'][0]:+6.2f} +/- {tr['duration'][1]:.2f}")
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    col = {'primary': 'tab:blue', 'secondary': 'tab:orange'}
    for kind, r in rows_all.items():
        ax[0].plot(r[:, 0], r[:, 1] / np.mean(r[:, 1]), 'o', ms=4, color=col[kind], label=kind)
        ax[1].plot(r[:, 0], r[:, 2], 'o', ms=4, color=col[kind], label=kind)
    ax[0].set_ylabel('depth / mean depth'); ax[1].set_ylabel('duration / mean duration')
    ax[1].set_xlabel('Time (BTJD)'); ax[0].legend(fontsize=8)
    for a in ax:
        a.axhline(1, color='0.6', lw=0.7)
    ax[0].set_title(f'TIC {tic}: eclipse depth and duration over time', fontsize=10)
    fig.tight_layout(); fig.savefig(f'depth_duration_{tic}.png', dpi=150)
    print(f'saved depth_duration_{tic}.png')

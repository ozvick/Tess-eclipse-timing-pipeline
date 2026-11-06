"""Look at the eclipses directly: are the primary and secondary durations equal, or do they differ as
Penev's e sin(omega) requires?

For each system: folds the light curve on the pipeline's final ephemeris, cuts out the primary and
the secondary eclipse, normalizes each to its own baseline and depth (0 = out of eclipse, 1 = bottom),
and overlays them against time from mid-eclipse. It then measures the duration ratio directly by
stretching the primary's profile to match the secondary's (also allowing a small timing offset),
and draws the secondary shape that Penev's |e sin w| would predict, ratio (1 + |e sin w|)/(1 - |e sin w|),
for comparison. Uses the pipeline's own loader (apsidal_pipeline.py in this folder).

Usage: python3 plot_eclipse_overlay.py [TIC ...]      (default: the systems flagged in the v2.2 comparison)
Writes eclipse_overlay.png and eclipse_overlay.csv.
"""
import csv, contextlib, io, json, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import apsidal_pipeline as ap

DEFAULT = [158728853, 269852699, 99028319, 281498280, 436592342, 199420503,
           162585265, 84342725, 130158361, 329770815, 121568625]
TICS = [int(x) for x in sys.argv[1:]] or DEFAULT
trig = {int(float(r['tic_id'])): r for r in csv.DictReader(open('tic_etrigwquants.csv'))}
widths = {int(r['tic']): r for r in csv.DictReader(open('widths_v22.csv'))} if os.path.exists('widths_v22.csv') else {}


def profile(time, flux, center_phase, P, T0, hw):
    """Eclipse cut out around center_phase: time from mid-eclipse [h], normalized depth (0 = baseline, 1 = bottom)."""
    ph = ((time - T0) / P - center_phase + 0.5) % 1.0 - 0.5
    m = np.abs(ph) < 3 * hw
    x, y = ph[m] * P * 24, flux[m]
    base = np.median(y[np.abs(ph[m]) > 1.8 * hw])
    nb = 160
    edges = np.linspace(-3 * hw * P * 24, 3 * hw * P * 24, nb + 1)
    idx = np.digitize(x, edges) - 1
    med = np.array([np.median(y[idx == i]) if np.sum(idx == i) > 2 else np.nan for i in range(nb)])
    depth = base - np.nanmin(med)
    return x, (base - y) / depth, 0.5 * (edges[1:] + edges[:-1]), (base - med) / depth


def best_stretch(xp, yp, xs, ys, hw_h):
    """Stretch k and shift dt so that primary_profile((x - dt)/k) matches the secondary profile."""
    ok = np.isfinite(yp)
    tmpl = lambda x: np.interp(x, xp[ok], yp[ok], left=0, right=0)
    best = (np.inf, 1.0, 0.0)
    for k in np.linspace(0.6, 1.6, 501):
        for dt in np.linspace(-0.1 * hw_h, 0.1 * hw_h, 21):
            chi = np.nansum((ys - tmpl((xs - dt) / k)) ** 2)
            if chi < best[0]:
                best = (chi, k, dt)
    return best[1], best[2]


rows = []
n = len(TICS); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)
for ax, t in zip(axes.flat, TICS):
    p = f'rerun_v2/TIC_{t}_out/TIC{t}_summary.json'
    data = next((d for d in (f'TIC_{t}_fits', f'TIC{t}_fits') if os.path.isdir(d)), None)
    if not os.path.exists(p) or data is None:
        ax.set_title(f'TIC {t}: no data'); ax.axis('off'); continue
    s = json.load(open(p))
    P, T0, phs = s['period_days'], s['t0_btjd'], s['secondary_phase']
    hw = max(float(widths[t]['hw_p']), float(widths[t]['hw_s'])) if t in widths else \
        max(s['primary_half_width_phase'], s['secondary_half_width_phase'])
    with contextlib.redirect_stdout(io.StringIO()):
        time, flux, _, _ = ap.load_and_stitch(data)
    _, _, xp, yp = profile(time, flux, 0.0, P, T0, hw)
    xs_raw, ys_raw, xs, ys = profile(time, flux, phs, P, T0, hw)
    k, dt = best_stretch(xp, yp, xs, ys, hw * P * 24)
    es_pen = abs(float(trig[t]['esinw_Q0.5'])) if t in trig else np.nan
    r_pen = (1 + es_pen) / (1 - es_pen)
    ok = np.isfinite(yp)
    tm = lambda x: np.interp(x, xp[ok], yp[ok], left=0, right=0)
    ax.plot(xp, yp, '-', color='tab:blue', lw=2, label='primary')
    ax.plot(xs - dt, ys, '-', color='tab:orange', lw=2, label=f'secondary (measured ratio {k:.3f})')
    ax.plot(xp * r_pen, tm(xp), '--', color='k', lw=1, label=f"Penev's e sin w: ratio {r_pen:.3f}")
    ax.plot(xp / r_pen, tm(xp), ':', color='k', lw=1)
    ax.invert_yaxis()
    ax.set_xlim(-2.2 * hw * P * 24, 2.2 * hw * P * 24)
    ax.set_title(f'TIC {t}  (P = {P:.3f} d)', fontsize=10)
    ax.set_xlabel('time from mid-eclipse (h)'); ax.set_ylabel('normalized depth')
    ax.legend(fontsize=7, loc='lower center')
    rows.append(dict(tic=t, P=P, esinw_penev=es_pen, ratio_penev=r_pen, ratio_measured=k,
                     esinw_from_ratio=abs((k - 1) / (k + 1)), shift_h=dt))
    print(f"TIC {t:>10}  P={P:6.3f}  Penev |e sin w| {es_pen:.4f} -> ratio {r_pen:.3f}   "
          f"measured ratio {k:.3f} -> |e sin w| {abs((k - 1) / (k + 1)):.4f}", flush=True)
for ax in list(axes.flat)[len(TICS):]:
    ax.axis('off')
fig.suptitle('Primary vs secondary eclipse, scaled to the same depth (dashed/dotted: durations implied by Penev\'s e sin w)',
             fontsize=11)
fig.tight_layout(); fig.savefig('eclipse_overlay.png', dpi=140)
if rows:
    with open('eclipse_overlay.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
print('saved eclipse_overlay.png and eclipse_overlay.csv')

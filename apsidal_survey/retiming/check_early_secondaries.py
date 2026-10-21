import glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

t = 289044378
s = json.load(open(f'TIC_{t}_out/TIC{t}_summary.json'))
P, T0, ph = s['period_days'], s['t0_btjd'], s['secondary_phase']
SLOPE, TREF = s['apsidal_slope_min_per_1000d'] / 1000, 3180.0   # min/day, BTJD where O-C ~ 0

files = sorted(set(glob.glob(f'TIC_{t}_fits/**/*_lc.fits', recursive=True) +
                   glob.glob(f'TIC{t}_fits/**/*_lc.fits', recursive=True)))
T, F, SEC = [], [], []
for f in files:
    with fits.open(f) as h:
        d, sec = h[1].data, h[0].header.get('SECTOR', -1)
        tt, ff, q = d['TIME'], d['PDCSAP_FLUX'], d['QUALITY']
        ok = np.isfinite(tt) & np.isfinite(ff) & (q == 0)
        T.append(tt[ok]); F.append(ff[ok] / np.nanmedian(ff[ok])); SEC.append(np.full(ok.sum(), sec))
T, F, SEC = map(np.concatenate, (T, F, SEC))
print(f"{len(files)} light-curve files, BTJD {T.min():.0f}-{T.max():.0f}")

wins = []
for n in range(int((T.min() - T0) / P) - 1, int((T.max() - T0) / P) + 2):
    tsec = T0 + (n + ph) * P
    m = np.abs(T - tsec) < 1.0
    if m.sum() >= 20:
        wins.append((n, tsec, m))

print(f"\n{'epoch':>6} {'BTJD':>8} {'sector':>6} {'points':>6} {'depth':>6} {'pred. shift [min]':>18}")
cols = 4
rows = (len(wins) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)
for ax, (n, tsec, m) in zip(axes.flat, wins):
    shift = SLOPE * (tsec - TREF)
    x = (T[m] - tsec) * 24
    ax.plot(x, F[m], '.', ms=2)
    ax.axvline(0, color='0.5', ls='--', lw=1)
    ax.axvline(shift / 60, color='red', lw=1)
    ax.set_title(f"epoch {n}  BTJD {tsec:.0f}  S{int(SEC[m][0])}", fontsize=9)
    ax.set_xlim(-24, 24)
    depth = 1 - np.percentile(F[m], 2)
    print(f"{n:6d} {tsec:8.1f} {int(SEC[m][0]):6d} {m.sum():6d} {depth:6.3f} {shift:18.1f}")
for ax in list(axes.flat)[len(wins):]:
    ax.axis('off')
for ax in axes[-1]:
    ax.set_xlabel('hours from nominal secondary')
fig.suptitle(f'TIC {t}: secondary eclipse windows (grey = no drift, red = drift extrapolated)')
fig.tight_layout()
fig.savefig(f'secondary_windows_{t}.png', dpi=110)
print(f"\nsaved secondary_windows_{t}.png")

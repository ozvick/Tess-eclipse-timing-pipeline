import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tics = [int(x) for x in sys.argv[1:]]
data = {}
for t in tics:
    p = f'rerun_v2/TIC_{t}_out/TIC{t}_secondary_oc.csv'
    if not os.path.exists(p):
        print(f'{t}: no secondary O-C file'); continue
    a = np.loadtxt(p, delimiter=',', skiprows=1, ndmin=2)
    tt, oc = a[:, 0], a[:, 1]
    sl, ic = np.polyfit(tt, oc, 1)
    res = oc - (sl * tt + ic)
    cuts = np.where(np.diff(tt) > 10)[0] + 1
    segs = [(g[0].mean(), g[1].mean(), g[1].std() / np.sqrt(len(g[1])) if len(g[1]) > 1 else np.nan)
            for g in zip(np.split(tt, cuts), np.split(res, cuts))]
    data[t] = dict(t=tt, oc=oc, slope=sl * 1000, res=res, segs=segs)

fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for t, d in data.items():
    ax[0].plot(d['t'], d['oc'] - d['oc'].mean(), 'o', ms=3, label=f"{t} ({d['slope']:+.2f}/1000 d)")
    st = np.array([s[0] for s in d['segs']]); sm = np.array([s[1] for s in d['segs']])
    se = np.array([s[2] for s in d['segs']])
    ax[1].errorbar(st, sm, se, fmt='o-', ms=5, label=str(t))
ax[0].set_ylabel('secondary O-C [min] (mean removed)'); ax[0].legend(fontsize=8)
ax[1].axhline(0, color='0.6'); ax[1].set_ylabel('season mean after removing own line [min]')
ax[1].set_xlabel('BTJD'); ax[1].legend(fontsize=8)
fig.suptitle('Same timing wiggles at the same times = shared systematic')
fig.tight_layout()
out = 'group_' + '_'.join(map(str, tics)) + '.png'
fig.savefig(out, dpi=120)

ids = list(data)
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        a, b = data[ids[i]]['segs'], data[ids[j]]['segs']
        pairs = [(x[1], y[1]) for x in a for y in b if abs(x[0] - y[0]) < 15]
        if len(pairs) >= 3:
            r = np.corrcoef(*zip(*pairs))[0, 1]
            print(f"{ids[i]} vs {ids[j]}: {len(pairs)} shared seasons, correlation of season residuals r = {r:+.2f}")
        else:
            print(f"{ids[i]} vs {ids[j]}: only {len(pairs)} shared seasons, too few to compare")
print(f"saved {out}")

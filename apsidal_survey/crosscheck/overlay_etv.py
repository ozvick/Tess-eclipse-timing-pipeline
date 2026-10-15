import csv, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def wline(x, y, e):
    w = 1 / e**2
    A = np.vstack([x, np.ones_like(x)]).T
    return np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (w * y))

def wmean(y, e):
    w = 1 / e**2
    return np.sum(w * y) / np.sum(w), 1 / np.sqrt(np.sum(w))

pen = {}
for r in csv.DictReader(open('penev_etv_points.csv')):
    pen.setdefault(int(r['tic']), []).append(r)

for t in [329770815, 257737073, 358232450, 18271803, 155540827, 224308601]:
    d = f'timing_dump/TIC_{t}_out'
    s = json.load(open(f'{d}/TIC{t}_summary.json'))
    P = s['period_days']
    vp = np.loadtxt(f'{d}/TIC{t}_primary_oc.csv', delimiter=',', skiprows=1, ndmin=2)
    vs_path = f'{d}/TIC{t}_secondary_oc.csv'
    vs = np.loadtxt(vs_path, delimiter=',', skiprows=1, ndmin=2) if os.path.exists(vs_path) else None
    c = vp[0, 1] % P

    rows = pen.get(t, [])
    def arr(kind):
        sel = [r for r in rows if r['kind'] == kind]
        ep = np.array([int(r['epoch']) for r in sel], float)
        y = -np.array([float(r['p50']) for r in sel]) * 1440
        e = np.array([(float(r['p84']) - float(r['p16'])) / 2 for r in sel]) * 1440
        off = 0.0 if kind == 'p' else 0.5 * P
        return c + ep * P + off, y, e
    tp, yp, ep_ = arr('p')
    ts, ys, es = arr('s')

    if len(tp) and (tp.max() < vp[:, 1].min() - 50 or tp.min() > vp[:, 1].max() + 50):
        print(f"WARNING TIC {t}: Penev times {tp.min():.0f}-{tp.max():.0f} do not overlap yours "
              f"{vp[:, 1].min():.0f}-{vp[:, 1].max():.0f}; time alignment is off")

    b = wline(tp, yp, ep_)
    yp_f = yp - (b[0] * tp + b[1])
    ys_f = ys - (b[0] * ts + b[1])
    ps = wline(ts, ys_f, es)[0] * 1000 if len(ts) > 3 else np.nan

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].errorbar(tp, yp_f, ep_, fmt='.', color='0.6', ms=4, label='Penev (-1x, primary trend removed)')
    ax[0].plot(vp[:, 1], vp[:, 2], 'o', ms=4, color='tab:blue', label='Victor')
    ax[0].set_ylabel('primary O-C [min]')
    ax[0].legend(fontsize=8)
    if len(ts):
        ax[1].errorbar(ts, ys_f - wmean(ys_f, es)[0], es, fmt='.', color='0.6', ms=4, label='Penev')
    if vs is not None:
        ax[1].plot(vs[:, 0], vs[:, 1] - vs[:, 1].mean(), 'o', ms=4, color='tab:red', label='Victor')
    ax[1].set_ylabel('secondary O-C [min] (mean removed)')
    ax[1].set_xlabel('BTJD')
    ax[1].legend(fontsize=8)
    vS = s.get('apsidal_slope_min_per_1000d')
    fig.suptitle(f"TIC {t}: Victor {vS if vS is None else round(vS, 3)}  vs  Penev {ps:+.3f} min/1000d")
    fig.tight_layout()
    fig.savefig(f'timing_dump/overlay_TIC{t}.png', dpi=110)
    plt.close(fig)

    allt = np.sort(np.concatenate([vp[:, 1], tp, ts] + ([vs[:, 0]] if vs is not None else [])))
    edges = [allt[0]] + [allt[i + 1] for i in np.where(np.diff(allt) > 20)[0]]
    gid = lambda x: np.searchsorted(edges, x, side='right') - 1
    print(f"\nTIC {t}   (Victor slope {vS}, Penev slope {ps:+.3f} min/1000d)")
    print(f"  {'BTJD':>6}  {'nV':>3} {'Victor sec':>13}  {'nP':>3} {'Penev sec':>13}  {'V-P':>6}")
    for g in range(len(edges)):
        vv = vs[gid(vs[:, 0]) == g, 1] - vs[:, 1].mean() if vs is not None else np.array([])
        m = gid(ts) == g
        mid = edges[g]
        vtxt = f"{vv.mean():+6.2f}+/-{vv.std() / np.sqrt(len(vv)):4.2f}" if len(vv) > 1 else '      --     '
        if m.sum() > 0:
            pm, pe = wmean(ys_f[m] - wmean(ys_f, es)[0], es[m])
            ptxt = f"{pm:+6.2f}+/-{pe:4.2f}"
        else:
            pm, ptxt = None, '      --     '
        dtxt = f"{vv.mean() - pm:+6.2f}" if len(vv) > 1 and pm is not None else '    --'
        print(f"  {mid:6.0f}  {len(vv):3d} {vtxt}  {int(m.sum()):3d} {ptxt}  {dtxt}")
print("\nplots saved as timing_dump/overlay_TIC*.png")

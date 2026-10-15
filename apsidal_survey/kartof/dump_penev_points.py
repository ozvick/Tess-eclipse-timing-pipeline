import pickle, csv
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

tics = [329770815, 257737073, 358232450, 18271803, 155540827, 224308601]
rows = []
for t in tics:
    d = pickle.load(open(f'/mnt/md2/TESS_EBs/ETV/tess{t}_etv_distros_none.pkl', 'rb'))
    n1 = d['num_primary_tasks']
    n = 0
    for i, (task, typ) in enumerate(zip(d['distributions'], d['task_types'])):
        if typ == 'eclipse':
            try:
                p16, p50, p84 = (pct(task, q) for q in (0.16, 0.5, 0.84))
            except Exception:
                continue
            rows.append(dict(tic=t, kind='p' if i < n1 else 's',
                             epoch=abs(int(task['eclipse_indices'][0])),
                             sector=d['task_sectors'][i], p16=p16, p50=p50, p84=p84))
            n += 1
    print(t, n, "eclipse timings")
with open('/home/victor/penev_etv_points.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['tic', 'kind', 'epoch', 'sector', 'p16', 'p50', 'p84'])
    w.writeheader()
    w.writerows(rows)
print("wrote", len(rows), "rows")

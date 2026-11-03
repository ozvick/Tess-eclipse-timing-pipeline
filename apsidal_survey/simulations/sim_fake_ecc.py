"""Tasks 3-4: can unmodeled precession make a static light-curve fit report spurious eccentricity?

For each orbital period and true eccentricity, simulate TESS-like 2-min light curves of a detached
binary (four sectors spread over five years, like TIC 468792279) with ellc, under three scenarios:
  static   no precession (control; with e = 1e-4 this is task 3's nearly circular reference)
  apsidal  omega turns at the tides + GR rate for these stars (k2 = 0.005)
  tilt     orbital inclination drifts (orbit-plane precession, e.g. from a misaligned spin or a third body)
Then fit each simulated light curve with a STATIC ellc model (no precession) and report the fitted e.

Usage: python sim_fake_ecc.py [--periods 1.5 3 6 12] [--ecc 1e-4 0.01 0.05] [--tilt 0.1]
Writes sim_fake_ecc.csv.
"""
import argparse, csv, time
from math import pi
import numpy as np
import ellc
from scipy.optimize import least_squares

ap = argparse.ArgumentParser()
ap.add_argument('--periods', type=float, nargs='+', default=[1.5, 3.0, 6.0, 12.0])
ap.add_argument('--ecc', type=float, nargs='+', default=[1e-4, 0.01, 0.05])
ap.add_argument('--tilt', type=float, default=0.1, help='inclination drift for the tilt scenario [deg/yr]')
ap.add_argument('--noise', type=float, default=5e-4, help='flux noise per 2-min point')
ap.add_argument('--omega', type=float, default=60.0, help='true longitude of periastron [deg]')
args = ap.parse_args()

# ---- the binary: two ~1.8 Msun stars (typical of the systems discussed), slightly unequal
M1, M2, R1, R2 = 1.9, 1.7, 1.9, 1.7            # Msun, Rsun
SB, INCL, K2 = 0.8, 87.0, 0.005
LD = dict(ld_1='quad', ld_2='quad', ldc_1=[0.3, 0.2], ldc_2=[0.3, 0.2])
GM_SUN, C, R_SUN = 1.32712440018e20, 2.99792458e8, 6.957e8
SECTORS = [(0, 27), (1090, 1117), (1630, 1657), (1820, 1847)]   # days, ~5-yr baseline
T_REF = 924.0                                   # reference epoch near the middle of the data
rng = np.random.default_rng(42)
t_all = np.concatenate([np.arange(a, b, 2 / 1440) for a, b in SECTORS])


def apsidal_rate_deg_per_orbit(P):
    a = ((GM_SUN * (M1 + M2) * (P * 86400) ** 2) / (4 * pi ** 2)) ** (1 / 3) / R_SUN
    gr = 3 * GM_SUN * (M1 + M2) / (C ** 2 * a * R_SUN)
    tid = sum((Ri / a) ** 5 * ((1 + mj / mi) + 15 * (mj / mi)) for mi, mj, Ri in ((M1, M2, R1), (M2, M1, R2)))
    return 360 * (gr + K2 * tid), a


def model(t, p, extra=None):
    t0, P, r1, r2, sb, inc, fc, fs = p
    kw = dict(radius_1=r1, radius_2=r2, sbratio=sb, incl=inc, t_zero=t0, period=P, q=M2 / M1, f_c=fc, f_s=fs, **LD)
    if extra:
        kw.update(extra)
    return ellc.lc(t, **kw)


rows = []
for P in args.periods:
    domdt, a = apsidal_rate_deg_per_orbit(P)
    for e in args.ecc:
        w = np.radians(args.omega)
        fc, fs = np.sqrt(e) * np.cos(w), np.sqrt(e) * np.sin(w)
        t0 = T_REF - (T_REF % P)
        truth = np.array([t0, P, R1 / a, R2 / a, SB, INCL, fc, fs])
        scenarios = {'static': None, 'apsidal': dict(domdt=domdt), 'tilt': dict(didt=args.tilt * P / 365.25)}
        for name, extra in scenarios.items():
            t1 = time.time()
            f_true = model(t_all, truth, extra)
            ecl = f_true < 1 - 1e-5                   # keep eclipses (+1 h margins) and a few baseline points
            keep = np.convolve(ecl, np.ones(61), mode='same') > 0
            keep |= rng.random(len(t_all)) < 0.02
            t, y = t_all[keep], f_true[keep] + args.noise * rng.standard_normal(keep.sum())
            res = lambda p: (model(t, p) - y) / args.noise
            lo = [t0 - 0.05, P * 0.999, 0.01, 0.01, 0.05, 70, -0.9, -0.9]
            hi = [t0 + 0.05, P * 1.001, 0.6, 0.6, 5.0, 90, 0.9, 0.9]
            fit = least_squares(res, np.clip(truth, lo, hi), bounds=(lo, hi), x_scale='jac', max_nfev=400)
            fc_f, fs_f = fit.x[6], fit.x[7]
            e_fit = fc_f ** 2 + fs_f ** 2
            try:
                cov = np.linalg.inv(fit.jac.T @ fit.jac)
                g = np.array([2 * fc_f, 2 * fs_f])
                e_err = float(np.sqrt(g @ cov[6:8, 6:8] @ g))
            except np.linalg.LinAlgError:
                e_err = float('nan')
            chi2r = 2 * fit.cost / max(len(t) - 8, 1)
            ecw_t, esw_t = e * np.cos(w), e * np.sin(w)
            ecw_f, esw_f = np.sqrt(e_fit) * fc_f, np.sqrt(e_fit) * fs_f
            row = dict(P=P, e_true=e, scenario=name, rate=(domdt if name == 'apsidal' else args.tilt if name == 'tilt' else 0),
                       e_fit=e_fit, e_err=e_err, ecosw_true=ecw_t, ecosw_fit=ecw_f, esinw_true=esw_t, esinw_fit=esw_f,
                       incl_fit=fit.x[5], chi2_red=chi2r, n=len(t), secs=time.time() - t1)
            rows.append(row)
            unit = 'deg/orbit' if name == 'apsidal' else 'deg/yr' if name == 'tilt' else ''
            print(f"P={P:5.2f} d  e_true={e:<7g} {name:8s} rate={row['rate']:.4g} {unit:9s} -> e_fit={e_fit:.5f} "
                  f"+/- {e_err:.5f}  (ecosw {ecw_t:+.5f}->{ecw_f:+.5f}, esinw {esw_t:+.5f}->{esw_f:+.5f})  "
                  f"chi2_red={chi2r:.2f}  [{row['secs']:.0f} s]", flush=True)

with open('sim_fake_ecc.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader(); wr.writerows(rows)
print('wrote sim_fake_ecc.csv')

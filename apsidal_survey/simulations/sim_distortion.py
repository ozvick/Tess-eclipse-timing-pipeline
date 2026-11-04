"""Can tidal distortion make a spherical-star eclipse fit report spurious eccentricity?

Truth: ellc light curves of tidally distorted stars (Roche shapes, gravity darkening) on CIRCULAR
orbits (true e cos w = e sin w = 0). Fit: spherical-star eclipses (as batman does in Penev's model)
times out-of-eclipse terms 1 + A1 cos(phi) + B1 sin(phi) + A2 cos(2 phi) (reflection, beaming,
ellipsoidal), with the radii either free or fixed at their true (volume-equivalent) values, as
isochrones would fix them. Control: the same fits to spherical-star light curves.
Usage: python sim_distortion.py [--periods 1.25 1.5 2 3 4 6] [--gdc 0.35] [--R1 1.9 --R2 1.7] [--out file.csv]
"""
import argparse, csv, time
from math import pi
import numpy as np
import ellc
from scipy.optimize import least_squares

ap = argparse.ArgumentParser()
ap.add_argument('--periods', type=float, nargs='+', default=[1.25, 1.5, 2.0, 3.0, 4.0, 6.0])
ap.add_argument('--gdc', type=float, default=0.35)
ap.add_argument('--noise', type=float, default=5e-4)
ap.add_argument('--incl', type=float, default=87.0)
ap.add_argument('--R1', type=float, default=1.9)
ap.add_argument('--R2', type=float, default=1.7)
ap.add_argument('--out', default='sim_distortion.csv')
args = ap.parse_args()

M1, M2, R1, R2, SB = 1.9, 1.7, args.R1, args.R2, 0.8
LD = dict(ld_1='quad', ld_2='quad', ldc_1=[0.3, 0.2], ldc_2=[0.3, 0.2])
GM_SUN, R_SUN = 1.32712440018e20, 6.957e8
SECTORS = [(0, 27), (1090, 1117)]
rng = np.random.default_rng(7)
t_all = np.concatenate([np.arange(a, b, 2 / 1440) for a, b in SECTORS])


def eggleton(q):
    return 0.49 * q ** (2 / 3) / (0.6 * q ** (2 / 3) + np.log(1 + q ** (1 / 3)))


def sphere_lc(t, p):
    t0, P, r1, r2, sb, inc, ec, es = p[:8]
    e = np.hypot(ec, es)
    fc, fs = (ec / np.sqrt(e), es / np.sqrt(e)) if e > 1e-12 else (0.0, 0.0)
    return ellc.lc(t, radius_1=r1, radius_2=r2, sbratio=sb, incl=inc, t_zero=t0, period=P,
                   q=M2 / M1, f_c=fc, f_s=fs, **LD)


def full_model(t, p):
    A1, B1, A2 = p[8:11]
    phi = 2 * pi * (t - p[0]) / p[1]
    return sphere_lc(t, p) * (1 + A1 * np.cos(phi) + B1 * np.sin(phi) + A2 * np.cos(2 * phi))


rows = []
for P in args.periods:
    a = ((GM_SUN * (M1 + M2) * (P * 86400) ** 2) / (4 * pi ** 2)) ** (1 / 3) / R_SUN
    r1, r2 = R1 / a, R2 / a
    fill1, fill2 = r1 / eggleton(M1 / M2), r2 / eggleton(M2 / M1)
    if max(fill1, fill2) > 0.95:
        print(f"P={P}: stars fill {fill1:.2f}/{fill2:.2f} of their Roche lobes, too close to contact, skipped"); continue
    t0 = 500.0 - (500.0 % P)
    for truth_shape in ('sphere', 'roche'):
        kw = dict(radius_1=r1, radius_2=r2, sbratio=SB, incl=args.incl, t_zero=t0, period=P, q=M2 / M1, **LD)
        if truth_shape == 'roche':
            kw.update(shape_1='roche', shape_2='roche', gdc_1=args.gdc, gdc_2=args.gdc)
        f_true = ellc.lc(t_all, **kw)
        ecl = sphere_lc(t_all, [t0, P, r1, r2, SB, args.incl, 0, 0]) < 1 - 1e-5
        keep = (np.convolve(ecl, np.ones(61), mode='same') > 0) | (rng.random(len(t_all)) < 0.15)
        t, y = t_all[keep], f_true[keep] + args.noise * rng.standard_normal(keep.sum())
        for radii in ('free', 'fixed'):
            t1 = time.time()
            p0 = np.array([t0, P, r1, r2, SB, args.incl, 0.0, 0.0, 0.0, 0.0, 0.0])
            free = np.ones(11, bool)
            if radii == 'fixed':
                free[2:4] = False
            lo = np.array([t0 - 0.05, P * 0.999, 0.01, 0.01, 0.05, 60, -0.3, -0.3, -0.1, -0.1, -0.1])
            hi = np.array([t0 + 0.05, P * 1.001, 0.6, 0.6, 5.0, 90, 0.3, 0.3, 0.1, 0.1, 0.1])

            def res(x):
                p = p0.copy(); p[free] = x
                return (full_model(t, p) - y) / args.noise

            fit = least_squares(res, p0[free], bounds=(lo[free], hi[free]), x_scale='jac', max_nfev=300)
            p = p0.copy(); p[free] = fit.x
            try:
                cov = np.linalg.inv(fit.jac.T @ fit.jac)
                idx = {k: int(np.sum(free[:k])) for k in (6, 7)}
                ecw_err, esw_err = np.sqrt(cov[idx[6], idx[6]]), np.sqrt(cov[idx[7], idx[7]])
            except np.linalg.LinAlgError:
                ecw_err = esw_err = float('nan')
            chi2r = 2 * fit.cost / max(len(t) - free.sum(), 1)
            row = dict(P=P, roche_fill_1=fill1, roche_fill_2=fill2, truth=truth_shape, radii=radii,
                       ecosw=p[6], ecosw_err=ecw_err, esinw=p[7], esinw_err=esw_err, e=np.hypot(p[6], p[7]),
                       r1_fit_over_true=p[2] / r1, r2_fit_over_true=p[3] / r2, incl_fit=p[5],
                       A_ellipsoidal=p[10], chi2_red=chi2r, secs=time.time() - t1)
            rows.append(row)
            print(f"P={P:5.2f} d (fill {fill1:.2f}/{fill2:.2f})  truth={truth_shape:6s} radii={radii:5s} -> "
                  f"e cosw={p[6]:+.5f}+/-{ecw_err:.5f}  e sinw={p[7]:+.5f}+/-{esw_err:.5f}  e={row['e']:.5f}  "
                  f"r1/true={row['r1_fit_over_true']:.3f}  chi2_red={chi2r:.2f}  [{row['secs']:.0f} s]", flush=True)

with open(args.out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f'wrote {args.out}')

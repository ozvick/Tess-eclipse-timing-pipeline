"""Synthetic TESS light curves with KNOWN e cos(omega) and e sin(omega), to test how well the
pipeline's eclipse-duration ratio recovers e sin(omega) (and the secondary phase e cos(omega)).

Grid: P = 2.5, 5, 10 d; true e sin w = 0, 0.01, 0.03, 0.1 (e cos w = 0.02 throughout);
similar (sbratio 0.9) or unequal (sbratio 0.3) eclipse depths; spherical stars, plus tidally
distorted (Roche) versions at P = 2.5 d. Two TESS-like sectors, 2-min cadence, noise 5e-4.

Writes val/TIC_<id>_fits/*_lc.fits, val/truth.csv and run_validation.sh.
Usage: python make_validation_set.py [--pipeline PATH_TO_apsidal_pipeline.py] [--python PYTHON]
"""
import argparse, csv, os
from math import atan2, cos, sin, sqrt, pi
import numpy as np
import ellc
from astropy.io import fits

ap = argparse.ArgumentParser()
ap.add_argument('--pipeline', default=os.path.expanduser(
    '~/Tess-eclipse-timing-pipeline/apsidal_survey/pipeline/apsidal_pipeline.py'))
ap.add_argument('--python', default=os.path.expanduser('~/venv_etv/bin/python'))
ap.add_argument('--noise', type=float, default=5e-4)
args = ap.parse_args()

M1, M2, R1, R2, INCL = 1.9, 1.7, 1.9, 1.7, 87.0
LD = dict(ld_1='quad', ld_2='quad', ldc_1=[0.3, 0.2], ldc_2=[0.3, 0.2])
GM_SUN, R_SUN = 1.32712440018e20, 6.957e8
SECTORS = [(31, 1500.0), (33, 2200.0)]          # (sector number, start time in BTJD)
rng = np.random.default_rng(11)
ECW = 0.02

grid = []
for P in (2.5, 5.0, 10.0):
    for esw in (0.0, 0.01, 0.03, 0.1):
        for sb in (0.9, 0.3):
            grid.append((P, esw, sb, 'sphere'))
            if P == 2.5:
                grid.append((P, esw, sb, 'roche'))

os.makedirs('val', exist_ok=True)
truth, lines = [], ['set -u']
for k, (P, esw, sb, shape) in enumerate(grid):
    tic = 990000000 + k
    e = sqrt(ECW ** 2 + esw ** 2)
    w = atan2(esw, ECW)
    a = ((GM_SUN * (M1 + M2) * (P * 86400) ** 2) / (4 * pi ** 2)) ** (1 / 3) / R_SUN
    kw = dict(radius_1=R1 / a, radius_2=R2 / a, sbratio=sb, incl=INCL, t_zero=1510.0, period=P,
              q=M2 / M1, f_c=sqrt(e) * cos(w), f_s=sqrt(e) * sin(w), **LD)
    if shape == 'roche':
        kw.update(shape_1='roche', shape_2='roche', gdc_1=0.35, gdc_2=0.35)
    d = f'val/TIC_{tic}_fits'
    os.makedirs(d, exist_ok=True)
    for sec, t0 in SECTORS:
        t = np.arange(t0, t0 + 27.0, 2 / 1440)
        f = ellc.lc(t, **kw) + args.noise * rng.standard_normal(len(t))
        cols = [fits.Column(name='TIME', format='D', array=t),
                fits.Column(name='PDCSAP_FLUX', format='D', array=f * 1000.0),
                fits.Column(name='PDCSAP_FLUX_ERR', format='D', array=np.full(len(t), args.noise * 1000)),
                fits.Column(name='QUALITY', format='J', array=np.zeros(len(t), int))]
        h0 = fits.PrimaryHDU(); h0.header['SECTOR'] = sec
        h1 = fits.BinTableHDU.from_columns(cols); h1.header['TSTART'] = t[0]; h1.header['TSTOP'] = t[-1]
        fits.HDUList([h0, h1]).writeto(f'{d}/synth-s{sec:04d}_lc.fits', overwrite=True)
    truth.append(dict(tic=tic, P=P, ecosw=ECW, esinw=esw, e=e, omega_deg=np.degrees(w), sbratio=sb, shape=shape))
    lines.append(f'echo "=== {tic}  P={P} esinw={esw} sbratio={sb} {shape} ==="')
    lines.append(f'{args.python} {args.pipeline} --tic {tic} --data-dir {d} --outdir val/out/TIC_{tic}_out '
                 f'--period-min {P * 0.999:.6f} --period-max {P * 1.001:.6f} > val/out/TIC_{tic}.log 2>&1')
    print(f'{tic}: P={P:5.1f}  e sin w={esw:<5}  sbratio={sb}  {shape}', flush=True)

os.makedirs('val/out', exist_ok=True)
with open('val/truth.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(truth[0].keys()))
    wr.writeheader(); wr.writerows(truth)
open('run_validation.sh', 'w').write('\n'.join(lines) + '\n')
print(f'\nwrote {len(truth)} systems, val/truth.csv and run_validation.sh')

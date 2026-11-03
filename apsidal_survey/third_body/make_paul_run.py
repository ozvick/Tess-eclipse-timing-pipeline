"""Download SPOC 2-minute light curves from MAST for Paul's systems and write run_paul.sh,
which runs the patched pipeline with each period fixed (+/-0.1%) at Penev's fitted value.
Needs binary_radii_paul.csv (from kartof) in the current folder. Already-downloaded systems are skipped.
"""
import csv, glob, os, re, shutil
from astroquery.mast import Observations

TICS = [286566392, 117175528, 468792279, 417728778, 72782654]
P = {int(r['tic']): float(r['per']) for r in csv.DictReader(open('binary_radii_paul.csv'))}

for t in TICS:
    d = f'TIC_{t}_fits'
    if glob.glob(f'{d}/*_lc.fits'):
        print(f'{t}: already downloaded ({len(glob.glob(f"{d}/*_lc.fits"))} files)')
        continue
    try:
        obs = Observations.query_criteria(obs_collection='TESS', dataproduct_type='timeseries',
                                          target_name=str(t), provenance_name='SPOC')
    except Exception as e:
        print(f'{t}: MAST query failed ({e.__class__.__name__}: {e})'); continue
    if len(obs) == 0:
        print(f'{t}: no SPOC 2-minute light curves at MAST'); continue
    prods = Observations.filter_products(Observations.get_product_list(obs), productSubGroupDescription='LC')
    tmp = d + '_tmp'
    Observations.download_products(prods, download_dir=tmp)
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(f'{tmp}/**/*_lc.fits', recursive=True):
        shutil.move(f, os.path.join(d, os.path.basename(f)))
    shutil.rmtree(tmp, ignore_errors=True)
    files = sorted(glob.glob(f'{d}/*_lc.fits'))
    secs = sorted({int(m.group(1)) for f in files for m in [re.search(r'-s(\d{4})-', f)] if m})
    print(f'{t}: {len(files)} light curves, sectors {secs}')

os.makedirs('paul_v2', exist_ok=True)
lines = ['cd ~/Desktop']
for t in TICS:
    if not glob.glob(f'TIC_{t}_fits/*_lc.fits'):
        continue
    if t not in P:
        print(f'{t}: no period in binary_radii_paul.csv, skipped'); continue
    lines.append(f'echo "=== TIC {t} ==="')
    lines.append(f'python3 apsidal_pipeline.py --tic {t} --data-dir TIC_{t}_fits --outdir paul_v2/TIC_{t}_out '
                 f'--period-min {P[t] * 0.999:.8f} --period-max {P[t] * 1.001:.8f} > paul_v2/TIC_{t}_out.log 2>&1')
open('run_paul.sh', 'w').write('\n'.join(lines) + '\n')
print(f'\nwrote run_paul.sh with {(len(lines) - 1) // 2} systems')

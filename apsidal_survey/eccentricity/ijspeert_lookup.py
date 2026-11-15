"""Look up systems in IJspeert et al. (2024, A&A 691, A242) TESS OBAF eclipsing-binary catalogue
(table obaf-eb1) by cone search on the SAAO VizieR mirror, one system at a time.

Positions: RA_OBJ/DEC_OBJ from the TIC_<tic>_fits light curves, or from a positions file.
  python3 ijspeert_lookup.py                     (Mac: systems in consistency_modelfree.csv, positions from FITS)
  python3 ijspeert_lookup.py --write-positions   (Mac: just write positions.csv, e.g. to copy to kartof)
  python3 ijspeert_lookup.py --positions positions.csv   (anywhere: use a positions file)
Writes ijspeert_matches.csv with every column returned for each matched system.
"""
import argparse, csv, glob, io, os, sys, urllib.request
import numpy as np

URL = 'http://vizieridia.saao.ac.za/viz-bin/conesearch/J/A+A/691/A242/obaf-eb1?RA={ra}&DEC={dec}&SR={sr}'
HIGH_E = [197755658, 89522181, 160328766, 424719589, 289044378, 341029467, 71531463, 427312571,
          12915124, 331183881, 468792279, 417728778]
ap = argparse.ArgumentParser()
ap.add_argument('--positions'); ap.add_argument('--write-positions', action='store_true')
ap.add_argument('--radius', type=float, default=5.0, help='match radius in arcsec')
args = ap.parse_args()

if args.positions:
    pos = {int(r['tic']): (float(r['ra']), float(r['dec'])) for r in csv.DictReader(open(args.positions))}
else:
    from astropy.io import fits
    tics = [int(r['tic']) for r in csv.DictReader(open('consistency_modelfree.csv'))]
    tics += [t for t in HIGH_E if t not in tics]
    pos = {}
    for t in tics:
        f = sorted(glob.glob(f'TIC_{t}_fits/**/*_lc.fits', recursive=True) + glob.glob(f'TIC{t}_fits/**/*_lc.fits', recursive=True))
        if f:
            h = fits.getheader(f[0], 0)
            if h.get('RA_OBJ') is not None:
                pos[t] = (float(h['RA_OBJ']), float(h['DEC_OBJ']))
    with open('positions.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['tic', 'ra', 'dec'])
        for t, (ra, dec) in pos.items():
            w.writerow([t, ra, dec])
    print(f'positions for {len(pos)} systems -> positions.csv')
    if args.write_positions:
        sys.exit(0)

from astropy.io.votable import parse_single_table
rows, cols, missing = [], None, []
for k, (t, (ra, dec)) in enumerate(pos.items(), 1):
    try:
        data = urllib.request.urlopen(URL.format(ra=ra, dec=dec, sr=args.radius / 3600), timeout=30).read()
        tab = parse_single_table(io.BytesIO(data)).to_table()
    except Exception as e:
        print(f'  {t}: query failed ({e.__class__.__name__})'); missing.append(t); continue
    if len(tab) == 0:
        missing.append(t); continue
    cols = tab.colnames
    rec = {c: (tab[c][0].item() if hasattr(tab[c][0], 'item') else tab[c][0]) for c in cols}
    rec['tic'] = t
    rows.append(rec)
    if k % 10 == 0:
        print(f'  {k}/{len(pos)} queried, {len(rows)} matched', flush=True)

print(f'\nmatched {len(rows)} of {len(pos)} systems; not in the catalogue: {len(missing)}')
if rows:
    keys = ['tic'] + [c for c in cols if c != 'tic']
    with open('ijspeert_matches.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    print('columns returned:', ', '.join(cols))
    ecc = [c for c in cols if any(s in c.lower() for s in ('ecos', 'esin', 'ecc', 'omega', 'per'))] or \
          [c for c in cols if c.lower() in ('e', 'w', 'p')]
    print('\nhigh-eccentricity systems in the catalogue:')
    by = {r['tic']: r for r in rows}
    for t in HIGH_E:
        if t in by:
            print(f'  {t}: ' + ', '.join(f'{c}={by[t][c]}' for c in ecc))
        else:
            print(f'  {t}: not in the catalogue')
    print('wrote ijspeert_matches.csv')

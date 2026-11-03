import glob
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad

TICS = [220430912, 161988269, 160328766, 260502102, 293954660, 343757299, 234491752,
        281498280, 154511165, 436592342, 89522181, 220420534, 184298625, 18271803,
        120626764, 231809798, 255908136, 271529021, 364186087, 63916460, 85785338,
        366972961, 180770456, 287669537, 358232450, 125857387]
CATALOGS = {'Mitnyan+2024 (TESS NCVZ triples)': 'J/A+A/685/A43',
            'Borkovits+2016 (Kepler triples)': 'J/MNRAS/455/4136',
            'Czavalinga+2023 (Gaia triples)': 'J/A+A/670/A75'}
viz = Vizier(columns=['**'], row_limit=5, timeout=30)
Simbad.TIMEOUT = 30

def coords(t):
    for f in glob.glob(f'TIC_{t}_fits/**/*_lc.fits', recursive=True) + glob.glob(f'TIC{t}_fits/**/*_lc.fits', recursive=True):
        h = fits.getheader(f, 0)
        if h.get('RA_OBJ') is not None:
            return SkyCoord(h['RA_OBJ'], h['DEC_OBJ'], unit='deg')
    return None

for t in TICS:
    c = coords(t)
    if c is None:
        print(f"{t}: no coordinates found"); continue
    line = []
    try:
        s = Simbad.query_region(c, radius=5 * u.arcsec)
        if s is not None and len(s):
            cols = {k.lower(): k for k in s.colnames}
            name = s[cols.get('main_id', s.colnames[0])][0]
            otype = s[cols['otype']][0] if 'otype' in cols else ''
            line.append(f"SIMBAD: {name} [{otype}]")
    except Exception as e:
        line.append(f"SIMBAD error: {e.__class__.__name__}")
    for label, cat in CATALOGS.items():
        print(f"  {t}: querying {label}...", flush=True)
        try:
            res = viz.query_region(c, radius=5 * u.arcsec, catalog=cat)
            n = sum(len(tab) for tab in res)
            if n:
                line.append(f"MATCH in {label} ({n} row(s))")
        except Exception as e:
            line.append(f"{label}: query failed ({e.__class__.__name__})")
    print(f"{t}: " + ('; '.join(line) if line else 'no matches'))

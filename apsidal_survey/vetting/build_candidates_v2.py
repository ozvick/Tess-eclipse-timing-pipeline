"""Assemble the vetting evidence for every candidate from the patched re-run (rerun_v2/).

Candidates: new verdict SOLID or TENTATIVE, plus every old detection (to see what happened to it).
For each: drift and significance (errors x1.2, the paper's noise-floor finding), |r|, retention
ratio, Penev's ETV drift and the mirror-method drift (with agreement z), eccentricity, implied k2
(tides + GR, Penev's radii), and same-field neighbours that also drift (co-drift or opposite sign).

Inputs on ~/Desktop: rerun_v2/, rerun_fixedP/, TIC_*_out/, TIC*_fits/, group_statistics.py,
etv_comparison.csv, retime_all.csv, tic_medians_table_add_med_flipped.csv, tic_etrigwquants.csv,
binary_radii.csv (+ binary_radii_more.csv if present).
Outputs: candidates_v2.csv, need_radii.txt (candidates still lacking radii).
"""
import csv, glob, json, os, re
from math import pi, sqrt, sin, cos, atan2, hypot, radians, degrees, acos
import numpy as np
from astropy.io import fits

GM_SUN, C, R_SUN = 1.32712440018e20, 2.99792458e8, 6.957e8
ERR_SCALE = 1.2
rng = np.random.default_rng(1)

# ------------------------------------------------------------------ inputs ---
def load_rows(fname, key='tic'):
    if not os.path.exists(fname):
        return {}
    return {int(float(r[key])): r for r in csv.DictReader(open(fname))}

new = {}
for p in glob.glob('rerun_v2/TIC_*_out/TIC*_summary.json'):
    s = json.load(open(p))
    new[int(s['tic'])] = s

def old_summary(t):
    for p in (f'rerun_fixedP/TIC_{t}_out/TIC{t}_summary.json', f'TIC_{t}_out/TIC{t}_summary.json'):
        if os.path.exists(p):
            return json.load(open(p))
    return {}

src = re.sub(r'#.*', '', open('group_statistics.py').read())
sets = {n: {int(x) for x in re.findall(r'\b\d{5,10}\b', b)}
        for n, b in re.findall(r'^([A-Z_]+)\s*=\s*\{(.*?)\}', src, re.S | re.M)}
def old_class(t):
    for n in ('PRECESSION_LTTE', 'PRECESSION', 'TENTATIVE', 'DEMOTED', 'NULL_CONSTRAINED'):
        if t in sets.get(n, ()):
            return n
    return 'other'
OLD_DET = sets.get('PRECESSION', set()) | sets.get('PRECESSION_LTTE', set())

penev = {t: (float(r['pen']), float(r['pen_err'])) for t, r in load_rows('etv_comparison.csv').items()}
mirror = {t: (float(r['mirror']), float(r['mirror_err'])) for t, r in load_rows('retime_all.csv').items()}
med = load_rows('tic_medians_table_add_med_flipped.csv', 'tic_id')
trig = load_rows('tic_etrigwquants.csv', 'tic_id')
radii = load_rows('binary_radii.csv')
radii.update(load_rows('binary_radii_more.csv'))
if med:
    cols = list(next(iter(med.values())).keys())
    cM = next(c for c in cols if 'M_1+M_2' in c)
    cP = next(c for c in cols if 'P_{orb}' in c)

# ------------------------------------------------- positions and sectors ---
pos = {}
for t in new:
    files = sorted(glob.glob(f'TIC_{t}_fits/**/*_lc.fits', recursive=True) +
                   glob.glob(f'TIC{t}_fits/**/*_lc.fits', recursive=True))
    secs, ra, dec = set(), None, None
    for f in files:
        try:
            h = fits.getheader(f, 0)
        except Exception:
            continue
        secs.add(h.get('SECTOR'))
        ra, dec = h.get('RA_OBJ', ra), h.get('DEC_OBJ', dec)
    if ra is not None and dec is not None:
        pos[t] = (float(ra), float(dec), secs)

def sep_deg(a, b):
    ra1, de1, ra2, de2 = map(radians, (a[0], a[1], b[0], b[1]))
    c = sin(de1) * sin(de2) + cos(de1) * cos(de2) * cos(ra1 - ra2)
    return degrees(acos(max(-1.0, min(1.0, c))))

def sig(s):
    S, Se = s.get('apsidal_slope_min_per_1000d'), s.get('apsidal_slope_err_min_per_1000d')
    return (S, Se, abs(S) / (ERR_SCALE * Se) if S is not None and Se else None)

def neighbours(t):
    """Significant drifters (>3 sigma) within 15 deg sharing >= 2 sectors."""
    if t not in pos:
        return []
    out = []
    for u, s in new.items():
        if u == t or u not in pos:
            continue
        S, Se, zz = sig(s)
        if zz is None or zz < 3:
            continue
        if len(pos[t][2] & pos[u][2]) >= 2 and sep_deg(pos[t], pos[u]) < 6:
            out.append((u, S))
    return out

# ------------------------------------------------------ physical test ---
def mean_anom(nu, e):
    E = 2 * atan2(sqrt(1 - e) * sin(nu / 2), sqrt(1 + e) * cos(nu / 2))
    return E - e * sin(E)

def dD_dw(w, e, P, h=1e-5):
    def sepf(w):
        return P * ((mean_anom(1.5 * pi - w, e) - mean_anom(0.5 * pi - w, e)) % (2 * pi)) / (2 * pi)
    return (sepf(w + h) - sepf(w - h)) / (2 * h)

def tidal_pu(e, M1, M2, R1, R2, a, rot):
    f = (1 + 1.5 * e**2 + 0.125 * e**4) / (1 - e**2) ** 5
    g = 1 / (1 - e**2) ** 2
    return sum((Ri / a) ** 5 * (rot**2 * (1 + mj / mi) * g + 15 * (mj / mi) * f)
               for mi, mj, Ri in ((M1, M2, R1), (M2, M1, R2)))

def implied_k2(t, S, Se, N=500):
    if t not in trig or t not in radii or S is None or not Se:
        return None
    Q = {k: float(v) for k, v in trig[t].items() if k.startswith('e')}
    ec, es = Q['ecosw_Q0.5'], Q['esinw_Q0.5']
    sec = (Q['ecosw_Q0.841'] - Q['ecosw_Q0.159']) / 2
    ses = (Q['esinw_Q0.841'] - Q['esinw_Q0.159']) / 2
    r = radii[t]
    M1, M2, R1, R2, a_R, P = (float(r[k]) for k in ('M1', 'M2', 'R1', 'R2', 'a_Rsun', 'per'))
    prot = float(r['prot'])
    rot = P / prot if prot > 0 else 1.0
    e0 = hypot(ec, es)
    pu_gr = 3 * GM_SUN * (M1 + M2) / (C**2 * a_R * R_SUN * (1 - e0**2))
    tid1 = tidal_pu(e0, M1, M2, R1, R2, a_R, rot)
    Sd = S + Se * ERR_SCALE * rng.standard_normal(N)
    ecd, esd = ec + sec * rng.standard_normal(N), es + ses * rng.standard_normal(N)
    k2 = []
    for i in range(N):
        wdot = (Sd[i] / 1000 / 1440) / dD_dw(atan2(esd[i], ecd[i]), min(hypot(ecd[i], esd[i]), 0.99), P)
        k2.append((abs(wdot) * P / (2 * pi) - pu_gr) / tid1)
    return tuple(np.percentile(k2, [2.5, 50, 97.5]))

# ------------------------------------------------------------ assemble ---
cands = sorted({t for t, s in new.items() if s.get('verdict') in ('SOLID', 'TENTATIVE')} | (OLD_DET & set(new)))
rows, need_radii = [], []
for t in cands:
    s, o = new[t], old_summary(t)
    S, Se, zz = sig(s)
    smp = s.get('sec_minus_primary') or {}
    ret = abs(smp['slope_min_per_1000d'] / S) if smp.get('slope_min_per_1000d') and S else None
    e = hypot(float(trig[t]['ecosw_Q0.5']), float(trig[t]['esinw_Q0.5'])) if t in trig else None
    zp = (S - penev[t][0]) / hypot(Se * ERR_SCALE, penev[t][1]) if t in penev and S is not None else None
    zm = (S - mirror[t][0]) / hypot(Se * ERR_SCALE, mirror[t][1]) if t in mirror and S is not None else None
    k2 = implied_k2(t, S, Se)
    if k2 is None and S is not None and t in trig:
        need_radii.append(t)
    nb = neighbours(t)
    co = [u for u, Su in nb if S is not None and Su * S > 0 and 0.67 < abs(Su / S) < 1.5]
    opp = [u for u, Su in nb if S is not None and Su * S < 0 and 0.67 < abs(Su / S) < 1.5]
    notes = []
    if zz is not None and zz < 5:
        notes.append('weak')
    if S is not None and abs(S) < 0.15:
        notes.append('small drift')
    if e is not None and e < 0.02:
        notes.append('near-circular')
    if k2 is not None and k2[0] > 0.75:
        notes.append('k2 > max')
    if k2 is not None and k2[2] < 0:
        notes.append('slower than GR')
    if zp is not None and abs(zp) > 3:
        notes.append('Penev disagrees')
    if zm is not None and abs(zm) > 3:
        notes.append('mirror disagrees')
    if co:
        notes.append('co-drift neighbour')
    if opp:
        notes.append('opposite-sign neighbour')
    rows.append(dict(
        tic=t, old_class=old_class(t), old_verdict=o.get('verdict'), new_verdict=s.get('verdict'),
        S=S, Se=Se, z_infl=zz, abs_r=abs(s['apsidal_r']) if s.get('apsidal_r') is not None else None,
        retention=ret, penev=penev.get(t, (None,))[0], z_penev=zp, mirror=mirror.get(t, (None,))[0],
        z_mirror=zm, e=e, k2_lo=k2[0] if k2 else None, k2_med=k2[1] if k2 else None,
        k2_hi=k2[2] if k2 else None, co_drift=' '.join(map(str, co)), opposite=' '.join(map(str, opp)),
        notes='; '.join(notes)))

def f(x, spec, w):
    return f"{'--':>{w}}" if x is None else format(x, spec)

order = {'SOLID': 0, 'TENTATIVE': 1}
rows.sort(key=lambda r: (order.get(r['new_verdict'], 2), -(r['z_infl'] or 0)))
print(f"{'TIC':>10} {'old class':16} {'new':10} {'drift':>16} {'z':>5} {'|r|':>5} {'Penev':>7} "
      f"{'mirror':>7} {'e':>5} {'k2':>7}  notes")
for r in rows:
    print(f"{r['tic']:>10} {r['old_class']:16} {str(r['new_verdict']):10} "
          f"{f(r['S'], '+8.2f', 8)}+/-{f(r['Se'], '5.2f', 5)} {f(r['z_infl'], '5.1f', 5)} "
          f"{f(r['abs_r'], '5.2f', 5)} {f(r['penev'], '+7.2f', 7)} {f(r['mirror'], '+7.2f', 7)} "
          f"{f(r['e'], '5.2f', 5)} {f(r['k2_med'], '7.4f', 7)}  {r['notes']}")

with open('candidates_v2.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
open('need_radii.txt', 'w').write('\n'.join(map(str, need_radii)) + '\n')
print(f"\n{len(rows)} candidates written to candidates_v2.csv")
print(f"{len(need_radii)} candidates need radii from Penev's fits (need_radii.txt): {need_radii}")

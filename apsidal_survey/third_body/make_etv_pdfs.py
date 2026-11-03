"""Penev-style ETV PDFs from the patched pipeline's per-eclipse timings.

Page 1: whole baseline, primary (blue) and secondary (orange) O-C, with season means as large
markers. Then one page per group of consecutive TESS sectors, showing individual eclipses.
Usage: python3 make_etv_pdfs.py OUTDIR TIC [TIC ...]      e.g. python3 make_etv_pdfs.py paul_v2 286566392
Writes tess<TIC>_etv_victor.pdf (and an overview PNG) in the current folder.
"""
import glob, os, sys
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT, TICS = sys.argv[1], [int(x) for x in sys.argv[2:]]
COL = {'Primary': 'tab:blue', 'Secondary': 'tab:orange'}


def load_oc(path):
    if not os.path.exists(path):
        return None
    head = open(path).readline().strip().lower().split(',')
    a = np.atleast_2d(np.genfromtxt(path, delimiter=',', skip_header=1))
    pick = lambda keys, default: next((i for i, h in enumerate(head) if any(k in h for k in keys)), default)
    it, io = pick(['time', 'btjd'], 0), pick(['oc', 'o-c', 'o_c'], 1)
    ie = pick(['err', 'sig', 'unc'], None)
    t, oc = a[:, it], a[:, io]
    err = a[:, ie] if ie is not None else np.full(len(t), np.nan)
    ok = np.isfinite(t) & np.isfinite(oc)
    return t[ok], oc[ok], err[ok]


def sector_ranges(tic):
    rng = []
    for f in glob.glob(f'TIC_{tic}_fits/**/*_lc.fits', recursive=True):
        h0, h1 = fits.getheader(f, 0), fits.getheader(f, 1)
        rng.append((int(h0['SECTOR']), h1['TSTART'], h1['TSTOP']))
    return sorted(rng)


def groups_of_consecutive(rng):
    out = []
    for s, a, b in rng:
        if out and s == out[-1][1] + 1:
            out[-1] = (out[-1][0], s, out[-1][2], b)
        else:
            out.append((s, s, a, b))
    return out


def plot_points(ax, data, means=False):
    for name, d in data.items():
        if d is None or not len(d[0]):
            continue
        t, oc, err = d
        ax.errorbar(t, oc, yerr=None if np.all(np.isnan(err)) else err, fmt='o', ms=3, lw=0.8,
                    color=COL[name], label=name, alpha=0.8 if means else 1)
        if means:
            cuts = np.where(np.diff(t) > 10)[0] + 1
            for tt, oo in zip(np.split(t, cuts), np.split(oc, cuts)):
                ax.plot(tt.mean(), np.median(oo), 's', ms=9, color=COL[name], mec='k', mew=0.6)
    ax.axhline(0, color='k', ls='-.', lw=0.7)
    ax.set_xlabel('Time (BTJD)')
    ax.set_ylabel('O$-$C (min)')


for tic in TICS:
    d = f'{OUT}/TIC_{tic}_out'
    data = {'Primary': load_oc(f'{d}/TIC{tic}_primary_oc.csv'),
            'Secondary': load_oc(f'{d}/TIC{tic}_secondary_oc.csv')}
    if all(v is None for v in data.values()):
        print(f'{tic}: no O-C files in {d}'); continue
    pdf_name = f'tess{tic}_etv_victor.pdf'
    with PdfPages(pdf_name) as pdf:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        plot_points(ax, data, means=True)
        ax.set_title(f'TIC {tic}: eclipse timings (large squares: season medians)', fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout(); pdf.savefig(fig); fig.savefig(f'tess{tic}_etv_victor_overview.png', dpi=150)
        plt.close(fig)
        for s1, s2, a, b in groups_of_consecutive(sector_ranges(tic)):
            sub = {k: (None if v is None else tuple(x[(v[0] >= a - 1) & (v[0] <= b + 1)] for x in v))
                   for k, v in data.items()}
            if all(v is None or not len(v[0]) for v in sub.values()):
                continue
            fig, ax = plt.subplots(figsize=(7, 4.2))
            plot_points(ax, sub)
            ax.set_title(f'TIC {tic}: Sectors {s1}\u2013{s2}', fontsize=10)
            ax.legend(fontsize=8)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f'{tic}: wrote {pdf_name}')

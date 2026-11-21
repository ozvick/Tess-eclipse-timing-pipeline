#!/usr/bin/env python
"""
Export TESS light curves for STAR SHADOW (IJspeert et al. 2024 method).

For each target, download one light curve per TESS sector (preferring SPOC 2-min,
then TESS-SPOC FFI, then QLP FFI), median-normalise each sector, and write one
3-column file (time [BTJD], flux, flux_err) per TESS observing cycle, as
IJspeert et al. analysed each cycle separately.

Outputs (in ~/sts_input):
  TIC<id>_c<cycle>.dat          time, flux, flux_err
  TIC<id>_c<cycle>_sectors.txt  start/end index pairs of each sector (STAR SHADOW i_sectors)
  manifest.csv                  which sectors/authors went into each file

Run with an environment that has lightkurve (e.g. ~/venv_lk on kartof):
  ~/venv_lk/bin/python export_lcs_starshadow.py
"""
import os
import csv
import numpy as np
import lightkurve as lk

# 11 high-e systems (same list as the IJspeert data request)
TICS = [197755658, 89522181, 160328766, 424719589, 289044378, 341029467,
        71531463, 427312571, 12915124, 468792279, 417728778]

OUT = os.path.expanduser("~/sts_input")
PREF = ["SPOC", "TESS-SPOC", "QLP"]          # preference order per sector
CYCLE_FIRST_SECTOR = [1, 14, 27, 40, 56, 70, 84, 97]   # first sector of cycles 1..8


def tess_cycle(sector):
    cyc = 0
    for i, s in enumerate(CYCLE_FIRST_SECTOR, start=1):
        if sector >= s:
            cyc = i
    return cyc


def choose_products(res):
    """Pick one product per sector, by author preference (skip 20-s SPOC)."""
    chosen = {}
    seq = res.table["sequence_number"]
    for idx in range(len(res)):
        author = str(res.author[idx])
        if author not in PREF:
            continue
        exptime = float(res.exptime[idx].value)
        if author == "SPOC" and exptime < 100:
            continue
        sector = int(seq[idx])
        rank = PREF.index(author)
        if sector not in chosen or rank < chosen[sector][0]:
            chosen[sector] = (rank, idx, author, exptime)
    return chosen


def clean_sector(lc):
    lc = lc.remove_nans().normalize()
    t = np.asarray(lc.time.value, dtype=float)
    f = np.asarray(lc.flux.value, dtype=float)
    e = np.asarray(lc.flux_err.value, dtype=float)
    ok = np.isfinite(t) & np.isfinite(f) & (f > 0)
    t, f, e = t[ok], f[ok], e[ok]
    bad_err = ~np.isfinite(e) | (e <= 0)
    if bad_err.all() and len(f) > 2:
        # no usable errors (some FFI products): point-to-point scatter
        sig = 1.4826 * np.median(np.abs(np.diff(f))) / np.sqrt(2)
        e = np.full_like(f, sig)
    elif bad_err.any():
        e[bad_err] = np.median(e[~bad_err])
    return t, f, e


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for tic in TICS:
        print(f"TIC {tic}: searching...", flush=True)
        try:
            res = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
        except Exception as exc:
            print(f"  search failed: {exc}")
            continue
        if len(res) == 0:
            print("  no light curves found")
            continue
        chosen = choose_products(res)
        cycles = {}
        for sector in sorted(chosen):
            rank, idx, author, exptime = chosen[sector]
            try:
                lc = res[idx].download(quality_bitmask="default")
            except Exception as exc:
                print(f"  S{sector} {author}: download failed ({exc})")
                continue
            if lc is None:
                continue
            t, f, e = clean_sector(lc)
            if len(t) < 100:
                print(f"  S{sector} {author}: too few points ({len(t)}), skipped")
                continue
            cycles.setdefault(tess_cycle(sector), []).append((sector, author, t, f, e))
            print(f"  S{sector:3d} {author:9s} {int(exptime):5d}s  n={len(t)}", flush=True)

        for cyc, parts in sorted(cycles.items()):
            parts.sort(key=lambda p: p[0])
            ts, fs, es, isec = [], [], [], []
            start = 0
            for sector, author, t, f, e in parts:
                ts.append(t); fs.append(f); es.append(e)
                isec.append([start, start + len(t)])
                start += len(t)
            data = np.column_stack([np.concatenate(ts), np.concatenate(fs), np.concatenate(es)])
            stem = f"TIC{tic}_c{cyc}"
            np.savetxt(os.path.join(OUT, stem + ".dat"), data, fmt="%.8f",
                       header="time_btjd flux_norm flux_err")
            np.savetxt(os.path.join(OUT, stem + "_sectors.txt"), np.array(isec), fmt="%d")
            manifest.append({"tic": tic, "cycle": cyc,
                             "sectors": " ".join(str(p[0]) for p in parts),
                             "authors": " ".join(p[1] for p in parts),
                             "n_points": len(data)})
            print(f"  -> {stem}.dat: {len(parts)} sector(s), {len(data)} points")

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tic", "cycle", "sectors", "authors", "n_points"])
        w.writeheader()
        w.writerows(manifest)
    print(f"\nWrote {len(manifest)} cycle files to {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Run STAR SHADOW v1.1.7a on the per-cycle light curves written by
export_lcs_starshadow.py, then collect e cos w and e sin w.

Run inside the micromamba 'sts' environment on kartof:
  micromamba activate sts
  python run_starshadow.py                 # known periods (default, avoids aliases)
  python run_starshadow.py --free-period   # let STAR SHADOW find the period, as IJspeert did
  python run_starshadow.py --collect-only  # just re-read finished results

Outputs (in ~/sts_output):
  <stem>_analysis/...        STAR SHADOW files for each TIC/cycle
  sts_results.csv            one row per TIC per cycle
  sts_combined.csv           one row per TIC: unweighted mean over cycles (as IJspeert et al.)
"""
import os
import sys
import csv
import glob
import argparse
from multiprocessing import Pool
import numpy as np

IN_DIR = os.path.expanduser("~/sts_input")
OUT_DIR = os.path.expanduser("~/sts_output")

# Orbital periods from the v2.2 pipeline (days)
PERIODS = {197755658: 3.15882, 89522181: 2.34698, 160328766: 10.318,
           424719589: 7.641, 289044378: 24.849446, 341029467: 13.220,
           71531463: 15.338, 427312571: 6.06633, 12915124: 3.164,
           468792279: 3.782, 417728778: 3.431}

# Model-free (e cos w, e sin w) from our pipeline (consistency_modelfree.csv).
# Used ONLY to fix the primary/secondary labelling of each cycle (a label swap
# flips the sign of both components), so all cycles and our pipeline share one convention.
REF = {197755658: (-0.017, 0.402), 12915124: (0.051, 0.071), 89522181: (-0.318, -0.039),
       424719589: (0.354, 0.365), 160328766: (0.431, -0.318), 71531463: (-0.057, -0.464),
       341029467: (-0.308, -0.265), 289044378: (0.305, 0.378), 427312571: (0.151, 0.135),
       468792279: (0.195, 0.075), 417728778: (-0.080, -0.143)}

KEYS = ["stage", "period", "p_err",
        "ecosw_form", "ecosw_low", "esinw_form", "esinw_low",
        "ecosw_phys", "esinw_phys", "i_phys"]


def run_one(args):
    path, free_period = args
    import star_shadow as sts
    stem = os.path.basename(path)[:-4]
    tic = int(stem[3:].split("_")[0])
    sec_file = path[:-4] + "_sectors.txt"
    i_sectors = np.loadtxt(sec_file, dtype=int).reshape(-1, 2) if os.path.exists(sec_file) else None
    p_orb = 0 if free_period else PERIODS.get(tic, 0)
    try:
        sts.analyse_lc_from_file(path, p_orb=p_orb, i_sectors=i_sectors, stage="all",
                                 method="fitter", data_id=stem, save_dir=OUT_DIR,
                                 overwrite=True, verbose=False)
        return stem, "done"
    except Exception as exc:
        return stem, f"error: {exc}"


def read_summary(stem):
    path = os.path.join(OUT_DIR, f"{stem}_analysis", f"{stem}_analysis_summary.csv")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#") or len(row) < 2:
                continue
            out[row[0].strip()] = row[1].strip()
    return out


def to_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def collect(files):
    rows = []
    for path in files:
        stem = os.path.basename(path)[:-4]
        tic, cyc = stem[3:].split("_c")
        s = read_summary(stem)
        row = {"tic": int(tic), "cycle": int(cyc)}
        if s is None:
            row["stage"] = "no output"
        else:
            for k in KEYS:
                row[k] = s.get(k, "")
        rows.append(row)
    rows.sort(key=lambda r: (r["tic"], r["cycle"]))
    with open(os.path.join(OUT_DIR, "sts_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tic", "cycle"] + KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # combine cycles: unweighted mean over cycles that reached the eclipse-parameter stage
    comb = []
    print(f"\n{'TIC':>10} {'n':>2} {'P':>9} {'ecosw_form':>11} {'esinw_form':>11} "
          f"{'ecosw_phys':>11} {'esinw_phys':>11} {'e_phys':>6} {'sd_esp':>6}  stages")
    for tic in sorted({r["tic"] for r in rows}):
        rs = [r for r in rows if r["tic"] == tic]
        good = [r for r in rs if to_float(r.get("stage")) >= 7 and np.isfinite(to_float(r.get("ecosw_form")))]
        stages = ",".join(str(r.get("stage")) for r in rs)
        if not good:
            print(f"{tic:>10}  0          no cycle reached eclipse parameters   [{stages}]")
            comb.append({"tic": tic, "n_cycles": 0, "stages": stages})
            continue
        # 1) align primary/secondary labelling of each cycle to our pipeline's convention
        n_flip = 0
        rx, ry = REF.get(tic, (np.nan, np.nan))
        for r in good:
            x, y = to_float(r["ecosw_form"]), to_float(r["esinw_form"])
            if np.isfinite(rx) and np.hypot(-x - rx, -y - ry) < np.hypot(x - rx, y - ry):
                for k in ("ecosw_form", "esinw_form", "ecosw_phys", "esinw_phys"):
                    r[k] = -to_float(r[k])
                n_flip += 1
        # 2) drop cycles whose e cos w disagrees with the median (e.g. too few eclipses)
        med = np.median([to_float(r["ecosw_form"]) for r in good])
        kept = [r for r in good
                if abs(to_float(r["ecosw_form"]) - med) <= max(0.02, 5 * to_float(r["ecosw_low"]))]
        n_drop = len(good) - len(kept)
        good = kept
        A = lambda k: np.array([to_float(r.get(k)) for r in good])
        ec, es, ecp, esp = A("ecosw_form"), A("esinw_form"), A("ecosw_phys"), A("esinw_phys")
        n = len(good)
        # formal error of the mean shrinks with sqrt(n) (as IJspeert et al.); also report cycle scatter
        err_c = np.nanmean(A("ecosw_low")) / np.sqrt(n)
        err_s = np.nanmean(A("esinw_low")) / np.sqrt(n)
        rec = {"tic": tic, "n_cycles": n, "period": np.nanmean(A("period")),
               "ecosw_form": np.nanmean(ec), "ecosw_form_err": err_c,
               "esinw_form": np.nanmean(es), "esinw_form_err": err_s,
               "ecosw_phys": np.nanmean(ecp), "esinw_phys": np.nanmean(esp),
               "ecosw_cycle_std": np.nanstd(ec) if n > 1 else np.nan,
               "esinw_cycle_std": np.nanstd(es) if n > 1 else np.nan,
               "esinw_phys_cycle_std": np.nanstd(esp) if n > 1 else np.nan,
               "e_form": np.hypot(np.nanmean(ec), np.nanmean(es)),
               "e_phys": np.hypot(np.nanmean(ecp), np.nanmean(esp)),
               "n_flipped": n_flip, "n_dropped": n_drop, "stages": stages}
        comb.append(rec)
        print(f"{tic:>10} {n:>2} {rec['period']:9.4f} {rec['ecosw_form']:+11.4f} {rec['esinw_form']:+11.4f} "
              f"{rec['ecosw_phys']:+11.4f} {rec['esinw_phys']:+11.4f} {rec['e_phys']:6.3f} "
              f"{rec['esinw_phys_cycle_std']:6.3f}  flip {n_flip} drop {n_drop}  [{stages}]")
    fields = ["tic", "n_cycles", "period", "ecosw_form", "ecosw_form_err", "esinw_form", "esinw_form_err",
              "ecosw_phys", "esinw_phys", "ecosw_cycle_std", "esinw_cycle_std", "esinw_phys_cycle_std",
              "e_form", "e_phys", "n_flipped", "n_dropped", "stages"]
    with open(os.path.join(OUT_DIR, "sts_combined.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(comb)
    print(f"\nwrote {OUT_DIR}/sts_results.csv and sts_combined.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-period", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(IN_DIR, "TIC*_c*.dat")))
    if not files:
        sys.exit(f"no light-curve files in {IN_DIR}; run export_lcs_starshadow.py first")
    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.collect_only:
        print(f"running STAR SHADOW on {len(files)} cycle files with {args.procs} processes "
              f"({'free' if args.free_period else 'known'} periods)", flush=True)
        with Pool(args.procs) as pool:
            for stem, status in pool.imap_unordered(run_one, [(f, args.free_period) for f in files]):
                print(f"  {stem}: {status}", flush=True)
    collect(files)


if __name__ == "__main__":
    main()

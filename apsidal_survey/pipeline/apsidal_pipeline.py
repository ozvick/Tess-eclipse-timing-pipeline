#!/usr/bin/env python3
"""
apsidal_pipeline.py — Reusable TESS ETV apsidal-motion analysis pipeline.

Usage:
    python3 apsidal_pipeline.py --tic 12915124 --data-dir TIC_12915124_fits --outdir TIC_12915124_out

Given a directory containing SPOC light curve FITS files (searched recursively
for *_lc.fits), this script:
  1. Loads, quality-filters, normalizes, and stitches all sectors.
  2. Runs a two-stage Box Least Squares search (coarse, then fine) to find
     the orbital period.
  3. Refines the period/epoch precisely using per-epoch dip-time fitting
     with iterative sigma-clipping (avoids single naive-dip-finder noise).
  4. Searches the full phase curve for a genuine secondary eclipse,
     distinguishing it from noise via depth/consistency checks.
  5. If primary/secondary depths are close, verifies which is truly deeper
     and swaps the reference epoch if needed.
  6. Builds eclipse templates and fits every individual eclipse.
  7. Applies a QC filter on the fitted depth-scale parameter (auto-relaxes
     if the pass rate is too low).
  8. Computes the refined primary ephemeris (closed-form, per-epoch).
  9. If a secondary exists: computes the secondary O-C series, fits a
     linear apsidal-motion slope, and runs mechanistic checks (baseline
     drift, depth stability, residual-vs-range ratio).
 10. Checks the primary O-C for curvature (LTTE), using both a naive
     per-epoch F-test and a fairer per-sector-averaged F-test to guard
     against pseudo-replication when sectors are sparse.
 11. Prints a full summary to the terminal and saves diagnostic plots as
     PNG files in --outdir.

This consolidates the manual, step-by-step methodology used interactively
earlier in this project into one repeatable tool.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from scipy import stats
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_eclipse_model import (
    phase01, phase_distance, build_eclipse_templates, fit_local_eclipses,
)


# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------

def load_and_stitch(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*_lc.fits"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No *_lc.fits files found under {data_dir}")

    all_time, all_flux, all_flux_err, sector_id = [], [], [], []
    print(f"Found {len(files)} SPOC light curve files")

    for f in files:
        with fits.open(f) as hdul:
            data = hdul[1].data
            sector = hdul[0].header.get("SECTOR", -1)
            time = data["TIME"]
            flux = data["PDCSAP_FLUX"]
            flux_err = data["PDCSAP_FLUX_ERR"]
            quality = data["QUALITY"]

            good = (quality == 0) & np.isfinite(time) & np.isfinite(flux)
            med = np.nanmedian(flux[good])

            all_time.append(time[good])
            all_flux.append(flux[good] / med)
            all_flux_err.append(flux_err[good] / med)
            sector_id.append(np.full(np.sum(good), sector))
            print(f"  Sector {sector}: {np.sum(good)}/{len(time)} good points")

    time = np.concatenate(all_time)
    flux = np.concatenate(all_flux)
    flux_err = np.concatenate(all_flux_err)
    sectors = np.concatenate(sector_id)

    order = np.argsort(time)
    time, flux, flux_err, sectors = time[order], flux[order], flux_err[order], sectors[order]

    print(f"\nTotal points: {len(time)}")
    print(f"Time span: {time.min():.2f} to {time.max():.2f} BTJD ({time.max()-time.min():.1f} days)")
    return time, flux, flux_err, sectors


# ----------------------------------------------------------------------
# 2. Period search (coarse -> fine BLS)
# ----------------------------------------------------------------------

def bls_search(time, flux, period_min, period_max, n_coarse=15000, n_fine=50000,
                durations=None):
    if durations is None:
        durations = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    durations = durations[durations < period_min]
    m = np.isfinite(time) & np.isfinite(flux)
    t, f = time[m], flux[m]
    model = BoxLeastSquares(t, f)

    coarse_periods = np.linspace(period_min, period_max, n_coarse)
    coarse = model.power(coarse_periods, durations)
    best_coarse = np.argmax(coarse.power)
    p_coarse = coarse.period[best_coarse]
    print(f"Coarse BLS: P={p_coarse:.6f} d, power={coarse.power[best_coarse]:.3f}")

    fine_lo = max(period_min, p_coarse * 0.94)
    fine_hi = min(period_max, p_coarse * 1.06)
    fine_periods = np.linspace(fine_lo, fine_hi, n_fine)
    fine = model.power(fine_periods, durations)
    best_fine = np.argmax(fine.power)
    P0 = fine.period[best_fine]
    T0 = fine.transit_time[best_fine]
    duration = fine.duration[best_fine]
    power = fine.power[best_fine]
    print(f"Fine BLS:   P={P0:.8f} d, T0={T0:.6f}, duration={duration:.4f} d, power={power:.3f}")
    return P0, T0, duration, power


# ----------------------------------------------------------------------
# 3. Precise ephemeris refinement via per-epoch dip fitting + sigma-clip
# ----------------------------------------------------------------------

def refine_ephemeris_dip_times(time, flux, P0, T0, half_window, n_sigma_clip=3,
                                min_points=15):
    """Find one observed dip time (and depth) per predicted epoch, then
    robustly fit a linear ephemeris with iterative sigma-clipping."""
    tmin, tmax = time.min(), time.max()
    e_min = int(np.floor((tmin - T0) / P0)) - 1
    e_max = int(np.ceil((tmax - T0) / P0)) + 1

    epochs, dip_times, dip_depths = [], [], []
    for e in range(e_min, e_max + 1):
        center = T0 + e * P0
        m = (time > center - half_window) & (time < center + half_window) & np.isfinite(flux)
        if np.sum(m) < min_points:
            continue
        te, fe = time[m], flux[m]
        idx = np.argmin(fe)
        epochs.append(e)
        dip_times.append(te[idx])
        dip_depths.append(1.0 - fe[idx])

    epochs = np.array(epochs)
    dip_times = np.array(dip_times)
    dip_depths = np.array(dip_depths)
    if len(epochs) < 3:
        return P0, T0, epochs, dip_times, dip_depths

    keep = np.ones(len(epochs), dtype=bool)
    for _ in range(4):
        if np.sum(keep) < 3:
            break
        coeffs = np.polyfit(epochs[keep], dip_times[keep], deg=1)
        P_r, T0_r = coeffs[0], coeffs[1]
        resid = (dip_times - (T0_r + epochs * P_r)) * 24 * 60
        sigma = np.std(resid[keep])
        if sigma == 0:
            break
        new_keep = np.abs(resid) < n_sigma_clip * sigma
        if new_keep.sum() == keep.sum():
            keep = new_keep
            break
        keep = new_keep

    coeffs = np.polyfit(epochs[keep], dip_times[keep], deg=1)
    P_refined, T0_refined = coeffs[0], coeffs[1]
    resid = (dip_times[keep] - (T0_refined + epochs[keep] * P_refined)) * 24 * 60
    print(f"Dip-time ephemeris refinement: kept {keep.sum()}/{len(epochs)} epochs, "
          f"residual std={np.std(resid):.3f} min")
    print(f"Refined (pre-template) P={P_refined:.8f} d, T0={T0_refined:.8f}")
    return P_refined, T0_refined, epochs[keep], dip_times[keep], dip_depths[keep]


def detect_half_period_alias(epochs, dip_depths, min_n=6):
    """Detect classic half-period BLS aliasing: if odd/even epochs show
    significantly different eclipse depths, the true period is 2x P0
    (primary and secondary are alternating onto the same nominal phase)."""
    if len(epochs) < min_n:
        return False, np.nan, np.nan, np.nan

    even_mask = (epochs % 2) == 0
    odd_mask = ~even_mask
    if even_mask.sum() < 3 or odd_mask.sum() < 3:
        return False, np.nan, np.nan, np.nan

    even_depths = dip_depths[even_mask]
    odd_depths = dip_depths[odd_mask]
    t_stat, p_val = stats.ttest_ind(even_depths, odd_depths, equal_var=False)

    mean_diff = abs(np.mean(even_depths) - np.mean(odd_depths))
    pooled_mean = np.mean(dip_depths)
    rel_diff = mean_diff / pooled_mean if pooled_mean else 0

    # Require both statistical significance AND a practically large
    # relative depth difference (>15%) to avoid false positives from
    # ordinary depth jitter.
    is_aliased = (p_val < 0.01) and (rel_diff > 0.15)
    return is_aliased, p_val, rel_diff, mean_diff


# ----------------------------------------------------------------------
# 4. Secondary eclipse search
# ----------------------------------------------------------------------

def measure_eclipse_half_width(time, flux, P0, T0, nbins=401, search_half_width=0.15,
                                depth_threshold_frac=0.15):
    """Measure the actual eclipse half-width from the binned folded profile,
    rather than trusting BLS's coarse duration grid (which can be far too
    wide if the true duration falls between grid points)."""
    phase = phase01(time, T0, P0)
    local = phase_distance(phase, 0.0)
    m = np.abs(local) < search_half_width
    if np.sum(m) < 20:
        return 0.01

    bins = np.linspace(-search_half_width, search_half_width, nbins)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = np.full(len(bc), np.nan)
    for i in range(len(bc)):
        bm = (local[m] >= bins[i]) & (local[m] < bins[i + 1])
        if np.sum(bm) > 3:
            med[i] = np.nanmedian(flux[m][bm])

    finite = np.isfinite(med)
    if finite.sum() < 10:
        return 0.01
    baseline = np.nanmedian(med[np.abs(bc) > search_half_width * 0.7])
    min_flux = np.nanmin(med[finite])
    full_depth = baseline - min_flux
    if full_depth <= 0:
        return 0.01

    threshold = baseline - depth_threshold_frac * full_depth
    below = finite & (med < threshold)
    if not np.any(below):
        return 0.01
    half_width = np.max(np.abs(bc[below])) * 1.3  # small padding
    return max(half_width, 0.003)


def find_secondary(time, flux, P0, T0, primary_half_width, nbins=1000,
                    min_pts_per_bin=8, exclude_margin_factor=2.0):
    phase = phase01(time, T0, P0)
    exclude = np.abs(phase_distance(phase, 0.0)) < primary_half_width * exclude_margin_factor

    bins = np.linspace(0, 1, nbins + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (phase >= bins[i]) & (phase < bins[i + 1]) & (~exclude)
        if np.sum(m) >= min_pts_per_bin:
            med[i] = np.nanmedian(flux[m])

    finite = np.isfinite(med)
    if finite.sum() < 10:
        return None

    baseline = np.nanmedian(med[finite])
    scatter = 1.4826 * np.nanmedian(np.abs(med[finite] - baseline))
    depth_arr = baseline - med
    best_idx = np.nanargmax(depth_arr)
    best_depth = depth_arr[best_idx]
    best_phase = bc[best_idx]

    # significance: depth relative to point-to-point scatter, and check
    # a few neighboring bins also show depression (not a single noisy bin)
    neighbor_idx = [(best_idx + k) % nbins for k in (-2, -1, 1, 2)]
    neighbor_depths = [depth_arr[i] for i in neighbor_idx if np.isfinite(depth_arr[i])]
    consistent = len(neighbor_depths) >= 2 and np.mean(neighbor_depths) > 3 * scatter

    significant = (best_depth > 6 * scatter) and consistent and (best_depth > 0.003)

    if not significant:
        return None
    return {"phase": best_phase, "depth": best_depth, "scatter": scatter}


def binned_depth(time, flux, P0, T0, center, half_width, nbins=61):
    phase = phase01(time, T0, P0)
    local = phase_distance(phase, center)
    m = np.abs(local) < half_width
    if np.sum(m) < 10:
        return np.nan
    bins = np.linspace(-half_width, half_width, nbins)
    bc = 0.5 * (bins[:-1] + bins[1:])
    bflux = np.full(len(bc), np.nan)
    for i in range(len(bc)):
        bm = (local[m] >= bins[i]) & (local[m] < bins[i + 1])
        if np.sum(bm) > 3:
            bflux[i] = np.nanmedian(flux[m][bm])
    if not np.any(np.isfinite(bflux)):
        return np.nan
    return 1.0 - np.nanmin(bflux)


# ----------------------------------------------------------------------
# 5. QC filter
# ----------------------------------------------------------------------

def sec_minus_primary_check(times_p, resid_p_min, sectors_p,
                             times_s, oc_min, sectors_s):
    """Isolate the apsidal-motion component from any shared LTTE/period-
    variation drift by computing the sec-minus-primary O-C difference,
    matched per sector. If the shared-drift hypothesis is correct, this
    should look flat; if apsidal motion is real and independent of the
    shared drift, it should reproduce a similar slope to the raw secondary
    O-C but with the shared component removed."""
    common_sectors = sorted(set(sectors_p) & set(sectors_s))
    if len(common_sectors) < 3:
        return None

    sec_mean, prim_mean, t_mean = [], [], []
    for s in common_sectors:
        mp = sectors_p == s
        ms = sectors_s == s
        if mp.sum() == 0 or ms.sum() == 0:
            continue
        prim_mean.append(np.mean(resid_p_min[mp]))
        sec_mean.append(np.mean(oc_min[ms]))
        t_mean.append(np.mean(np.concatenate([times_p[mp], times_s[ms]])))

    if len(t_mean) < 3:
        return None

    prim_mean = np.array(prim_mean)
    sec_mean = np.array(sec_mean)
    t_mean = np.array(t_mean)
    diff = sec_mean - prim_mean

    slope, intercept, r, p, se = stats.linregress(t_mean, diff)
    return {
        "n_sectors": len(t_mean),
        "sector_times": t_mean,
        "sector_diff": diff,
        "slope_min_per_1000d": slope * 1000,
        "slope_err_min_per_1000d": se * 1000,
        "p": p,
        "r": r,
        "intercept": intercept,
    }


def qc_filter(fit_list, lo=0.85, hi=1.15):
    return [f for f in fit_list if (f["scale"] >= lo) and (f["scale"] <= hi) and f["success"]]


def qc_filter_adaptive(fit_list, standard=(0.85, 1.15), relaxed=(0.7, 1.35),
                        min_pass_fraction=0.5):
    if not fit_list:
        return [], standard
    passed = qc_filter(fit_list, *standard)
    if len(passed) / len(fit_list) >= min_pass_fraction:
        return passed, standard
    passed_relaxed = qc_filter(fit_list, *relaxed)
    print(f"  Standard QC window {standard} passed only "
          f"{len(passed)}/{len(fit_list)} ({100*len(passed)/len(fit_list):.0f}%) -- "
          f"using relaxed window {relaxed} instead "
          f"({len(passed_relaxed)}/{len(fit_list)} pass)")
    return passed_relaxed, relaxed


# ----------------------------------------------------------------------
# 6. Curvature (LTTE) check, naive and per-sector
# ----------------------------------------------------------------------

def curvature_check(times, resid_min):
    n = len(times)
    if n < 5:
        return np.nan, np.nan
    coeffs2 = np.polyfit(times, resid_min, deg=2)
    quad_pred = np.polyval(coeffs2, times)
    ss_flat = np.sum((resid_min - np.mean(resid_min)) ** 2)
    ss_quad = np.sum((resid_min - quad_pred) ** 2)
    if ss_quad <= 0:
        return np.nan, np.nan
    F = ((ss_flat - ss_quad) / 2) / (ss_quad / (n - 3))
    p = 1 - stats.f.cdf(F, 2, n - 3)
    return F, p


def per_sector_curvature_check(times, resid_min, sectors_for_times):
    unique_sectors = np.unique(sectors_for_times)
    if len(unique_sectors) < 5:
        return np.nan, np.nan, len(unique_sectors)
    sector_t = np.array([np.mean(times[sectors_for_times == s]) for s in unique_sectors])
    sector_r = np.array([np.mean(resid_min[sectors_for_times == s]) for s in unique_sectors])
    F, p = curvature_check(sector_t, sector_r)
    return F, p, len(unique_sectors)


def nearest_sector(times_query, time_all, sectors_all):
    idx = np.searchsorted(time_all, times_query)
    idx = np.clip(idx, 0, len(time_all) - 1)
    # check neighbor for closer match
    idx_prev = np.clip(idx - 1, 0, len(time_all) - 1)
    use_prev = np.abs(time_all[idx_prev] - times_query) < np.abs(time_all[idx] - times_query)
    idx_final = np.where(use_prev, idx_prev, idx)
    return sectors_all[idx_final]


# ----------------------------------------------------------------------
# 7. Mechanistic checks
# ----------------------------------------------------------------------

def mechanistic_checks(time, flux, P0, T0, sec_phase, primary_half_width,
                        sf_qc, times_s, oc_min, primary_depth):
    phase = phase01(time, T0, P0)
    # For very wide eclipses the standard 2x mask can consume the whole phase
    # curve. Narrow the mask progressively; if no usable baseline remains,
    # skip gracefully instead of emitting NaN warnings.
    t_oe = f_oe = None
    for mask_factor in (2.0, 1.2, 1.0):
        out_of_eclipse = (np.abs(phase_distance(phase, 0.0)) > primary_half_width * mask_factor) & \
                          (np.abs(phase_distance(phase, sec_phase)) > primary_half_width * mask_factor)
        if np.sum(out_of_eclipse) >= 50:
            t_oe, f_oe = time[out_of_eclipse], flux[out_of_eclipse]
            if mask_factor != 2.0:
                print(f"  (baseline-drift mask narrowed to {mask_factor:.1f}x "
                      f"half-width to retain out-of-eclipse points)")
            break
    good_oe = (np.isfinite(t_oe) & np.isfinite(f_oe)) if t_oe is not None else None
    if good_oe is not None and np.sum(good_oe) >= 50:
        slope_b, intercept_b, r_b, p_b, se_b = stats.linregress(t_oe[good_oe], f_oe[good_oe])
    else:
        print("  (baseline-drift check skipped: eclipses too wide, no usable "
              "out-of-eclipse baseline)")
        slope_b = intercept_b = r_b = p_b = se_b = np.nan
    total_drift_pct = slope_b * (time.max() - time.min()) * 100

    scales_s = np.array([f["scale"] for f in sf_qc])
    mid_t = np.median(times_s)
    early_scale = np.median(scales_s[times_s < mid_t]) if np.any(times_s < mid_t) else np.nan
    late_scale = np.median(scales_s[times_s >= mid_t]) if np.any(times_s >= mid_t) else np.nan
    rel_change = abs(late_scale - early_scale) / early_scale * 100 if early_scale else np.nan

    slope, intercept, r, p, se = stats.linregress(times_s, oc_min)
    resid = oc_min - (slope * times_s + intercept)
    raw_range = oc_min.max() - oc_min.min()
    resid_range = resid.max() - resid.min()
    ratio = raw_range / resid_range if resid_range > 0 else np.inf

    return {
        "baseline_drift_slope_per_day": slope_b,
        "baseline_drift_p": p_b,
        "baseline_drift_pct_of_flux": total_drift_pct,
        "baseline_drift_pct_of_depth": abs(total_drift_pct) / (primary_depth * 100) * 100 if primary_depth else np.nan,
        "depth_stability_early": early_scale,
        "depth_stability_late": late_scale,
        "depth_stability_pct_change": rel_change,
        "oc_slope_min_per_1000d": slope * 1000,
        "oc_slope_err_min_per_1000d": se * 1000,
        "oc_p": p,
        "oc_r": r,
        "resid_std_min": np.std(resid),
        "raw_vs_resid_ratio": ratio,
    }


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tic", required=True, help="TIC number (digits only)")
    ap.add_argument("--data-dir", required=True,
                     help="Directory containing SPOC *_lc.fits files (searched recursively)")
    ap.add_argument("--outdir", default=None,
                     help="Output directory for plots/summary (default: TIC_<tic>_pipeline_out)")
    ap.add_argument("--period-min", type=float, default=0.4,
                     help="Minimum period (days) for BLS search (default 0.4)")
    ap.add_argument("--period-max", type=float, default=45.0,
                     help="Maximum period (days) for BLS search (default 45)")
    args = ap.parse_args()

    tic = args.tic
    outdir = args.outdir or f"TIC_{tic}_pipeline_out"
    os.makedirs(outdir, exist_ok=True)

    print("=" * 70)
    print(f"TIC {tic} — Apsidal Motion Pipeline")
    print("=" * 70)

    # 1. Load data
    time, flux, flux_err, sectors = load_and_stitch(args.data_dir)

    # 2. BLS period search
    print("\n--- Period search (BLS) ---")
    P0, T0, duration, power = bls_search(time, flux, args.period_min, args.period_max)

    # 3. Precise ephemeris refinement
    print("\n--- Precise ephemeris refinement ---")
    half_window = max(duration * 3, P0 * 0.01)
    P0, T0, dip_epochs, dip_times, dip_depths = refine_ephemeris_dip_times(
        time, flux, P0, T0, half_window=half_window
    )

    # 3b. Check for classic BLS half-period aliasing (primary/secondary
    # alternating onto the same nominal phase). If detected, double the
    # period and re-refine.
    is_aliased, alias_p, alias_reldiff, alias_meandiff = detect_half_period_alias(
        dip_epochs, dip_depths
    )
    alias_caveat = False
    if is_aliased:
        print(f"\n*** Half-period aliasing detected *** "
              f"(odd/even epoch depths differ by {alias_reldiff*100:.1f}%, p={alias_p:.4g})")
        print(f"Doubling period: {P0:.8f} -> {P0*2:.8f} d, and re-refining ephemeris.")
        P0 = P0 * 2
        half_window = max(duration * 3, P0 * 0.01)
        P0, T0, dip_epochs, dip_times, dip_depths = refine_ephemeris_dip_times(
            time, flux, P0, T0, half_window=half_window
        )
        # Re-check once more in case of residual aliasing at a different harmonic
        is_aliased2, alias_p2, alias_reldiff2, _ = detect_half_period_alias(
            dip_epochs, dip_depths
        )
        if is_aliased2:
            print(f"  Note: residual odd/even depth difference persists "
                  f"({alias_reldiff2*100:.1f}%, p={alias_p2:.4g}) -- proceeding, "
                  f"but this may indicate additional structure worth manual review.")
    else:
        print(f"\nNo half-period aliasing detected "
              f"(odd/even epoch depth difference: {alias_reldiff*100 if not np.isnan(alias_reldiff) else float('nan'):.1f}%"
              f"{f', p={alias_p:.4g}' if not np.isnan(alias_p) else ''})")
        if (not np.isnan(alias_p)) and alias_p < 1e-3 and alias_reldiff > 0.02:
            alias_caveat = True
            print(f"  CAVEAT: odd/even depth difference is small but highly "
                  f"significant ({alias_reldiff*100:.1f}%, p={alias_p:.3g}) -- "
                  f"the true period may be 2x the adopted value. Flagging "
                  f"period_alias_caveat in the JSON summary.")

    primary_half_width = measure_eclipse_half_width(time, flux, P0, T0)
    print(f"Measured primary eclipse half-width (from profile): {primary_half_width:.4f} phase units")

    # 4. Measure primary depth precisely
    primary_depth = binned_depth(time, flux, P0, T0, 0.0, primary_half_width)
    print(f"\nPrimary depth (binned): {primary_depth:.4f}")

    # 5. Secondary search
    print("\n--- Secondary eclipse search ---")
    sec_info = find_secondary(time, flux, P0, T0, primary_half_width)
    has_secondary = sec_info is not None

    if has_secondary:
        print(f"Candidate secondary found: phase={sec_info['phase']:.4f}, "
              f"depth~{sec_info['depth']:.4f}")
        sec_depth_precise = binned_depth(time, flux, P0, T0, sec_info["phase"], primary_half_width)
        print(f"Secondary depth (binned): {sec_depth_precise:.4f}")

        # Check whether "primary" is actually the deeper eclipse
        if sec_depth_precise > primary_depth:
            print("Secondary is deeper than nominal primary -- swapping reference epoch.")
            new_T0 = T0 + sec_info["phase"] * P0
            new_sec_phase = ((T0 - new_T0) / P0) % 1.0
            T0 = new_T0
            primary_depth, sec_depth_precise = sec_depth_precise, primary_depth
            sec_phase_rough = new_sec_phase
        else:
            sec_phase_rough = sec_info["phase"]
    else:
        print("No significant secondary eclipse detected -- flat baseline elsewhere in phase.")
        sec_phase_rough = 0.5
        sec_depth_precise = np.nan

    # 6. Full phase-fold plot
    phase_plot = phase01(time, T0, P0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(phase_plot, flux, ".", ms=1.2, alpha=0.25, color="tab:blue")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalized flux")
    ax.set_title(f"TIC {tic} full phase fold (P={P0:.6f} d)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"TIC{tic}_full_phasefold.png"), dpi=120)
    plt.close(fig)

    # 7-9. Templates, eclipse fits and refined ephemeris, iterated.
    # Each eclipse type gets its own measured half-width, fit window and shift
    # bound. After each pass, templates and fits are redone with the refined
    # period, epoch and secondary phase, so fit windows are never centred on a
    # drifting first-pass prediction (this produced fake secondary drifts for
    # long-period systems with wide secondaries, e.g. TIC 289044378).
    from local_eclipse_model import median_bin_template
    WINDOW_FACTOR = 2.0   # template/fit window = WINDOW_FACTOR x eclipse half-width
    SHIFT_FACTOR = 0.4    # max timing shift = SHIFT_FACTOR x eclipse half-width
    MAX_PASSES = 6
    CONVERGED_MIN = 0.5   # stop when predicted eclipse times move less than this [min] ...
    CONVERGED_FRAC = 0.02 # ... or this fraction of the narrower eclipse half-width, if larger
    tol_min = CONVERGED_MIN
    P_first = P0
    P_use, T0_use, sec_phase_use = P0, T0, sec_phase_rough
    n_passes, converged_shift = 0, np.nan
    for n_pass in range(1, MAX_PASSES + 1):
        n_passes = n_pass
        print(f"\n--- Pass {n_pass}: templates and eclipse fits with "
              f"P={P_use:.8f} d, T0={T0_use:.6f} ---")
        hw_p = measure_eclipse_half_width(time, flux, P_use, T0_use)
        hw_s = (measure_eclipse_half_width(time, flux, P_use, T0_use + sec_phase_use * P_use)
                if has_secondary else hw_p)
        win = {"primary": min(WINDOW_FACTOR * hw_p, 0.2),
               "secondary": min(WINDOW_FACTOR * hw_s, 0.2)}
        max_shift = {"primary": SHIFT_FACTOR * hw_p, "secondary": SHIFT_FACTOR * hw_s}
        print(f"Eclipse half-widths: primary {hw_p:.4f}, secondary {hw_s:.4f} phase "
              f"(secondary/primary = {hw_s / hw_p:.2f})")

        phase_use = phase01(time, T0_use, P_use)
        templates = {}
        for kind, center in (("primary", 0.0), ("secondary", sec_phase_use)):
            local = phase_distance(phase_use, center)
            m = np.abs(local) < win[kind]
            try:
                templates[kind] = median_bin_template(
                    local[m], np.asarray(flux)[m], half_width=win[kind], nbins=121)
            except ValueError:
                templates[kind] = None
        if templates["primary"] is None:
            print("\nFATAL: primary eclipse template could not be built.")
            sys.exit(1)
        if templates["secondary"] is None:
            if has_secondary:
                print("Secondary template could not be built (too few points in window) -- "
                      "demoting to no-secondary target (primary-only template).")
            has_secondary = False
            templates["secondary"] = templates["primary"]
            win["secondary"], max_shift["secondary"] = win["primary"], max_shift["primary"]
        print(f"Primary template depth:   {templates['primary']['depth']:.4f}")
        print(f"Secondary template depth: {templates['secondary']['depth']:.4f}")

        fits_all = fit_local_eclipses(
            time, flux, T0_use, P_use, templates,
            secondary_phase=sec_phase_use,
            half_window_phase=win,
            max_shift_phase=max_shift,
            min_points=min(max(int(P_use * win["primary"] * 2 * 24 * 60 / 2 * 0.3), 15), 60),
        )
        primary_fits = [f for f in fits_all if f["kind"] == "primary"]
        secondary_fits = [f for f in fits_all if f["kind"] == "secondary"]
        print(f"Total fits: {len(fits_all)} (primary={len(primary_fits)}, "
              f"secondary={len(secondary_fits)})")

        # 8. QC filter (adaptive)
        pf_qc, pf_window = qc_filter_adaptive(primary_fits)
        print(f"Primary: {len(pf_qc)}/{len(primary_fits)} pass QC (window {pf_window})")
        if has_secondary:
            sf_qc, sf_window = qc_filter_adaptive(secondary_fits)
            print(f"Secondary: {len(sf_qc)}/{len(secondary_fits)} pass QC (window {sf_window})")
        else:
            sf_qc = []
        if len(pf_qc) < 3:
            print("\nFATAL: fewer than 3 QC-passed primary eclipses -- cannot proceed.")
            sys.exit(1)

        # 9. Refined primary ephemeris
        epochs_p = np.array([f["epoch"] for f in pf_qc])
        obs_centers_p = np.array([f["obs_center"] for f in pf_qc])
        times_p = np.array([f["center_time"] for f in pf_qc])
        coeffs = np.polyfit(epochs_p, obs_centers_p, deg=1)
        P_refined, T0_refined = coeffs[0], coeffs[1]
        pred_p = T0_refined + epochs_p * P_refined
        resid_p_min = (obs_centers_p - pred_p) * 24 * 60

        n_bound_p = int(sum(f["at_bound"] for f in pf_qc))
        n_bound_s = int(sum(f["at_bound"] for f in sf_qc))
        if has_secondary and len(sf_qc) >= 3:
            obs_s = np.array([f["obs_center"] for f in sf_qc])
            sec_phase_new = float(np.median(((obs_s - T0_refined) / P_refined) % 1.0))
        else:
            sec_phase_new = sec_phase_use

        # how far the predicted eclipse times move between passes, over the data span
        e_lo, e_hi = epochs_p.min() - 1, epochs_p.max() + 1
        moves = [abs((T0_refined + e * P_refined) - (T0_use + e * P_use)) for e in (e_lo, e_hi)]
        moves.append(abs(sec_phase_new - sec_phase_use) * P_refined)
        converged_shift = max(moves) * 24 * 60
        print(f"Pass {n_pass}: refined P={P_refined:.8f} d; predicted eclipse times move "
              f"by up to {converged_shift:.2f} min; shift-bound hits: "
              f"{n_bound_p} primary, {n_bound_s} secondary")

        P_use, T0_use, sec_phase_use = P_refined, T0_refined, sec_phase_new
        tol_min = max(CONVERGED_MIN, CONVERGED_FRAC * min(hw_p, hw_s) * P_refined * 24 * 60)
        if n_pass >= 2 and converged_shift < tol_min:
            break
    else:
        print(f"WARNING: ephemeris not converged after {MAX_PASSES} passes "
              f"(last change {converged_shift:.2f} min)")

    primary_half_width = hw_p
    secondary_half_width = hw_s
    print("\n--- Refined primary ephemeris (final pass) ---")
    print(f"Refined P  = {P_refined:.8f} d   (first-pass P0 = {P_first:.8f} d)")
    print(f"Refined T0 = {T0_refined:.8f} BTJD")
    print(f"Primary O-C: std={np.std(resid_p_min):.4f} min, "
          f"range {resid_p_min.min():.4f} to {resid_p_min.max():.4f} min, n={len(epochs_p)}")

    F_naive, p_naive = curvature_check(times_p, resid_p_min)
    sectors_p = nearest_sector(times_p, time, sectors)
    F_sector, p_sector, n_sectors_used = per_sector_curvature_check(times_p, resid_p_min, sectors_p)
    print(f"Primary curvature (naive, per-epoch): F={F_naive:.3f}, p={p_naive:.4f}")
    if not np.isnan(F_sector):
        print(f"Primary curvature (fairer, per-sector n={n_sectors_used}): F={F_sector:.3f}, p={p_sector:.4f}")
    else:
        print("Primary curvature (per-sector): too few independent sectors for a fair test")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axhline(0, color="0.5", lw=1)
    ax.plot(times_p, resid_p_min, "o", ms=5, color="tab:blue", alpha=0.7)
    ax.set_xlabel("Time [BTJD]")
    ax.set_ylabel("Primary O-C [min]")
    ax.set_title(f"TIC {tic} primary O-C (n={len(times_p)})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"TIC{tic}_primary_oc.png"), dpi=120)
    np.savetxt(os.path.join(outdir, f"TIC{tic}_primary_oc.csv"), np.column_stack([epochs_p, times_p, resid_p_min]), delimiter=",", header="epoch,time_btjd,oc_min", comments="")
    plt.close(fig)

    result = {
        "tic": tic,
        "period_days": P_refined,
        "t0_btjd": T0_refined,
        "primary_depth": primary_depth,
        "primary_oc_std_min": float(np.std(resid_p_min)),
        "primary_curvature_naive_p": float(p_naive) if not np.isnan(p_naive) else None,
        "primary_curvature_sector_p": float(p_sector) if not np.isnan(p_sector) else None,
        "n_primary_qc": len(pf_qc),
        "has_secondary": has_secondary,
        "period_alias_caveat": alias_caveat,
        "pipeline_version": 2,
        "n_passes": n_passes,
        "converged_shift_min": converged_shift,
        "converged_tolerance_min": tol_min,
        "first_pass_period_days": P_first,
        "primary_half_width_phase": primary_half_width,
        "secondary_half_width_phase": secondary_half_width,
        "n_primary_at_shift_bound": n_bound_p,
        "n_secondary_at_shift_bound": n_bound_s,
    }

    # 10. Secondary O-C and apsidal-motion test
    if has_secondary and len(sf_qc) >= 3:
        print("\n--- Secondary eclipse timing and O-C ---")
        epochs_s = np.array([f["epoch"] for f in sf_qc])
        obs_centers_s = np.array([f["obs_center"] for f in sf_qc])
        times_s = np.array([f["center_time"] for f in sf_qc])
        sectors_s = nearest_sector(times_s, time, sectors)

        measured_phase = ((obs_centers_s - T0_refined) / P_refined) % 1.0
        sec_phase_final = np.median(measured_phase)
        ecosw = (sec_phase_final - 0.5) * np.pi / 2
        print(f"Adopted secondary phase (median, n={len(measured_phase)}) = {sec_phase_final:.6f}")
        print(f"Implied e*cos(w) = {ecosw:+.6f}")

        predicted_sec = T0_refined + epochs_s * P_refined + sec_phase_final * P_refined
        oc_min = (obs_centers_s - predicted_sec) * 24 * 60

        slope, intercept, r, p, se = stats.linregress(times_s, oc_min)
        print(f"\n=== APSIDAL MOTION RESULT: TIC {tic} ===")
        print(f"Slope   = {slope*1000:+.4f} +/- {se*1000:.4f} min/1000d")
        print(f"p-value = {p:.6g}")
        print(f"r       = {r:.4f}")
        print(f"n       = {len(times_s)} secondary eclipses over {times_s.max()-times_s.min():.1f} days")

        checks = mechanistic_checks(time, flux, P_refined, T0_refined, sec_phase_final,
                                     max(primary_half_width, secondary_half_width),
                                     sf_qc, times_s, oc_min, primary_depth)
        print("\n--- Mechanistic checks ---")
        print(f"Baseline drift: slope={checks['baseline_drift_slope_per_day']:.3e}/day, "
              f"p={checks['baseline_drift_p']:.4g}, "
              f"{checks['baseline_drift_pct_of_flux']:.4f}% of flux "
              f"({checks['baseline_drift_pct_of_depth']:.4f}% of depth)")
        print(f"Depth stability: early={checks['depth_stability_early']:.4f}, "
              f"late={checks['depth_stability_late']:.4f}, "
              f"change={checks['depth_stability_pct_change']:.3f}%")
        print(f"Residual std={checks['resid_std_min']:.4f} min, "
              f"raw/residual range ratio={checks['raw_vs_resid_ratio']:.1f}x")

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.axhline(0, color="0.5", lw=1)
        ax.plot(times_s, oc_min, "o", ms=6, color="tab:red", label="secondary O-C")
        tt = np.linspace(times_s.min(), times_s.max(), 50)
        ax.plot(tt, slope * tt + intercept, "-", color="k", lw=1.5,
                label=f"fit: {slope*1000:+.3f} min/1000d")
        ax.set_xlabel("Time [BTJD]")
        ax.set_ylabel("Secondary O-C [min]")
        ax.set_title(f"TIC {tic} secondary O-C: slope={slope*1000:+.3f}"
                     f"$\\pm${se*1000:.3f} min/1000d, p={p:.2g}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"TIC{tic}_secondary_oc.png"), dpi=120)
        np.savetxt(os.path.join(outdir, f"TIC{tic}_secondary_oc.csv"), np.column_stack([times_s, oc_min]), delimiter=",", header="time_btjd,oc_min", comments="")
        plt.close(fig)

        # If the primary shows significant curvature, the secondary drift
        # may be contaminated by a shared LTTE/period-variation effect
        # rather than being purely apsidal. Cross-check with the
        # sec-minus-primary difference, which cancels any shared drift.
        curvature_flagged = (not np.isnan(p_sector) and p_sector < 0.05) or \
                             (not np.isnan(p_naive) and p_naive < 0.05)
        secpm = None
        if curvature_flagged:
            print("\n*** Primary O-C shows significant curvature -- the secondary")
            print("    drift above may be partly or wholly a shared LTTE effect,")
            print("    not genuine apsidal motion. Running sec-minus-primary")
            print("    cross-check (isolates the apsidal component). ***")
            secpm = sec_minus_primary_check(times_p, resid_p_min, sectors_p,
                                             times_s, oc_min, sectors_s)
            if secpm is not None:
                print(f"\nsec-minus-primary (n={secpm['n_sectors']} sectors): "
                      f"slope={secpm['slope_min_per_1000d']:+.4f} "
                      f"+/- {secpm['slope_err_min_per_1000d']:.4f} min/1000d, "
                      f"p={secpm['p']:.4g}, r={secpm['r']:.4f}")
                print("Compare this to the raw secondary-only slope above -- if")
                print("similar, the apsidal signal is real and independent of the")
                print("shared drift. If it vanishes here, the raw slope was likely")
                print("just riding on the shared LTTE effect.")

                fig, ax = plt.subplots(figsize=(9, 4.5))
                ax.axhline(0, color="0.5", lw=1)
                ax.plot(times_s, oc_min, "o", ms=5, color="tab:red", alpha=0.4,
                        label=f"raw secondary O-C ({slope*1000:+.3f} min/1000d)")
                ax.plot(secpm["sector_times"], secpm["sector_diff"], "s", ms=10,
                        color="tab:purple",
                        label=f"sec-minus-primary ({secpm['slope_min_per_1000d']:+.3f} min/1000d)")
                tt2 = np.linspace(times_s.min(), times_s.max(), 50)
                ax.plot(tt2, secpm["slope_min_per_1000d"]/1000 * tt2 + secpm["intercept"],
                        "-", color="tab:purple", lw=1.8)
                ax.set_xlabel("Time [BTJD]")
                ax.set_ylabel("O-C [min]")
                ax.set_title(f"TIC {tic}: raw secondary O-C vs. LTTE-robust sec-minus-primary")
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(outdir, f"TIC{tic}_secminusprimary.png"), dpi=120)
                plt.close(fig)
            else:
                print("Not enough overlapping sectors between primary and secondary "
                      "fits to run the sec-minus-primary cross-check.")

        result.update({
            "secondary_depth": sec_depth_precise,
            "secondary_phase": sec_phase_final,
            "implied_ecosw": ecosw,
            "n_secondary_qc": len(sf_qc),
            "apsidal_slope_min_per_1000d": slope * 1000,
            "apsidal_slope_err_min_per_1000d": se * 1000,
            "apsidal_p": p,
            "apsidal_r": r,
            "mechanistic_checks": checks,
            "primary_curvature_flagged": curvature_flagged,
            "sec_minus_primary": ({
                "n_sectors": secpm["n_sectors"],
                "slope_min_per_1000d": secpm["slope_min_per_1000d"],
                "slope_err_min_per_1000d": secpm["slope_err_min_per_1000d"],
                "p": secpm["p"],
                "r": secpm["r"],
            } if secpm is not None else None),
        })

        if p < 0.001:
            verdict = "SOLID"
        elif p < 0.05:
            verdict = "TENTATIVE"
        else:
            verdict = "INCONCLUSIVE"
        # Auto-demotion: if the sec-minus-primary cross-check ran, the apsidal
        # magnitude must survive it (same sign, >=50% of the raw slope). If it
        # collapsed or flipped, the raw slope was likely riding on the shared
        # LTTE drift -- demote regardless of the raw p-value.
        if secpm is not None and verdict in ("SOLID", "TENTATIVE"):
            raw_slope_1000d = slope * 1000
            xs = secpm["slope_min_per_1000d"]
            retained = (raw_slope_1000d != 0 and xs * raw_slope_1000d > 0
                        and abs(xs) >= 0.5 * abs(raw_slope_1000d))
            if retained:
                print(f"Cross-check magnitude retained ({xs:+.3f} vs raw "
                      f"{raw_slope_1000d:+.3f} min/1000d) -- verdict stands.")
            else:
                print(f"Cross-check magnitude NOT retained ({xs:+.3f} vs raw "
                      f"{raw_slope_1000d:+.3f} min/1000d) -- auto-demoting "
                      f"{verdict} -> INCONCLUSIVE (cross-check failed).")
                verdict = "INCONCLUSIVE (cross-check failed)"
        result["verdict"] = verdict
        print(f"\nVERDICT: {verdict} (p={p:.4g})")
    else:
        if has_secondary:
            print(f"\nSecondary detected but only {len(sf_qc)} pass QC -- too few for an O-C test.")
        result["verdict"] = "NO_SECONDARY_ANALYSIS"

    # Save summary JSON
    summary_path = os.path.join(outdir, f"TIC{tic}_summary.json")
    def json_default(o):
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, float):
            return None if np.isnan(o) else o
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(summary_path, "w") as fh:
        json.dump(result, fh, indent=2, default=json_default)
    print(f"\nSummary written to {summary_path}")
    print(f"Plots written to {outdir}/")


if __name__ == "__main__":
    main()

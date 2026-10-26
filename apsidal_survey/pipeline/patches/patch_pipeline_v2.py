"""Patch apsidal_pipeline.py and local_eclipse_model.py (fix of the eclipse-fitting setup).

Changes:
  1. The secondary eclipse gets its own measured half-width, fit window and
     shift bound (previously everything used the primary's width).
  2. Templates and fit windows include out-of-eclipse baseline on both sides
     (window = 2 x eclipse half-width).
  3. Templates, fits and the ephemeris are iterated: after each pass the
     templates and all eclipse fits are redone with the refined period, epoch
     and secondary phase, until the predicted eclipse times stop changing.
  4. Template minima are located between bins (parabola), so template
     centres do not jump by a bin width between passes.
  5. Each fit records whether its timing shift hit the bound; counts, widths,
     passes and the first-pass period are saved in the summary JSON.

Run from the folder that contains both files. Backups: *.py.bak_v2.
Nothing is written unless every replacement matches exactly once.
"""
import sys

PIPE, MODEL = "apsidal_pipeline.py", "local_eclipse_model.py"
pipe = open(PIPE).read()
model = open(MODEL).read()
new_pipe, new_model = pipe, model
problems = []


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        problems.append(f"{label}: expected 1 match, found {n}")
        return text
    return text.replace(old, new)


# ---------------------------------------------------------------- model ----
new_model = replace_once(
    new_model,
    '        "shift_phase": float(result.x[3]),\n',
    '        "shift_phase": float(result.x[3]),\n'
    '        "max_shift_phase": float(max_shift_phase),\n'
    '        "at_bound": bool(abs(result.x[3]) > 0.95 * max_shift_phase),\n',
    "model: record shift bound",
)
new_model = replace_once(
    new_model,
    '        template = templates[kind]\n'
    '        row = fit_single_eclipse(\n'
    '            time,\n'
    '            flux,\n'
    '            center_time,\n'
    '            period,\n'
    '            template,\n'
    '            half_window_phase=half_window_phase,\n'
    '            max_shift_phase=max_shift_phase,\n',
    '        template = templates[kind]\n'
    '        hw_kind = half_window_phase[kind] if isinstance(half_window_phase, dict) else half_window_phase\n'
    '        ms_kind = max_shift_phase[kind] if isinstance(max_shift_phase, dict) else max_shift_phase\n'
    '        row = fit_single_eclipse(\n'
    '            time,\n'
    '            flux,\n'
    '            center_time,\n'
    '            period,\n'
    '            template,\n'
    '            half_window_phase=hw_kind,\n'
    '            max_shift_phase=ms_kind,\n',
    "model: per-eclipse-type window and shift bound",
)

new_model = replace_once(
    new_model,
    '    min_index = int(np.nanargmin(bflux_interp))\n'
    '    min_phase = float(centers[min_index])\n',
    '    min_index = int(np.nanargmin(bflux_interp))\n'
    '    min_phase = float(centers[min_index])\n'
    '    # Refine the minimum below the bin width with a parabola through the\n'
    '    # neighbouring bins, so the template centre does not jump by whole bins.\n'
    '    lo_i, hi_i = max(min_index - 5, 0), min(min_index + 6, len(centers))\n'
    '    if hi_i - lo_i >= 5:\n'
    '        qa, qb, _ = np.polyfit(centers[lo_i:hi_i], bflux_interp[lo_i:hi_i], 2)\n'
    '        if qa > 0 and centers[lo_i] <= -qb / (2 * qa) <= centers[hi_i - 1]:\n'
    '            min_phase = float(-qb / (2 * qa))\n',
    "model: sub-bin template minimum",
)

# ------------------------------------------------------------- pipeline ----
start_marker = "    # 7. Build templates and fit all eclipses\n"
end_marker = "    F_naive, p_naive = curvature_check(times_p, resid_p_min)\n"
if new_pipe.count(start_marker) != 1 or new_pipe.count(end_marker) != 1:
    problems.append("pipeline: step 7-9 markers not found exactly once")
else:
    i0 = new_pipe.index(start_marker)
    i1 = new_pipe.index(end_marker)
    NEW_BLOCK = '''    # 7-9. Templates, eclipse fits and refined ephemeris, iterated.
    # Each eclipse type gets its own measured half-width, fit window and shift
    # bound. After each pass, templates and fits are redone with the refined
    # period, epoch and secondary phase, so fit windows are never centred on a
    # drifting first-pass prediction (this produced fake secondary drifts for
    # long-period systems with wide secondaries, e.g. TIC 289044378).
    from local_eclipse_model import median_bin_template
    WINDOW_FACTOR = 2.0   # template/fit window = WINDOW_FACTOR x eclipse half-width
    SHIFT_FACTOR = 0.4    # max timing shift = SHIFT_FACTOR x eclipse half-width
    MAX_PASSES = 6
    CONVERGED_MIN = 0.5   # stop when predicted eclipse times move less than this [min]
    P_first = P0
    P_use, T0_use, sec_phase_use = P0, T0, sec_phase_rough
    n_passes, converged_shift = 0, np.nan
    for n_pass in range(1, MAX_PASSES + 1):
        n_passes = n_pass
        print(f"\\n--- Pass {n_pass}: templates and eclipse fits with "
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
            print("\\nFATAL: primary eclipse template could not be built.")
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
            print("\\nFATAL: fewer than 3 QC-passed primary eclipses -- cannot proceed.")
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
        if n_pass >= 2 and converged_shift < CONVERGED_MIN:
            break
    else:
        print(f"WARNING: ephemeris not converged after {MAX_PASSES} passes "
              f"(last change {converged_shift:.2f} min)")

    primary_half_width = hw_p
    secondary_half_width = hw_s
    print("\\n--- Refined primary ephemeris (final pass) ---")
    print(f"Refined P  = {P_refined:.8f} d   (first-pass P0 = {P_first:.8f} d)")
    print(f"Refined T0 = {T0_refined:.8f} BTJD")
    print(f"Primary O-C: std={np.std(resid_p_min):.4f} min, "
          f"range {resid_p_min.min():.4f} to {resid_p_min.max():.4f} min, n={len(epochs_p)}")

'''
    new_pipe = new_pipe[:i0] + NEW_BLOCK + new_pipe[i1:]

new_pipe = replace_once(
    new_pipe,
    '        "period_alias_caveat": alias_caveat,\n    }\n',
    '        "period_alias_caveat": alias_caveat,\n'
    '        "pipeline_version": 2,\n'
    '        "n_passes": n_passes,\n'
    '        "converged_shift_min": converged_shift,\n'
    '        "first_pass_period_days": P_first,\n'
    '        "primary_half_width_phase": primary_half_width,\n'
    '        "secondary_half_width_phase": secondary_half_width,\n'
    '        "n_primary_at_shift_bound": n_bound_p,\n'
    '        "n_secondary_at_shift_bound": n_bound_s,\n'
    '    }\n',
    "pipeline: diagnostics in summary JSON",
)
new_pipe = replace_once(
    new_pipe,
    "                                     primary_half_width, sf_qc, times_s, oc_min, primary_depth)",
    "                                     max(primary_half_width, secondary_half_width),\n"
    "                                     sf_qc, times_s, oc_min, primary_depth)",
    "pipeline: mechanistic checks mask both eclipse widths",
)

if problems:
    print("NOT PATCHED. Problems:")
    for p in problems:
        print("  -", p)
    sys.exit(1)

open(PIPE + ".bak_v2", "w").write(pipe)
open(MODEL + ".bak_v2", "w").write(model)
open(PIPE, "w").write(new_pipe)
open(MODEL, "w").write(new_model)
print("Patched apsidal_pipeline.py and local_eclipse_model.py")
print("Backups: apsidal_pipeline.py.bak_v2, local_eclipse_model.py.bak_v2")

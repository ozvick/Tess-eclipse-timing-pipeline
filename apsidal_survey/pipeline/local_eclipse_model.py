import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares


def phase01(time, epoch, period):
    """Orbital phase in [0, 1)."""
    return ((np.asarray(time, dtype=float) - epoch) / period) % 1.0


def phase_distance(phase, center):
    """Shortest signed phase distance from center, in [-0.5, 0.5)."""
    return ((np.asarray(phase, dtype=float) - center + 0.5) % 1.0) - 0.5


def split_time_segments(time, gap_days=10.0, min_points=100):
    """Split a light curve into continuous observing segments using time gaps."""
    time = np.asarray(time, dtype=float)
    good = np.isfinite(time)
    idx = np.where(good)[0]
    if len(idx) == 0:
        return []

    order = idx[np.argsort(time[idx])]
    gaps = np.diff(time[order])
    breaks = np.where(gaps > gap_days)[0] + 1
    groups = np.split(order, breaks)
    return [g for g in groups if len(g) >= min_points]


def local_baseline_normalize(
    time,
    flux,
    epoch,
    period,
    segments,
    eclipse_centers=(0.0, 0.5),
    mask_width_phase=0.10,
    poly_order=1,
):
    """
    Normalize each segment with its own out-of-eclipse baseline.

    This removes slow segment-to-segment offsets without forcing one global
    baseline through all sectors.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    phase = phase01(time, epoch, period)
    norm_flux = np.full_like(flux, np.nan, dtype=float)
    baseline = np.full_like(flux, np.nan, dtype=float)

    for seg in segments:
        seg = np.asarray(seg, dtype=int)
        out = np.isfinite(time[seg]) & np.isfinite(flux[seg])
        for center in eclipse_centers:
            out &= np.abs(phase_distance(phase[seg], center)) > mask_width_phase

        if np.sum(out) < poly_order + 3:
            base_level = np.nanmedian(flux[seg])
            baseline[seg] = base_level
            norm_flux[seg] = flux[seg] / base_level
            continue

        x = time[seg] - np.nanmedian(time[seg])
        coeff = np.polyfit(x[out], flux[seg][out], deg=poly_order)
        base = np.polyval(coeff, x)
        bad_base = ~np.isfinite(base) | (base == 0)
        if np.any(bad_base):
            base[bad_base] = np.nanmedian(flux[seg][out])
        baseline[seg] = base
        norm_flux[seg] = flux[seg] / base

    return norm_flux, baseline


def magnitude_segment_normalize(
    time,
    flux,
    epoch,
    period,
    segments,
    eclipse_centers=(0.0, 0.5),
    mask_width_phase=0.10,
    poly_order=0,
):
    """
    Normalize each segment in magnitude space using out-of-eclipse points.

    This follows the common light-curve procedure:
      1. mask eclipse windows,
      2. convert flux to magnitude,
      3. estimate the out-of-eclipse level per segment,
      4. subtract that magnitude level,
      5. convert back to normalized relative flux.

    With poly_order=0 this is equivalent to dividing by the segment's
    out-of-eclipse median flux, but it makes the magnitude-space operation
    explicit. With poly_order=1, it removes a linear magnitude trend per
    segment using only out-of-eclipse points.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    phase = phase01(time, epoch, period)

    norm_flux = np.full_like(flux, np.nan, dtype=float)
    baseline_mag = np.full_like(flux, np.nan, dtype=float)

    positive = np.isfinite(flux) & (flux > 0)
    mag = np.full_like(flux, np.nan, dtype=float)
    mag[positive] = -2.5 * np.log10(flux[positive])

    for seg in segments:
        seg = np.asarray(seg, dtype=int)
        out = np.isfinite(time[seg]) & np.isfinite(mag[seg])
        for center in eclipse_centers:
            out &= np.abs(phase_distance(phase[seg], center)) > mask_width_phase

        if np.sum(out) < max(poly_order + 3, 5):
            level = np.nanmedian(mag[seg][np.isfinite(mag[seg])])
            baseline_mag[seg] = level
        elif poly_order == 0:
            level = np.nanmedian(mag[seg][out])
            baseline_mag[seg] = level
        else:
            x = time[seg] - np.nanmedian(time[seg])
            coeff = np.polyfit(x[out], mag[seg][out], deg=poly_order)
            baseline_mag[seg] = np.polyval(coeff, x)

        delta_mag = mag[seg] - baseline_mag[seg]
        norm_flux[seg] = 10.0 ** (-0.4 * delta_mag)

    return norm_flux, baseline_mag


def median_bin_template(local_phase, flux, half_width=0.12, nbins=241, min_points=5):
    """Build a median-binned eclipse template in local phase coordinates.

    The returned template is recentered so its measured minimum is at phase 0.
    That way the fitted phase shift corresponds to the eclipse minimum timing,
    not just the arbitrary phase used to collect the template points.
    """
    local_phase = np.asarray(local_phase, dtype=float)
    flux = np.asarray(flux, dtype=float)
    bins = np.linspace(-half_width, half_width, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bflux = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        m = (
            (local_phase >= bins[i])
            & (local_phase < bins[i + 1])
            & np.isfinite(flux)
        )
        counts[i] = int(np.sum(m))
        if counts[i] >= min_points:
            bflux[i] = np.nanmedian(flux[m])

    good = np.isfinite(bflux)
    if np.sum(good) < 5:
        raise ValueError("Not enough points to build an eclipse template.")

    bflux_interp = np.interp(centers, centers[good], bflux[good])
    edge = np.abs(centers) > 0.75 * half_width
    base = np.nanmedian(bflux_interp[edge])
    if not np.isfinite(base) or base == 0:
        base = np.nanmedian(bflux_interp)
    bflux_interp = bflux_interp / base

    min_index = int(np.nanargmin(bflux_interp))
    min_phase = float(centers[min_index])
    centered_phase = centers - min_phase
    order = np.argsort(centered_phase)

    return {
        "phase": centered_phase[order],
        "flux": bflux_interp[order],
        "counts": counts,
        "half_width": half_width,
        "depth": float(1.0 - np.nanmin(bflux_interp)),
        "phase_shift_to_minimum": min_phase,
    }


def build_eclipse_templates(
    time,
    flux,
    epoch,
    period,
    primary_center=0.0,
    secondary_center=0.5,
    half_width=0.12,
    nbins=241,
):
    """Build separate primary and secondary templates from the folded data."""
    phase = phase01(time, epoch, period)
    primary_local = phase_distance(phase, primary_center)
    secondary_local = phase_distance(phase, secondary_center)

    primary_mask = np.abs(primary_local) < half_width
    secondary_mask = np.abs(secondary_local) < half_width

    primary = median_bin_template(
        primary_local[primary_mask],
        np.asarray(flux)[primary_mask],
        half_width=half_width,
        nbins=nbins,
    )
    secondary = median_bin_template(
        secondary_local[secondary_mask],
        np.asarray(flux)[secondary_mask],
        half_width=half_width,
        nbins=nbins,
    )
    return {"primary": primary, "secondary": secondary}


def _template_interp(template):
    return interp1d(
        template["phase"],
        template["flux"],
        kind="linear",
        bounds_error=False,
        fill_value=1.0,
    )


def predicted_eclipse_centers(time, epoch, period, secondary_phase=0.5):
    """Return predicted primary and secondary eclipse centers within the time span."""
    time = np.asarray(time, dtype=float)
    tmin = np.nanmin(time)
    tmax = np.nanmax(time)
    e_min = int(np.floor((tmin - epoch) / period)) - 1
    e_max = int(np.ceil((tmax - epoch) / period)) + 1

    centers = []
    for epoch_number in range(e_min, e_max + 1):
        primary_time = epoch + epoch_number * period
        secondary_time = epoch + (epoch_number + secondary_phase) * period
        if tmin <= primary_time <= tmax:
            centers.append((epoch_number, "primary", primary_time, 0.0))
        if tmin <= secondary_time <= tmax:
            centers.append((epoch_number, "secondary", secondary_time, secondary_phase))

    centers.sort(key=lambda row: row[2])
    return centers


def fit_single_eclipse(
    time,
    flux,
    center_time,
    period,
    template,
    half_window_phase=0.12,
    max_shift_phase=0.04,
    min_points=25,
):
    """
    Fit one eclipse using a local baseline and a shifted eclipse template.

    Parameters fitted per eclipse:
      baseline, linear slope, depth scale, phase shift
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    xphase = (time - center_time) / period
    m = np.isfinite(time) & np.isfinite(flux) & (np.abs(xphase) < half_window_phase)
    if np.sum(m) < min_points:
        return None

    x = xphase[m]
    y = flux[m]
    t_rel = time[m] - center_time
    interp_template = _template_interp(template)

    base0 = float(np.nanmedian(y[np.abs(x) > 0.75 * half_window_phase]))
    if not np.isfinite(base0):
        base0 = float(np.nanmedian(y))
    scale0 = 1.0
    p0 = np.array([base0, 0.0, scale0, 0.0], dtype=float)
    lower = np.array([0.5, -5.0, 0.0, -max_shift_phase], dtype=float)
    upper = np.array([1.5, 5.0, 3.0, max_shift_phase], dtype=float)

    def model(p):
        baseline, slope, scale, shift = p
        template_flux = interp_template(x - shift)
        return baseline + slope * t_rel + scale * (template_flux - 1.0)

    def resid(p):
        return model(p) - y

    result = least_squares(
        resid,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.003,
        max_nfev=10000,
    )

    fitted = model(result.x)
    rms = float(np.sqrt(np.nanmean((fitted - y) ** 2)))
    obs_center = center_time + result.x[3] * period

    order = np.argsort(x)
    return {
        "success": bool(result.success),
        "center_time": float(center_time),
        "obs_center": float(obs_center),
        "oc_days": float(obs_center - center_time),
        "oc_min": float((obs_center - center_time) * 24.0 * 60.0),
        "baseline": float(result.x[0]),
        "slope": float(result.x[1]),
        "scale": float(result.x[2]),
        "shift_phase": float(result.x[3]),
        "rms": rms,
        "n_points": int(np.sum(m)),
        "xphase": x[order],
        "flux": y[order],
        "model": fitted[order],
        "residual": (y - fitted)[order],
    }


def fit_local_eclipses(
    time,
    flux,
    epoch,
    period,
    templates,
    secondary_phase=0.5,
    half_window_phase=0.12,
    max_shift_phase=0.04,
    min_points=25,
):
    """Fit all predicted primary and secondary eclipses independently."""
    fits = []
    for epoch_number, kind, center_time, center_phase in predicted_eclipse_centers(
        time, epoch, period, secondary_phase=secondary_phase
    ):
        template = templates[kind]
        row = fit_single_eclipse(
            time,
            flux,
            center_time,
            period,
            template,
            half_window_phase=half_window_phase,
            max_shift_phase=max_shift_phase,
            min_points=min_points,
        )
        if row is None:
            continue
        row["epoch"] = int(epoch_number)
        row["kind"] = kind
        row["center_phase"] = float(center_phase)
        fits.append(row)
    return fits


def plot_template_check(time, flux, epoch, period, templates, secondary_phase=0.5):
    """Plot folded data against the separate primary and secondary templates."""
    phase = phase01(time, epoch, period)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, kind, center in [
        (axes[0], "primary", 0.0),
        (axes[1], "secondary", secondary_phase),
    ]:
        local = phase_distance(phase, center)
        m = np.abs(local) < templates[kind]["half_width"]
        ax.plot(local[m], np.asarray(flux)[m], ".", ms=2, alpha=0.25, label="data")
        ax.plot(templates[kind]["phase"], templates[kind]["flux"], color="black", lw=2, label="template")
        ax.axvline(0.0, color="0.35", ls="--", lw=1)
        ax.set_title(f"{kind.capitalize()} eclipse template")
        ax.set_xlabel(r"$\Delta$Phase")
        ax.set_ylabel("Normalized flux")
        ax.legend()

    fig.tight_layout()
    return fig


def plot_eclipse_fit_grid(fits, n_show=8):
    """Plot individual eclipse data, local model, and residuals."""
    rows = min(int(n_show), len(fits))
    if rows == 0:
        raise ValueError("No eclipse fits available to plot.")

    fig, axes = plt.subplots(rows, 2, figsize=(10, 2.3 * rows), sharex=False)
    if rows == 1:
        axes = np.asarray([axes])

    for axrow, fit in zip(axes, fits[:rows]):
        ax = axrow[0]
        ax.plot(fit["xphase"], fit["flux"], ".", ms=3, label="data")
        ax.plot(fit["xphase"], fit["model"], color="tab:orange", lw=2, label="local model")
        ax.axvline(0.0, color="0.45", ls="--", lw=1, label="calculated center")
        ax.axvline(fit["shift_phase"], color="black", lw=1.5, label="observed center")
        ax.set_title(f"{fit['kind']} epoch {fit['epoch']}: O-C = {fit['oc_min']:.2f} min")
        ax.set_ylabel("Normalized flux")

        rax = axrow[1]
        rax.axhline(0.0, color="0.5", lw=1)
        rax.plot(fit["xphase"], fit["residual"], ".", ms=3)
        rax.set_title(f"residual RMS = {fit['rms']:.4f}")
        rax.set_ylabel("data - model")

    axes[-1, 0].set_xlabel(r"$\Delta$Phase")
    axes[-1, 1].set_xlabel(r"$\Delta$Phase")
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    return fig


def summarize_fit_table(fits):
    """Return a pandas DataFrame summary if pandas is available."""
    rows = []
    for fit in fits:
        rows.append({
            "epoch": fit["epoch"],
            "kind": fit["kind"],
            "calculated_center": fit["center_time"],
            "observed_center": fit["obs_center"],
            "oc_days": fit["oc_days"],
            "oc_min": fit["oc_min"],
            "baseline": fit["baseline"],
            "slope": fit["slope"],
            "scale": fit["scale"],
            "shift_phase": fit["shift_phase"],
            "rms": fit["rms"],
            "n_points": fit["n_points"],
        })

    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except Exception:
        return rows

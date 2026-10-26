"""Diagnostics describing how much spatial regularization changed a map.

Tool-agnostic: operates on plain arrays of strictly positive quantities
(e.g. density, radiation field), so it is usable by
:class:`~pdrtpy.tool.lineratiofit.LineRatioFit` today and, eventually,
:class:`~pdrtpy.tool.excitation.ExcitationFit`. Changes are measured in dex
(``log10(after) - log10(before)``) because the regularization penalty acts in
log10 space.
"""

import numpy as np

_PERCENTILES = (50, 90, 99)


def change_summary(before, after, threshold=0.0625, valid_mask=None, sigma=None):
    """Summarize how many pixels changed between two maps, and by how much.

    The change at each pixel is ``delta = log10(after) - log10(before)``
    (dex). A pixel counts as "changed" when ``|delta| > threshold``. Pixels
    that are NaN, non-positive, or infinite in either map, or are False in
    ``valid_mask``, are excluded from all statistics.

    The default threshold is set to 0.0625 which is half the step
    size of the model space.

    If ``sigma`` is given, the number of pixels whose change exceeds their own
    1-sigma uncertainty (converted to dex as ``sigma / (ln(10) * before)``) is
    also counted. This is the more physically meaningful measure of a
    "real" change.

    Parameters
    ----------
    before : `~numpy.ndarray`
        The map before regularization. Must be positive where valid.
    after : `~numpy.ndarray`
        The map after regularization; same shape as ``before``.
    threshold : float, optional
        Minimum ``|delta|`` in dex for a pixel to count as changed.
        Default: 0.0625 (half the 0.125 dex model grid step, about a 15%
        change).
    valid_mask : `~numpy.ndarray`, optional
        Boolean array, same shape as ``before``; True where a pixel is
        eligible. If None, all finite positive pixels are used.
    sigma : `~numpy.ndarray`, optional
        1-sigma uncertainty of ``before``, same shape and units as
        ``before``. Pixels where it is non-finite or non-positive are left
        out of the sigma comparison.

    Returns
    -------
    dict
        With keys ``n_valid``, ``n_changed``, ``fraction_changed``,
        ``mean_abs``, ``median_abs``, ``max_abs``, ``p90_abs``, ``p99_abs``
        (statistics of ``|delta|`` in dex), ``threshold``, and ``delta``
        (signed change in dex, 1-D array over the valid pixels). If ``sigma``
        is given, also ``n_sigma_valid`` (pixels with a usable uncertainty)
        and ``n_beyond_sigma`` (those whose ``|delta|`` exceeds their
        uncertainty in dex); both are None otherwise. If there are no valid
        pixels, counts are 0 and the statistics are NaN.

    Raises
    ------
    ValueError
        If ``before``, ``after``, ``valid_mask`` (or ``sigma``) differ in shape, or
        ``threshold`` is negative.
    """
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.shape != after.shape:
        raise ValueError(f"before and after must have the same shape; got {before.shape} and {after.shape}.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    good = np.isfinite(before) & np.isfinite(after) & (before > 0) & (after > 0)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != before.shape:
            raise ValueError(f"valid_mask must have shape {before.shape}; got {valid_mask.shape}.")
        good &= valid_mask

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape != before.shape:
            raise ValueError(f"sigma must have shape {before.shape}; got {sigma.shape}.")

    delta = np.log10(after[good]) - np.log10(before[good])
    absdelta = np.abs(delta)
    n_valid = int(delta.size)
    n_changed = int(np.count_nonzero(absdelta > threshold))

    summary = {
        "n_valid": n_valid,
        "n_changed": n_changed,
        "fraction_changed": n_changed / n_valid if n_valid else np.nan,
        "threshold": threshold,
        "delta": delta,
        "n_sigma_valid": None,
        "n_beyond_sigma": None,
    }
    if sigma is not None:
        sigma_good = sigma[good]
        usable = np.isfinite(sigma_good) & (sigma_good > 0)
        sigma_dex = sigma_good[usable] / (np.log(10.0) * before[good][usable])
        summary["n_sigma_valid"] = int(np.count_nonzero(usable))
        summary["n_beyond_sigma"] = int(np.count_nonzero(absdelta[usable] > sigma_dex))
    if n_valid:
        summary["mean_abs"] = float(np.mean(absdelta))
        summary["median_abs"] = float(np.median(absdelta))
        summary["max_abs"] = float(np.max(absdelta))
        p90, p99 = np.percentile(absdelta, _PERCENTILES[1:])
        summary["p90_abs"] = float(p90)
        summary["p99_abs"] = float(p99)
    else:
        for key in ("mean_abs", "median_abs", "max_abs", "p90_abs", "p99_abs"):
            summary[key] = np.nan
    return summary


def format_summary(summary, name):
    """Format the output of :func:`change_summary` as a brief text block.

    Parameters
    ----------
    summary : dict
        A dictionary returned by :func:`change_summary`.
    name : str
        Label for the quantity, such as ``"density"``.

    Returns
    -------
    str
        A multi-line, human-readable report.
    """
    text = (
        f"{name}: {summary['n_changed']} of {summary['n_valid']} pixels changed by more than "
        f"{summary['threshold']:g} dex ({100 * summary['fraction_changed']:.1f}%)\n"
        f"  |change| [dex]: median {summary['median_abs']:.3g}, mean {summary['mean_abs']:.3g}, "
        f"90th pct {summary['p90_abs']:.3g}, 99th pct {summary['p99_abs']:.3g}, max {summary['max_abs']:.3g}"
    )
    if summary["n_beyond_sigma"] is not None:
        text += f"\n  {summary['n_beyond_sigma']} of {summary['n_sigma_valid']} pixels changed by more than 1 sigma"
    return text

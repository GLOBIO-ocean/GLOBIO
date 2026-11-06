"""Error metrics, statistical tests and MC dropout uncertainty metrics.

All errors are in the physical units of the target. The profile-wise RMSE is
the root mean square error over the levels of one profile.
"""
import warnings

import numpy as np
import pandas as pd

from .constants import LATITUDE_BANDS, REGIONS


def profile_rmse(prediction, target, levels=None, mask=None):
    """RMSE of each profile, over ``levels`` (indices) or the ``mask``-ed levels."""
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    if prediction.shape != target.shape:
        raise ValueError(f"Shapes differ: {prediction.shape} vs {target.shape}")
    if levels is not None:
        prediction, target = prediction[:, levels], target[:, levels]
        mask = None if mask is None else np.asarray(mask)[:, levels]
    squared = (prediction - target) ** 2
    if mask is None:
        return np.sqrt(squared.mean(axis=1))
    mask = np.asarray(mask, bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.where(mask, squared, 0).sum(axis=1) / mask.sum(axis=1))


def describe(values):
    values = np.asarray(values, float)
    if not values.size:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan}
    return {"n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std())}


def group_masks(meta, groups=("all", "summer", "winter") + LATITUDE_BANDS):
    """Boolean masks of profile groups: ``all``, a season, or a region of
    :data:`globio.constants.REGIONS` (e.g. the latitude bands ``r1``-``r6``)."""
    masks = {}
    for group in groups:
        if group == "all":
            masks[group] = np.ones(len(meta), dtype=bool)
        elif group in ("winter", "spring", "summer", "autumn"):
            masks[group] = (meta["season"] == group).to_numpy()
        elif group in REGIONS:
            from .data import region_mask
            masks[group] = region_mask(meta["lat"], meta["lon"], group)
        else:
            raise ValueError(f"Unknown group {group!r}")
    return masks


def summarize(errors, meta, groups=("all", "summer", "winter") + LATITUDE_BANDS):
    """mean / median / std of each error column of ``errors`` (dict name -> array)
    over the profile groups of :func:`group_masks`."""
    rows = []
    for group, mask in group_masks(meta, groups).items():
        for method, values in errors.items():
            rows.append({"group": group, "method": method, **describe(np.asarray(values)[mask])})
    return pd.DataFrame(rows)


def depth_statistics(prediction, target, meta, groups=LATITUDE_BANDS,
                     seasons=("all", "winter", "summer"), pressure=None):
    """Depth-resolved errors for each group x season and level.

    With ``e = prediction - target`` at one level, reports across profiles the
    mean absolute error and its standard deviation, the bias (mean of ``e``)
    and the RMSE.
    """
    errors = np.asarray(prediction, float) - np.asarray(target, float)
    pressure = np.arange(errors.shape[1]) if pressure is None else np.asarray(pressure)
    region_masks = group_masks(meta, groups)
    season_masks = group_masks(meta, seasons)
    rows = []
    for group, group_mask in region_masks.items():
        for season_name, season_mask in season_masks.items():
            e = errors[group_mask & season_mask]
            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                stats = {"mean_abs_error": np.abs(e).mean(0), "sd_abs_error": np.abs(e).std(0),
                         "bias": e.mean(0), "rmse": np.sqrt((e ** 2).mean(0))}
            frame = pd.DataFrame({"group": group, "season": season_name, "level": np.arange(len(pressure)),
                                  "pressure": pressure, "n_profiles": len(e), **stats})
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistical comparison of methods
# ---------------------------------------------------------------------------

def holm(pvalues, alpha=0.05):
    """Holm step-down correction: boolean array of rejected null hypotheses."""
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    reject = np.zeros(pvalues.size, dtype=bool)
    for rank, index in enumerate(order):
        if pvalues[index] > alpha / (pvalues.size - rank):
            break
        reject[index] = True
    return reject


def wilcoxon_best_vs_all(errors, alpha=0.05, decimals=None):
    """Paired one-sided Wilcoxon signed-rank tests between methods.

    For each method m1 and every other method m2, tests whether the errors of
    m1 are smaller (H1: m1 < m2); the p-values of m1 are Holm-corrected and m1
    is *significantly best* if it is better than all the others. ``errors``
    maps method names to per-profile errors on the same profiles. With
    ``decimals``, errors are rounded before the test (the reference evaluation uses 2).

    Returns ``(best, pvalues)``: ``best[m]`` is a bool, ``pvalues[m1][m2]`` the
    raw p-value of m1 < m2.
    """
    from scipy.stats import wilcoxon
    names = list(errors)
    data = {m: np.asarray(errors[m], float) for m in names}
    if decimals is not None:
        data = {m: np.round(v, decimals) for m, v in data.items()}
    sizes = {v.size for v in data.values()}
    if len(sizes) != 1:
        raise ValueError("All methods must have errors on the same profiles")
    best, pvalues = {}, {m: {} for m in names}
    for m1 in names:
        others = [m2 for m2 in names if m2 != m1]
        for m2 in others:
            a, b = data[m1], data[m2]
            pvalues[m1][m2] = 1.0 if np.array_equal(a, b) else float(wilcoxon(a, b, alternative="less").pvalue)
        best[m1] = bool(others) and bool(holm([pvalues[m1][m2] for m2 in others], alpha).all())
    return best, pvalues


def compare_methods(errors, alpha=0.05, decimals=2):
    """Ranking by median error and Wilcoxon + Holm test (see
    :func:`wilcoxon_best_vs_all`). Returns a table with one row per method."""
    best, pvalues = wilcoxon_best_vs_all(errors, alpha=alpha, decimals=decimals)
    rows = [{"method": m, **describe(v), "significantly_best": best[m],
             **{f"p_less_than_{other}": p for other, p in pvalues[m].items()}}
            for m, v in errors.items()]
    table = pd.DataFrame(rows).sort_values("median", kind="stable").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


# ---------------------------------------------------------------------------
# MC dropout uncertainty
# ---------------------------------------------------------------------------

def uncertainty_metrics(target, mean, std, mask=None, pressure=None, ks=(1, 2, 3)):
    """Calibration of MC dropout predictions.

    Returns three tables:

    * per profile: number of valid levels, RMSE of the MC mean, RMS of the
      per-level MC standard deviation (``uncertainty_rms``) and, for k in
      ``ks``, whether all the valid levels are within ``k`` sigma
      (``covered_entire_<k>sigma``) and the fraction of levels that are
      (``covered_fraction_<k>sigma``);
    * coverage: for each k, the percentage of profiles entirely covered and
      the point-wise coverage (fractions averaged with equal weight per
      profile);
    * vertical: median absolute error and median sigma at each level.

    ``mask`` marks valid target levels (default: finite values).
    """
    target, mean, std = (np.asarray(a, float) for a in (target, mean, std))
    mask = np.isfinite(target) if mask is None else np.asarray(mask, bool)
    if not (target.shape == mean.shape == std.shape == mask.shape):
        raise ValueError("target, mean, std and mask must have the same shape")
    if not mask.any(axis=1).all():
        raise ValueError("Every profile must have at least one valid level")
    if not (np.isfinite(mean).all() and np.isfinite(std).all() and (std >= 0).all()):
        raise ValueError("Invalid predictive statistics")
    pressure = np.arange(target.shape[1]) if pressure is None else np.asarray(pressure)
    error = np.where(mask, np.abs(target - mean), np.nan)
    sigma = np.where(mask, std, np.nan)
    per_profile = pd.DataFrame({
        "valid_levels": mask.sum(1),
        "rmse": np.sqrt(np.nanmean(error ** 2, axis=1)),
        "uncertainty_rms": np.sqrt(np.nanmean(sigma ** 2, axis=1)),
    })
    coverage = []
    for k in ks:
        with np.errstate(invalid="ignore"):
            inside = (error <= k * std) & mask
        entire = (inside | ~mask).all(axis=1)
        fraction = inside.sum(1) / mask.sum(1)
        per_profile[f"covered_entire_{k}sigma"] = entire
        per_profile[f"covered_fraction_{k}sigma"] = fraction
        coverage.append({"k": k, "n_profiles": len(target),
                         "entire_profile_percent": 100 * entire.mean(),
                         "pointwise_percent": 100 * fraction.mean()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # levels without observations stay NaN
        vertical = pd.DataFrame({"pressure": pressure, "n_profiles": mask.sum(0),
                                 "median_absolute_error": np.nanmedian(error, axis=0),
                                 "median_sigma": np.nanmedian(sigma, axis=0)})
    return per_profile, pd.DataFrame(coverage), vertical

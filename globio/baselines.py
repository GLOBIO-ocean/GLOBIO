"""Baselines: CANYON-B (NITRATE) and SOCA (CHLA).

* CANYON-B (Bittig et al., 2018) through ``canyonbpy``, run on the same
  temperature, salinity, oxygen, position and date as the network.
* SOCA: the Copernicus product ``cmems_obs-mob_glo_bgc-chl-poc_my_0.25deg_P7D-m``
  at the 22 SOCA depths, downloaded for each profile with
  ``tools/download_soca.py``; models are compared with SOCA on the CHLA levels
  :data:`globio.constants.SOCA_LEVEL_INDICES`.
"""
import csv
import sys

import numpy as np

from .constants import SOCA_LEVEL_INDICES, pressure_levels
from .data import profile_date

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

#: SOCA profiles with values >= this (fill values) or all zeros (failed download)
#: are invalid.
SOCA_MAX_VALID = 100.0
SOCA_COORD_ATOL = 1e-4


def canyonb_nitrate(dataset, verbose=True):
    """CANYON-B NO3 and its uncertainty ``NO3_ci`` for every profile of a
    NITRATE :class:`globio.data.ProfileDataset`, shape ``(n_profiles, 200)``."""
    try:
        from canyonbpy import canyonb
    except ImportError as error:
        raise ImportError("The CANYON-B baseline needs canyonbpy: pip install canyonbpy") from error
    if dataset.variable != "NITRATE":
        raise ValueError("CANYON-B is the NITRATE baseline")
    pres = pressure_levels("NITRATE")
    n = pres.size
    means, sigmas = [], []
    for i in range(len(dataset)):
        out = canyonb(gtime=[profile_date(dataset.year[i], dataset.day_rad[i])] * n,
                      lat=np.full(n, dataset.lat[i]), lon=np.full(n, dataset.lon[i]), pres=pres,
                      temp=dataset.temp[i].astype(float), psal=dataset.psal[i].astype(float),
                      doxy=dataset.doxy[i].astype(float), param=["NO3"])
        means.append(np.asarray(out["NO3"], dtype=float).reshape(n))
        sigmas.append(np.asarray(out["NO3_ci"], dtype=float).reshape(n))
        if verbose and (i + 1) % 500 == 0:
            print(f"CANYON-B: {i + 1}/{len(dataset)}", flush=True)
    if not means:
        return np.empty((0, n)), np.empty((0, n))
    return np.stack(means), np.stack(sigmas)


def _parse_soca(text):
    text = str(text).strip().strip("[]").replace(",", " ")
    return np.asarray([float(v) for v in text.split()], dtype=float)


def read_soca_csv(path):
    """Read a SOCA CSV written by ``tools/download_soca.py``.

    One column per profile (header: profile id) and the rows ``year``,
    ``day_rad``, ``lat``, ``lon`` and ``chla_SOCA`` (22 values). Returns a dict
    profile id -> (lat, lon, values).
    """
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    labels = [row[0].strip() for row in rows[1:]]
    if labels[:5] != ["year", "day_rad", "lat", "lon", "chla_SOCA"]:
        raise ValueError(f"{path}: unexpected row labels {labels[:5]}")
    header, lat, lon, values = rows[0], rows[3], rows[4], rows[5]
    return {header[j]: (float(lat[j]), float(lon[j]), _parse_soca(values[j]))
            for j in range(1, len(header))}


def soca_for_dataset(path, dataset):
    """SOCA profiles aligned with the profiles of ``dataset`` (by profile id).

    Returns ``(values, valid)``: ``values`` has shape ``(n_profiles, 22)`` (NaN
    for invalid profiles) and ``valid`` marks profiles with a usable SOCA
    estimate. Raises if a profile is missing from the SOCA file or if the
    coordinates differ.
    """
    soca = read_soca_csv(path)
    missing = [p for p in dataset.profile_ids if p not in soca]
    if missing:
        raise ValueError(f"{len(missing)} profiles of {dataset.source} are not in {path}, e.g. "
                         f"{missing[0]}: download them with tools/download_soca.py")
    n_levels = len(SOCA_LEVEL_INDICES)
    values = np.full((len(dataset), n_levels), np.nan)
    valid = np.zeros(len(dataset), dtype=bool)
    for i, profile in enumerate(dataset.profile_ids):
        lat, lon, profile_values = soca[profile]
        if abs(lat - dataset.lat[i]) > SOCA_COORD_ATOL or abs(lon - dataset.lon[i]) > SOCA_COORD_ATOL:
            raise ValueError(f"{profile}: coordinates in {path} ({lat}, {lon}) differ from the dataset "
                             f"({dataset.lat[i]}, {dataset.lon[i]})")
        if profile_values.size != n_levels:
            continue
        # Fill values mark missing levels; all zeros a failed download.
        if np.all(np.isfinite(profile_values)) and np.all(profile_values < SOCA_MAX_VALID) \
                and not np.all(profile_values == 0):
            values[i], valid[i] = profile_values, True
    return values, valid


def canyonb_range_analysis(prediction, target, canyonb_mean, canyonb_sigma, rmse_model, rmse_canyonb):
    """Where model predictions and measurements fall with respect to the
    CANYON-B range ``NO3 +/- NO3_ci``, and how the range compares with the
    errors. Returns ``(per_profile, summary)`` DataFrames; the summary covers
    all profiles and those where the model beats CANYON-B."""
    import pandas as pd
    pred_inside = np.abs(prediction - canyonb_mean) <= canyonb_sigma
    target_inside = np.abs(target - canyonb_mean) <= canyonb_sigma
    rmse_model, rmse_canyonb = np.asarray(rmse_model), np.asarray(rmse_canyonb)
    per_profile = pd.DataFrame({
        "pred_inside_fraction": pred_inside.mean(1),
        "target_inside_fraction": target_inside.mean(1),
        "pred_inside_entire": pred_inside.all(1),
        "target_inside_entire": target_inside.all(1),
        "canyonb_sigma_rms": np.sqrt(np.mean(canyonb_sigma ** 2, axis=1)),  # RMS over levels, like the RMSE
    })
    per_profile["sigma_gt_rmse_canyonb"] = per_profile.canyonb_sigma_rms.to_numpy() > rmse_canyonb
    per_profile["sigma_gt_rmse_model"] = per_profile.canyonb_sigma_rms.to_numpy() > rmse_model
    per_profile["model_better"] = rmse_model < rmse_canyonb
    rows = []
    for subset, mask in (("all profiles", np.ones(len(per_profile), dtype=bool)),
                         ("model better than CANYON-B", per_profile.model_better.to_numpy())):
        part = per_profile[mask]
        rows.append({
            "subset": subset, "n_profiles": int(mask.sum()),
            "prediction inside range, % of levels": 100 * part.pred_inside_fraction.mean(),
            "prediction inside range, % of profiles (all levels)": 100 * part.pred_inside_entire.mean(),
            "target inside range, % of levels": 100 * part.target_inside_fraction.mean(),
            "target inside range, % of profiles (all levels)": 100 * part.target_inside_entire.mean(),
            "median CANYON-B sigma (RMS over levels)": part.canyonb_sigma_rms.median(),
            "median RMSE CANYON-B": np.median(rmse_canyonb[mask]) if mask.any() else np.nan,
            "median RMSE model": np.median(rmse_model[mask]) if mask.any() else np.nan,
            "% profiles sigma > RMSE CANYON-B": 100 * part.sigma_gt_rmse_canyonb.mean(),
            "% profiles sigma > RMSE model": 100 * part.sigma_gt_rmse_model.mean(),
        })
    return per_profile, pd.DataFrame(rows)

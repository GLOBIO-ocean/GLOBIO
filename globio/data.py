"""Dataset CSVs, input encodings and normalization.

Dataset CSV format
------------------
One column per profile and one row per field::

    ,<profile id 1>,<profile id 2>,...
    year,2019,2021,...
    day_rad,1.58,4.78,...
    lat,21.79,-18.64,...
    lon,-157.08,99.75,...
    temp,"tensor([24.15, 24.15, ...])",...      # 200 levels
    psal,"tensor([34.80, ...])",...             # 200 levels
    doxy,"tensor([210.81, ...])",...            # 200 levels
    CHLA,"tensor([0.025, ...])",...             # target, 200 levels (optional)

Profiles are lists of 200 comma-separated numbers, optionally wrapped in
``tensor([...])`` (the format written by the dataset-creation code) or ``[...]``.
The row-label column is optional. The target row can be omitted for inference
on new data. Profile ids are paths ``.../<float id>/<profile name>``: the parent
directory is the float id, used by float-wise splits.

``day_rad`` encodes the day of the year as ``((month - 1) * 30 + day) * 2 pi / 365``
(30-day pseudo-months, the convention used to build the datasets); see
:func:`day_rad_from_date` and :func:`profile_date`.
"""
import csv
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import (FILL_VALUE, INPUT_RANGES, INPUT_ROWS, METADATA_ROWS, N_LEVELS,
                        REGIONS, YEAR_OFFSET, check_variable)

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

FIELDS = ("year", "day", "space", "temp", "psal", "doxy", "target")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_profile(text, n_levels=N_LEVELS):
    """Parse ``"tensor([v1, v2, ...])"``, ``"[v1, ...]"`` or ``"v1, v2, ..."``."""
    text = str(text).strip()
    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        text = text[start + 1:end]
    try:
        values = np.fromstring(text, sep=",", dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"Cannot parse profile {text[:60]!r}...: {error}") from None
    if n_levels is not None and values.size != n_levels:
        raise ValueError(f"Expected {n_levels} levels, found {values.size}: {text[:60]!r}...")
    return values.astype(np.float32)


def read_profile_csv(path, variable=None):
    """Read a dataset CSV.

    Returns a dict with ``profile_ids`` (array of str), ``year``, ``day_rad``,
    ``lat``, ``lon`` (float64 arrays), ``temp``, ``psal``, ``doxy`` (float32,
    shape ``(n_profiles, 200)``), ``target`` (same shape, or ``None`` when the CSV
    has no target row) and ``variable`` (target row label, or ``variable``).
    """
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 1 + len(METADATA_ROWS) + len(INPUT_ROWS):
        raise ValueError(f"{path}: expected at least 8 rows (header, year, day_rad, lat, lon, "
                         f"temp, psal, doxy), found {len(rows)}")
    header, data = rows[0], rows[1:]
    labelled = data[0] and data[0][0].strip().lower() == "year"
    first = 1 if labelled else 0
    if labelled:
        labels = [row[0].strip() for row in data]
        expected = list(METADATA_ROWS + INPUT_ROWS)
        if [label.lower() for label in labels[:7]] != expected:
            raise ValueError(f"{path}: row labels {labels[:7]} differ from {expected}")
    if len(data) > 8:
        raise ValueError(f"{path}: expected at most 8 data rows, found {len(data)}")
    n_profiles = len(header) - first
    for i, row in enumerate(data):
        if len(row) != len(header):
            raise ValueError(f"{path}: row {i + 1} has {len(row)} fields, header has {len(header)}")

    target_name = None
    if len(data) == 8:
        target_name = labels[7].upper() if labelled else None
        if variable is not None:
            variable = check_variable(variable)
            if target_name is not None and target_name != variable:
                raise ValueError(f"{path}: target row is {target_name}, expected {variable}")
        target_name = target_name or (variable and check_variable(variable))

    out = {"profile_ids": np.asarray(header[first:], dtype=str),
           "variable": target_name or (variable and check_variable(variable))}
    for i, name in enumerate(METADATA_ROWS):
        out[name] = np.asarray([float(v) for v in data[i][first:]], dtype=np.float64)
    for i, name in enumerate(INPUT_ROWS + ("target",), start=len(METADATA_ROWS)):
        if i >= len(data):
            out[name] = None
            continue
        try:
            out[name] = (np.stack([parse_profile(v) for v in data[i][first:]])
                         if n_profiles else np.empty((0, N_LEVELS), np.float32))
        except ValueError as error:
            raise ValueError(f"{path}, row {name}: {error}") from None
    return out


# ---------------------------------------------------------------------------
# Encodings and decodings
# ---------------------------------------------------------------------------

def encode_year(year):
    return np.asarray(year, dtype=np.float64) - YEAR_OFFSET


def encode_day(day_rad):
    """(sin, cos) of the day angle, shape ``(..., 2)``."""
    day_rad = np.asarray(day_rad, dtype=np.float64)
    return np.stack([np.sin(day_rad), np.cos(day_rad)], axis=-1)


def encode_space(lat, lon):
    """Position on the unit sphere (x, y, z), shape ``(..., 3)``."""
    lat, lon = np.radians(np.asarray(lat, dtype=np.float64)), np.radians(np.asarray(lon, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def decode_space(xyz):
    """Inverse of :func:`encode_space`: (lat, lon) in degrees."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.degrees(np.arcsin(np.clip(xyz[..., 2], -1, 1))), np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))


def day_rad_from_date(date):
    """``day_rad`` of a date, with the 30-day pseudo-month convention of the datasets."""
    return ((date.month - 1) * 30 + date.day) * 2 * np.pi / 365


def day_of_year(day_rad):
    """Day index ``(month - 1) * 30 + day`` encoded by ``day_rad`` (float)."""
    return np.asarray(day_rad, dtype=np.float64) * 365 / (2 * np.pi)


def profile_date(year, day_rad):
    """Calendar date of a profile, inverse of :func:`day_rad_from_date`.

    Days 31 map to the first day of the next month and the last days of the
    year to 31 December, as in the original code (the date is used
    for the CANYON-B and SOCA baselines).
    """
    angle = math.atan2(math.sin(day_rad), math.cos(day_rad)) % (2 * math.pi)
    day_total = round(angle * 365 / (2 * math.pi))
    month, day = divmod(day_total, 30)
    if day == 0:
        day, month = 30, month - 1
    if month > 11:
        return dt.datetime(int(year), 12, 31)
    return dt.datetime(int(year), max(0, month) + 1, 1) + dt.timedelta(days=day - 1)


def season(day_rad, lat):
    """Season (winter, spring, summer, autumn) of a profile, inverted south of the equator."""
    day = int(day_of_year(day_rad))
    if day < 80 or day >= 355:
        name = "winter"
    elif day < 172:
        name = "spring"
    elif day < 264:
        name = "summer"
    else:
        name = "autumn"
    if lat >= 0:
        return name
    return {"winter": "summer", "spring": "autumn", "summer": "winter", "autumn": "spring"}[name]


def float_id(profile_id):
    """Float id of a profile id ``.../<float id>/<profile name>``."""
    parts = str(profile_id).replace("\\", "/").rstrip("/").split("/")
    if len(parts) < 2 or not parts[-2]:
        raise ValueError(f"Profile id {profile_id!r} has no float directory")
    return parts[-2]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def region_mask(lat, lon, region=None, lat_range=None, lon_range=None):
    """Boolean mask of the profiles inside a region of :data:`REGIONS` or a box."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if region is not None:
        if region not in REGIONS:
            raise ValueError(f"Unknown region {region!r}; expected one of {list(REGIONS)}")
        lat_range, lon_range = REGIONS[region]["lat"], REGIONS[region]["lon"]
    if lat_range is None and lon_range is None:
        return np.ones(lat.shape, dtype=bool)
    mask = np.ones(lat.shape, dtype=bool)
    if lat_range is not None:
        mask &= (lat >= lat_range[0]) & (lat <= lat_range[1])
    if lon_range is not None:
        ranges = [lon_range] if np.isscalar(lon_range[0]) else lon_range
        lon_mask = np.zeros(lon.shape, dtype=bool)
        for low, high in ranges:
            lon_mask |= (lon >= low) & (lon <= high)
        mask &= lon_mask
    return mask


def valid_mask(temp, psal, doxy, target=None):
    """Profiles whose inputs are finite and inside :data:`INPUT_RANGES` and whose
    target (if given) is finite and has no fill values."""
    mask = np.ones(len(temp), dtype=bool)
    for name, values in (("temp", temp), ("psal", psal), ("doxy", doxy)):
        low, high = INPUT_RANGES[name]
        with np.errstate(invalid="ignore"):
            mask &= (np.isfinite(values) & (values >= low) & (values <= high)).all(axis=1)
    if target is not None:
        with np.errstate(invalid="ignore"):
            mask &= (np.isfinite(target) & (target < FILL_VALUE)).all(axis=1)
    return mask


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------

def compute_normalization_stats(dataset):
    """Mean and standard deviation (ddof=1) of temp, psal and doxy over all levels
    of all profiles of ``dataset`` (a :class:`ProfileDataset`)."""
    stats = {}
    for name in INPUT_ROWS:
        values = getattr(dataset, name).astype(np.float64).ravel()
        if values.size < 2:
            raise ValueError("At least two values are needed to compute normalization statistics")
        stats[name] = [float(values.mean()), float(values.std(ddof=1))]
    return stats


def check_normalization_stats(stats):
    for name in INPUT_ROWS:
        if name not in stats:
            raise ValueError(f"Normalization statistics have no entry {name!r}")
        mean, std = stats[name]
        if not (np.isfinite(mean) and np.isfinite(std) and std > 0):
            raise ValueError(f"Invalid normalization statistics for {name}: {stats[name]}")
    return {name: [float(stats[name][0]), float(stats[name][1])] for name in INPUT_ROWS}


def load_normalization_stats(path):
    with open(path) as handle:
        return check_normalization_stats(json.load(handle))


def save_normalization_stats(stats, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(check_normalization_stats(stats), indent=4) + "\n")


def default_stats_path(ds_name, variable):
    """Statistics shipped with the repository for the reference models, e.g.
    ``normalization_stats/SEANOE_CHLA.json``."""
    root = Path(__file__).resolve().parent.parent
    return root / "normalization_stats" / f"{ds_name}_{check_variable(variable)}.json"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProfileDataset(Dataset):
    """Profiles of a dataset CSV, with encoded metadata and normalized inputs.

    Args:
        csv_path: dataset CSV (see the module docstring).
        variable: target variable; checked against the target row label.
        norm_stats: normalization statistics (dict or path to a JSON file);
            ``None`` leaves the inputs unnormalized (only to compute statistics).
        region: name of a region of :data:`globio.constants.REGIONS`, or
            ``lat_range``/``lon_range`` for a custom box (inclusive bounds).
        drop_invalid: drop profiles with inputs outside the Argo global range
            or with non-finite or fill values (inputs or target).
        max_profiles: keep only the first ``max_profiles`` profiles (after the
            selections above), e.g. for quick tests.

    Each item is a dict of float32 tensors ``year`` (scalar), ``day`` (2,),
    ``space`` (3,), ``temp``, ``psal``, ``doxy``, ``target`` (200,), and the
    ``index`` of the profile in the dataset. ``target`` is NaN when the CSV
    has no target row. Raw (unnormalized) arrays and metadata are available as
    attributes (``temp``, ``target``, ``lat``, ``profile_ids``, ...).
    """

    def __init__(self, csv_path=None, variable=None, norm_stats=None, region=None,
                 lat_range=None, lon_range=None, drop_invalid=True, max_profiles=None,
                 verbose=True, _data=None):
        super().__init__()
        if _data is None:
            if csv_path is None:
                raise ValueError("csv_path is required")
            _data = read_profile_csv(csv_path, variable)
        self.source = str(csv_path) if csv_path is not None else "<arrays>"
        self.variable = _data["variable"] or (variable and check_variable(variable))
        self.has_target = _data["target"] is not None
        n = len(_data["profile_ids"])
        if not self.has_target:
            _data = {**_data, "target": np.full((n, N_LEVELS), np.nan, dtype=np.float32)}
        keep = region_mask(_data["lat"], _data["lon"], region, lat_range, lon_range)
        n_region = int(keep.sum())
        if drop_invalid:
            keep &= valid_mask(_data["temp"], _data["psal"], _data["doxy"],
                               _data["target"] if self.has_target else None)
        indices = np.flatnonzero(keep)
        if verbose and drop_invalid and len(indices) < n_region:
            print(f"[ProfileDataset] {os.path.basename(self.source)}: dropped {n_region - len(indices)} "
                  f"of {n_region} profiles with invalid values", flush=True)
        if max_profiles is not None:
            if max_profiles < 0:
                raise ValueError("max_profiles must be >= 0")
            indices = indices[:max_profiles]
        self.region = region
        for name in ("profile_ids", "year", "day_rad", "lat", "lon", "temp", "psal", "doxy", "target"):
            setattr(self, name, np.asarray(_data[name])[indices])
        for name in ("temp", "psal", "doxy", "target"):
            setattr(self, name, getattr(self, name).astype(np.float32, copy=False))
        self._encoded = {
            "year": torch.from_numpy(encode_year(self.year).astype(np.float32)),
            "day": torch.from_numpy(encode_day(self.day_rad).astype(np.float32)),
            "space": torch.from_numpy(encode_space(self.lat, self.lon).astype(np.float32)),
            "target": torch.from_numpy(self.target),
        }
        self.set_normalization(norm_stats)

    @classmethod
    def from_arrays(cls, year, day_rad, lat, lon, temp, psal, doxy, target=None, profile_ids=None,
                    variable=None, norm_stats=None, drop_invalid=False, **kwargs):
        """Dataset from arrays: metadata of shape ``(n,)``, profiles ``(n, 200)``.

        ``day_rad`` follows the dataset convention, see :func:`day_rad_from_date`.
        """
        temp = np.atleast_2d(np.asarray(temp, dtype=np.float32))
        n = temp.shape[0]
        data = {
            "profile_ids": (np.asarray(profile_ids, dtype=str) if profile_ids is not None
                            else np.asarray([f"profile_{i}" for i in range(n)])),
            "year": np.broadcast_to(np.asarray(year, dtype=np.float64), (n,)),
            "day_rad": np.broadcast_to(np.asarray(day_rad, dtype=np.float64), (n,)),
            "lat": np.broadcast_to(np.asarray(lat, dtype=np.float64), (n,)),
            "lon": np.broadcast_to(np.asarray(lon, dtype=np.float64), (n,)),
            "temp": temp,
            "psal": np.atleast_2d(np.asarray(psal, dtype=np.float32)),
            "doxy": np.atleast_2d(np.asarray(doxy, dtype=np.float32)),
            "target": None if target is None else np.atleast_2d(np.asarray(target, dtype=np.float32)),
            "variable": variable and check_variable(variable),
        }
        for name in ("temp", "psal", "doxy", "target"):
            if data[name] is not None and data[name].shape != (n, N_LEVELS):
                raise ValueError(f"{name} must have shape ({n}, {N_LEVELS}), got {data[name].shape}")
        if len(data["profile_ids"]) != n:
            raise ValueError("profile_ids must have one entry per profile")
        return cls(variable=variable, norm_stats=norm_stats, drop_invalid=drop_invalid,
                   _data=data, **kwargs)

    def set_normalization(self, norm_stats):
        """(Re)normalize the inputs with ``norm_stats`` (dict, JSON path or None)."""
        if isinstance(norm_stats, (str, os.PathLike)):
            norm_stats = load_normalization_stats(norm_stats)
        elif norm_stats is not None:
            norm_stats = check_normalization_stats(norm_stats)
        self.norm_stats = norm_stats
        for name in INPUT_ROWS:
            values = getattr(self, name)
            if norm_stats is not None:
                mean, std = norm_stats[name]
                values = (values - np.float32(mean)) / np.float32(std)
            self._encoded[name] = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
        return self

    def subset(self, indices):
        """New dataset with the profiles ``indices`` (same normalization)."""
        indices = np.asarray(indices)
        data = {name: getattr(self, name)[indices]
                for name in ("profile_ids", "year", "day_rad", "lat", "lon", "temp", "psal", "doxy")}
        data["target"] = self.target[indices] if self.has_target else None
        data["variable"] = self.variable
        out = type(self)(variable=self.variable, norm_stats=self.norm_stats, drop_invalid=False,
                         verbose=False, _data=data)
        out.source, out.region = self.source, self.region
        return out

    def metadata(self):
        """One row per profile: id, float id, year, day_rad, day, season, lat, lon."""
        import pandas as pd
        meta = pd.DataFrame({
            "profile_id": self.profile_ids,
            "float_id": [float_id(p) if "/" in p or "\\" in p else "" for p in self.profile_ids],
            "year": self.year.astype(int),
            "day_rad": self.day_rad,
            "day": day_of_year(self.day_rad),
            "lat": self.lat,
            "lon": self.lon,
        })
        meta["season"] = [season(d, l) for d, l in zip(self.day_rad, self.lat)]
        return meta

    def __len__(self):
        return len(self.profile_ids)

    def __getitem__(self, index):
        item = {name: self._encoded[name][index] for name in FIELDS}
        item["index"] = index
        return item

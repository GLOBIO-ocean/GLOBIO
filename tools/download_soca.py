"""Download the SOCA CHLA estimate of every profile of a dataset CSV.

For each profile, the weekly 0.25 degree Copernicus product
``cmems_obs-mob_glo_bgc-chl-poc_my_0.25deg_P7D-m`` (variable ``chl``, 22
depths between 0 and 200 m) is subset at the profile position and date and
cached as NetCDF in --cache_dir/<year>/<month>/<day>/. Existing valid files
are reused, so an interrupted download can be restarted. Profiles outside the
product (e.g. after its last date) get all-zero values, which the evaluation
treats as invalid.

Output CSV: one column per profile (header: profile id) and the rows year,
day_rad, lat, lon, chla_SOCA, as read by globio.baselines.read_soca_csv.
Needs a Copernicus Marine account (``copernicusmarine login``).

Example:
    python -m tools.download_soca --csv data/test_profile_wise_SEANOE.csv \
        --cache_dir data/soca_cache/profile --output_csv data/SOCA_test_profile.csv
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from globio.constants import SOCA_LEVEL_INDICES
from globio.data import ProfileDataset, profile_date

COPERNICUS_DATASET = "cmems_obs-mob_glo_bgc-chl-poc_my_0.25deg_P7D-m"
N_DEPTHS = len(SOCA_LEVEL_INDICES)
# Copernicus writes the grid-cell centre in the file name, e.g. ..._chl_67.88W_24.12N_0.00-200.00m_...
NC_COORDS = re.compile(r"_chl_([0-9.]+)([EW])_([0-9.]+)([NS])_")
# Half of the 0.25 deg grid step, plus the rounding of the two decimals in the name.
MAX_CELL_DISTANCE = 0.125 + 0.01


def is_valid_nc(path):
    import netCDF4 as nc
    try:
        with nc.Dataset(path) as ds:
            if "chl" not in ds.variables:
                return False
            chl = ds.variables["chl"]
            if len(chl.shape) != 4 or chl.shape[0] < 1 or chl.shape[1] < N_DEPTHS or min(chl.shape[2:]) < 1:
                return False
            return bool(np.isfinite(np.asarray(chl[:])).any())
    except Exception as error:  # corrupted or truncated file
        print(f"Invalid NetCDF {path}: {error}")
        return False


def find_matching_nc(directory, lat, lon):
    """NetCDF of ``directory`` for the grid cell containing (lat, lon), or None."""
    best, best_distance = None, MAX_CELL_DISTANCE
    for name in os.listdir(directory):
        match = NC_COORDS.search(name)
        if not name.endswith(".nc") or match is None:
            continue
        file_lon = float(match[1]) * (-1 if match[2] == "W" else 1)
        file_lat = float(match[3]) * (-1 if match[4] == "S" else 1)
        distance = max(abs(file_lat - lat), abs((file_lon - lon + 180) % 360 - 180))
        if distance <= best_distance:
            best, best_distance = name, distance
    return None if best is None else os.path.join(directory, best)


def soca_profile(lat, lon, date, cache_dir, max_depth=200):
    """22-level SOCA CHLA at (lat, lon, date); None if not available."""
    import copernicusmarine
    import netCDF4 as nc
    from copernicusmarine.core_functions.exceptions import CoordinatesOutOfDatasetBounds
    directory = os.path.join(cache_dir, f"{date:%Y}", f"{date:%m}", f"{date:%d}")
    os.makedirs(directory, exist_ok=True)
    path = find_matching_nc(directory, lat, lon)
    if path is not None and not is_valid_nc(path):
        os.remove(path)
        path = None
    if path is None:
        try:
            copernicusmarine.subset(
                dataset_id=COPERNICUS_DATASET, variables=["chl"],
                minimum_longitude=lon, maximum_longitude=lon, minimum_latitude=lat, maximum_latitude=lat,
                start_datetime=f"{date:%Y-%m-%d}T00:00:00", end_datetime=f"{date:%Y-%m-%d}T00:00:00",
                minimum_depth=0, maximum_depth=max_depth, output_directory=directory)
        except CoordinatesOutOfDatasetBounds:
            return None
        path = find_matching_nc(directory, lat, lon)
        if path is None or not is_valid_nc(path):
            return None
    with nc.Dataset(path) as ds:
        chl = np.squeeze(np.asarray(ds.variables["chl"][:], dtype=float), axis=(0, 2, 3))
    return chl[:N_DEPTHS]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, type=Path, help="CHLA dataset CSV")
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path, help="Directory of the downloaded NetCDF files")
    args = parser.parse_args()

    dataset = ProfileDataset(args.csv, "CHLA", drop_invalid=False)
    values, failed = [], []
    for i in range(len(dataset)):
        date = profile_date(dataset.year[i], dataset.day_rad[i])
        print(f"[{i + 1}/{len(dataset)}] {dataset.profile_ids[i]} {date:%Y-%m-%d}", flush=True)
        chl = soca_profile(float(dataset.lat[i]), float(dataset.lon[i]), date, str(args.cache_dir))
        if chl is None or chl.size != N_DEPTHS:
            failed.append(i)
            chl = np.zeros(N_DEPTHS)
        values.append(chl)
    table = pd.DataFrame([dataset.year, dataset.day_rad, dataset.lat, dataset.lon,
                          [np.array2string(v, max_line_width=10 ** 6) for v in values]],
                         index=["year", "day_rad", "lat", "lon", "chla_SOCA"], columns=dataset.profile_ids)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv)
    print(f"Saved {args.output_csv}: {len(dataset)} profiles, {len(failed)} without a SOCA estimate")


if __name__ == "__main__":
    main()

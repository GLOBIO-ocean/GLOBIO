"""Build dataset CSVs from quality-controlled BGC-Argo profile NetCDF files.

The input directory holds one NetCDF per profile (``<float>/<profile>.nc``,
variables LATITUDE, LONGITUDE, TEMP, PSAL, DOXY, <VARIABLE> and their
PRES_<name> pressures) and a ``Float_Index.txt`` listing them (comma-separated:
file, lat, lon, ``YYYYMMDD-hh:mm:ss``, variables), as in the SUPERFLOAT /
CORIOLIS / SEANOE collections used for the reference datasets.

Every variable is discretized on the 200-level grid of the target (CHLA and
BBP700: 0-199 dbar every 1 dbar; NITRATE: 0-995 dbar every 5 dbar): each level
is the mean of the measurements within half a step, or the linear
interpolation between the closest measurements above and below (the first or
last value beyond the measured range); a 5-level centred moving average is then
applied (the two levels at each end are kept). Profiles are rejected when CHLA
(BBP700) has no measurement above 10 (30) dbar or below 180 dbar, or fewer
than 50 measurements. BBP700 is multiplied by 1000.

The accepted profiles are shuffled (--seed) and split profile-wise into
<prefix>_train.csv (80%) and <prefix>_test.csv; rejected profiles go to
<prefix>_removed.csv. Values are written with 4 decimals as ``tensor([...])``,
the format of the reference datasets. Profile ids are the file paths
relative to the input directory, without extension. Then run
tools.clean_dataset and tools.split_dataset (see README).

Example:
    python -m tools.make_dataset --input_dir /data/SEANOE_QCed --variable CHLA --output_prefix data/CHLA_SEANOE
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from globio.constants import MAX_PRESSURE, PRESSURE_STEP, check_variable

QC_RULES = {"CHLA": (10, 180, 50), "BBP700": (30, 180, 50), "NITRATE": None}
SMOOTHING_WINDOW = 5


def read_date(text):
    """(year, day_rad) of ``YYYYMMDD...``, with 30-day pseudo-months."""
    year, month, day = int(text[0:4]), int(text[4:6]), int(text[6:8])
    return year, ((month - 1) * 30 + day) * 2 * np.pi / 365


def moving_average(values, window=SMOOTHING_WINDOW):
    """Centred moving average; the window // 2 values at each end are kept."""
    values = np.asarray(values, dtype=np.float32)
    if window > len(values):
        raise ValueError("window must not exceed the profile length")
    half = window // 2
    out = values.copy()
    out[half:len(values) - half] = np.convolve(values.astype(np.float64), np.ones(window) / window, mode="valid")
    return out


def discretize(pressure, values, max_pressure, step):
    """Profile on the grid ``0, step, ..., max_pressure - step`` (see module docstring)."""
    pressure, values = np.asarray(pressure, dtype=np.float64), np.asarray(values, dtype=np.float64)
    grid = np.arange(0, max_pressure, step)
    out = np.zeros(grid.size, dtype=np.float32)
    for i, level in enumerate(grid):
        near = np.abs(pressure - level) < step / 2
        if near.any():
            out[i] = values[near].mean()
        elif level <= pressure[0]:
            out[i] = values[0]
        elif level >= pressure[-1]:
            out[i] = values[-1]
        else:
            j = np.searchsorted(pressure, level)
            out[i] = np.interp(level, pressure[[j - 1, j]], values[[j - 1, j]])
    return moving_average(out)


def _measurements(ds, name):
    """Valid (pressure, values) of one variable, sorted by pressure."""
    values = np.ma.filled(np.ma.asarray(ds[name][:], dtype=float), np.nan).ravel()
    pressure = np.ma.filled(np.ma.asarray(ds[f"PRES_{name}"][:], dtype=float), np.nan).ravel()
    keep = np.isfinite(values) & np.isfinite(pressure) & (np.abs(values) < 99999) & (pressure < 99999)
    order = np.argsort(pressure[keep], kind="stable")
    return pressure[keep][order], values[keep][order]


def read_profile(path, variable):
    """(fields, accepted) for one NetCDF, or (None, False) if it cannot be used.
    ``fields`` is [lat, lon, temp, psal, doxy, target]."""
    import netCDF4 as nc
    try:
        with nc.Dataset(path) as ds:
            if any(name not in ds.variables for name in ("TEMP", "PSAL", "DOXY", variable)):
                return None, False
            lat, lon = float(ds["LATITUDE"][:].data[0]), float(ds["LONGITUDE"][:].data[0])
            data = {name: _measurements(ds, name) for name in ("TEMP", "PSAL", "DOXY", variable)}
    except (OSError, KeyError, IndexError, ValueError) as error:
        print(f"Skipping {path}: {error!r}")
        return None, False
    if any(p.size == 0 for p, _ in data.values()):
        return None, False
    accepted = True
    rule = QC_RULES[variable]
    if rule is not None:
        top, bottom, count = rule
        pressure, values = data[variable]
        accepted = pressure[0] <= top and pressure[-1] >= bottom and values.size >= count
    if variable == "BBP700":
        data[variable] = (data[variable][0], data[variable][1] * 1000)
    fields = [discretize(p, v, MAX_PRESSURE[variable], PRESSURE_STEP[variable]) for p, v in data.values()]
    return [lat, lon] + fields, accepted


def format_profile(values):
    import torch
    return repr(torch.from_numpy(np.asarray(values, dtype=np.float32)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory with Float_Index.txt")
    parser.add_argument("--variable", required=True, type=check_variable)
    parser.add_argument("--output_prefix", required=True, help="Writes <prefix>_train/_test/_removed.csv")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_files", type=int, help="Debug: read only the first files of the index")
    args = parser.parse_args()
    outputs = {part: Path(f"{args.output_prefix}_{part}.csv") for part in ("train", "test", "removed")}
    for path in outputs.values():
        if path.exists():
            parser.error(f"{path} already exists")

    index = pd.read_csv(args.input_dir / "Float_Index.txt", header=None, dtype=str)
    if args.max_files:
        index = index.iloc[:args.max_files]
    accepted, removed = {}, {}
    for n, (name, date) in enumerate(zip(index[0], index[3])):
        path = args.input_dir / name.strip()
        if not path.exists():
            continue
        fields, ok = read_profile(path, args.variable)
        if fields is None:
            continue
        year, day_rad = read_date(date.strip())
        profile_id = os.path.splitext(name.strip())[0]
        record = [year, day_rad] + fields[:2] + [format_profile(f) for f in fields[2:]]
        (accepted if ok else removed)[profile_id] = record
        if (n + 1) % 1000 == 0:
            print(f"{n + 1}/{len(index)} files, {len(accepted)} accepted", flush=True)

    items = list(accepted.items())
    random.Random(args.seed).shuffle(items)
    n_test = len(items) - int(args.train_fraction * len(items))
    rows = ["year", "day_rad", "lat", "lon", "temp", "psal", "doxy", args.variable]
    parts = {"test": dict(items[:n_test]), "train": dict(items[n_test:]), "removed": removed}
    for part, profiles in parts.items():
        outputs[part].parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(profiles, index=rows).to_csv(outputs[part])
        print(f"{outputs[part]}: {len(profiles)} profiles")


if __name__ == "__main__":
    main()

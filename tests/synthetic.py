"""Synthetic dataset CSVs for the tests."""
import csv

import numpy as np

from globio.constants import N_LEVELS


def profile_text(values):
    return "tensor([" + ", ".join(f"{v:.4f}" for v in values) + "])"


def make_profiles(n, seed=0, variable="CHLA"):
    """Plausible random profiles; the target depends on temperature and depth."""
    rng = np.random.default_rng(seed)
    depth = np.linspace(0, 1, N_LEVELS)
    lat = rng.uniform(-70, 70, n)
    lon = rng.uniform(-180, 180, n)
    surface = 26 - 0.25 * np.abs(lat)
    temp = surface[:, None] - 8 * depth[None, :] + rng.normal(0, 0.1, (n, N_LEVELS))
    psal = 35 + 0.5 * depth[None, :] + rng.normal(0, 0.05, (n, N_LEVELS))
    doxy = 220 - 60 * depth[None, :] + rng.normal(0, 1, (n, N_LEVELS))
    if variable == "NITRATE":
        target = 2 + 30 * depth[None, :] * (1 + 0.01 * np.abs(lat)[:, None])
    else:
        target = 0.05 + 0.5 * np.exp(-((depth[None, :] - 0.3) / 0.1) ** 2) * (temp[:, :1] / 25)
    return {
        "profile_ids": [f"root/{1000 * (seed + 1) + i // 3}/SD{1000 * (seed + 1) + i // 3}_{i:03d}"
                        for i in range(n)],
        "year": rng.integers(2012, 2024, n), "day_rad": rng.uniform(0.02, 2 * np.pi * 360 / 365, n),
        "lat": lat, "lon": lon, "temp": temp, "psal": psal, "doxy": doxy, "target": np.abs(target),
    }


def write_csv(path, data, variable="CHLA", with_target=True, labelled=True):
    rows = [["year"] + [str(int(v)) for v in data["year"]],
            ["day_rad"] + [repr(float(v)) for v in data["day_rad"]],
            ["lat"] + [repr(float(v)) for v in data["lat"]],
            ["lon"] + [repr(float(v)) for v in data["lon"]]]
    names = ["temp", "psal", "doxy"] + (["target"] if with_target else [])
    for name in names:
        rows.append([variable if name == "target" else name] + [profile_text(p) for p in data[name]])
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        if labelled:
            writer.writerow([""] + list(data["profile_ids"]))
            writer.writerows(rows)
        else:
            writer.writerow(list(data["profile_ids"]))
            writer.writerows(row[1:] for row in rows)
    return path

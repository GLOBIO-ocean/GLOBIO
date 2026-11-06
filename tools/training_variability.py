"""Variability of the training targets per region (used to relate the gain of
fine-tuning to the regional training data).

For every dataset CSV and region, reports the number of profiles and the
coefficient of variation (CV = std / mean across profiles, ddof=0) of the
target at each level, summarized by its median and mean over levels, and the
pooled CV of all values. Non-finite and fill values are excluded from the
statistics (profiles are still counted); the CV is undefined for means <=
--mean_epsilon or fewer than two values.

Example:
    python -m tools.training_variability --csv CHLA_profile=data/train_profile_wise_SEANOE.csv \
        --csv NITRATE_profile=data/train_profile_wise_CORIOLIS.csv --output_dir results/variability
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from globio.cli import named_path
from globio.constants import BASINS, FILL_VALUE, REGIONS
from globio.data import read_profile_csv, region_mask


def moments(values, epsilon):
    values = values[np.isfinite(values)]
    n = values.size
    mean = float(values.mean()) if n else np.nan
    std = float(values.std(ddof=0)) if n else np.nan
    return n, mean, std, (std / mean if n >= 2 and mean > epsilon else np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, action="append", type=named_path, metavar="NAME=CSV")
    parser.add_argument("--regions", nargs="+", default=list(BASINS), choices=list(REGIONS))
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--mean_epsilon", type=float, default=1e-12)
    args = parser.parse_args()

    summary, depths = [], []
    for name, path in args.csv:
        print(f"Reading {path}", flush=True)
        data = read_profile_csv(path)
        if data["target"] is None:
            parser.error(f"{path} has no target row")
        target = data["target"].astype(float)
        target[~np.isfinite(target) | (target >= FILL_VALUE)] = np.nan
        for region in args.regions:
            matrix = target[region_mask(data["lat"], data["lon"], region)]
            cvs = []
            for level in range(matrix.shape[1]):
                n, mean, std, cv = moments(matrix[:, level], args.mean_epsilon)
                depths.append({"dataset": name, "region": region, "level": level, "n_valid": n,
                               "mean": mean, "std": std, "cv": cv})
                if np.isfinite(cv):
                    cvs.append(cv)
            n, mean, std, cv = moments(matrix.ravel(), args.mean_epsilon)
            summary.append({"dataset": name, "variable": data["variable"], "region": region,
                            "n_profiles": len(matrix), "n_valid_values": n,
                            "cv_depth_median": float(np.median(cvs)) if cvs else np.nan,
                            "cv_depth_mean": float(np.mean(cvs)) if cvs else np.nan,
                            "target_mean": mean, "target_std": std, "cv_pooled": cv, "csv": str(path)})
            print(f"  {region}: n={len(matrix)}, median CV over levels={summary[-1]['cv_depth_median']:.4g}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(args.output_dir / "summary.csv", index=False, na_rep="NaN")
    pd.DataFrame(depths).to_csv(args.output_dir / "by_level.csv", index=False, na_rep="NaN")
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()

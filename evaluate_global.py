#!/usr/bin/env python3
"""Evaluate a global model.

For every dataset given with --data NAME=CSV (e.g. train, val, test), computes
the profile-wise RMSE of the model and summarizes it (mean, median, std) over
all profiles, summer, winter and the latitude bands r1-r6 (60-90N, 30-60N,
0-30N, 0-30S, 30-60S, 60-90S). Seasons are inverted in the southern hemisphere.

Optional analyses:
  --depth_stats       depth-resolved errors (mean |error|, its SD, bias, RMSE
                      at each level) per latitude band and season
  --baseline soca     CHLA: comparison with SOCA on the 22 SOCA depths, on the
                      profiles with a valid SOCA estimate (needs --soca_csv,
                      see tools/download_soca.py)
  --baseline canyonb  NITRATE: comparison with CANYON-B on all levels, and
                      analysis of the CANYON-B uncertainty range NO3 +/- NO3_ci
The baseline is computed on the dataset named by --baseline_on (default: test).

Outputs in --output_dir: profiles_<NAME>.csv (per-profile metadata and errors),
summary.csv, predictions_<NAME>.npz, depth_<NAME>.csv, baseline_*.csv,
config.json.

Example:
    python evaluate_global.py --checkpoint global_chla.pt --variable CHLA \
        --stats_path normalization_stats/SEANOE_CHLA.json \
        --data train=data/train_profile_wise_SEANOE.csv --data val=data/val_profile_wise_SEANOE.csv \
        --data test=data/test_profile_wise_SEANOE.csv \
        --baseline soca --soca_csv data/SOCA_test_profile.csv --depth_stats --output_dir results/eval_global_chla
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from globio.baselines import canyonb_nitrate, canyonb_range_analysis, soca_for_dataset
from globio.cli import load_model, named_path, unique_names, write_json
from globio.constants import LATITUDE_BANDS, REGIONS, SOCA_LEVEL_INDICES, VARIABLES, pressure_levels
from globio.data import ProfileDataset
from globio.inference import default_device, predict
from globio.metrics import compare_methods, depth_statistics, profile_rmse, summarize

GROUPS = ("all", "summer", "winter") + LATITUDE_BANDS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, action="append", type=named_path, metavar="NAME=CSV",
                        help="Dataset to evaluate; repeat for several (e.g. train=..., val=..., test=...)")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--variable", choices=VARIABLES, type=str.upper,
                        help="Needed only for CHLA/BBP700 checkpoints without metadata")
    parser.add_argument("--stats_path", type=Path, help="Needed only for checkpoints without statistics")
    parser.add_argument("--region", choices=list(REGIONS), help="Evaluate only the profiles of a region")
    parser.add_argument("--groups", nargs="+", default=list(GROUPS),
                        help="Profile groups of the summary: all, seasons, region names (default: %(default)s)")
    parser.add_argument("--depth_stats", action="store_true", help="Also write depth-resolved errors")
    parser.add_argument("--baseline", choices=["none", "soca", "canyonb"], default="none")
    parser.add_argument("--baseline_on", default="test", help="Dataset NAME on which to compute the baseline")
    parser.add_argument("--soca_csv", type=Path, help="SOCA profiles of the --baseline_on dataset")
    parser.add_argument("--wilcoxon_decimals", type=int, default=2,
                        help="Round errors to these decimals before the Wilcoxon test (reference: 2; -1: no rounding)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--max_profiles", type=int, help="Debug: evaluate only the first profiles of each set")
    args = parser.parse_args()
    data = unique_names(args.data, parser, "--data")
    if args.baseline != "none" and args.baseline_on not in data:
        parser.error(f"--baseline_on {args.baseline_on!r} is not one of the --data names {list(data)}")
    if args.baseline == "soca" and (args.soca_csv is None or not args.soca_csv.is_file()):
        parser.error("--baseline soca needs an existing --soca_csv")

    device = default_device(args.device)
    model, stats, info = load_model(args.checkpoint, args.variable, args.stats_path, device)
    variable = model.variable
    if args.baseline == "soca" and variable != "CHLA":
        parser.error("SOCA is a CHLA baseline")
    if args.baseline == "canyonb" and variable != "NITRATE":
        parser.error("CANYON-B is a NITRATE baseline")
    known = {"all", "winter", "spring", "summer", "autumn", *REGIONS}
    unknown = [g for g in args.groups if g not in known]
    if unknown:
        parser.error(f"Unknown --groups {unknown}; use all, a season or a region of {list(REGIONS)}")
    decimals = None if args.wilcoxon_decimals < 0 else args.wilcoxon_decimals
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pressure = pressure_levels(variable)

    summaries, counts = [], {}
    for name, path in data.items():
        print(f"\n=== {name}: {path}", flush=True)
        dataset = ProfileDataset(path, variable, stats, region=args.region, max_profiles=args.max_profiles)
        if len(dataset) == 0:
            if args.baseline != "none" and name == args.baseline_on:
                parser.error(f"No profiles in the --baseline_on set {name}: the baseline cannot be computed")
            print(f"No profiles in {path}, skipped")
            continue
        counts[name] = len(dataset)
        meta = dataset.metadata()
        prediction = predict(model, dataset, args.batch_size, device)
        meta["rmse_globio"] = profile_rmse(prediction, dataset.target)
        arrays = {"profile_id": dataset.profile_ids, "prediction": prediction, "target": dataset.target,
                  "pressure": pressure}
        summary = summarize({"globio": meta.rmse_globio}, meta, args.groups)
        summary.insert(0, "set", name)
        summaries.append(summary)
        if args.depth_stats:
            depth_statistics(prediction, dataset.target, meta, pressure=pressure).to_csv(
                args.output_dir / f"depth_{name}.csv", index=False)

        if args.baseline != "none" and name == args.baseline_on:
            if args.baseline == "soca":
                soca, valid = soca_for_dataset(args.soca_csv, dataset)
                target = dataset.target[:, SOCA_LEVEL_INDICES]
                meta["soca_valid"] = valid
                meta["rmse_globio_soca_levels"] = profile_rmse(prediction, dataset.target, SOCA_LEVEL_INDICES)
                meta["rmse_soca"] = np.sqrt(np.mean((soca - target) ** 2, axis=1))
                arrays.update(soca=soca, soca_valid=valid)
                errors = {"globio": meta.rmse_globio_soca_levels[valid], "soca": meta.rmse_soca[valid]}
                subset = meta[valid].reset_index(drop=True)
                print(f"Profiles with a valid SOCA estimate: {valid.sum()} of {len(valid)} "
                      f"(errors on the {len(SOCA_LEVEL_INDICES)} SOCA depths)")
            else:
                canyon, sigma = canyonb_nitrate(dataset)
                meta["rmse_canyonb"] = profile_rmse(canyon, dataset.target)
                arrays.update(canyonb=canyon, canyonb_sigma=sigma)
                per_profile, range_summary = canyonb_range_analysis(
                    prediction, dataset.target, canyon, sigma, meta.rmse_globio.to_numpy(),
                    meta.rmse_canyonb.to_numpy())
                pd.concat([meta[["profile_id"]], per_profile], axis=1).to_csv(
                    args.output_dir / "baseline_canyonb_range_profiles.csv", index=False)
                range_summary.to_csv(args.output_dir / "baseline_canyonb_range_summary.csv", index=False)
                print("\n--- CANYON-B uncertainty range (NO3 +/- NO3_ci)")
                print(range_summary.T.to_string(header=False))
                errors = {"globio": meta.rmse_globio, "canyonb": meta.rmse_canyonb}
                subset = meta
            baseline_summary = summarize({k: np.asarray(v) for k, v in errors.items()}, subset, args.groups)
            baseline_summary.insert(0, "set", name)
            baseline_summary.to_csv(args.output_dir / "baseline_summary.csv", index=False)
            tests = compare_methods({k: np.asarray(v) for k, v in errors.items()}, decimals=decimals)
            tests.to_csv(args.output_dir / "baseline_tests.csv", index=False)
            print(f"\n--- {args.baseline} comparison (profile-wise RMSE)")
            print(baseline_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print(tests[["rank", "method", "median", "significantly_best"]].to_string(index=False))

        meta.to_csv(args.output_dir / f"profiles_{name}.csv", index=False)
        np.savez_compressed(args.output_dir / f"predictions_{name}.npz", **arrays)

    if not summaries:
        parser.error("No profiles evaluated")
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print("\n=== Profile-wise RMSE: mean / median / std")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    write_json(args.output_dir / "config.json", {
        **{k: v for k, v in vars(args).items() if k != "data"}, "data": data, "variable": variable,
        "norm_stats": stats, "n_profiles": counts,
        "groups_definition": {g: REGIONS[g] for g in args.groups if g in REGIONS}})
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()

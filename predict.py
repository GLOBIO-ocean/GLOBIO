#!/usr/bin/env python3
"""Predict profiles with a trained model, optionally with MC dropout uncertainty.

The input CSV has the dataset format (see README); the target row is optional.
Outputs:
  <output>.json run configuration
  <output>.npz  arrays: profile_id, year, day_rad, lat, lon, pressure,
                prediction (eval mode), target (if present) and, with
                --mc_samples N, mc_mean, mc_std (and mc_samples with --save_samples)
  <output>.csv  one row per profile: metadata and, if available, RMSE of the
                prediction (and of the MC mean) and RMS of the MC standard deviation

Examples:
    python predict.py --checkpoint best.pt --csv test.csv --output out/predictions
    python predict.py --checkpoint best.pt --csv test.csv --output out/predictions --mc_samples 100
    # reference checkpoints (weights only): give variable and statistics
    python predict.py --checkpoint global_chla.pt --variable CHLA \
        --stats_path normalization_stats/SEANOE_CHLA.json --csv test.csv --output out/chla
"""
import argparse
from pathlib import Path

import numpy as np

from globio.cli import load_model, write_json
from globio.constants import REGIONS, VARIABLES, pressure_levels
from globio.data import ProfileDataset
from globio.inference import default_device, predict, predict_mc
from globio.metrics import profile_rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path, help="Dataset CSV with the profiles to predict")
    parser.add_argument("--output", required=True, type=Path, help="Output prefix (writes .npz and .csv)")
    parser.add_argument("--variable", choices=VARIABLES, type=str.upper,
                        help="Needed only for CHLA/BBP700 checkpoints without metadata")
    parser.add_argument("--stats_path", type=Path, help="Normalization statistics, needed only for "
                        "checkpoints without them (e.g. normalization_stats/SEANOE_CHLA.json)")
    parser.add_argument("--region", choices=list(REGIONS), help="Predict only the profiles of a region")
    parser.add_argument("--keep_invalid", action="store_true",
                        help="Also predict profiles with inputs outside the Argo ranges (not recommended)")
    parser.add_argument("--max_profiles", type=int)
    parser.add_argument("--mc_samples", type=int, default=0,
                        help="Number of MC dropout passes (0: deterministic only; reference: 100)")
    parser.add_argument("--save_samples", action="store_true", help="Also save every MC pass")
    parser.add_argument("--seed", type=int, default=123, help="Seed of the MC dropout masks")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", help="cpu, cuda, cuda:1, ... (default: cuda if available)")
    args = parser.parse_args()
    if args.mc_samples == 1 or args.mc_samples < 0:
        parser.error("--mc_samples must be 0 or >= 2")

    device = default_device(args.device)
    model, stats, info = load_model(args.checkpoint, args.variable, args.stats_path, device)
    dataset = ProfileDataset(args.csv, model.variable, stats, region=args.region,
                             drop_invalid=not args.keep_invalid, max_profiles=args.max_profiles)
    if len(dataset) == 0:
        parser.error("No profiles to predict")
    print(f"Predicting {len(dataset)} {model.variable} profiles with {args.checkpoint} on {device}", flush=True)

    arrays = {"profile_id": dataset.profile_ids, "year": dataset.year, "day_rad": dataset.day_rad,
              "lat": dataset.lat, "lon": dataset.lon, "pressure": pressure_levels(model.variable)}
    table = dataset.metadata()
    if args.mc_samples:
        result = predict_mc(model, dataset, args.mc_samples, args.batch_size, device, args.seed,
                            return_samples=args.save_samples)
        arrays.update(prediction=result["deterministic"], mc_mean=result["mean"], mc_std=result["std"])
        if args.save_samples:
            arrays["mc_samples"] = result["samples"]
        table["uncertainty_rms"] = np.sqrt((result["std"] ** 2).mean(axis=1))
    else:
        arrays["prediction"] = predict(model, dataset, args.batch_size, device)
    if dataset.has_target:
        arrays["target"] = dataset.target
        table["rmse"] = profile_rmse(arrays["prediction"], dataset.target)
        if args.mc_samples:
            table["rmse_mc_mean"] = profile_rmse(arrays["mc_mean"], dataset.target)
        print(f"median RMSE: {np.median(table['rmse']):.4f}")

    prefix = str(args.output)
    for extension in (".npz", ".csv", ".json"):
        if prefix.endswith(extension):
            prefix = prefix[:-len(extension)]
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prefix + ".npz", **arrays)
    table.to_csv(prefix + ".csv", index=False)
    write_json(prefix + ".json", {
        "checkpoint": str(args.checkpoint.resolve()), "csv": str(args.csv.resolve()),
        "variable": model.variable, "norm_stats": stats, "region": args.region,
        "mc_samples": args.mc_samples, "seed": args.seed, "batch_size": args.batch_size,
        "n_profiles": len(dataset)})
    print(f"Saved {prefix}.npz, {prefix}.csv and {prefix}.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MC dropout uncertainty of one or more models on a test set.

Every profile is predicted --n_samples times (reference: 100) with dropout active
and BatchNorm in eval mode. The MC mean is the prediction and the per-level
sample standard deviation (ddof=1) its uncertainty sigma. For each model:

  profiles_<NAME>.csv  per profile: RMSE of the MC mean, RMS of sigma
                       (uncertainty_rms), whether all levels are within k sigma
                       and the fraction of levels that are (k = 1, 2, 3)
  coverage_<NAME>.csv  % of profiles entirely within k sigma and point-wise
                       coverage (level fractions averaged per profile)
  vertical_<NAME>.csv  median |error| and median sigma at each pressure level
  coverage.csv         coverage of all models side by side

predictions.npz keeps targets, MC means, sigmas and deterministic predictions;
pass it to --results to recompute the tables without inference. The sigma is
the dispersion of the model (epistemic), not the measurement uncertainty.

Examples:
    python evaluate_uncertainty.py --model global=global.pt --test_csv test.csv --output_dir results/unc
    python evaluate_uncertainty.py --model global=global.pt --model fine_tuned=ft_med.pt \
        --region mediterranean --test_csv test.csv --output_dir results/unc_med
    python evaluate_uncertainty.py --results results/unc/predictions.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from globio.cli import load_model, named_path, unique_names, write_json
from globio.constants import FILL_VALUE, REGIONS, VARIABLES, pressure_levels
from globio.data import ProfileDataset
from globio.inference import default_device, predict_mc
from globio.metrics import uncertainty_metrics


def run_inference(args, parser):
    checkpoints = unique_names(args.model, parser, "--model")
    device = default_device(args.device)
    models = {name: load_model(path, args.variable, args.stats_path, device) for name, path in checkpoints.items()}
    variables = {model.variable for model, _, _ in models.values()}
    if len(variables) != 1:
        parser.error(f"All models must predict the same variable, got {sorted(variables)}")
    variable = variables.pop()
    dataset = ProfileDataset(args.test_csv, variable, None, region=args.region, max_profiles=args.max_profiles)
    if not dataset.has_target or len(dataset) == 0:
        parser.error("The test CSV must contain targets and at least one valid profile")
    fields = {"profile_id": dataset.profile_ids, "year": dataset.year, "day_rad": dataset.day_rad,
              "lat": dataset.lat, "lon": dataset.lon, "pressure": pressure_levels(variable),
              "target": dataset.target,
              "mask": np.isfinite(dataset.target) & (dataset.target < FILL_VALUE)}
    for name, (model, stats, _) in models.items():
        print(f"{name}: {len(dataset)} profiles, {args.n_samples} MC passes, device {device}", flush=True)
        dataset.set_normalization(stats)
        result = predict_mc(model, dataset, args.n_samples, args.batch_size, device, args.seed)
        fields[f"{name}_mean"], fields[f"{name}_std"] = result["mean"], result["std"]
        fields[f"{name}_deterministic"] = result["deterministic"]
    config = {**{k: v for k, v in vars(args).items() if k not in ("model", "results")},
              "models": checkpoints, "model_names": list(checkpoints), "variable": variable,
              "norm_stats": {name: stats for name, (_, stats, _) in models.items()}, "n_profiles": len(dataset)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz", **fields,
                        config=np.asarray(json.dumps(config, default=str)))
    write_json(args.output_dir / "config.json", config)
    return fields, config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", type=named_path, metavar="NAME=CHECKPOINT",
                        help="Model to evaluate; repeat for several")
    parser.add_argument("--test_csv", type=Path)
    parser.add_argument("--output_dir", type=Path, help="Default with --results: the directory of the archive")
    parser.add_argument("--region", choices=list(REGIONS))
    parser.add_argument("--variable", choices=VARIABLES, type=str.upper,
                        help="Needed only for CHLA/BBP700 checkpoints without metadata")
    parser.add_argument("--stats_path", type=Path, help="Statistics for the checkpoints without them")
    parser.add_argument("--n_samples", type=int, default=100, help="MC dropout passes (reference: 100)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size; with --seed it determines the dropout masks")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device")
    parser.add_argument("--max_profiles", type=int, help="Debug: evaluate only the first profiles")
    parser.add_argument("--results", type=Path, help="predictions.npz of a previous run: recompute the tables")
    args = parser.parse_args()

    if args.results:
        with np.load(args.results, allow_pickle=False) as archive:
            fields = dict(archive)
        config = json.loads(str(fields.pop("config")))
        args.output_dir = args.output_dir or args.results.parent
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not args.model or args.test_csv is None or args.output_dir is None:
            parser.error("--model, --test_csv and --output_dir are required (or --results)")
        if args.n_samples < 2 or args.batch_size < 1:
            parser.error("Require --n_samples >= 2 and --batch_size >= 1")
        fields, config = run_inference(args, parser)

    coverages = []
    for name in config["model_names"]:
        per_profile, coverage, vertical = uncertainty_metrics(
            fields["target"], fields[f"{name}_mean"], fields[f"{name}_std"], fields["mask"], fields["pressure"])
        for key in ("profile_id", "year", "day_rad", "lat", "lon"):
            per_profile.insert(0 if key == "profile_id" else len(per_profile.columns), key, fields[key])
        per_profile.to_csv(args.output_dir / f"profiles_{name}.csv", index=False)
        coverage.to_csv(args.output_dir / f"coverage_{name}.csv", index=False)
        vertical.to_csv(args.output_dir / f"vertical_{name}.csv", index=False)
        coverage.insert(0, "model", name)
        coverage["median_rmse"] = per_profile.rmse.median()
        coverage["median_uncertainty_rms"] = per_profile.uncertainty_rms.median()
        coverages.append(coverage)
    coverage = pd.concat(coverages, ignore_index=True)
    coverage.to_csv(args.output_dir / "coverage.csv", index=False)
    print("\n=== MC dropout coverage (% of profiles entirely within k sigma, % of levels within k sigma)")
    print(coverage.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare models on a region: global, global fine-tuned on the region
and regional trained from scratch, optionally with a baseline.

All models are evaluated on the same test profiles of --region. For each model
the profile-wise RMSE is computed; methods are ranked by median RMSE and a
paired one-sided Wilcoxon signed-rank test with Holm correction tells whether
the best one is significantly better than all the others.

Two comparisons are written:
  models         the models only, on all 200 levels of all profiles
  with_baseline  (with --baseline) models and baseline on the same profiles and
                 levels: CANYON-B (NITRATE) on all levels; SOCA (CHLA) on the
                 22 SOCA depths of the profiles with a valid SOCA estimate

Outputs in --output_dir: profiles.csv (per-profile metadata and errors),
comparison_models.csv, comparison_with_baseline.csv, predictions.npz,
config.json.

Example:
    python evaluate_regional.py --variable CHLA --region mediterranean \
        --test_csv data/test_profile_wise_SEANOE.csv --stats_path normalization_stats/SEANOE_CHLA.json \
        --model global=ckpt/global.pt --model fine_tuned=ckpt/ft_med.pt --model local=ckpt/local_med.pt \
        --baseline soca --soca_csv data/SOCA_test_profile.csv --output_dir results/eval_med_chla
"""
import argparse
from pathlib import Path

import numpy as np

from globio.baselines import canyonb_nitrate, soca_for_dataset
from globio.cli import load_model, named_path, unique_names, write_json
from globio.constants import REGIONS, SOCA_LEVEL_INDICES, VARIABLES, pressure_levels
from globio.data import ProfileDataset
from globio.inference import default_device, predict
from globio.metrics import compare_methods, profile_rmse


def report(title, table):
    print(f"\n=== {title}")
    columns = ["rank", "method", "n", "mean", "median", "std", "significantly_best"]
    print(table[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best = table.iloc[0]
    verdict = "IS" if best.significantly_best else "is NOT"
    print(f"Best by median RMSE: {best.method}, which {verdict} significantly better than all the others")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, action="append", type=named_path, metavar="NAME=CHECKPOINT",
                        help="Model to compare; repeat (e.g. global=..., fine_tuned=..., local=...)")
    parser.add_argument("--test_csv", required=True, type=Path)
    parser.add_argument("--region", choices=list(REGIONS), help="Region of the test profiles (default: all)")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--variable", choices=VARIABLES, type=str.upper,
                        help="Needed only for CHLA/BBP700 checkpoints without metadata")
    parser.add_argument("--stats_path", type=Path,
                        help="Statistics for the checkpoints without them (e.g. the reference ones)")
    parser.add_argument("--baseline", choices=["none", "soca", "canyonb"], default="none")
    parser.add_argument("--soca_csv", type=Path, help="SOCA profiles of --test_csv (tools/download_soca.py)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--wilcoxon_decimals", type=int, default=2,
                        help="Round errors to these decimals before the Wilcoxon test (reference: 2; -1: no rounding)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--max_profiles", type=int, help="Debug: evaluate only the first profiles")
    args = parser.parse_args()
    checkpoints = unique_names(args.model, parser, "--model")
    if args.baseline == "soca" and (args.soca_csv is None or not args.soca_csv.is_file()):
        parser.error("--baseline soca needs an existing --soca_csv")
    if args.baseline != "none" and args.baseline in checkpoints:
        parser.error(f"A model cannot be named {args.baseline!r}")
    decimals = None if args.wilcoxon_decimals < 0 else args.wilcoxon_decimals

    device = default_device(args.device)
    models = {name: load_model(path, args.variable, args.stats_path, device) for name, path in checkpoints.items()}
    variables = {model.variable for model, _, _ in models.values()}
    if len(variables) != 1:
        parser.error(f"All models must predict the same variable, got {sorted(variables)}")
    variable = variables.pop()
    if args.baseline == "soca" and variable != "CHLA":
        parser.error("SOCA is a CHLA baseline")
    if args.baseline == "canyonb" and variable != "NITRATE":
        parser.error("CANYON-B is a NITRATE baseline")

    dataset = ProfileDataset(args.test_csv, variable, None, region=args.region, max_profiles=args.max_profiles)
    if len(dataset) < 2:
        parser.error(f"Only {len(dataset)} test profiles in the region: nothing to compare")
    print(f"{len(dataset)} {variable} test profiles" + (f" in {args.region}" if args.region else ""), flush=True)
    meta = dataset.metadata()
    arrays = {"profile_id": dataset.profile_ids, "target": dataset.target, "pressure": pressure_levels(variable)}
    errors = {}
    for name, (model, stats, _) in models.items():
        dataset.set_normalization(stats)
        prediction = predict(model, dataset, args.batch_size, device)
        arrays[f"prediction_{name}"] = prediction
        errors[name] = meta[f"rmse_{name}"] = profile_rmse(prediction, dataset.target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = {"models": compare_methods(errors, args.alpha, decimals)}
    if args.baseline == "canyonb":
        canyon, sigma = canyonb_nitrate(dataset)
        arrays.update(canyonb=canyon, canyonb_sigma=sigma)
        meta["rmse_canyonb"] = profile_rmse(canyon, dataset.target)
        comparisons["with_baseline"] = compare_methods({**errors, "canyonb": meta.rmse_canyonb.to_numpy()},
                                                       args.alpha, decimals)
    elif args.baseline == "soca":
        soca, valid = soca_for_dataset(args.soca_csv, dataset)
        arrays.update(soca=soca, soca_valid=valid)
        meta["soca_valid"] = valid
        meta["rmse_soca"] = np.sqrt(np.mean((soca - dataset.target[:, SOCA_LEVEL_INDICES]) ** 2, axis=1))
        on_soca = {}
        for name in errors:
            meta[f"rmse_{name}_soca_levels"] = profile_rmse(arrays[f"prediction_{name}"], dataset.target,
                                                           SOCA_LEVEL_INDICES)
            on_soca[name] = meta[f"rmse_{name}_soca_levels"].to_numpy()[valid]
        print(f"Profiles with a valid SOCA estimate: {valid.sum()} of {len(valid)}")
        if valid.sum() >= 2:
            comparisons["with_baseline"] = compare_methods({**on_soca, "soca": meta.rmse_soca.to_numpy()[valid]},
                                                           args.alpha, decimals)
        else:
            print("Fewer than 2 valid SOCA profiles: no baseline comparison")

    for title, table in comparisons.items():
        table.to_csv(args.output_dir / f"comparison_{title}.csv", index=False)
        report(f"{title} ({variable}, {args.region or 'all regions'})", table)
    meta.to_csv(args.output_dir / "profiles.csv", index=False)
    np.savez_compressed(args.output_dir / "predictions.npz", **arrays)
    write_json(args.output_dir / "config.json", {
        **{k: v for k, v in vars(args).items() if k != "model"}, "models": checkpoints, "variable": variable,
        "norm_stats": {name: stats for name, (_, stats, _) in models.items()}, "n_profiles": len(dataset),
        "region_definition": REGIONS.get(args.region)})
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()

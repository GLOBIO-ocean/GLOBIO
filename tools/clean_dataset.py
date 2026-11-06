"""Remove corrupted profiles from dataset CSVs.

A profile is dropped when temp, psal, doxy or the target have non-finite
values, fill values (>= 99999), values outside the Argo global range, or the
same value on all levels (e.g. a profile interpolated from one valid point).
Cells of the kept profiles are copied unchanged.

Usage (from the repository root):
    python -m tools.clean_dataset train.csv test.csv             # writes train_clean.csv, test_clean.csv
    python -m tools.clean_dataset train.csv test.csv --dry_run   # only report
    python -m tools.clean_dataset train.csv --inplace            # overwrite, keeping a .bak copy

The reference datasets were cleaned with the default ranges, except
NITRATE, cleaned with --target_range -10 60.
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd

from globio.constants import FILL_VALUE, INPUT_RANGES, INPUT_ROWS, TARGET_RANGES
from globio.data import float_id, parse_profile
from tools.csv_columns import ColumnTable

PROFILE_ROWS = {4: "temp", 5: "psal", 6: "doxy"}


def check_profile(values, low, high, drop_constant):
    """Reasons why one profile is invalid (empty list if it is valid)."""
    reasons = []
    finite = np.isfinite(values)
    fill = finite & (values >= FILL_VALUE)
    valid = finite & ~fill
    if not finite.all():
        reasons.append("nonfinite")
    if fill.any():
        reasons.append("fill")
    if (valid & (values < low)).any():
        reasons.append("below_range")
    if (valid & (values > high)).any():
        reasons.append("above_range")
    if drop_constant and valid.any() and np.ptp(values[valid]) == 0:
        reasons.append("constant")
    return reasons


def inspect(table, target, target_range, drop_constant):
    """One record per invalid profile, with the offending variables and reasons."""
    ranges = {row: (name, *INPUT_RANGES[name]) for row, name in PROFILE_ROWS.items()}
    if len(table.rows) > 7:
        ranges[7] = (target, *target_range)
    records = []
    for i, profile in enumerate(table.ids):
        issues = []
        for row, (name, low, high) in ranges.items():
            values = parse_profile(table.rows[row][i], n_levels=None).astype(float)
            issues += [f"{name}:{reason}" for reason in check_profile(values, low, high, drop_constant)]
        if issues:
            try:
                fid = float_id(profile)
            except ValueError:
                fid = ""
            records.append({"profile": profile, "float": fid, "year": table.rows[0][i],
                            "lat": float(table.rows[2][i]), "lon": float(table.rows[3][i]),
                            "issues": ";".join(issues)})
    return pd.DataFrame(records, columns=["profile", "float", "year", "lat", "lon", "issues"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", help="Dataset CSVs to clean")
    parser.add_argument("--target", choices=list(TARGET_RANGES), type=str.upper,
                        help="Target variable, needed only for CSVs without row labels")
    parser.add_argument("--target_range", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="Valid range of the target (default: Argo global range of the target)")
    parser.add_argument("--keep_constant", action="store_true", help="Do not drop constant profiles")
    parser.add_argument("--suffix", default="_clean", help="Suffix of the cleaned CSVs (ignored with --inplace)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the CSVs, keeping a .bak copy")
    parser.add_argument("--dry_run", action="store_true", help="Only report, write no dataset")
    parser.add_argument("--report", help="CSV listing the dropped profiles "
                        "(default: removed_profiles.csv next to the first CSV)")
    args = parser.parse_args()
    assert tuple(PROFILE_ROWS.values()) == INPUT_ROWS

    reports = []
    for path in args.csv:
        table = ColumnTable.read(path)
        target = (table.labels[7].upper() if table.labels is not None and len(table.labels) > 7
                  else args.target)
        if target is None:
            parser.error(f"{path} has no row labels: pass --target")
        target_range = args.target_range or TARGET_RANGES.get(target)
        if target_range is None:
            parser.error(f"No range for target {target}: pass --target_range")
        report = inspect(table, target, target_range, drop_constant=not args.keep_constant)
        report.insert(0, "file", os.path.basename(path))
        reports.append(report)
        print(f"{path}: target {target} {tuple(target_range)}, dropping {len(report)} of {len(table)} profiles",
              flush=True)
        if len(report):
            for issue, count in report["issues"].str.split(";").explode().value_counts().items():
                print(f"    {issue:<24} {count}")
        if args.dry_run:
            continue
        dropped = set(report["profile"])
        cleaned = table.select(i for i, p in enumerate(table.ids) if p not in dropped)
        if args.inplace:
            if not os.path.exists(path + ".bak"):  # a second run must not overwrite the original backup
                shutil.copy2(path, path + ".bak")
            out = path
        else:
            root, ext = os.path.splitext(path)
            out = root + args.suffix + ext
        cleaned.write(out, overwrite=args.inplace)
        print(f"    -> {out} ({len(cleaned)} profiles)", flush=True)

    report_path = args.report or os.path.join(os.path.dirname(os.path.abspath(args.csv[0])), "removed_profiles.csv")
    pd.concat(reports, ignore_index=True).to_csv(report_path, index=False)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

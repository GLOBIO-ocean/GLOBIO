"""Split dataset CSVs into two parts, profile-wise or float-wise.

The input CSVs are concatenated (in the given order) and a fraction of the
profiles (--split profile) or of the floats (--split float: all the profiles
of a float go to the same part) is written to --out_fraction, the rest to
--out_rest. The selection uses scikit-learn's ShuffleSplit / GroupShuffleSplit
with --seed, so the reference splits can be reproduced. Profiles keep
their original order and cells are copied unchanged. Existing files are never
overwritten.

Reference splits (seed 42):
  # float-wise train/test from the profile-wise ones (20% of the floats to test)
  python -m tools.split_dataset train_profile_wise.csv test_profile_wise.csv --split float \
      --fraction 0.2 --out_rest train_float_wise.csv --out_fraction test_float_wise.csv
  # validation/test from a test set (80-10-10 overall)
  python -m tools.split_dataset test_full.csv --split profile --fraction 0.5 \
      --out_rest test_profile_wise.csv --out_fraction val_profile_wise.csv
"""
import argparse
from pathlib import Path

from globio.data import float_id
from tools.csv_columns import ColumnTable


def split_indices(ids, split, fraction, seed):
    try:
        from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
    except ImportError as error:
        raise ImportError("tools.split_dataset needs scikit-learn: pip install scikit-learn") from error
    if split == "float":
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        rest, part = next(splitter.split(ids, groups=[float_id(p) for p in ids]))
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        rest, part = next(splitter.split(ids))
    return sorted(rest), sorted(part)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", help="Input dataset CSVs")
    parser.add_argument("--split", required=True, choices=["profile", "float"])
    parser.add_argument("--fraction", type=float, required=True,
                        help="Fraction of the profiles (or floats) written to --out_fraction")
    parser.add_argument("--out_rest", required=True, type=Path)
    parser.add_argument("--out_fraction", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.fraction < 1:
        parser.error("--fraction must be in (0, 1)")
    for path in (args.out_rest, args.out_fraction):
        if path.exists():
            parser.error(f"{path} already exists")

    table = ColumnTable.concat([ColumnTable.read(path) for path in args.csv])
    rest, part = split_indices(table.ids, args.split, args.fraction, args.seed)
    if args.split == "float":
        assert {float_id(table.ids[i]) for i in rest}.isdisjoint({float_id(table.ids[i]) for i in part})
    for indices, path in ((rest, args.out_rest), (part, args.out_fraction)):
        subset = table.select(indices)
        subset.write(path)
        n_floats = len({float_id(p) for p in subset.ids})
        print(f"{path}: {len(subset)} profiles, {n_floats} floats")


if __name__ == "__main__":
    main()

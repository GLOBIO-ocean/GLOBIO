"""Select the profiles of dataset CSVs by year and/or by membership of their
float in a reference dataset.

Used to build test sets of new data (e.g. only the 2025 profiles), optionally
split float-wise with respect to an existing training set: with --reference,
profiles whose float is in the reference CSV go to --out_in, the others to
--out_out. Cells are copied unchanged; existing files are never overwritten.

Examples:
    # keep only the profiles of 2025
    python -m tools.select_profiles test.csv --year 2025 --out test_2025.csv
    # 2025 profiles, split by the floats of the float-wise reference training set
    python -m tools.select_profiles train_2025.csv test_2025.csv --year 2025 \
        --reference train_float_wise_CORIOLIS.csv --out_in train_float_wise_2025.csv \
        --out_out test_float_wise_2025.csv
"""
import argparse
from pathlib import Path

from globio.data import float_id
from tools.csv_columns import ColumnTable


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", help="Input dataset CSVs (concatenated)")
    parser.add_argument("--year", type=int, nargs="+", help="Keep only these years")
    parser.add_argument("--reference", type=Path, help="Dataset CSV whose floats define --out_in")
    parser.add_argument("--out", type=Path, help="Output without --reference")
    parser.add_argument("--out_in", type=Path, help="With --reference: profiles of reference floats")
    parser.add_argument("--out_out", type=Path, help="With --reference: profiles of the other floats")
    args = parser.parse_args()
    outputs = [args.out_in, args.out_out] if args.reference else [args.out]
    if None in outputs:
        parser.error("Give --out, or --reference with --out_in and --out_out")
    for path in outputs:
        if path.exists():
            parser.error(f"{path} already exists")

    table = ColumnTable.concat([ColumnTable.read(path) for path in args.csv])
    keep = range(len(table))
    if args.year:
        years = table.years()
        if not all(float(y).is_integer() for y in years):
            parser.error("Non-integer years in the input")
        keep = [i for i in keep if int(years[i]) in args.year]
        print(f"Years {args.year}: {len(keep)} of {len(table)} profiles")
    if not keep:
        parser.error("No profile selected")
    if args.reference is None:
        table.select(keep).write(args.out)
        print(f"{args.out}: {len(keep)} profiles")
        return
    reference = {float_id(p) for p in ColumnTable.read(args.reference).ids}
    inside = [i for i in keep if float_id(table.ids[i]) in reference]
    outside = [i for i in keep if float_id(table.ids[i]) not in reference]
    for indices, path in ((inside, args.out_in), (outside, args.out_out)):
        table.select(indices).write(path)
        print(f"{path}: {len(indices)} profiles, {len({float_id(table.ids[i]) for i in indices})} floats")


if __name__ == "__main__":
    main()

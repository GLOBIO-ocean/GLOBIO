"""Make an old checkpoint (weights only, as the reference ones) self-contained
by storing in it the variable, the dropout rate and the normalization
statistics, so that it can be used without --variable and --stats_path.

Example:
    python -m tools.upgrade_checkpoint --checkpoint global_profile_checkpoint_116.pt \
        --variable CHLA --stats_path normalization_stats/SEANOE_CHLA.json --output global_chla_profile.pt
"""
import argparse
from pathlib import Path

import torch

from globio.constants import VARIABLES
from globio.data import load_normalization_stats
from globio.models import CHECKPOINT_VERSION, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--variable", required=True, choices=VARIABLES, type=str.upper)
    parser.add_argument("--stats_path", required=True, type=Path)
    parser.add_argument("--dp_rate", type=float, default=0.2, help="Dropout rate used in training")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--note", help="Free text stored in the checkpoint (e.g. dataset and split)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists")

    load_checkpoint(args.checkpoint, args.variable)  # check that weights and variable match
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("norm_stats") not in (None, load_normalization_stats(args.stats_path)):
        parser.error(f"{args.checkpoint} already stores different normalization statistics")
    checkpoint.update(format_version=CHECKPOINT_VERSION, variable=args.variable, dp_rate=args.dp_rate,
                      norm_stats=load_normalization_stats(args.stats_path), source=str(args.checkpoint))
    checkpoint.pop("optimizer", None)
    if args.note:
        checkpoint["note"] = args.note
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

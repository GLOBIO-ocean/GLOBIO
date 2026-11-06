#!/usr/bin/env python3
"""Train a GlobIO model: global, regional from scratch, or fine-tuned from a
pretrained checkpoint.

Run with plain ``python``: one process per visible GPU is started internally
(restrict them with CUDA_VISIBLE_DEVICES). For several nodes use
``torchrun --nnodes ... train.py ...``. Without GPUs training runs on CPU.

Examples:
    # global model
    python train.py --variable CHLA --train_csv data/train.csv --val_csv data/val.csv --epochs 200

    # regional model from scratch
    python train.py ... --region mediterranean

    # fine-tuning of a global model on a region
    python train.py ... --region mediterranean --init_checkpoint global.pt --epochs 100

    # continue an interrupted run
    python train.py --resume results/CHLA/<run>
"""
import argparse

from globio.constants import REGIONS, VARIABLES
from globio.training import FT_STRATEGIES, TrainConfig, resume, train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = parser.add_argument_group("data")
    data.add_argument("--variable", choices=VARIABLES, type=str.upper)
    data.add_argument("--train_csv", help="Training dataset CSV")
    data.add_argument("--val_csv", help="Validation dataset CSV (used for model selection)")
    data.add_argument("--region", choices=list(REGIONS),
                      help="Train only on the profiles of this region (training and validation)")
    data.add_argument("--stats_path", help="Normalization statistics JSON. Default: those of --init_checkpoint "
                      "when fine-tuning, otherwise computed on --train_csv (before the region selection)")
    data.add_argument("--max_profiles", type=int, help="Debug: use only the first profiles of each CSV")

    optim = parser.add_argument_group("optimization (defaults: reference values)")
    optim.add_argument("--epochs", type=int, help="Number of epochs (default 200; reference: CHLA 200 global and "
                       "regional, 100 fine-tuning; NITRATE 50 global and regional, 25 fine-tuning)")
    optim.add_argument("--batch_size", type=int, default=32, help="Batch size per process (GPU)")
    optim.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    optim.add_argument("--dp_rate", type=float, default=0.2, help="Dropout rate of the CNN")
    optim.add_argument("--alpha_smooth_reg", type=float, default=1.99e-2,
                       help="Weight of the smoothness penalty (mean over levels of |p[i+1] - p[i]|, "
                            "summed over the profiles of the batch); reference: 1.99e-2")
    optim.add_argument("--seed", type=int, default=0)
    optim.add_argument("--recalibrate_bn", action="store_true",
                       help="Recompute the BatchNorm statistics on the training set before each validation")

    ft = parser.add_argument_group("fine-tuning")
    ft.add_argument("--init_checkpoint", help="Checkpoint of the model to fine-tune")
    ft.add_argument("--ft_strategy", choices=FT_STRATEGIES,
                    help="aggressive (default, used in the reference runs): train all weights; freeze_backbone: "
                         "freeze the metadata MLPs, the first CNN blocks and the BatchNorm layers")

    out = parser.add_argument_group("output and execution")
    out.add_argument("--output_dir", default="results", help="Runs are saved in <output_dir>/<VARIABLE>/<run_name>")
    out.add_argument("--run_name", help="Default: <date>_<time>_<global|local_REGION|ft_REGION>")
    out.add_argument("--save_every", type=int, default=0,
                     help="Also keep a checkpoint every N epochs (best.pt and last.pt are always written)")
    out.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    out.add_argument("--n_procs", type=int, help="Number of processes (default: number of GPUs, 1 on CPU)")
    out.add_argument("--num_workers", type=int, default=0, help="DataLoader workers per process")
    out.add_argument("--resume", metavar="RUN_DIR", help="Continue the run in RUN_DIR from checkpoints/last.pt; "
                     "only --epochs can be changed")
    args = parser.parse_args()

    if args.resume:
        resume(args.resume, epochs=args.epochs)
        return
    missing = [name for name in ("variable", "train_csv", "val_csv") if getattr(args, name) is None]
    if missing:
        parser.error("the following arguments are required: " + ", ".join("--" + m for m in missing))
    options = {k: v for k, v in vars(args).items() if k != "resume"}
    options["epochs"] = 200 if args.epochs is None else args.epochs
    try:
        config = TrainConfig(**options).validate()
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    train(config)


if __name__ == "__main__":
    main()

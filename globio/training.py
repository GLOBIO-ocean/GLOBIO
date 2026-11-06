"""Distributed training and fine-tuning.

The loss is mean squared error over all levels of the batch plus
``alpha_smooth_reg`` times the smoothness penalty: for each profile the mean
absolute difference between consecutive levels (199 differences), summed over
the profiles of the batch. With the default ``alpha_smooth_reg = 1.99e-2`` this
is numerically identical to the loss used to train the reference models
(whose original code summed the differences without dividing by 199, with weight 1e-4).

Training runs with DistributedDataParallel on all visible GPUs of a node
(one process per GPU, NCCL), on several nodes with ``torchrun``, or on CPU
(gloo) when no GPU is available.
"""
import dataclasses
import datetime as dt
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .constants import REGIONS, check_variable
from .data import (ProfileDataset, compute_normalization_stats, load_normalization_stats,
                   save_normalization_stats)
from .inference import recalibrate_batchnorm
from .models import EARLY_LAYERS, GlobioNet, load_checkpoint, save_checkpoint, unwrap

FT_STRATEGIES = ("aggressive", "freeze_backbone")


@dataclasses.dataclass
class TrainConfig:
    """Training options; see ``python train.py --help`` for their meaning."""
    variable: str
    train_csv: str
    val_csv: str
    output_dir: str = "results"
    run_name: str = None
    region: str = None
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    dp_rate: float = 0.2
    alpha_smooth_reg: float = 1.99e-2
    save_every: int = 0
    init_checkpoint: str = None
    ft_strategy: str = None
    stats_path: str = None
    recalibrate_bn: bool = False
    seed: int = 0
    num_workers: int = 0
    max_profiles: int = None
    device: str = "auto"
    n_procs: int = None

    def validate(self, resuming=False):
        self.variable = check_variable(self.variable)
        if self.region is not None and self.region not in REGIONS:
            raise ValueError(f"Unknown region {self.region!r}; expected one of {list(REGIONS)}")
        if self.ft_strategy is not None and self.ft_strategy not in FT_STRATEGIES:
            raise ValueError(f"ft_strategy must be one of {FT_STRATEGIES}")
        if self.ft_strategy is not None and self.init_checkpoint is None:
            raise ValueError("ft_strategy needs init_checkpoint (the model to fine-tune)")
        if self.init_checkpoint is not None and self.ft_strategy is None:
            self.ft_strategy = "aggressive"
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError("device must be auto, cuda or cpu")
        for name in ("epochs", "batch_size", "n_procs", "max_profiles"):
            if getattr(self, name) is not None and getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.recalibrate_bn and self.ft_strategy == "freeze_backbone":
            raise ValueError("recalibrate_bn would change the BatchNorm layers frozen by freeze_backbone")
        if self.lr <= 0 or not 0 <= self.dp_rate < 1 or self.alpha_smooth_reg < 0 or self.save_every < 0:
            raise ValueError("Require lr > 0, 0 <= dp_rate < 1, alpha_smooth_reg >= 0, save_every >= 0")
        # On resume the weights and statistics come from the run directory.
        paths = (self.train_csv, self.val_csv) if resuming else \
            (self.train_csv, self.val_csv, self.init_checkpoint, self.stats_path)
        for path in paths:
            if path is not None and not Path(path).is_file():
                raise FileNotFoundError(path)
        for name in ("train_csv", "val_csv", "init_checkpoint", "stats_path", "output_dir"):
            if getattr(self, name) is not None:
                setattr(self, name, str(Path(getattr(self, name)).resolve()))
        return self

    @property
    def kind(self):
        if self.init_checkpoint is not None:
            return f"ft_{self.region}" if self.region else "ft"
        return f"local_{self.region}" if self.region else "global"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def loss_function(prediction, target, alpha_smooth_reg):
    """MSE + alpha_smooth_reg * sum over profiles of mean_i |p[i+1] - p[i]|."""
    smooth = torch.abs(prediction[:, 1:] - prediction[:, :-1]).mean(dim=1).sum()
    return torch.nn.functional.mse_loss(prediction, target) + alpha_smooth_reg * smooth


def apply_ft_strategy(model, strategy):
    """``aggressive``: train all weights. ``freeze_backbone``: freeze the metadata
    MLPs, the first two CNN blocks and every BatchNorm layer (kept in eval mode)."""
    if strategy in (None, "aggressive"):
        return
    for mlp in (model.mlp_day, model.mlp_year, model.mlp_space):
        mlp.requires_grad_(False)
    early = EARLY_LAYERS[model.variable]
    for name, module in model.cnn.named_children():
        if name in early or isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.requires_grad_(False)


def set_train_mode(model, strategy):
    model.train()
    if strategy == "freeze_backbone":
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _use_cuda(config):
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda but no GPU is available")
    return config.device != "cpu" and torch.cuda.is_available()


def _new_run_dir(config):
    root = Path(config.output_dir) / config.variable
    name = config.run_name or f"{dt.datetime.now():%Y-%m-%d_%H%M%S}_{config.kind}"
    run_dir = root / name
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def resolve_stats(config):
    """Normalization statistics for a run (see the README)."""
    from_checkpoint = None
    if config.init_checkpoint is not None:
        _, info = load_checkpoint(config.init_checkpoint, config.variable)
        if info["dp_rate"] != config.dp_rate:
            print(f"[train] note: {config.init_checkpoint} was trained with dp_rate={info['dp_rate']}, "
                  f"fine-tuning with dp_rate={config.dp_rate}", flush=True)
        from_checkpoint = info["norm_stats"]
    if config.stats_path is not None:
        stats = load_normalization_stats(config.stats_path)
        if from_checkpoint is not None and from_checkpoint != stats:
            raise ValueError(f"{config.stats_path} differs from the statistics stored in "
                             f"{config.init_checkpoint}: fine-tuning must use the same normalization")
        return stats, config.stats_path
    if from_checkpoint is not None:
        return from_checkpoint, config.init_checkpoint
    if config.init_checkpoint is not None:
        raise ValueError(f"{config.init_checkpoint} has no normalization statistics: pass the JSON "
                         "used to train it with --stats_path")
    print(f"[train] computing normalization statistics on {config.train_csv}", flush=True)
    dataset = ProfileDataset(config.train_csv, config.variable, norm_stats=None, verbose=False)
    return compute_normalization_stats(dataset), config.train_csv


def prepare_run(config, resume_dir=None):
    """Create the run directory, write ``config.json`` and the normalization
    statistics. Returns ``(run_dir, stats)``."""
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        stats = load_normalization_stats(run_dir / "normalization_stats.json")
        return run_dir, stats
    stats, stats_source = resolve_stats(config)
    run_dir = _new_run_dir(config)
    save_normalization_stats(stats, run_dir / "normalization_stats.json")
    info = {"config": dataclasses.asdict(config), "kind": config.kind, "stats_source": str(stats_source),
            "git_commit": _git_commit(), "torch": torch.__version__, "created": dt.datetime.now().isoformat()}
    (run_dir / "config.json").write_text(json.dumps(info, indent=2) + "\n")
    return run_dir, stats


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, config, run_dir, stats, rank, world_size, device, resume=False):
        self.config, self.run_dir, self.stats = config, Path(run_dir), stats
        self.rank, self.world_size, self.device = rank, world_size, device
        self.is_main = rank == 0
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        kwargs = dict(variable=config.variable, norm_stats=stats, region=config.region,
                      max_profiles=config.max_profiles, verbose=self.is_main)
        self.train_set = ProfileDataset(config.train_csv, **kwargs)
        self.val_set = ProfileDataset(config.val_csv, **kwargs)
        for name, dataset in (("training", self.train_set), ("validation", self.val_set)):
            if len(dataset) == 0:
                raise ValueError(f"No {name} profiles" + (f" in region {config.region}" if config.region else ""))
        self.log(f"training profiles: {len(self.train_set)}, validation profiles: {len(self.val_set)}, "
                 f"processes: {world_size}, device: {device}")

        model = GlobioNet(config.variable, config.dp_rate)
        if config.init_checkpoint is not None and not resume:
            pretrained, _ = load_checkpoint(config.init_checkpoint, config.variable, dp_rate=config.dp_rate)
            model.load_state_dict(pretrained.state_dict())
            self.log(f"fine-tuning {config.init_checkpoint} ({config.ft_strategy})")
        apply_ft_strategy(model, config.ft_strategy)
        if device.type == "cuda" and world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)
        self.model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
        self.optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.lr)

        self.start_epoch, self.history = 1, []
        self.best_val_loss, self.best_epoch = float("inf"), None
        if resume:
            last = self.run_dir / "checkpoints" / "last.pt"
            checkpoint = torch.load(last, map_location="cpu", weights_only=True)
            unwrap(self.model).load_subnet_state_dicts(checkpoint)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.history = checkpoint.get("history", [])
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.best_epoch = checkpoint.get("best_epoch")
            self.log(f"resuming {last} from epoch {self.start_epoch}")

        pin = device.type == "cuda"
        self.train_sampler = DistributedSampler(self.train_set, world_size, rank, shuffle=True, seed=config.seed)
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, sampler=self.train_sampler,
                                       num_workers=config.num_workers, pin_memory=pin)
        # Validation runs on rank 0 over the whole set, so the loss (and the best
        # epoch) does not depend on the number of processes.
        self.val_loader = DataLoader(self.val_set, batch_size=config.batch_size, shuffle=False,
                                     num_workers=config.num_workers, pin_memory=pin)

    def log(self, message):
        if self.is_main:
            print(f"[train] {message}", flush=True)

    def _forward(self, batch):
        return self.model(*(batch[name].to(self.device, torch.float32, non_blocking=True)
                            for name in ("year", "day", "space", "temp", "psal", "doxy")))

    def _mean_over_processes(self, losses):
        total = torch.tensor([sum(losses), len(losses)], dtype=torch.float64, device=self.device)
        dist.all_reduce(total)
        return (total[0] / total[1]).item()

    def train_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
        set_train_mode(self.model, self.config.ft_strategy)
        losses = []
        for batch in self.train_loader:
            target = batch["target"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            loss = loss_function(self._forward(batch), target, self.config.alpha_smooth_reg)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        if not np.isfinite(losses).all():
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        return self._mean_over_processes(losses)

    @torch.no_grad()
    def validate(self):
        result = torch.zeros(1, dtype=torch.float64, device=self.device)
        if self.is_main:
            model = unwrap(self.model)  # no collective operations: eval-mode (Sync)BatchNorm is local
            model.eval()
            losses = [loss_function(model.forward_batch(batch, self.device), batch["target"].to(self.device),
                                    self.config.alpha_smooth_reg).item()
                      for batch in self.val_loader]
            result[0] = float(np.mean(losses))
        dist.broadcast(result, src=0)
        return result.item()

    def _save(self, path, epoch):
        save_checkpoint(path, self.model, self.stats, optimizer=self.optimizer, epoch=epoch,
                        config=dataclasses.asdict(self.config), history=self.history,
                        best_val_loss=self.best_val_loss, best_epoch=self.best_epoch,
                        n_train_profiles=len(self.train_set), n_val_profiles=len(self.val_set))

    def train(self):
        config, checkpoints = self.config, self.run_dir / "checkpoints"
        for epoch in range(self.start_epoch, config.epochs + 1):
            start = time.time()
            train_loss = self.train_epoch(epoch)
            if config.recalibrate_bn:
                recalibrate_batchnorm(self.model, self.train_loader, self.device)
            val_loss = self.validate()
            if self.is_main:
                improved = val_loss < self.best_val_loss
                if improved:
                    self.best_val_loss, self.best_epoch = val_loss, epoch
                self.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                                     "seconds": round(time.time() - start, 2)})
                if improved:
                    self._save(checkpoints / "best.pt", epoch)
                if config.save_every and epoch % config.save_every == 0:
                    self._save(checkpoints / f"epoch_{epoch:04d}.pt", epoch)
                self._save(checkpoints / "last.pt", epoch)
                (self.run_dir / "history.json").write_text(json.dumps(
                    {"best_epoch": self.best_epoch, "best_val_loss": self.best_val_loss,
                     "epochs": self.history}, indent=2) + "\n")
                print(f"[epoch {epoch}/{config.epochs}] train loss {train_loss:.6f} | val loss {val_loss:.6f}"
                      f"{' (best)' if improved else ''} | {time.time() - start:.1f} s", flush=True)
            dist.barrier()
        self.log(f"done: best epoch {self.best_epoch}, val loss {self.best_val_loss:.6f}, "
                 f"checkpoint {checkpoints / 'best.pt'}")


def _worker(local_rank, world_size, config, run_dir, stats, resume, use_cuda, init_method=None, rank_offset=0):
    rank = rank_offset + local_rank
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if not dist.is_initialized():
        dist.init_process_group("nccl" if use_cuda else "gloo", init_method=init_method, rank=rank,
                                world_size=world_size)
    try:
        Trainer(config, run_dir, stats, rank, world_size, device, resume).train()
    finally:
        dist.destroy_process_group()


def train(config, resume_dir=None):
    """Train (or fine-tune, or resume) a model. Returns the run directory.

    Without ``torchrun``, one process is started per visible GPU (or
    ``config.n_procs`` processes, on CPU when no GPU is available). Under
    ``torchrun`` the processes are those started by torchrun.
    """
    config.validate(resuming=resume_dir is not None)
    use_cuda = _use_cuda(config)
    if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:  # torchrun
        rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if use_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl" if use_cuda else "gloo")
        shared = [None, None]
        if rank == 0:
            run_dir, stats = prepare_run(config, resume_dir)
            shared = [str(run_dir), stats]
        dist.broadcast_object_list(shared, src=0)
        _worker(local_rank, world_size, config, shared[0], shared[1], resume_dir is not None, use_cuda,
                rank_offset=rank - local_rank)  # process group already initialized by torchrun's env
        return Path(shared[0])

    run_dir, stats = prepare_run(config, resume_dir)
    world_size = config.n_procs or (torch.cuda.device_count() if use_cuda else 1)
    if use_cuda and world_size > torch.cuda.device_count():
        raise ValueError(f"n_procs={world_size} but only {torch.cuda.device_count()} GPUs are visible")
    print(f"[train] run directory: {run_dir}", flush=True)
    init_method = f"tcp://127.0.0.1:{_free_port()}"  # single node: rendezvous on a free local port
    args = (world_size, config, str(run_dir), stats, resume_dir is not None, use_cuda, init_method)
    if world_size == 1:
        _worker(0, *args)
    else:
        torch.multiprocessing.spawn(_worker, args=args, nprocs=world_size, join=True)
    return run_dir


def resume(run_dir, epochs=None):
    """Continue the run in ``run_dir`` from its last checkpoint (optionally
    extending the number of epochs)."""
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "config.json").read_text())
    config = TrainConfig(**info["config"])
    if not (run_dir / "checkpoints" / "last.pt").is_file():
        raise FileNotFoundError(f"{run_dir}/checkpoints/last.pt not found")
    if epochs is not None and epochs != config.epochs:
        config.epochs = epochs
        info["config"]["epochs"] = epochs
        (run_dir / "config.json").write_text(json.dumps(info, indent=2) + "\n")
    return train(config, resume_dir=run_dir)


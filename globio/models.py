"""Network architectures and checkpoints.

A GlobIO model (:class:`GlobioNet`) has three small MLPs that embed the day of
the year, the year and the position into 200-level channels, and a 1D CNN that
maps the six channels (day, year, position, temperature, salinity, oxygen) to
the 200-level target profile. Layer names match the reference checkpoints.
"""
import warnings
from pathlib import Path

import torch
from torch import nn

from .constants import N_LEVELS, check_variable
from .data import check_normalization_stats

IN_CHANNELS = 6
CHECKPOINT_VERSION = 2


class Conv1dMed_CHLA(nn.Module):
    """CNN for CHLA and BBP700 (0-200 dbar, 1 dbar step)."""

    def __init__(self, dp_rate):
        super().__init__()
        self.dp_rate = dp_rate
        self.conv1 = nn.Conv1d(IN_CHANNELS, 53, kernel_size=8, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(53)
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(p=dp_rate)
        self.conv2 = nn.Conv1d(53, 77, kernel_size=8, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(77)
        self.af2 = nn.ReLU()
        self.do2 = nn.Dropout(p=dp_rate)
        self.deconv3 = nn.ConvTranspose1d(77, 46, kernel_size=7, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(46)
        self.af3 = nn.ReLU()
        self.do3 = nn.Dropout(p=dp_rate)
        self.conv4 = nn.Conv1d(46, 128, kernel_size=4, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.af4 = nn.ReLU()
        self.do4 = nn.Dropout(p=dp_rate)
        self.conv5 = nn.Conv1d(128, 98, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm1d(98)
        self.af5 = nn.ReLU()
        self.do5 = nn.Dropout(p=dp_rate)
        self.deconv6 = nn.ConvTranspose1d(98, 12, kernel_size=2, stride=2, padding=1)
        self.do6 = nn.Dropout(p=dp_rate)
        self.flatten = nn.Flatten()
        self.fc7 = nn.Linear(600, N_LEVELS)
        self.af7 = nn.Softplus()  # positive output

    def forward(self, x):
        x = self.do1(self.af1(self.bn1(self.conv1(x))))
        x = self.do2(self.af2(self.bn2(self.conv2(x))))
        x = self.do3(self.af3(self.bn3(self.deconv3(x))))
        x = self.do4(self.af4(self.bn4(self.conv4(x))))
        x = self.do5(self.af5(self.bn5(self.conv5(x))))
        x = self.do6(self.deconv6(x))
        return self.af7(self.fc7(self.flatten(x)))


class Conv1dMed_NITRATE(nn.Module):
    """CNN for NITRATE (0-1000 dbar, 5 dbar step)."""

    def __init__(self, dp_rate):
        super().__init__()
        self.dp_rate = dp_rate
        self.deconv1 = nn.ConvTranspose1d(IN_CHANNELS, 61, kernel_size=10, stride=2, padding=1)
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(p=dp_rate)
        self.deconv2 = nn.ConvTranspose1d(61, 128, kernel_size=2, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.af2 = nn.ReLU()
        self.do2 = nn.Dropout(p=dp_rate)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=2, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.af3 = nn.ReLU()
        self.do3 = nn.Dropout(p=dp_rate)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=2, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.af4 = nn.ReLU()
        self.do4 = nn.Dropout(p=dp_rate)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=2)
        self.bn5 = nn.BatchNorm1d(128)
        self.af5 = nn.ReLU()
        self.do5 = nn.Dropout(p=dp_rate)
        self.deconv6 = nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2, padding=2)
        self.bn6 = nn.BatchNorm1d(128)
        self.af6 = nn.ReLU()
        self.do6 = nn.Dropout(p=dp_rate)
        self.conv7 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm1d(128)
        self.af7 = nn.ReLU()
        self.do7 = nn.Dropout(p=dp_rate)
        self.conv8 = nn.Conv1d(128, 84, kernel_size=2, stride=1, padding=0)
        self.af8 = nn.ReLU()
        self.do8 = nn.Dropout(p=dp_rate)
        self.conv9 = nn.Conv1d(84, 111, kernel_size=4, stride=1, padding=1)
        self.bn9 = nn.BatchNorm1d(111)
        self.af9 = nn.ReLU()
        self.do9 = nn.Dropout(p=dp_rate)
        self.conv10 = nn.Conv1d(111, 1, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc11 = nn.Linear(808, N_LEVELS)
        self.af11 = nn.Softplus()  # positive output

    def forward(self, x):
        x = self.do1(self.af1(self.deconv1(x)))
        x = self.do2(self.af2(self.bn2(self.deconv2(x))))
        x = self.do3(self.af3(self.bn3(self.conv3(x))))
        x = self.do4(self.af4(self.bn4(self.conv4(x))))
        x = self.do5(self.af5(self.bn5(self.conv5(x))))
        x = self.do6(self.af6(self.bn6(self.deconv6(x))))
        x = self.do7(self.af7(self.bn7(self.conv7(x))))
        x = self.do8(self.af8(self.conv8(x)))
        x = self.do9(self.af9(self.bn9(self.conv9(x))))
        x = self.conv10(x)
        return self.af11(self.fc11(self.flatten(x)))


def _mlp(n_inputs):
    return nn.Sequential(nn.Linear(n_inputs, 80), nn.ReLU(), nn.Linear(80, 140), nn.ReLU(),
                         nn.Linear(140, N_LEVELS), nn.ReLU())


class _MLP(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.network = _mlp(n_inputs)

    def forward(self, x):
        return self.network(x)


class MLPDay(_MLP):
    """Embedding of the day of the year (sin, cos)."""

    def __init__(self):
        super().__init__(2)


class MLPYear(_MLP):
    """Embedding of the year (``year - 2010``)."""

    def __init__(self):
        super().__init__(1)


class MLPSpace(_MLP):
    """Embedding of the position on the unit sphere (x, y, z)."""

    def __init__(self):
        super().__init__(3)


CNNS = {"CHLA": Conv1dMed_CHLA, "NITRATE": Conv1dMed_NITRATE, "BBP700": Conv1dMed_CHLA}

#: Layers frozen by the ``freeze_backbone`` fine-tuning strategy (with the MLPs
#: and all the BatchNorm layers).
EARLY_LAYERS = {"CHLA": ("conv1", "bn1", "conv2", "bn2"),
                "BBP700": ("conv1", "bn1", "conv2", "bn2"),
                "NITRATE": ("deconv1", "deconv2", "bn2")}


class GlobioNet(nn.Module):
    """Complete model: metadata MLPs + CNN. ``forward`` returns ``(batch, 200)``."""

    def __init__(self, variable, dp_rate=0.2):
        super().__init__()
        self.variable = check_variable(variable)
        self.dp_rate = float(dp_rate)
        self.cnn = CNNS[self.variable](dp_rate)
        self.mlp_day = MLPDay()
        self.mlp_year = MLPYear()
        self.mlp_space = MLPSpace()

    def forward(self, year, day, space, temp, psal, doxy):
        """year (B,), day (B, 2), space (B, 3), temp/psal/doxy (B, 200), normalized."""
        channels = (self.mlp_day(day), self.mlp_year(year.unsqueeze(1)), self.mlp_space(space),
                    temp, psal, doxy)
        return self.cnn(torch.stack(channels, dim=1))

    def forward_batch(self, batch, device=None):
        """Forward pass on a batch of :class:`globio.data.ProfileDataset` items."""
        device = device or next(self.parameters()).device
        return self(*(batch[name].to(device, torch.float32, non_blocking=True)
                      for name in ("year", "day", "space", "temp", "psal", "doxy")))

    def subnet_state_dicts(self):
        return {"model": self.cnn.state_dict(), "mlp_day": self.mlp_day.state_dict(),
                "mlp_year": self.mlp_year.state_dict(), "mlp_space": self.mlp_space.state_dict()}

    def load_subnet_state_dicts(self, checkpoint):
        self.cnn.load_state_dict(checkpoint["model"])
        self.mlp_day.load_state_dict(checkpoint["mlp_day"])
        self.mlp_year.load_state_dict(checkpoint["mlp_year"])
        self.mlp_space.load_state_dict(checkpoint["mlp_space"])


def unwrap(model):
    """The module inside a DistributedDataParallel wrapper (or the model itself)."""
    return model.module if hasattr(model, "module") else model


def infer_architecture(state_dict):
    """``"NITRATE"`` or ``"CHLA"`` (also used for BBP700) from the CNN weights."""
    if "fc11.weight" in state_dict:
        return "NITRATE"
    if "fc7.weight" in state_dict:
        return "CHLA"
    raise ValueError("Unrecognized CNN weights")


def save_checkpoint(path, model, norm_stats, optimizer=None, epoch=None, **extra):
    """Save a self-contained checkpoint.

    It holds the weights (keys ``model``, ``mlp_day``, ``mlp_year``, ``mlp_space``,
    as the reference checkpoints), the variable, the dropout rate and the
    normalization statistics, so it can be used without other files.
    """
    model = unwrap(model)
    checkpoint = {
        "format_version": CHECKPOINT_VERSION,
        **model.subnet_state_dicts(),
        "variable": model.variable,
        "dp_rate": model.dp_rate,
        "norm_stats": check_normalization_stats(norm_stats),
        "epoch": epoch,
        **extra,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)  # never leave a truncated checkpoint behind


def load_checkpoint(path, variable=None, device="cpu", dp_rate=None):
    """Load a checkpoint into a :class:`GlobioNet` in eval mode.

    Works with self-contained checkpoints written by :func:`save_checkpoint`
    and with the older reference ones, which hold only the weights: for
    those, ``variable`` is needed to tell CHLA from BBP700 (NITRATE is detected
    from the weights) and the normalization statistics must be given separately.

    Returns ``(model, info)``; ``info`` is the checkpoint without the weights
    (``variable``, ``dp_rate``, ``norm_stats`` - ``None`` for old checkpoints -,
    ``epoch``, ``config``, ...).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    missing = {"model", "mlp_day", "mlp_year", "mlp_space"} - set(checkpoint)
    if missing:
        raise ValueError(f"{path} is not a GlobIO checkpoint: missing {sorted(missing)}")
    architecture = infer_architecture(checkpoint["model"])
    stored = checkpoint.get("variable")
    if variable is not None:
        variable = check_variable(variable)
        if stored is not None and stored != variable:
            raise ValueError(f"{path} is a {stored} model, not {variable}")
    variable = variable or stored
    if variable is None:
        if architecture == "CHLA":
            raise ValueError(f"{path} has no variable: pass variable='CHLA' or 'BBP700'")
        variable = architecture
    if CNNS[variable] is not CNNS[architecture]:
        raise ValueError(f"{path} has {architecture} weights, not {variable}")
    dp_rate = dp_rate if dp_rate is not None else checkpoint.get("dp_rate", 0.2)
    model = GlobioNet(variable, dp_rate)
    model.load_subnet_state_dicts(checkpoint)
    model.to(device).eval()
    info = {k: v for k, v in checkpoint.items()
            if k not in ("model", "mlp_day", "mlp_year", "mlp_space", "optimizer")}
    info.update(variable=variable, dp_rate=dp_rate, path=str(path),
                norm_stats=checkpoint.get("norm_stats"))
    if "optimizer" in checkpoint:
        info["optimizer"] = checkpoint["optimizer"]
    return model, info


def resolve_norm_stats(info, stats_path=None):
    """Normalization statistics for a loaded checkpoint: ``stats_path`` if given
    (warning if it differs from the statistics stored in the checkpoint),
    otherwise the stored ones. Raises if neither is available."""
    from .data import load_normalization_stats
    stored = info.get("norm_stats")
    if stats_path is not None:
        stats = load_normalization_stats(stats_path)
        if stored is not None and stored != stats:
            warnings.warn(f"{stats_path} differs from the statistics stored in {info.get('path')}; "
                          f"using {stats_path}")
        return stats
    if stored is None:
        raise ValueError(f"{info.get('path')} has no normalization statistics: pass the JSON used "
                         "to train it (e.g. normalization_stats/SEANOE_CHLA.json), or embed it with "
                         "tools/upgrade_checkpoint.py")
    return stored

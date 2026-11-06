"""Deterministic and Monte Carlo dropout predictions."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import unwrap

DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)


def default_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset, batch_size=256, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict(model, dataset, batch_size=256, device=None):
    """Predictions of ``model`` in eval mode, shape ``(len(dataset), 200)``.

    The dataset must be normalized with the statistics of the model
    (``dataset.set_normalization(...)``).
    """
    device = default_device(device)
    model = unwrap(model).to(device)
    was_training = model.training
    model.eval()
    try:
        out = [model.forward_batch(batch, device).cpu().numpy()
               for batch in make_loader(dataset, batch_size)]
    finally:
        model.train(was_training)
    return np.concatenate(out) if out else np.empty((0, 200), np.float32)


def enable_mc_dropout(model):
    """Eval mode everywhere (BatchNorm uses its running statistics) except the
    dropout layers, which stay stochastic. Returns the number of dropout layers."""
    model.eval()
    count = 0
    for module in model.modules():
        if isinstance(module, DROPOUT_TYPES):
            module.train()
            count += module.p > 0
    if not count:
        raise ValueError("The model has no dropout layer with p > 0: MC dropout is not possible")
    return count


@torch.no_grad()
def predict_mc(model, dataset, n_samples=100, batch_size=32, device=None, seed=123,
               return_samples=False):
    """Monte Carlo dropout predictions.

    Every profile is predicted ``n_samples`` times with dropout active and
    BatchNorm in eval mode. Returns a dict with ``mean`` and ``std`` (sample
    standard deviation, ddof=1) of the passes, the ``deterministic`` eval-mode
    prediction and, with ``return_samples``, all the passes
    ``(n_samples, n_profiles, 200)``. Results depend on ``seed`` and
    ``batch_size`` (which set the dropout masks).
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    device = default_device(device)
    model = unwrap(model).to(device)
    was_training = model.training
    generator_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    means, stds, deterministic, samples = [], [], [], []
    try:
        for batch in make_loader(dataset, batch_size):
            enable_mc_dropout(model)
            draws = torch.stack([model.forward_batch(batch, device) for _ in range(n_samples)])
            if not torch.isfinite(draws).all():
                raise FloatingPointError("Non-finite MC dropout predictions")
            means.append(draws.mean(0).cpu().numpy())
            stds.append(draws.std(0, unbiased=True).cpu().numpy())
            if return_samples:
                samples.append(draws.cpu().numpy())
            model.eval()
            deterministic.append(model.forward_batch(batch, device).cpu().numpy())
    finally:
        model.train(was_training)
        torch.random.set_rng_state(generator_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    out = {"mean": np.concatenate(means), "std": np.concatenate(stds),
           "deterministic": np.concatenate(deterministic)}
    if return_samples:
        out["samples"] = np.concatenate(samples, axis=1)
    return out


@torch.no_grad()
def recalibrate_batchnorm(model, loader, device=None):
    """Recompute the BatchNorm running statistics with the current weights.

    Running statistics are an exponential average over batches seen with older
    weights; here they are reset and recomputed as an exact average
    (``momentum=None``) over ``loader``. Only the BatchNorm layers are put in
    train mode, so dropout stays off and no weight changes. ``model`` can be a
    DistributedDataParallel wrapper (with SyncBatchNorm the statistics are
    averaged over all processes).
    """
    device = default_device(device)
    inner = unwrap(model)
    batchnorms = [m for m in inner.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not batchnorms:
        return
    was_training = inner.training
    momenta = [bn.momentum for bn in batchnorms]
    inner.eval()
    for bn in batchnorms:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    try:
        for batch in loader:
            model(*(batch[name].to(device, torch.float32)
                    for name in ("year", "day", "space", "temp", "psal", "doxy")))
    finally:
        for bn, momentum in zip(batchnorms, momenta):
            bn.momentum = momentum
        inner.train(was_training)

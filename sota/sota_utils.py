"""Small shared helpers for the SOTA models (wandb + progress + early stop)."""

import torch


def train_val_split(n, val_frac, seed, device, max_val=200_000):
    """Split ``n`` row indices into (train_idx, val_idx) on ``device``.

    The validation set is carved from the TRAINING data so the held-out test set
    is never used for model selection / early stopping. ``val_frac<=0`` disables
    it (empty val_idx). The val size is capped at ``max_val`` so per-epoch val
    evaluation stays cheap on multi-million-row splits.
    """
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    n_val = min(int(n * val_frac), max_val) if val_frac and val_frac > 0 else 0
    va = perm[:n_val].to(device)
    tr = perm[n_val:].to(device)
    return tr, va


class EarlyStopper:
    """Best-checkpoint early stopping.

    Tracks the lowest monitored metric (e.g. validation MSE), snapshots the
    module's weights at that best epoch (kept on CPU), and can restore them so
    the FINAL model is the best epoch -- not the last (which, for a
    high-capacity field on a cross-specimen split, is typically the worst).

    ``patience=0`` never terminates early but still keeps the best snapshot;
    ``patience>0`` also stops after that many epochs without improvement.
    """
    def __init__(self, patience=0, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.best_state = None
        self.best_epoch = -1
        self.bad = 0

    def step(self, metric, module, epoch):
        """Record ``metric`` for ``epoch``; returns True if training should stop."""
        if metric < self.best - self.min_delta:
            self.best = float(metric)
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in module.state_dict().items()}
            self.best_epoch = int(epoch)
            self.bad = 0
            return False
        self.bad += 1
        return self.patience > 0 and self.bad >= self.patience

    def restore(self, module):
        """Load the best snapshot back into ``module`` (device-preserving)."""
        if self.best_state is not None:
            module.load_state_dict(self.best_state)


def get_wandb():
    """Return the ``wandb`` module iff a run is active, else None.

    The driver (run_sota.py) calls ``wandb.init`` when --wandb is set; the models
    then log through this guard, so they stay decoupled -- running a model
    directly (e.g. a unit test) without an active run simply skips logging.
    """
    try:
        import wandb
        return wandb if getattr(wandb, "run", None) is not None else None
    except Exception:
        return None


def wandb_log(metrics, step=None):
    """Log ``metrics`` (a flat dict) to wandb if a run is active; else no-op."""
    wb = get_wandb()
    if wb is not None:
        wb.log(metrics, step=step)

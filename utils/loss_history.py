"""Loss-curve tracking and plotting for both base training and unlearning phases.

Plugs into the existing `build_trainer(..., additional_callbacks=[...])` hook
already defined in utils.py -- no notebook cells need to change, only the
run*.py scripts that call `build_trainer`.

CAVEAT: strategy classes (regun.py, neggrad.py, salun.py, ...) build their
trainer via `self.new_trainer()`, defined in base.py, which is not available
here. If `new_trainer()` internally wraps `build_trainer` and forwards
`additional_callbacks`, wiring the unlearning-phase half of this in is a
one-line change per strategy; if it doesn't forward callbacks, `base.py`
needs a small update first. Worth checking `base.py` before relying on the
"loss during unlearn" plot.
"""
from typing import Dict, List, Optional
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt


class LossHistoryCallback(Callback):
    """Records every epoch-level scalar Lightning logs (`on_epoch=True`), per epoch."""

    def __init__(self, tracked_keys: Optional[List[str]] = None) -> None:
        """Initialize the history buffer; if `tracked_keys` is None, track everything logged."""
        super().__init__()
        self.tracked_keys = tracked_keys
        self.history: Dict[str, List[float]] = {}

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Snapshot the current epoch's logged metrics into `self.history`."""
        metrics = trainer.callback_metrics
        keys = self.tracked_keys or [k for k in metrics.keys() if not k.startswith("lr-")]
        for k in keys:
            if k in metrics:
                v = metrics[k]
                v = float(v.detach().cpu()) if hasattr(v, "detach") else float(v)
                self.history.setdefault(k, []).append(v)


def plot_loss_curves(
    history: Dict[str, List[float]],
    keys: List[str],
    title: str,
    save_path: str,
) -> None:
    """Plot one or more tracked metrics (e.g. ["train/loss", "val/loss"]) and save to disk."""
    plt.figure(figsize=(7, 5))
    for k in keys:
        if k in history and len(history[k]) > 0:
            plt.plot(range(1, len(history[k]) + 1), history[k], marker="o", label=k, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_train_then_unlearn(
    train_history: Dict[str, List[float]],
    unlearn_history: Dict[str, List[float]],
    train_key: str,
    unlearn_key: str,
    title: str,
    save_path: str,
) -> None:
    """Concatenate a base-training loss curve and an unlearning loss curve on one
    timeline, with a dashed line marking the handoff between the two phases."""
    train_vals = train_history.get(train_key, [])
    unlearn_vals = unlearn_history.get(unlearn_key, [])

    plt.figure(figsize=(8, 5))
    x_train = range(1, len(train_vals) + 1)
    x_unlearn = range(len(train_vals) + 1, len(train_vals) + len(unlearn_vals) + 1)
    plt.plot(x_train, train_vals, marker="o", label=f"train ({train_key})", color="tab:blue")
    plt.plot(x_unlearn, unlearn_vals, marker="o", label=f"unlearn ({unlearn_key})", color="tab:red")
    if train_vals:
        plt.axvline(len(train_vals) + 0.5, linestyle="--", color="gray", alpha=0.7, label="unlearning starts")
    plt.xlabel("Epoch (concatenated)")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

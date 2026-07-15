"""Unlearning + post-unlearning continuation on the held-out set Dh.

Drop-in replacement for run3_unlearning.py. Runs the SAME unlearning strategy
via the same `run_mul_strategy`, evaluates it (logged under the `unlearning/`
prefix, byte-for-byte comparable with run3's output), then -- for every method
EXCEPT ReGUn -- continues training the unlearned model on Dh and evaluates
again (logged under `unlearning_plus_heldout/`).

Rationale: ReGUn is the only strategy that touches Dh during unlearning (via
`heldout_eval_dataloader()`). At forget_frac=0.01, |Dh|/|Df| ~ 11x, so any
"ReGUn > X" gap partly reflects a data budget it alone has. Giving the other
methods a Dh phase equalizes the budget and isolates the algorithmic effect.

Both evaluations land in ONE wandb run, so `read_wandb_run(...)` returns both
prefixes from a single summary.

Usage (identical CLI to run3):
    python run4_unlearning_heldout.py unlearn=finetune seed=42 split.forget_frac=0.01 ...

Dh phase config: optional block `heldout_finetune` in conf/unlearn/<method>.yaml:
    heldout_finetune:
      enabled: true
      epochs: 5
      optim: {...}       # optional; defaults to the method's own cfg.unlearn.optim
      scheduler: {...}   # optional; defaults to the method's own cfg.unlearn.scheduler
If the block is absent, the defaults below apply (enabled, epochs=5, method's
own optim/scheduler), so no YAML edit is strictly required.

Loss curves: `LossHistoryCallback` is injected into every trainer built by
`UnlearningStrategy.new_trainer()` via monkeypatch -- base.py already forwards
`additional_callbacks`, but strategies call `new_trainer()` with no arguments,
so patching is the only way in without editing each strategy file.
"""

import os
import uuid
from typing import Any, Optional

import hydra
import torch
import lightning.pytorch as pl
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from data import build_datamodule
from models import load_model
from mul import MULEvaluator, run_mul_strategy
from mul.base import UnlearningStrategy
from utils.utils import build_optimizer
from utils import (
    seed_everything,
    build_trainer,
    wandb_init,
    wandb_log,
    wandb_finish,
)

# --- optional loss-curve capture -------------------------------------------------
try:
    from utils.loss_history import LossHistoryCallback, plot_loss_curves
    _HAS_LOSS_HISTORY = True
except ImportError:
    _HAS_LOSS_HISTORY = False

# Fixed default for the Dh phase. Deliberately NOT inherited from
# cfg.unlearn.optim: ssd.yaml has `optim.name: none` and no scheduler block at
# all, so inheriting would crash SSD -- and more importantly, a method-
# independent Dh budget is what makes the "everyone gets Dh" comparison fair.
# Override per-method via conf/unlearn/<m>.yaml or CLI:
#   +unlearn.heldout_finetune.epochs=3 +unlearn.heldout_finetune.optim.lr=0.005
_DEFAULT_HELDOUT = OmegaConf.create({
    "enabled": True,
    "epochs": 5,
    "optim": {
        "name": "sgd",
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "nesterov": False,
        "betas": [0.9, 0.999],
        "exclude_wd_norm_bias": True,
    },
    "scheduler": {"name": "none"},
})


class _HeldoutFinetuneModule(pl.LightningModule):
    """Plain supervised fine-tuning on Dh (cross-entropy, no distillation)."""

    def __init__(self, model: pl.LightningModule, optim_cfg: Any, scheduler_cfg: Any) -> None:
        """Initialize the held-out fine-tuning module."""
        super().__init__()
        self.model = model
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
        self.loss_fn = nn.CrossEntropyLoss()
        num_classes = self.model.num_classes
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        """Run a forward pass through the wrapped model."""
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        """Run a shared heldout/val step."""
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        metric = {"heldout": self.train_acc, "val": self.val_acc}[stage]
        metric(preds, y)
        self.log(f"{stage}/acc", metric, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        return loss

    def training_step(self, batch, batch_idx):
        """Run one training step on Dh."""
        del batch_idx
        return self._shared_step(batch, stage="heldout")

    def validation_step(self, batch, batch_idx):
        """Run one validation step."""
        del batch_idx
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        """Configure the optimizer and scheduler."""
        return build_optimizer(self.model.parameters(), self.optim_cfg, self.scheduler_cfg)


def _save_state_dict(cfg: DictConfig, model: pl.LightningModule, tag: str) -> str:
    """Persist an unlearned model's state_dict so the notebook can reload it later.

    run3/run4 run `enable_checkpointing: false`, so without this the unlearned
    model dies with the subprocess and the representation-flip analysis has
    nothing to compare against the retrain model.
    """
    out_dir = os.path.join(str(cfg.paths.cache_dir), "models")
    os.makedirs(out_dir, exist_ok=True)
    fname = (
        f"unlearned_{tag}_{cfg.model.model.name}_{cfg.data.name}_"
        f"{cfg.split.scheme}_forget{cfg.split.forget_frac}_{cfg.seed}.pt"
    )
    path = os.path.join(out_dir, fname)
    torch.save(model.state_dict(), path)
    print(f"[MAIN] Saved state_dict -> {path}")
    return path


def _heldout_train_loader(dm: Any) -> DataLoader:
    """Return the Dh loader to fine-tune on.

    Uses `heldout_dataloader()` (train_tf + shuffle=True), NOT
    `heldout_eval_dataloader()`: the latter wraps `ds_val`, i.e. eval_tf, so
    training on it would silently drop the augmentation every other training
    phase in this codebase uses. It is also shuffle=True despite the `_eval_`
    name, which is a separate inconsistency in cifar.py worth being aware of.

    Swap for `dm.combined_retain_heldout_dataloader()` to continue on Dr U Dh
    instead of Dh alone.
    """
    return dm.heldout_dataloader()


def _heldout_settings(cfg: DictConfig):
    """Resolve (enabled, epochs, optim_cfg, scheduler_cfg) for the Dh phase."""
    user = OmegaConf.select(cfg, "unlearn.heldout_finetune")
    ho = _DEFAULT_HELDOUT if user is None else OmegaConf.merge(_DEFAULT_HELDOUT, user)
    return bool(ho.enabled), int(ho.epochs), ho.optim, ho.scheduler


def apply_heldout_finetune(
    cfg: DictConfig,
    model: pl.LightningModule,
    dm: Any,
    logger: Optional[Any] = None,
    additional_callbacks: Optional[list] = None,
) -> pl.LightningModule:
    """Continue training `model` on Dh after its unlearning phase has finished."""
    _, epochs, optim_cfg, sched_cfg = _heldout_settings(cfg)

    heldout_loader = _heldout_train_loader(dm)
    module = _HeldoutFinetuneModule(model, optim_cfg, sched_cfg)

    trainer_cfg = OmegaConf.merge(cfg, {"trainer": {"max_epochs": epochs}})
    trainer = build_trainer(
        trainer_cfg,
        job_type="heldout-finetune",
        logger=logger,
        additional_callbacks=additional_callbacks or [],
    )
    trainer.fit(module, train_dataloaders=heldout_loader, val_dataloaders=dm.val_dataloader())
    return module.model


def _patch_new_trainer_for_loss_history():
    """Inject a LossHistoryCallback into every strategy trainer; return the callback."""
    if not _HAS_LOSS_HISTORY:
        return None

    loss_cb = LossHistoryCallback()
    original = UnlearningStrategy.new_trainer

    def patched(self, additional_callbacks=None):
        """new_trainer with the loss-history callback always attached."""
        cbs = list(additional_callbacks or []) + [loss_cb]
        return original(self, additional_callbacks=cbs)

    UnlearningStrategy.new_trainer = patched
    return loss_cb


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"[MAIN] Config:\n{OmegaConf.to_yaml(cfg)}")
    seed_everything(cfg.seed)
    cfg.run.experiment_id = cfg.run.experiment_id or str(uuid.uuid4().hex)[:8]

    method = str(cfg.unlearn.name)
    dm = build_datamodule(cfg)
    model_base = load_model(cfg, cfg.run.base_weights)
    evaluator = MULEvaluator(cfg=cfg, datamodule=dm)

    unlearn_loss_cb = _patch_new_trainer_for_loss_history()

    try:
        logger = wandb_init(cfg, job_type="unlearning")

        print(f"[MAIN] --- Starting Unlearning ({method}, Group: {cfg.run.experiment_id}) ---")
        model_unlearned = run_mul_strategy(cfg, model_base, dm, logger, evaluator=evaluator)

        print("[MAIN] --- Evaluation (no Dh continuation) ---")
        results = evaluator.run(model_unlearned)
        print("[MAIN] Results:", results)
        wandb_log(results, prefix="unlearning", summary=True)
        _save_state_dict(cfg, model_unlearned, method)

        enabled, epochs, _, _ = _heldout_settings(cfg)
        if method == "regun":
            print("[MAIN] --- Skipping Dh continuation: ReGUn already consumes Dh during unlearning. ---")
        elif not enabled:
            print(f"[MAIN] --- Skipping Dh continuation: disabled for '{method}'. ---")
        else:
            print(f"[MAIN] --- Continuing training on Dh ({epochs} epochs, method={method}) ---")
            heldout_loss_cb = LossHistoryCallback() if _HAS_LOSS_HISTORY else None
            model_plus_dh = apply_heldout_finetune(
                cfg,
                model_unlearned,
                dm,
                logger=logger,
                additional_callbacks=[heldout_loss_cb] if heldout_loss_cb else None,
            )

            print("[MAIN] --- Evaluation (after Dh continuation) ---")
            results_plus_dh = evaluator.run(model_plus_dh)
            print("[MAIN] Results (+Dh):", results_plus_dh)
            wandb_log(results_plus_dh, prefix="unlearning_plus_heldout", summary=True)
            _save_state_dict(cfg, model_plus_dh, f"{method}_plus_dh")

            if heldout_loss_cb is not None:
                plot_loss_curves(
                    heldout_loss_cb.history,
                    keys=["heldout/loss", "val/loss"],
                    title=f"Dh continuation loss ({method})",
                    save_path=f"{cfg.paths.outputs_dir}/loss_heldout_{method}.png",
                )

        if unlearn_loss_cb is not None and unlearn_loss_cb.history:
            plot_loss_curves(
                unlearn_loss_cb.history,
                keys=list(unlearn_loss_cb.history.keys()),
                title=f"Unlearning-phase loss ({method})",
                save_path=f"{cfg.paths.outputs_dir}/loss_unlearn_{method}.png",
            )
    finally:
        wandb_finish()


if __name__ == "__main__":
    main()

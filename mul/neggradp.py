from typing import Any, Dict, Optional, Tuple
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from utils.utils import build_optimizer
from .base import UnlearningStrategy


class _NegGradPlusModule(pl.LightningModule):
    """Single-module implementation of NegGrad+ with sequential or joint updates."""

    def __init__(
        self,
        model: pl.LightningModule,
        cfg: Dict[str, Any],
    ) -> None:
        """Initialize the NegGrad+ module."""
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()

        num_classes = self.model.num_classes
        self.forget_acc = MulticlassAccuracy(num_classes=num_classes)
        self.retain_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        self.sequential = bool(getattr(self.cfg, "sequential", True))
        self.neg_cfg = self.cfg.neggrad
        self.fin_cfg = self.cfg.finetune

        if self.sequential:
            self._forget_epochs = int(self.neg_cfg.epochs)
            self._retain_epochs = int(self.fin_cfg.epochs)
        else:
            self.retain_weight = float(self.cfg.retain_weight)
            self.forget_weight = float(self.cfg.forget_weight)

        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the wrapped model."""
        return self.model(x)

    def _stage_for_epoch(self, epoch: int) -> str:
        """Resolve the training stage for a given epoch."""
        if not self.sequential:
            return "joint"
        return "forget" if epoch < self._forget_epochs else "retain"

    def _current_stage(self) -> str:
        """Return the current training stage."""
        return self._stage_for_epoch(self.current_epoch)

    def _build_optimizer(self, optim_cfg: Any, scheduler_cfg: Any) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Build an optimizer (and optional scheduler) from config."""
        built = build_optimizer(self.model.parameters(), optim_cfg, scheduler_cfg)
        if isinstance(built, dict):
            optimizer = built["optimizer"]
            scheduler = built["lr_scheduler"]["scheduler"]
        else:
            optimizer = built
            scheduler = None
        return optimizer, scheduler

    def configure_optimizers(self) -> Any:
        """Configure optimizers for sequential or joint training."""
        if self.sequential:
            forget_opt, forget_sched = self._build_optimizer(self.neg_cfg.optim, self.neg_cfg.scheduler)
            retain_opt, retain_sched = self._build_optimizer(self.fin_cfg.optim, self.fin_cfg.scheduler)
            self._forget_scheduler = forget_sched
            self._retain_scheduler = retain_sched
            return [forget_opt, retain_opt]

        joint_opt, joint_sched = self._build_optimizer(self.cfg.optim, self.cfg.scheduler)
        self._joint_scheduler = joint_sched
        return joint_opt

    def _shared_eval_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Run a shared validation/test step."""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)

        acc_metric = {
            "val": self.val_acc,
            "test": self.test_acc,
        }[stage]

        acc_metric(preds, y)
        self.log(f"{stage}/acc", acc_metric, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        return loss

    def training_step(self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """Run one training step for sequential or joint mode."""
        del batch_idx
        stage = self._current_stage()
        optimizers = self.optimizers()

        if self.sequential:
            if stage == "forget":
                x_forget, y_forget = batch["forget"]
                x_forget, y_forget = x_forget.to(self.device), y_forget.to(self.device)
                logits = self.model(x_forget)

                ce_loss = self.loss_fn(logits, y_forget)
                preds = logits.argmax(dim=1)

                self.forget_acc(preds, y_forget)
                self.log("forget/acc", self.forget_acc, on_epoch=True, on_step=False, prog_bar=True)
                self.log("forget/ce_loss", ce_loss, on_epoch=True, on_step=False, prog_bar=True)

                ascent_loss = -ce_loss
                optimizers[0].zero_grad()
                self.manual_backward(ascent_loss)
                optimizers[0].step()

                return ascent_loss

            x_retain, y_retain = batch["retain"]
            x_retain, y_retain = x_retain.to(self.device), y_retain.to(self.device)
            logits = self.model(x_retain)

            retain_loss = self.loss_fn(logits, y_retain)
            preds = logits.argmax(dim=1)

            self.retain_acc(preds, y_retain)
            self.log("retain/acc", self.retain_acc, on_epoch=True, on_step=False, prog_bar=True)
            self.log("retain/ce_loss", retain_loss, on_epoch=True, on_step=False, prog_bar=True)

            optimizers[1].zero_grad()
            self.manual_backward(retain_loss)
            optimizers[1].step()

            return retain_loss

        # joint mode
        x_forget, y_forget = batch["forget"]
        x_retain, y_retain = batch["retain"]
        x_forget, y_forget = x_forget.to(self.device), y_forget.to(self.device)
        x_retain, y_retain = x_retain.to(self.device), y_retain.to(self.device)

        logits_forget = self.model(x_forget)
        logits_retain = self.model(x_retain)

        forget_loss = self.loss_fn(logits_forget, y_forget)
        retain_loss = self.loss_fn(logits_retain, y_retain)

        preds_forget = logits_forget.argmax(dim=1)
        preds_retain = logits_retain.argmax(dim=1)

        self.forget_acc(preds_forget, y_forget)
        self.retain_acc(preds_retain, y_retain)

        self.log("joint/forget_acc", self.forget_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log("joint/retain_acc", self.retain_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log("joint/forget_ce", forget_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("joint/retain_ce", retain_loss, on_epoch=True, on_step=False, prog_bar=True)

        total_loss = self.forget_weight * (-forget_loss) + self.retain_weight * retain_loss

        optimizer = optimizers if isinstance(optimizers, torch.optim.Optimizer) else optimizers[0]
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        optimizer.step()

        self.log("joint/total_loss", total_loss, on_epoch=True, on_step=False, prog_bar=False)
        return total_loss

    def on_train_epoch_end(self) -> None:
        """Step the appropriate scheduler after each epoch."""
        if self.sequential:
            stage = self._stage_for_epoch(self.current_epoch)
            if stage == "forget" and getattr(self, "_forget_scheduler", None) is not None:
                self._forget_scheduler.step()
            if stage == "retain" and getattr(self, "_retain_scheduler", None) is not None:
                self._retain_scheduler.step()
        else:
            if getattr(self, "_joint_scheduler", None) is not None:
                self._joint_scheduler.step()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one validation step."""
        del batch_idx
        return self._shared_eval_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one test step."""
        del batch_idx
        return self._shared_eval_step(batch, "test")


class NegGradPlus(UnlearningStrategy):
    """NegGrad+ with optional sequential or joint forgetting/fine-tuning."""

    def run(self, base_model: pl.LightningModule) -> pl.LightningModule:
        """Execute NegGrad+ and return the updated model."""

        neggradp_module = _NegGradPlusModule(
            base_model,
            self.cfg.unlearn,
        )

        if neggradp_module.sequential:
            total_epochs = neggradp_module._forget_epochs + neggradp_module._retain_epochs
            self.cfg.unlearn.trainer.max_epochs = total_epochs

        combined_train = CombinedLoader(
            {"forget": self.dm.forget_dataloader(), "retain": self.dm.retain_dataloader()},
            mode="min_size",
        )

        trainer = self.new_trainer()
        trainer.fit(
            neggradp_module,
            train_dataloaders=combined_train,
            val_dataloaders=self.dm.val_dataloader(),
        )
        return neggradp_module.model

from typing import Any, Dict, Tuple
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy

from utils.utils import build_optimizer
from .base import UnlearningStrategy


class _L1SparseModule(pl.LightningModule):
    """Fine-tune on retain data with an L1 sparsity penalty."""

    def __init__(self, model: pl.LightningModule, cfg: Dict[str, Any]) -> None:
        """Initialize the L1-sparse fine-tuning module."""
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.l1_cfg = cfg.l1

        self.gamma = float(self.l1_cfg.gamma)
        self.schedule = str(self.l1_cfg.schedule).lower()
        self._total_epochs = int(cfg.trainer.max_epochs)

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=self.model.num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=self.model.num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=self.model.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the wrapped model."""
        return self.model(x)

    def _gamma_at_epoch(self, epoch: int) -> float:
        """Return the L1 coefficient for a given epoch."""
        denom = max(self._total_epochs, 1)
        frac = float(epoch) / float(denom)
        if self.schedule == "constant":
            return self.gamma
        elif self.schedule == "linear_grow":
            return self.gamma * (2.0 * frac)
        elif self.schedule == "linear_decay":
            return self.gamma * (2.0 - 2.0 * frac)

    def _l1_penalty(self) -> torch.Tensor:
        """Compute the L1 penalty over trainable parameters."""
        penalty = torch.zeros((), device=self.device)
        for param in self.model.parameters():
            if param.requires_grad:
                penalty = penalty + param.abs().sum()
        return penalty

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Run a shared train/val/test step."""
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)

        metric = {
            "retain": self.train_acc,
            "val": self.val_acc,
            "test": self.test_acc,
        }[stage]
        metric(preds, y)

        self.log(f"{stage}/acc", metric, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one training step with L1 penalty."""
        del batch_idx
        ce_loss = self._shared_step(batch, stage="retain")
        l1_penalty = self._l1_penalty()
        gamma_t = self._gamma_at_epoch(self.current_epoch + 1)
        total_loss = ce_loss + gamma_t * l1_penalty

        self.log("train/l1_penalty", l1_penalty, on_epoch=True, on_step=False, prog_bar=False)
        self.log("train/gamma", gamma_t, on_epoch=True, on_step=False, prog_bar=False)
        self.log("train/loss", total_loss, on_epoch=True, on_step=False, prog_bar=False)
        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one validation step."""
        del batch_idx
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one test step."""
        del batch_idx
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and scheduler."""
        return build_optimizer(self.model.parameters(), self.cfg.optim, self.cfg.scheduler)


class L1Sparse(UnlearningStrategy):
    """L1-sparsity-aware unlearning via fine-tuning on retain data."""

    def run(self, base_model: pl.LightningModule) -> pl.LightningModule:
        """Execute L1-sparse fine-tuning and return the updated model."""
        l1_module = _L1SparseModule(base_model, self.cfg.unlearn)
        trainer = self.new_trainer()

        trainer.fit(
            l1_module,
            train_dataloaders=self.dm.retain_dataloader_subset(),
            val_dataloaders=self.dm.val_dataloader(),
        )

        return l1_module.model

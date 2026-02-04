from typing import Any, Dict, Tuple
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy

from utils.utils import build_optimizer
from .base import UnlearningStrategy


class _FinetuneModule(pl.LightningModule):
    """Lightweight LightningModule that fine-tunes on retain data."""

    def __init__(self, model: pl.LightningModule, cfg: Dict[str, Any]) -> None:
        """Initialize the fine-tuning module."""
        super().__init__()
        self.model = model
        self.cfg = cfg

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=self.model.num_classes)
        self.val_acc   = MulticlassAccuracy(num_classes=self.model.num_classes)
        self.test_acc  = MulticlassAccuracy(num_classes=self.model.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the wrapped model."""
        return self.model(x)

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
        """Run one training step on retain data."""
        del batch_idx
        return self._shared_step(batch, stage="retain")

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


class Finetune(UnlearningStrategy):
    """Fine-tune the base model on the retain split."""

    def run(self, base_model: pl.LightningModule) -> pl.LightningModule:
        """Execute fine-tuning and return the updated model."""

        finetune_module = _FinetuneModule(base_model, self.cfg.unlearn)
        trainer = self.new_trainer()

        trainer.fit(
            finetune_module,
            train_dataloaders=self.dm.retain_dataloader_subset(),
            val_dataloaders=self.dm.val_dataloader(),
        )

        return finetune_module.model

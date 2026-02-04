from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader

from utils.utils import build_optimizer
from .base import UnlearningStrategy


class _DummyLoader(DataLoader):
    """One-step loader so Lightning triggers `training_step` once per epoch."""

    def __init__(self) -> None:
        """Initialize a single-batch loader."""
        super().__init__(torch.zeros(1), batch_size=1, shuffle=False)


class _SCRUBModule(pl.LightningModule):
    """Teacher-student SCRUB implementation with alternating forget/retain phases."""

    def __init__(
        self,
        model: pl.LightningModule,
        teacher: pl.LightningModule,
        cfg: Dict[str, Any],
        forget_loader: DataLoader,
        retain_loader: DataLoader,
    ) -> None:
        """Initialize the SCRUB training module."""
        super().__init__()
        self.model = model
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.cfg = cfg
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader

        self.alpha = float(cfg.alpha)
        self.gamma = float(getattr(cfg, "gamma", 1.0))
        self.temperature = float(getattr(cfg, "temperature", 1.0))
        self._forget_epochs: Optional[int] = None

        num_classes = self.model.num_classes
        self.forget_acc = MulticlassAccuracy(num_classes=num_classes)
        self.retain_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        self.ce_loss = nn.CrossEntropyLoss()
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the student model."""
        return self.model(x)

    def on_fit_start(self) -> None:
        """Compute the number of forget epochs from trainer config."""
        self.teacher.to(self.device)
        max_epochs = None
        if self.trainer is not None and self.trainer.max_epochs is not None:
            max_epochs = int(self.trainer.max_epochs)
        if max_epochs is None:
            max_epochs = int(getattr(getattr(self.cfg, "trainer", None), "max_epochs", 1) or 1)
        self._forget_epochs = max(1, max_epochs // 2)

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and scheduler."""
        built = build_optimizer(self.model.parameters(), self.cfg.optim, self.cfg.scheduler)
        return built

    def _kl_distance(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute temperature-scaled KL divergence between logits."""
        temperature = self.temperature
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)

    def _run_forget_epoch(self, optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        """Run a full forget phase over the forget loader."""
        total_kl = total_acc = torch.zeros(1, device=self.device)
        steps = 0
        for x, y in self.forget_loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad(set_to_none=True)

            logits_student = self.model(x)
            with torch.no_grad():
                logits_teacher = self.teacher(x)

            kl = self._kl_distance(logits_student, logits_teacher)
            loss = -kl
            self.manual_backward(loss)
            optimizer.step()

            preds = logits_student.argmax(dim=1)
            acc = self.forget_acc(preds, y)

            total_kl += kl.detach()
            total_acc += acc.detach()
            steps += 1

        if steps == 0:
            return {}
        return {
            "forget/kl": total_kl / steps,
            "forget/acc": total_acc / steps,
        }

    def _run_retain_epoch(self, optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        """Run a full retain phase over the retain loader."""
        total_loss = total_kl = total_ce = total_acc = torch.zeros(1, device=self.device)
        steps = 0
        for x, y in self.retain_loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad(set_to_none=True)

            logits_student = self.model(x)
            with torch.no_grad():
                logits_teacher = self.teacher(x)

            kl = self._kl_distance(logits_student, logits_teacher)
            ce = self.ce_loss(logits_student, y)
            loss = self.alpha * kl + self.gamma * ce

            self.manual_backward(loss)
            optimizer.step()

            preds = logits_student.argmax(dim=1)
            acc = self.retain_acc(preds, y)

            total_loss += loss.detach()
            total_kl += kl.detach()
            total_ce += ce.detach()
            total_acc += acc.detach()
            steps += 1

        if steps == 0:
            return {}
        inv = 1.0 / steps
        return {
            "retain/loss": total_loss * inv,
            "retain/kl": total_kl * inv,
            "retain/ce": total_ce * inv,
            "retain/acc": total_acc * inv,
        }

    def _should_run_forget(self) -> bool:
        """Return True if the current epoch includes a forget phase."""
        if self._forget_epochs is None:
            return True
        return self.current_epoch < self._forget_epochs

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Run a full SCRUB cycle within a single Lightning step."""
        del batch, batch_idx
        optimizer = self.optimizers()

        stats = {}
        if self._should_run_forget():
            stats.update(self._run_forget_epoch(optimizer))

        # Always run retain to keep accuracy on retained data.
        stats.update(self._run_retain_epoch(optimizer))

        for key, value in stats.items():
            self.log(key, value, on_epoch=True, on_step=False, prog_bar=("retain/loss" in key or "forget/kl" in key))

        return torch.zeros(1, device=self.device)

    def on_train_epoch_end(self) -> None:
        """Step any attached schedulers after each epoch."""
        try:
            schedulers = self.lr_schedulers()
        except Exception:
            schedulers = None
        if schedulers is None:
            return
        if isinstance(schedulers, (list, tuple)):
            for scheduler in schedulers:
                scheduler.step()
        else:
            schedulers.step()

    def _shared_eval_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Run a shared validation/test step."""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        loss = self.ce_loss(logits, y)
        preds = logits.argmax(dim=1)

        metric = {"val": self.val_acc, "test": self.test_acc}[stage]
        metric(preds, y)
        self.log(f"{stage}/acc", metric, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one validation step."""
        del batch_idx
        return self._shared_eval_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one test step."""
        del batch_idx
        return self._shared_eval_step(batch, "test")



class SCRUB(UnlearningStrategy):
    """Lightning wrapper around the SCRUB teacher-student alternating procedure."""

    def run(self, base_model: pl.LightningModule) -> pl.LightningModule:
        """Execute SCRUB and return the updated model."""
        scrub_cfg = self.cfg.unlearn
        forget_loader = self.dm.forget_dataloader()
        retain_loader = self.dm.retain_dataloader()

        teacher = deepcopy(base_model)

        scrub_module = _SCRUBModule(
            model=base_model,
            teacher=teacher,
            cfg=scrub_cfg,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
        )

        trainer = self.new_trainer()
        trainer.fit(
            scrub_module,
            train_dataloaders=_DummyLoader(),
            val_dataloaders=self.dm.val_dataloader(),
        )

        return scrub_module.model

from typing import Any, Dict, Iterable, Tuple
import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torchmetrics.classification import MulticlassAccuracy

from utils.utils import build_optimizer
from .base import UnlearningStrategy


class _SalUnModule(pl.LightningModule):
    """Saliency-guided random-label unlearning (SalUn)."""

    def __init__(
        self,
        model: pl.LightningModule,
        cfg: Dict[str, Any],
        saliency_mask: Dict[str, torch.Tensor],
        original_state: Dict[str, torch.Tensor],
    ) -> None:
        """Initialize the SalUn training module."""
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.saliency_mask = saliency_mask
        self.original_state = original_state

        self.retain_weight = float(cfg.retain_weight)
        self.forget_weight = float(cfg.forget_weight)

        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = int(self.model.num_classes)
        self.forget_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.retain_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the wrapped model."""
        return self.model(x)

    def on_fit_start(self) -> None:
        """Move saliency buffers to the training device."""
        for name, param in self.model.named_parameters():
            if name in self.saliency_mask:
                self.saliency_mask[name] = self.saliency_mask[name].to(device=param.device, dtype=param.dtype)
                self.original_state[name] = self.original_state[name].to(device=param.device, dtype=param.dtype)

    def _random_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Generate random labels different from the originals."""
        offsets = torch.randint(1, self.num_classes, y.shape, device=y.device)
        return (y + offsets) % self.num_classes

    def _shared_eval_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Run a shared validation/test step."""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)

        metric = {"val": self.val_acc, "test": self.test_acc}[stage]
        metric(preds, y)
        self.log(f"{stage}/acc", metric, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        return loss

    def training_step(self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """Run one SalUn training step with random labels."""
        del batch_idx
        x_forget, y_forget = batch["forget"]
        x_retain, y_retain = batch["retain"]
        x_forget, y_forget = x_forget.to(self.device), y_forget.to(self.device)
        x_retain, y_retain = x_retain.to(self.device), y_retain.to(self.device)

        logits_forget = self.model(x_forget)
        rand_y = self._random_labels(y_forget)
        forget_loss = self.loss_fn(logits_forget, rand_y)

        preds_forget = logits_forget.argmax(dim=1)
        self.forget_acc(preds_forget, y_forget)
        self.log("forget/acc", self.forget_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log("forget/rl_loss", forget_loss, on_epoch=True, on_step=False, prog_bar=True)

        logits_retain = self.model(x_retain)
        retain_loss = self.loss_fn(logits_retain, y_retain)

        preds_retain = logits_retain.argmax(dim=1)
        self.retain_acc(preds_retain, y_retain)
        self.log("retain/acc", self.retain_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log("retain/ce_loss", retain_loss, on_epoch=True, on_step=False, prog_bar=True)

        total = self.forget_weight * forget_loss + self.retain_weight * retain_loss
        self.log("train/loss", total, on_epoch=True, on_step=False, prog_bar=False)
        return total

    def on_after_backward(self) -> None:
        """Mask gradients to keep only salient parameters."""
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            mask = self.saliency_mask.get(name)
            if mask is not None:
                param.grad.mul_(mask)

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Restore masked parameters after each optimizer step."""
        del outputs, batch, batch_idx, args, kwargs
        for name, param in self.model.named_parameters():
            mask = self.saliency_mask.get(name)
            if mask is None:
                continue
            param.data.mul_(mask).add_(self.original_state[name] * (1.0 - mask))

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one validation step."""
        del batch_idx
        return self._shared_eval_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one test step."""
        del batch_idx
        return self._shared_eval_step(batch, "test")

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and scheduler."""
        return build_optimizer(self.model.parameters(), self.cfg.optim, self.cfg.scheduler)


class SalUn(UnlearningStrategy):
    """Saliency unlearning with random-label updates on forget data."""

    def _saliency_threshold(self, grad_accum: Iterable[torch.Tensor], quantile: float) -> torch.Tensor:
        """Compute a saliency threshold from accumulated gradients."""
        total = sum(g.numel() for g in grad_accum)
        max_exact = int(getattr(self.cfg.unlearn, "saliency_max_exact_params", 16_000_000))
        sample_size = int(getattr(self.cfg.unlearn, "saliency_sample_size", 16_000_000))

        if total <= max_exact:
            all_abs = torch.cat([g.abs().view(-1) for g in grad_accum])
            return torch.quantile(all_abs, quantile)

        # Sample gradients to estimate the threshold when the model is large.
        samples = []
        for g in grad_accum:
            n = g.numel()
            if n == 0:
                continue
            k = max(1, int(round(sample_size * n / total)))
            k = min(k, n)
            idx = torch.randint(0, n, (k,), device=g.device)
            samples.append(g.abs().view(-1)[idx])
        sample = torch.cat(samples)
        return torch.quantile(sample, quantile)

    def run(self, base_model: pl.LightningModule) -> pl.LightningModule:
        """Execute SalUn and return the updated model."""

        saliency_mask = self._compute_saliency_mask(base_model)
        original_state = {
            name: param.detach().clone()
            for name, param in base_model.named_parameters()
            if name in saliency_mask
        }

        salun_module = _SalUnModule(base_model, self.cfg.unlearn, saliency_mask, original_state)
        combined_train = CombinedLoader(
            {"forget": self.dm.forget_dataloader(), "retain": self.dm.retain_dataloader()},
            mode="min_size",
        )
        trainer = self.new_trainer()
        trainer.fit(
            salun_module,
            train_dataloaders=combined_train,
            val_dataloaders=self.dm.val_dataloader(),
        )
        return salun_module.model

    def _compute_saliency_mask(self, model: pl.LightningModule) -> Dict[str, torch.Tensor]:
        """Compute the saliency mask using forget data gradients."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        loader = self.dm.forget_eval_dataloader()

        params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not params:
            raise ValueError("SalUn requires trainable parameters to compute saliency.")

        grad_accum = [torch.zeros_like(p, device=p.device) for _, p in params]
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        total = 0
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            grads = torch.autograd.grad(
                loss,
                [p for _, p in params],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            for acc, grad in zip(grad_accum, grads):
                if grad is not None:
                    acc.add_(grad.detach())
            total += y.size(0)

        if was_training:
            model.train()

        if total == 0:
            raise ValueError("SalUn requires a non-empty forget set to compute saliency.")

        inv_total = 1.0 / float(total)
        for acc in grad_accum:
            acc.mul_(inv_total)

        quantile = 1.0 - self.cfg.unlearn.saliency_keep_ratio
        gamma = self._saliency_threshold(grad_accum, quantile)

        saliency_mask = {
            name: (grad.abs() >= gamma).float()
            for (name, _), grad in zip(params, grad_accum)
        }
        return saliency_mask

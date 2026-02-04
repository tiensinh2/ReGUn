from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torchmetrics.classification import MulticlassAccuracy
try:
    from torch.func import functional_call as _functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call as _functional_call

from utils.utils import build_optimizer
from .base import UnlearningStrategy


class _LURModule(pl.LightningModule):
    """LUR (Learning to Unlearn while Retaining) with SNIP-style pruning at init."""

    def __init__(
        self,
        model: pl.LightningModule,
        cfg: Dict[str, Any],
        full_cfg: Dict[str, Any],
        forget_loader: Iterable,
    ) -> None:
        """Initialize the LUR training module."""
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.full_cfg = full_cfg
        self.forget_loader = forget_loader

        self.alpha = float(cfg.alpha)
        self.retain_weight = float(cfg.retain_weight)
        self.forget_weight = float(cfg.forget_weight)

        num_classes = self.model.num_classes
        self.forget_acc = MulticlassAccuracy(num_classes=num_classes)
        self.retain_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the wrapped model."""
        return self.model(x)

    def on_fit_start(self) -> None:
        """Apply optional SNIP-style pruning at fit start."""
        pruning_cfg = getattr(self.cfg, "pruning", None)
        if pruning_cfg is None or not bool(getattr(pruning_cfg, "enabled", True)):
            return
        self._apply_snip_pruning(pruning_cfg)

    def _resolve_pruning_sparsity(self, pruning_cfg: Any) -> float:
        """Resolve pruning sparsity from config and dataset settings."""
        if getattr(pruning_cfg, "sparsity", None) is not None:
            return float(pruning_cfg.sparsity)
        if not bool(getattr(pruning_cfg, "auto", False)):
            raise ValueError("LUR pruning requires `pruning.sparsity` or `pruning.auto=true`.")

        data_name = str(self.full_cfg.data.name).lower()
        forget_frac = float(self.full_cfg.split.forget_frac)

        # params from LUR paper and for tiny_imagenet interpolated
        if data_name == "cifar10":
            return 0.97
        if data_name == "cifar100":
            if forget_frac <= 0.1 + 1e-6:
                return 0.99
            if forget_frac >= 0.5 - 1e-6:
                return 0.90
        if data_name == "tiny_imagenet":
            if forget_frac <= 0.1 + 1e-6:
                return 0.95
            if forget_frac >= 0.5 - 1e-6:
                return 0.85

    def _resolve_layer_patterns(self, pruning_cfg: Any) -> Optional[List[str]]:
        """Resolve layer name filters for pruning."""
        patterns = getattr(pruning_cfg, "layer_patterns", None)
        if patterns is not None:
            return [str(p) for p in list(patterns)]

        if not bool(getattr(pruning_cfg, "auto", False)):
            return None

        data_name = str(self.full_cfg.data.name).lower()
        if data_name == "cifar10":
            return ["layer3", "layer4", "fc"]
        if data_name == "cifar100":
            return ["layer4", "fc"]
        return None

    def _apply_snip_pruning(self, pruning_cfg: Any) -> None:
        """Apply SNIP-style pruning and reinitialize pruned weights."""
        sparsity = self._resolve_pruning_sparsity(pruning_cfg)
        prune_num = int(getattr(pruning_cfg, "prune_num", 1))
        layer_patterns = self._resolve_layer_patterns(pruning_cfg)
        exclude_bn = bool(getattr(pruning_cfg, "exclude_bn", True))

        if sparsity <= 0.0:
            return

        device = self.device
        named_params = list(self.model.named_parameters())
        params = [p for _, p in named_params]
        grads = [torch.zeros_like(p) for p in params]

        loader_iter = iter(self.forget_loader)
        for _ in range(max(prune_num, 1)):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.forget_loader)
                batch = next(loader_iter)
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            self.model.zero_grad()
            loss.backward()

            for idx, param in enumerate(self.model.parameters()):
                if param.grad is not None:
                    grads[idx].add_(param.grad.detach().abs())

        with torch.no_grad():
            # Saliency is |grad * weight|, restricted to eligible layers.
            abs_saliences = [(grad * weight).abs() for weight, grad in zip(params, grads)]
            eligible_flags = []
            eligible_flat = []
            for (name, _), sal in zip(named_params, abs_saliences):
                eligible = True
                if exclude_bn and "bn" in name:
                    eligible = False
                if layer_patterns is not None and not any(pat in name for pat in layer_patterns):
                    eligible = False
                eligible_flags.append(eligible)
                if eligible:
                    eligible_flat.append(sal.view(-1))

            if not eligible_flat:
                return

            flat = torch.cat(eligible_flat)
            k = int(sparsity * flat.numel())
            k = max(1, min(k, flat.numel() - 1))
            threshold = float(flat.kthvalue(k)[0])
            if threshold >= flat.max() - 1e-12 or threshold <= flat.min() + 1e-12:
                threshold = float((flat.max() - flat.min()) / 2.0)

            masks = []
            for eligible, sal in zip(eligible_flags, abs_saliences):
                mask = torch.ones_like(sal)
                if eligible:
                    mask[sal > threshold] = 0.0
                masks.append(mask)

            for (name, param), mask, eligible in zip(named_params, masks, eligible_flags):
                if not eligible:
                    continue
                if "weight" in name:
                    reinit = self._reinit_weight(param)
                elif "bias" in name:
                    reinit = torch.zeros_like(param)
                else:
                    continue
                param.data = param.data * mask + reinit * (1.0 - mask)

    @staticmethod
    def _reinit_weight(param: torch.Tensor) -> torch.Tensor:
        """Reinitialize a weight tensor with Kaiming uniform."""
        reinit = torch.empty_like(param)
        if reinit.ndim < 2:
            reinit = reinit.unsqueeze(1)
            nn.init.kaiming_uniform_(reinit, a=math.sqrt(5))
            return reinit.squeeze(1)
        nn.init.kaiming_uniform_(reinit, a=math.sqrt(5))
        return reinit

    def training_step(self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """Run one bilevel LUR training step."""
        del batch_idx
        x_forget, y_forget = batch["forget"]
        x_retain, y_retain = batch["retain"]
        x_forget, y_forget = x_forget.to(self.device), y_forget.to(self.device)
        x_retain, y_retain = x_retain.to(self.device), y_retain.to(self.device)

        logits_retain = self.model(x_retain)
        retain_loss = self.loss_fn(logits_retain, y_retain)

        params = dict(self.model.named_parameters())
        params_for_grad = {k: v for k, v in params.items() if v.requires_grad}
        grads = torch.autograd.grad(
            retain_loss,
            params_for_grad.values(),
            create_graph=True,
            allow_unused=True,
        )
        fast_params = {}
        for (name, param), grad in zip(params_for_grad.items(), grads):
            if grad is None:
                grad = torch.zeros_like(param)
            fast_params[name] = param - self.alpha * grad

        buffers = dict(self.model.named_buffers())
        params_and_buffers = {**params, **buffers}
        params_and_buffers.update(fast_params)

        logits_forget = _functional_call(self.model, params_and_buffers, (x_forget,))
        forget_ce = self.loss_fn(logits_forget, y_forget)
        forget_loss = -forget_ce

        total_loss = self.retain_weight * retain_loss + self.forget_weight * forget_loss

        preds_forget = logits_forget.argmax(dim=1)
        preds_retain = logits_retain.argmax(dim=1)
        self.forget_acc(preds_forget, y_forget)
        self.retain_acc(preds_retain, y_retain)
        self.log("forget/acc", self.forget_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log("retain/acc", self.retain_acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log("retain/ce_loss", retain_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("forget/ce_loss", forget_ce, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train/loss", total_loss, on_epoch=True, on_step=False, prog_bar=False)
        return total_loss

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


class LUR(UnlearningStrategy):
    """Bilevel LUR objective for classification unlearning."""

    def run(self, base_model: pl.LightningModule) -> pl.LightningModule:
        """Execute LUR and return the updated model."""
        lur_cfg = self.cfg.unlearn
        combined_train = CombinedLoader(
            {"forget": self.dm.forget_dataloader(), "retain": self.dm.retain_dataloader()},
            mode="min_size",
        )

        lur_module = _LURModule(
            model=base_model,
            cfg=lur_cfg,
            full_cfg=self.cfg,
            forget_loader=self.dm.forget_dataloader(),
        )

        trainer = self.new_trainer()
        trainer.fit(
            lur_module,
            train_dataloaders=combined_train,
            val_dataloaders=self.dm.val_dataloader(),
        )
        return lur_module.model

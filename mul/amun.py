from typing import Any, Dict, List, Optional, Tuple
import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import ConcatDataset, Dataset, TensorDataset
from tqdm import tqdm

from utils.utils import build_optimizer, resolve_amun_adv_cache_path
from .base import UnlearningStrategy


class _AMUNModule(pl.LightningModule):
    """Fine-tuning module for AMUN."""

    def __init__(self, model: pl.LightningModule, cfg: Dict[str, Any]) -> None:
        """Initialize the AMUN fine-tuning module."""
        super().__init__()
        self.model = model
        self.cfg = cfg

        self.loss_fn = nn.CrossEntropyLoss()
        self.use_weighted_loss = bool(getattr(self.cfg, "use_weighted_loss", False))
        if self.use_weighted_loss:
            self.retain_weight = float(self.cfg.retain_weight)
            self.forget_weight = float(self.cfg.forget_weight)
        num_classes = self.model.num_classes
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

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
            "train": self.train_acc,
            "val": self.val_acc,
            "test": self.test_acc,
        }[stage]
        metric(preds, y)

        self.log(f"{stage}/acc", metric, on_epoch=True, on_step=False, prog_bar=(stage != "train"))
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Run one training step with optional weighted loss."""
        del batch_idx
        if not self.use_weighted_loss:
            return self._shared_step(batch, "train")

        x_forget, y_forget = batch["forget"]
        x_retain, y_retain = batch["retain"]

        logits_forget = self.model(x_forget)
        logits_retain = self.model(x_retain)

        forget_loss = self.loss_fn(logits_forget, y_forget)
        retain_loss = self.loss_fn(logits_retain, y_retain)
        total_loss = self.forget_weight * forget_loss + self.retain_weight * retain_loss

        self.log("train/loss", total_loss, on_epoch=True, on_step=False, prog_bar=False)
        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one validation step."""
        del batch_idx
        return self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one test step."""
        del batch_idx
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and scheduler."""
        return build_optimizer(self.model.parameters(), self.cfg.optim, self.cfg.scheduler)


class _AdvDataset(Dataset):
    """TensorDataset wrapper that normalizes label types."""

    def __init__(self, base: TensorDataset) -> None:
        """Wrap an adversarial TensorDataset."""
        self.base = base

    def __len__(self) -> int:
        """Return the dataset length."""
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return a sample and integer label."""
        x, y = self.base[idx]
        return x, int(y)


class AMUN(UnlearningStrategy):
    """Adversarial Machine Unlearning (AMUN)."""

    def run(self, base_model: pl.LightningModule) -> pl.LightningModule:
        """Execute AMUN and return the updated model."""
        adv_dataset = self._build_adversarial_dataset(base_model)
        use_weighted_loss = bool(getattr(self.cfg.unlearn, "use_weighted_loss", False))
        if use_weighted_loss:
            adv_train = _AdvDataset(adv_dataset)
            adv_loader = self._build_loader(adv_train, shuffle=True)
            retain_loader = self.dm.retain_dataloader()
            train_loader = CombinedLoader(
                {"forget": adv_loader, "retain": retain_loader},
                mode="min_size",
            )
        else:
            train_dataset = self._build_finetune_dataset(adv_dataset)
            train_loader = self._build_loader(train_dataset, shuffle=True)

        amun_module = _AMUNModule(base_model, self.cfg.unlearn)
        trainer = self.new_trainer()
        trainer.fit(
            amun_module,
            train_dataloaders=train_loader,
            val_dataloaders=self.dm.val_dataloader(),
        )
        return amun_module.model

    def _build_finetune_dataset(self, adv_dataset: TensorDataset) -> torch.utils.data.Dataset:
        """Create the fine-tuning dataset for AMUN."""

        datasets: List[torch.utils.data.Dataset] = []
        if bool(self.cfg.unlearn.use_retain):
            datasets.append(self.dm.train_retain)
        if bool(self.cfg.unlearn.use_forget):
            datasets.append(self.dm.train_forget)
        datasets.append(_AdvDataset(adv_dataset))

        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    def _build_loader(self, dataset: torch.utils.data.Dataset, shuffle: bool) -> torch.utils.data.DataLoader:
        """Build a DataLoader using datamodule settings when available."""
        if hasattr(self.dm, "_dl"):
            return self.dm._dl(dataset, shuffle)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=int(self.cfg.data.batch_size),
            shuffle=shuffle,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            persistent_workers=bool(self.cfg.data.persistent_workers),
            prefetch_factor=int(self.cfg.data.prefetch_factor),
        )

    def _build_adversarial_dataset(self, model: pl.LightningModule) -> TensorDataset:
        """Build or load the cached AMUN adversarial dataset."""
        cache_path = resolve_amun_adv_cache_path(self.cfg)
        if cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu")
            print(f"[AMUN] Loaded AMUN adversarial set from {cache_path}")
            return TensorDataset(payload["images"], payload["labels"])

        loader = self.dm.forget_eval_dataloader()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        was_training = model.training
        model.eval()

        attack_cfg = self.cfg.unlearn.attack
        clamp_min, clamp_max = self._normalized_bounds(device)

        diag = None
        if bool(getattr(attack_cfg, "diagnostics", True)):
            max_rounds = int(attack_cfg.max_rounds)
            diag = {"total": [0] * max_rounds, "mis": [0] * max_rounds, "failed": 0}

        adv_tensor: Optional[torch.Tensor] = None
        label_tensor: Optional[torch.Tensor] = None
        offset = 0
        total = len(loader.dataset)
        for x, y in tqdm(loader, desc="building AMUN adversarial set"):
            x = x.to(device)
            y = y.to(device)
            adv_batch, adv_pred, _ = self._pgd_l2_until_misclassified(
                model,
                x,
                y,
                float(attack_cfg.epsilon_init),
                int(attack_cfg.steps),
                float(attack_cfg.step_factor),
                clamp_min,
                clamp_max,
                diag,
            )
            adv_cpu = adv_batch.detach().cpu()
            pred_cpu = adv_pred.detach().cpu().long()
            if adv_tensor is None:
                adv_tensor = torch.empty(
                    (total,) + adv_cpu.shape[1:],
                    dtype=adv_cpu.dtype,
                    device="cpu",
                )
                label_tensor = torch.empty((total,), dtype=torch.long, device="cpu")
            bs = adv_cpu.shape[0]
            adv_tensor[offset:offset + bs].copy_(adv_cpu)
            label_tensor[offset:offset + bs].copy_(pred_cpu)
            offset += bs
        if offset != total:
            adv_tensor = adv_tensor[:offset]
            label_tensor = label_tensor[:offset]

        if was_training:
            model.train()
        if diag is not None and diag["total"]:
            total_samples = diag["total"][0]
            for idx, (total, mis) in enumerate(zip(diag["total"], diag["mis"])):
                if total == 0:
                    continue
                rate = mis / total
                print(f"[AMUN] round {idx + 1}: misclassified {mis}/{total} ({rate:.3f})")
            if total_samples:
                failed = diag["failed"]
                if failed:
                    rate = failed / total_samples
                    print(
                        f"[AMUN] not misclassified after {int(attack_cfg.max_rounds)} rounds: "
                        f"{failed}/{total_samples} ({rate:.3f})"
                    )
                else:
                    print(f"[AMUN] all samples misclassified within {int(attack_cfg.max_rounds)} rounds")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save({"images": adv_tensor, "labels": label_tensor}, tmp_path)
        print(f"[AMUN] Saved AMUN adversarial set to {tmp_path}")
        tmp_path.replace(cache_path)
        return TensorDataset(adv_tensor, label_tensor)

    def _normalized_bounds(self, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute normalized min/max bounds for clamping."""
        norm_cfg = self.cfg.data.transforms.normalize
        mean = torch.tensor(norm_cfg.mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(norm_cfg.std, device=device).view(1, -1, 1, 1)
        clamp_min = (0.0 - mean) / std
        clamp_max = (1.0 - mean) / std
        return clamp_min, clamp_max

    def _pgd_l2_until_misclassified(
        self,
        model: pl.LightningModule,
        x: torch.Tensor,
        y: torch.Tensor,
        eps_init: float,
        steps: int,
        step_factor: float,
        clamp_min: Optional[torch.Tensor],
        clamp_max: Optional[torch.Tensor],
        diag: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run PGD L2 attack until samples are misclassified or max rounds."""
        batch_size = x.size(0)
        eps_all = torch.full((batch_size,), eps_init, device=x.device)
        adv_all = x.clone().detach()
        adv_labels = torch.full_like(y, fill_value=-1)
        remaining = torch.arange(batch_size, device=x.device)

        rounds = 0
        while remaining.numel() > 0:
            rounds += 1
            cur_x = x[remaining]
            cur_y = y[remaining]
            cur_eps = eps_all[remaining]

            adv_cur = self._pgd_l2_attack(model, cur_x, cur_y, cur_eps, steps, step_factor, clamp_min, clamp_max)

            with torch.no_grad():
                preds = model(adv_cur).argmax(dim=1)
            misclassified = preds.ne(cur_y)
            if diag is not None:
                round_idx = rounds - 1
                if round_idx < len(diag["total"]):
                    diag["total"][round_idx] += int(cur_y.numel())
                    diag["mis"][round_idx] += int(misclassified.sum().item())

            if misclassified.any():
                idx = remaining[misclassified]
                adv_all[idx] = adv_cur[misclassified].detach()
                adv_labels[idx] = preds[misclassified].detach()
                eps_all[idx] = cur_eps[misclassified]

            if (~misclassified).any():
                remaining = remaining[~misclassified]
                eps_all[remaining] = cur_eps[~misclassified] * 2.0
                if rounds >= self.cfg.unlearn.attack.max_rounds:
                    if diag is not None:
                        diag["failed"] += int(remaining.numel())
                    adv_labels[remaining] = cur_y[~misclassified].detach()
                    break
            else:
                break

        return adv_all, adv_labels, eps_all

    def _pgd_l2_attack(
        self,
        model: pl.LightningModule,
        x: torch.Tensor,
        y: torch.Tensor,
        eps: torch.Tensor,
        steps: int,
        step_factor: float,
        clamp_min: Optional[torch.Tensor],
        clamp_max: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run a fixed-step PGD L2 attack with optional clamping."""
        loss_fn = nn.CrossEntropyLoss()
        adv = x.clone().detach()
        eps = eps.view(-1)

        for _ in range(steps):
            adv.requires_grad_(True)
            logits = model(adv)
            loss = loss_fn(logits, y)
            grad = torch.autograd.grad(loss, adv)[0]

            grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1).clamp_min(1e-12)
            grad = grad / grad_norm.view(-1, 1, 1, 1)

            step = step_factor * eps
            adv = adv + step.view(-1, 1, 1, 1) * grad

            delta = adv - x
            delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1).clamp_min(1e-12)
            factor = torch.minimum(torch.ones_like(delta_norm), eps / delta_norm)
            adv = x + delta * factor.view(-1, 1, 1, 1)

            if clamp_min is not None and clamp_max is not None:
                adv = torch.max(torch.min(adv, clamp_max), clamp_min)
            adv = adv.detach()

        return adv

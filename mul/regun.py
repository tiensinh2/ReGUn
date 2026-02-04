import copy
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.utils import build_optimizer
from .base import UnlearningStrategy, cycle


class _ReGUnModule(pl.LightningModule):
    """ReGUn (Random Held-out Supervision)."""

    def __init__(
        self,
        model: pl.LightningModule,
        cfg: Dict[str, Any],
        heldout_loader: DataLoader,
        retain_loader: DataLoader,
    ) -> None:
        """Initialize the ReGUn module."""
        super().__init__()
        self.model = model
        self.optim_cfg = cfg.optim
        self.scheduler_cfg = cfg.scheduler
        self.regun_cfg = cfg.regun
        self.reference_model = None
        if bool(self.regun_cfg.fixed_reference):
            self.reference_model = copy.deepcopy(model)
            self.reference_model.requires_grad_(False)
            self.reference_model.eval()

        self.temperature = float(self.regun_cfg.temperature)
        self.retain_weight = float(self.regun_cfg.retain_weight)
        self.forget_weight = float(self.regun_cfg.forget_weight)
        self.selector = str(self.regun_cfg.selector)
        self.aggregation = str(self.regun_cfg.aggregation)

        self.heldout_iter = cycle(heldout_loader) if self.selector == "uniform" else None
        self.retain_iter = cycle(retain_loader)
        self.heldout_loader = heldout_loader
        self.heldout_logits = None

        self.ce_loss = nn.CrossEntropyLoss()
        num_classes = self.model.num_classes
        self.num_classes = num_classes
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        if self.selector == "class_informed":
            self.heldout_dataset = heldout_loader.dataset
            self.class_to_indices, self.all_heldout_indices = self._build_class_indices(self.heldout_dataset, num_classes)
        if self.selector == "feature_1nn":
            self.cls_layer = self._find_classifier_layer(self.model)
            if self.reference_model is not None:
                self.ref_cls_layer = self._find_classifier_layer(self.reference_model)
                ref_model = self.reference_model
                keep_eval = True
            else:
                self.ref_cls_layer = self.cls_layer
                ref_model = self.model
                keep_eval = False
            self.heldout_features, self.heldout_logits = self._cache_heldout_features(
                heldout_loader,
                ref_model,
                self.ref_cls_layer,
                keep_eval=keep_eval,
            )

    def on_fit_start(self) -> None:
        """Cache held-out logits when needed for class-informed selection."""
        if self.selector != "class_informed":
            return
        if self.reference_model is None:
            print("[ReGUn] Skipping held-out cache (fixed_reference=false).")
            return
        if self.heldout_logits is None:
            print("[ReGUn] Caching held-out logits for class-informed ReGUn...")
            self.heldout_logits = self._cache_heldout_logits(self.heldout_loader, self.reference_model, keep_eval=True)
            print(f"[ReGUn] Cached held-out logits: {self.heldout_logits.shape}.")
        else:
            print(f"[ReGUn] Using existing held-out logits cache: {self.heldout_logits.shape}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate forward to wrapped model."""
        return self.model(x)

    def _get_reference_model(self) -> pl.LightningModule:
        """Return the model to use for ReGUn reference logits."""
        if self.reference_model is None:
            return self.model
        self.reference_model.eval()
        return self.reference_model

    def _unwrap_subset(self, ds: Any) -> Tuple[Sequence[int], Any]:
        """Return (indices, base_dataset) for nested Subset chains."""
        indices = None
        while isinstance(ds, Subset):
            cur_idx = ds.indices
            indices = cur_idx if indices is None else [cur_idx[i] for i in indices]
            ds = ds.dataset
        return (list(range(len(ds))) if indices is None else indices), ds

    def _build_class_indices(self, ds: Any, num_classes: int) -> Tuple[Dict[int, List[int]], List[int]]:
        """Precompute per-class pools for held-out data."""
        base_indices, base = self._unwrap_subset(ds)
        labels = [int(base.targets[i]) for i in base_indices]

        class_to_indices = {c: [] for c in range(num_classes)}
        for pos, y in enumerate(labels):
            class_to_indices[y].append(pos)
        return class_to_indices, list(range(len(base_indices)))

    def _sample_class_informed_indices(self, y_forget: torch.Tensor) -> List[int]:
        """Sample held-out indices matching the class histogram of the forget batch."""
        counts = torch.bincount(y_forget.detach().cpu(), minlength=self.num_classes)
        chosen = []
        for cls, k in enumerate(counts.tolist()):
            if k == 0:
                continue
            pool = self.class_to_indices.get(cls, [])
            source = pool if pool else self.all_heldout_indices
            chosen.extend(random.choices(source, k=k))
        random.shuffle(chosen)
        return chosen

    def _sample_class_informed(self, y_forget: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Sample held-out images matching the class histogram of the forget batch."""
        chosen = self._sample_class_informed_indices(y_forget)
        imgs = [self.heldout_dataset[i][0] for i in chosen]
        return torch.stack(imgs, dim=0).to(device)

    def _sample_class_informed_logits(self, y_forget: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Sample cached held-out logits matching the forget batch class histogram."""
        chosen = self._sample_class_informed_indices(y_forget)
        logits = self.heldout_logits[chosen]
        return logits.to(device, non_blocking=True)

    def _build_cache_loader(self, loader: DataLoader) -> DataLoader:
        """Create a deterministic loader for caching."""
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            persistent_workers=loader.persistent_workers,
            prefetch_factor=loader.prefetch_factor,
        )

    def _cache_heldout_logits(self, loader: DataLoader, model: nn.Module, keep_eval: bool) -> torch.Tensor:
        """Cache held-out logits once (dataset order)."""
        cache_loader = self._build_cache_loader(loader)
        prev_mode = model.training
        model.eval()
        device = next(model.parameters()).device
        logits_all = []
        with torch.no_grad():
            for x, _ in tqdm(cache_loader, desc="[ReGUn]Caching held-out logits"):
                x = x.to(device)
                logits = model(x)
                logits_all.append(logits.detach().cpu())
        if prev_mode and not keep_eval:
            model.train()
        return torch.cat(logits_all, dim=0)

    def _cache_heldout_features(
        self,
        loader: Iterable,
        model: nn.Module,
        cls_layer: nn.Module,
        keep_eval: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache held-out penultimate features and logits once."""
        feats, logits_all = [], []
        prev_mode = model.training
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits, feat = self._forward_with_features(x, model, cls_layer)
                feats.append(F.normalize(feat, dim=1).cpu())
                logits_all.append(logits.detach().cpu())
        if prev_mode and not keep_eval:
            model.train()
        return torch.cat(feats, dim=0), torch.cat(logits_all, dim=0)

    def _find_classifier_layer(self, model: nn.Module) -> nn.Module:
        """Locate the final classifier layer to hook penultimate features."""
        layer = None
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.out_features == model.num_classes:
                layer = m
        if layer is None:
            raise ValueError("Classifier layer not found for feature extraction.")
        return layer

    def _forward_with_features(self, x: torch.Tensor, model: nn.Module, cls_layer: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also captures penultimate features."""
        feats = []

        def _hook(_m, inp, _out):
            feats.append(inp[0])

        handle = cls_layer.register_forward_hook(_hook)
        logits = model(x)
        handle.remove()
        return logits, feats[0]

    def _supervision_distribution(
        self,
        device: torch.device,
        batch_size: int,
        y_forget: Optional[torch.Tensor],
        feats_forget: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build ReGUn target distribution from held-out supervision."""
        ref_model = self._get_reference_model()
        if self.selector == "class_informed":
            if self.heldout_logits is not None:
                logits = self._sample_class_informed_logits(y_forget, device)
            else:
                x_held = self._sample_class_informed(y_forget, device)
                with torch.no_grad():
                    logits = ref_model(x_held)
        elif self.selector == "uniform":
            x_held, _ = next(self.heldout_iter)
            x_held = x_held.to(device)
            with torch.no_grad():
                logits = ref_model(x_held)
        elif self.selector == "feature_1nn":
            feats_forget = F.normalize(feats_forget.detach(), dim=1)
            feats_held = self.heldout_features.to(device)
            sims = torch.matmul(feats_forget, feats_held.t())
            nn_idx = sims.argmax(dim=1).cpu()
            logits = self.heldout_logits[nn_idx].to(device)

        if self.aggregation == "mean_probs":
            probs = F.softmax(logits / self.temperature, dim=1)
            agg = probs.mean(dim=0)
        elif self.aggregation == "mean_logits":
            mean_logits = logits.mean(dim=0)
            agg = F.softmax(mean_logits / self.temperature, dim=0)
        agg = agg / agg.sum()
        return agg.unsqueeze(0).repeat(batch_size, 1)

    def _retain_loss(self, device: torch.device) -> torch.Tensor:
        """Compute CE on retain batch for regularization."""
        x_ret, y_ret = next(self.retain_iter)
        x_ret, y_ret = x_ret.to(device), y_ret.to(device)
        logits = self.model(x_ret)
        return self.ce_loss(logits, y_ret)

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Shared step for forget/val/test phases."""
        x, y = batch
        if stage == "forget" and self.selector == "feature_1nn":
            if self.reference_model is None:
                logits, feats = self._forward_with_features(x, self.model, self.cls_layer)
            else:
                logits = self.model(x)
                with torch.no_grad():
                    ref_model = self._get_reference_model()
                    _, feats = self._forward_with_features(x, ref_model, self.ref_cls_layer)
        else:
            logits = self.model(x)
            feats = None
        preds = logits.argmax(dim=1)
        metric = {
            "forget": self.train_acc,
            "val": self.val_acc,
            "test": self.test_acc,
        }[stage]
        metric(preds, y)
        self.log(f"{stage}/acc", metric, on_epoch=True, on_step=False, prog_bar=(stage == "val"))

        if stage != "forget":
            loss = self.ce_loss(logits, y)
            self.log(f"{stage}/loss", loss, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
            return loss

        target_dist = self._supervision_distribution(x.device, x.size(0), y_forget=y, feats_forget=feats)
        regun_loss = F.kl_div(
            F.log_softmax(logits / self.temperature, dim=1),
            target_dist.to(x.device),
            reduction="batchmean",
        )
        self.log("forget/regun_loss", regun_loss, on_step=False, on_epoch=True, prog_bar=True)

        retain_loss = self._retain_loss(x.device)
        self.log("forget/retain_ce", retain_loss, on_step=False, on_epoch=True, prog_bar=False)

        total = self.forget_weight * regun_loss + self.retain_weight * retain_loss
        self.log("forget/loss", total, on_step=False, on_epoch=True, prog_bar=False)
        return total

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one training step on the forget split."""
        del batch_idx
        return self._shared_step(batch, "forget")

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
        return build_optimizer(self.model.parameters(), self.optim_cfg, self.scheduler_cfg)


class ReGUn(UnlearningStrategy):
    """Entry point for ReGUn strategy."""

    def run(self, base_model: pl.LightningModule) -> pl.LightningModule:
        """Execute ReGUn and return the updated model."""
        regun_module = _ReGUnModule(
            base_model,
            self.cfg.unlearn,
            heldout_loader=self.dm.heldout_eval_dataloader(),
            retain_loader=self.dm.retain_dataloader(),
        )
        trainer = self.new_trainer()
        trainer.fit(
            regun_module,
            train_dataloaders=self.dm.forget_dataloader(),
            val_dataloaders=self.dm.val_dataloader(),
        )
        return regun_module.model

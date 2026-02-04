import torch
import torch.nn as nn
import lightning.pytorch as pl
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import (
    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
)
from utils.utils import build_optimizer


_VITS = {
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_b_32": (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1),
    "vit_l_16": (vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1),
    "vit_l_32": (vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1),
}


class ViTClassifier(pl.LightningModule):
    """Minimal Lightning wrapper for torchvision ViT (B/L, 16/32 patch)."""

    @classmethod
    def build(cls, cfg: DictConfig) -> "ViTClassifier":
        return cls(cfg.model)

    @classmethod
    def load(cls, checkpoint_path: str, strict: bool = True) -> "ViTClassifier":
        return cls.load_from_checkpoint(checkpoint_path, strict=strict, map_location="cpu", weights_only=False)

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = cfg.model.num_classes

        # Build model with optional pretrained weights
        ctor, weights_enum = _VITS[cfg.model.name.lower()]
        weights = (weights_enum if cfg.model.pretrained else None)
        self.model = ctor(weights=weights, image_size=int(cfg.model.image_size))
        # Adjusted new head
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.num_classes)

        if getattr(cfg.model, "freeze_backbone", False):
            for n, p in self.model.named_parameters():
                if not n.startswith("heads."):
                    p.requires_grad = False

        label_smoothing = float(getattr(cfg.model, "label_smoothing", 0.0))
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_acc   = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_acc  = MulticlassAccuracy(num_classes=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)

        metric = {
            "train": self.train_acc,
            "val": self.val_acc,
            "test": self.test_acc,
        }[stage]
        metric(preds, y)

        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=False,  prog_bar=(stage != "train"))
        self.log(f"{stage}/acc", metric, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def training_step(self, batch, _):      return self._shared_step(batch, "train")
    def validation_step(self, batch, _):    return self._shared_step(batch, "val")
    def test_step(self, batch, _):          return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return build_optimizer(self.parameters(), self.cfg.optim, self.cfg.scheduler, named_params=self.named_parameters())

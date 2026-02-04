import torch
import torch.nn as nn
import lightning.pytorch as pl
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)
from utils.utils import build_optimizer

_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
}

def _cifar_stem(m: nn.Module) -> None:
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()


def _tiny_imagenet_stem(m: nn.Module) -> None:
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
    m.maxpool = nn.Identity()

def _resolve_stem(cfg: DictConfig) -> str:
    model_cfg = cfg.model
    stem = model_cfg.get("stem")
    if stem is None:
        stem = "cifar" if model_cfg.get("cifar_stem", False) else "none"
    return stem

class ResNetClassifier(pl.LightningModule):
    """Minimal Lightning wrapper for torchvision ResNet-18/34/50."""

    @classmethod
    def build(cls, cfg: DictConfig) -> "ResNetClassifier":
        return cls(cfg.model)

    @classmethod
    def load(cls, checkpoint_path: str, strict: bool = True) -> "ResNetClassifier":
        return cls.load_from_checkpoint(checkpoint_path, strict=strict, map_location="cpu", weights_only=False)

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = cfg.model.num_classes
        
        # Build model with optional pretrained weights
        ctor, weights_enum = _BACKBONES[cfg.model.name.lower()]
        weights = (weights_enum if cfg.model.pretrained else None)
        self.model = ctor(weights=weights)
        # Optional stem adjustment (backward-compatible with legacy configs).
        stem = _resolve_stem(cfg)
        if stem == "cifar":
            _cifar_stem(self.model)
        elif stem == "tiny_imagenet":
            _tiny_imagenet_stem(self.model)

        # Adjusted new head
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        if getattr(cfg.model, "freeze_backbone", False):
            for name, p in self.model.named_parameters():
                if not name.startswith("fc."):
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

        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=False, prog_bar=(stage != "train"))
        self.log(f"{stage}/acc",  metric,  on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def training_step(self, batch, _):      return self._shared_step(batch, "train")
    def validation_step(self, batch, _):    return self._shared_step(batch, "val")
    def test_step(self, batch, _):          return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return build_optimizer(self.parameters(), self.cfg.optim, self.cfg.scheduler)

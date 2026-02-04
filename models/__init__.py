import os
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from .resnet import ResNetClassifier
from .swin import SwinClassifier
from .vit import ViTClassifier

_MODEL_REGISTRY = {
    "resnet18": ResNetClassifier,
    "resnet34": ResNetClassifier,
    "resnet50": ResNetClassifier,
    "swin_t": SwinClassifier,
    "swin_s": SwinClassifier,
    "swin_b": SwinClassifier,
    "vit_b_16": ViTClassifier,
    "vit_b_32": ViTClassifier,
    "vit_l_16": ViTClassifier,
    "vit_l_32": ViTClassifier,
}


def _ensure_torch_home(cfg: DictConfig) -> None:
    os.environ["TORCH_HOME"] = os.path.join(cfg.paths.data_dir, "torch_cache")


def build_model(cfg: DictConfig) -> LightningModule:
    """Construct a fresh model from configuration."""
    _ensure_torch_home(cfg)
    model_cls = _MODEL_REGISTRY[cfg.model.model.name.lower()]
    return model_cls.build(cfg)


def load_model(
    cfg: DictConfig,
    checkpoint_path: str,
    strict: bool = True,
) -> LightningModule:
    """Load a checkpoint for the configured architecture."""
    _ensure_torch_home(cfg)
    model_cls = _MODEL_REGISTRY[cfg.model.model.name.lower()]
    return model_cls.load(checkpoint_path, strict=strict)

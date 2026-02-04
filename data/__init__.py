from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from .cifar import CIFARDataModule
from .tiny_imagenet import TinyImageNetDataModule

_DM_BUILDERS = {
    "cifar10":  lambda cfg: CIFARDataModule(cfg),
    "cifar100": lambda cfg: CIFARDataModule(cfg),
    "tiny_imagenet": lambda cfg: TinyImageNetDataModule(cfg),
}

def build_datamodule(cfg: DictConfig) -> LightningDataModule:
    """Build a datamodule from configuration."""
    name = cfg.data.name.lower()
    builder = _DM_BUILDERS[name]
    dm = builder(cfg)
    dm.prepare_data()
    dm.setup(stage=None)
    return dm

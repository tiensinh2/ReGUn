from copy import deepcopy
from typing import Dict, Optional, Type

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from evaluation.evaluator import MULEvaluator
from .base import UnlearningStrategy
from .amun import AMUN
from .finetune import Finetune
from .l1sparse import L1Sparse
from .neggrad import NegGrad
from .neggradp import NegGradPlus
from .regun import ReGUn
from .salun import SalUn
from .scrub import SCRUB
from .ssd import SSD
from .lur import LUR


_MUL_STRATEGIES: Dict[str, Type[UnlearningStrategy]] = {
    "amun": AMUN,
    "finetune": Finetune,
    "l1sparse": L1Sparse,
    "neggrad": NegGrad,
    "neggradp": NegGradPlus,
    "regun": ReGUn,
    "salun": SalUn,
    "scrub": SCRUB,
    "ssd": SSD,
    "lur": LUR,
}

def run_mul_strategy(
    cfg: DictConfig,
    base_model: pl.LightningModule,
    dm: pl.LightningDataModule,
    wandb_run: WandbLogger,
    evaluator: Optional[MULEvaluator] = None,
) -> pl.LightningModule:
    """Run an unlearning strategy by name."""
    name = str(cfg.unlearn.name).lower()
    strategy = _MUL_STRATEGIES[name](cfg, dm, wandb_run, evaluator)
    return strategy.run(deepcopy(base_model))

__all__ = [
    "MULEvaluator",
    "run_mul_strategy",
]

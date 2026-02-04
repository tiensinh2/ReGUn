from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Iterable, Iterator, List, Optional

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from evaluation.evaluator import MULEvaluator, UnlearningEpochEvaluationCallback


def cycle(loader: Iterable) -> Iterator:
    """Endless iterator over a finite data loader."""
    while True:
        for batch in loader:
            yield batch


class UnlearningStrategy(ABC):
    """Base interface for unlearning strategies."""

    def __init__(
        self,
        cfg: DictConfig,
        dm: pl.LightningDataModule,
        wandb_run: WandbLogger,
        evaluator: Optional[MULEvaluator] = None,
    ) -> None:
        """Initialize shared state for strategies."""
        super().__init__()
        self.cfg = cfg
        self.dm = dm
        self.dm.setup()
        self.wandb_run = wandb_run
        self._enable_epoch_eval: bool = bool(getattr(self.cfg.unlearn, "evaluate_each_epoch", False))
        self.evaluator: Optional[MULEvaluator] = evaluator

    def clone_fresh_model(self) -> pl.LightningModule:
        """Build a fresh randomly initialized model from cfg."""
        from models import build_model
        cfg_copy = deepcopy(self.cfg)
        cfg_copy.run.base_weights = None
        return build_model(cfg_copy)

    def new_trainer(self, additional_callbacks: Optional[List[Callback]] = None) -> Trainer:
        """Create a per-strategy trainer with unlearn overrides."""
        from utils import build_trainer

        cfg_copy = deepcopy(self.cfg)
        u_trainer = OmegaConf.select(self.cfg, "unlearn.trainer")

        if u_trainer is not None:
            merged_trainer = OmegaConf.merge(self.cfg.trainer, u_trainer)
            cfg_copy.trainer = merged_trainer
        else:
            cfg_copy.trainer = deepcopy(self.cfg.trainer)

        callbacks: List[Callback] = list(additional_callbacks or [])
        if self._enable_epoch_eval:
            callbacks.append(
                UnlearningEpochEvaluationCallback(
                    prefix="unlearning",
                    evaluator=self.evaluator,
                )
            )

        strategy_job_type = f"unlearning-strategy-{self.__class__.__name__.lower()}"
        trainer = build_trainer(
            cfg_copy,
            job_type=strategy_job_type,
            logger=self.wandb_run,
            additional_callbacks=callbacks,
        )
        return trainer

    @abstractmethod
    def run(self, base_model: pl.LightningModule) -> pl.LightningModule:
        """Return the unlearned model."""
        raise NotImplementedError

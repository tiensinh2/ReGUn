"""Base training and retraining script.

Usage:
  python run2_base.py  (Hydra args optional)

Outputs:
  Trains a base model and a retrained model, saves both checkpoints, and logs metrics/evaluations to W&B.
"""

import uuid
import hydra
from omegaconf import DictConfig, OmegaConf
from data import build_datamodule
from models import build_model, load_model
from mul import MULEvaluator
from utils import (
    seed_everything,
    build_trainer,
    wandb_init,
    wandb_log,
    wandb_finish,
    export_base_checkpoint,
)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"[MAIN] Config:\n{OmegaConf.to_yaml(cfg)}")
    seed_everything(cfg.seed)
    cfg.run.experiment_id = cfg.run.experiment_id or str(uuid.uuid4().hex)[:8]
    dm = build_datamodule(cfg)
    logger = wandb_init(cfg, job_type="base-and-retrain")
    try:
        base_model = build_model(cfg)

        print(f"[MAIN] --- Starting Initial Training (Group: {cfg.run.experiment_id}) ---")
        trainer = build_trainer(cfg, job_type="initial-training", logger=logger)
        trainer.fit(base_model, datamodule=dm)
        cfg.run.base_weights = export_base_checkpoint(cfg, trainer)
        trainer.test(base_model, datamodule=dm)

        print(f"[MAIN] --- Starting Retrain-from-Scratch (Group: {cfg.run.experiment_id}) ---")
        retrain_model = build_model(cfg)
        retrain_trainer = build_trainer(cfg, job_type="retrain", logger=logger)
        retrain_trainer.fit(
            retrain_model,
            train_dataloaders=dm.retain_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )
        cfg.run.retrain_weights = export_base_checkpoint(cfg, retrain_trainer, retrain=True)

        model_base = load_model(cfg, cfg.run.base_weights)
        model_retrained = load_model(cfg, cfg.run.retrain_weights)

        evaluator = MULEvaluator(cfg=cfg, datamodule=dm)

        print("[MAIN] --- Evaluating Base Model ---")
        base_results = evaluator.run(model_base)
        print("[MAIN] Base Results:", base_results)

        print("[MAIN] --- Evaluating Retrained Model ---")
        retrained_results = evaluator.run(model_retrained)
        print("[MAIN] Retrained Results:", retrained_results)

        wandb_log(base_results, prefix="base", summary=True)
        wandb_log(retrained_results, prefix="retrained", summary=True)
    finally:
        wandb_finish()


if __name__ == "__main__":
    main()

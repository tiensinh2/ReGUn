"""Unlearning script.

Usage:
  python run3_unlearning.py  (Hydra args optional)

Inputs:
  Expects `cfg.run.base_weights` to point to a trained checkpoint.

Outputs:
  Runs the configured unlearning strategy, evaluates the result, and logs metrics to W&B.
"""

import uuid
import hydra
from omegaconf import DictConfig, OmegaConf
from data import build_datamodule
from models import load_model
from mul import MULEvaluator, run_mul_strategy
from utils import (
    seed_everything,
    wandb_init,
    wandb_log,
    wandb_finish,
)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"[MAIN] Config:\n{OmegaConf.to_yaml(cfg)}")
    seed_everything(cfg.seed)
    cfg.run.experiment_id = cfg.run.experiment_id or str(uuid.uuid4().hex)[:8]

    dm = build_datamodule(cfg)
    model_base = load_model(cfg, cfg.run.base_weights)
    evaluator = MULEvaluator(cfg=cfg, datamodule=dm)

    try:
        print(f"[MAIN] --- Starting Unlearning (Group: {cfg.run.experiment_id}) ---")
        logger = wandb_init(cfg, job_type="unlearning")
        model_unlearned = run_mul_strategy(cfg, model_base, dm, logger, evaluator=evaluator)
        print("[MAIN] --- Starting Final Evaluation ---")
        results = evaluator.run(model_unlearned)
        print("[MAIN] Results:", results)
        wandb_log(results, prefix="unlearning", summary=True)
    finally:
        wandb_finish()


if __name__ == "__main__":
    main()

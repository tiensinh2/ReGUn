"""Reference training script.

Usage:
  python run1_reference.py --run-idx <int>  (Hydra args optional)

Inputs:
  --run-idx (1-based) selects which reference run to produce.

Outputs:
  Writes a retrained checkpoint to the cache/models path and logs to W&B.
"""

import argparse
import sys
import uuid
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
from data import build_datamodule
from models import build_model
from utils import (
    seed_everything,
    build_trainer,
    wandb_init,
    wandb_finish,
    export_base_checkpoint,
)


def _parse_run_idx() -> Optional[int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-idx", type=int, required=False, help="1-based reference run index")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args.run_idx

REFERENCE_RUN_IDX = _parse_run_idx()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if REFERENCE_RUN_IDX is None:
        raise ValueError("[MAIN] Reference run index missing. Provide it via '--run-idx <idx>'.")
    total_refs = int(cfg.mul_eval.mia_num_models)
    if not (1 <= REFERENCE_RUN_IDX <= total_refs):
        raise ValueError(f"[MAIN] Reference run index {REFERENCE_RUN_IDX} outside expected range [1, {total_refs}].")

    print(f"[MAIN] Config:\n{OmegaConf.to_yaml(cfg)}")
    cfg.run.experiment_id = cfg.run.experiment_id or str(uuid.uuid4().hex)[:8]

    base_seed = int(cfg.seed)
    run_seed = base_seed + REFERENCE_RUN_IDX
    seed_everything(run_seed)

    dm = build_datamodule(cfg)

    print(f"[MAIN] --- Starting Reference Training (Group: {cfg.run.experiment_id}) ---")
    logger = wandb_init(cfg, job_type="reference")
    try:
        model = build_model(cfg)
        trainer = build_trainer(cfg, job_type="reference", logger=logger)
        trainer.fit(
            model,
            train_dataloaders=dm.retain_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )

        ckpt_path = export_base_checkpoint(
            cfg,
            trainer,
            retrain=True,
            reference_run=REFERENCE_RUN_IDX,
        )
        print(f"[MAIN] Saved checkpoint -> {ckpt_path}")
    finally:
        wandb_finish()


if __name__ == "__main__":
    main()

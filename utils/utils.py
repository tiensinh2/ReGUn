import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2 as T

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver(
        "eval",
        lambda expr: eval(expr, {"__builtins__": {}}, {}),
    )


def seed_everything(seed: int, seed_torch_backends: bool = False) -> None:
    """Set all random seeds."""
    torch.set_float32_matmul_precision("high")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed_torch_backends:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)


def build_transforms(cfg: DictConfig) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Build training and evaluation transforms from config."""
    t = cfg.data.transforms
    ir = getattr(t, "imagenet_resize", None)
    use_imagenet_resize = bool(ir is not None and ir.enabled)

    if use_imagenet_resize:
        mean, std = ir.normalize.mean, ir.normalize.std
        train_size = int(ir.train_size)
        eval_resize = int(ir.eval_resize)
        eval_crop = int(ir.eval_crop)
    else:
        mean, std = t.normalize.mean, t.normalize.std

    spatial_ops = []
    if use_imagenet_resize:
        spatial_ops.append(
            T.RandomResizedCrop(
                size=train_size,
                scale=(0.6, 1.0),
                ratio=(3 / 4, 4 / 3),
                antialias=True,
            )
        )
        if t.horizontal_flip.enabled:
            spatial_ops.append(T.RandomHorizontalFlip(p=float(t.horizontal_flip.p)))
    else:
        if t.random_crop.enabled:
            spatial_ops += [T.Pad(int(t.random_crop.padding)), T.RandomCrop(int(t.random_crop.size))]
        if t.horizontal_flip.enabled:
            spatial_ops.append(T.RandomHorizontalFlip(p=float(t.horizontal_flip.p)))

    color_ops = []
    color_jitter = getattr(t, "color_jitter", None)
    if color_jitter is not None and color_jitter.enabled:
        color_ops.append(
            T.ColorJitter(
                brightness=float(color_jitter.brightness),
                contrast=float(color_jitter.contrast),
                saturation=float(color_jitter.saturation),
                hue=float(color_jitter.hue),
            )
        )

    randaugment = getattr(t, "randaugment", None)
    randaugment_op = None
    if randaugment is not None and randaugment.enabled:
        randaugment_op = T.RandAugment(
            num_ops=int(getattr(randaugment, "num_ops", 2)),
            magnitude=int(getattr(randaugment, "magnitude", 9)),
            num_magnitude_bins=int(getattr(randaugment, "num_magnitude_bins", 31)),
        )

    random_erasing = getattr(t, "random_erasing", None)
    erasing_ops = []
    if random_erasing is not None and random_erasing.enabled:
        erasing_ops.append(
            T.RandomErasing(
                p=float(getattr(random_erasing, "p", 0.25)),
                scale=tuple(float(x) for x in getattr(random_erasing, "scale", (0.02, 0.33))),
                ratio=tuple(float(x) for x in getattr(random_erasing, "ratio", (0.3, 3.3))),
                value=getattr(random_erasing, "value", 0),
            )
        )

    train_ops = [T.ToImage(), *spatial_ops]
    if randaugment_op is not None:
        train_ops.append(randaugment_op)
    train_ops += [
        T.ToDtype(torch.float32, scale=True),
        *color_ops,
        *erasing_ops,
        T.Normalize(mean, std),
    ]
    train_tf = T.Compose(train_ops)

    if use_imagenet_resize:
        eval_tf = T.Compose(
            [
                T.ToImage(),
                T.Resize(size=eval_resize, antialias=True),
                T.CenterCrop(size=eval_crop),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean, std),
            ]
        )
    else:
        eval_tf = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean, std),
            ]
        )

    return train_tf, eval_tf


def build_trainer(
    cfg: DictConfig,
    job_type: str,
    logger: Optional[WandbLogger] = None,
    additional_callbacks: Optional[List[Callback]] = None,
) -> pl.Trainer:
    """Build Lightning Trainer."""
    callbacks = []
    if cfg.trainer.enable_checkpointing:
        monitor_metric = cfg.checkpoint.monitor.replace("/", "_")
        ckpt_cb = ModelCheckpoint(
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=1,
            save_weights_only=True,
            save_last=True,
            filename=f"{job_type}-{{epoch:03d}}-{{{monitor_metric}}}",
            auto_insert_metric_name=False,
        )
        callbacks.append(ckpt_cb)

    callbacks.append(LearningRateMonitor(logging_interval="step"))

    if additional_callbacks:
        callbacks.extend(additional_callbacks)

    return pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        default_root_dir=cfg.paths.outputs_dir,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        callbacks=callbacks,
        logger=logger if logger is not None else cfg.logging.enable,
    )


def _format_fraction_for_path(value: Any) -> str:
    """Convert fractions to compact strings for checkpoint filenames."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        formatted = f"{value:.6f}".rstrip("0").rstrip(".")
        return formatted if formatted else "0"
    return str(value)


def _default_base_checkpoint_path(
    cfg: DictConfig,
    retrain: bool = False,
    reference_run: Optional[int] = None,
) -> Path:
    """Return the default base checkpoint path after initial training."""
    model_name = cfg.model.model.name
    data_name = cfg.data.name
    split_scheme = cfg.split.scheme
    forget_frac = _format_fraction_for_path(cfg.split.forget_frac)
    suffix = "" if reference_run is None else f"-{int(reference_run):03d}"
    cache_dir = cfg.paths.cache_dir
    relative_path = (
        f"{cache_dir}/models/{model_name}_{data_name}_{split_scheme}_forget{forget_frac}_"
        f"{'retrained_' if retrain else ''}{cfg.seed}{suffix}.ckpt"
    )
    return Path(to_absolute_path(relative_path))


def export_base_checkpoint(
    cfg: DictConfig,
    trainer: pl.Trainer,
    retrain: bool = False,
    reference_run: Optional[int] = None,
) -> str:
    """Save the last checkpoint into the standard cache/models location."""
    target_path = _default_base_checkpoint_path(
        cfg,
        retrain=retrain,
        reference_run=reference_run,
    )
    target_path = target_path.resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(target_path))
    return str(target_path.resolve())


def resolve_checkpoint_path(
    cfg: DictConfig,
    retrain: bool = False,
    reference_run: Optional[int] = None,
) -> Path:
    """Return expected checkpoint path (no filesystem interaction)."""
    return _default_base_checkpoint_path(
        cfg,
        retrain=retrain,
        reference_run=reference_run,
    ).resolve()


def resolve_amun_adv_cache_path(cfg: DictConfig) -> Path:
    """Return default cache path for AMUN adversarial set."""
    attack_cfg = cfg.unlearn.attack
    model_name = cfg.model.model.name
    data_name = cfg.data.name

    forget_frac = _format_fraction_for_path(cfg.split.forget_frac)
    eps = _format_fraction_for_path(attack_cfg.epsilon_init)
    step_factor = _format_fraction_for_path(attack_cfg.step_factor)
    max_rounds = int(attack_cfg.max_rounds)

    cache_dir = cfg.paths.cache_dir
    relative_path = (
        f"{cache_dir}/adv_sets/amun_adv_{model_name}_{data_name}_{cfg.split.scheme}_"
        f"forget{forget_frac}_seed{cfg.seed}_eps{eps}_steps{attack_cfg.steps}_stepf{step_factor}_rounds{max_rounds}.pt"
    )
    return Path(to_absolute_path(relative_path)).resolve()


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    cfg_optim: Dict[str, Any],
    cfg_scheduler: Dict[str, Any],
    named_params: Optional[Iterable[Tuple[str, torch.nn.Parameter]]] = None,
) -> Any:
    """Build optimizer and (optionally) scheduler from cfg."""
    optimizer_name = cfg_optim.name.lower()
    scheduler_name = cfg_scheduler.name.lower()

    exclude_wd = bool(getattr(cfg_optim, "exclude_wd_norm_bias", False)) and (named_params is not None)

    if optimizer_name == "adamw" and exclude_wd:
        decay, no_decay = [], []
        for name, p in named_params:
            if not p.requires_grad:
                continue
            if name.endswith(".bias") or ".ln" in name.lower() or "layernorm" in name.lower() or "norm" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        param_groups = [
            {"params": decay, "weight_decay": float(cfg_optim.weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        wd_for_opt = 0.0
        opt_params = param_groups
    else:
        opt_params = params
        wd_for_opt = cfg_optim.weight_decay

    if optimizer_name == "sgd":
        opt = torch.optim.SGD(
            opt_params,
            lr=cfg_optim.lr,
            momentum=cfg_optim.momentum,
            weight_decay=wd_for_opt,
            nesterov=cfg_optim.nesterov,
        )
    elif optimizer_name == "adamw":
        opt = torch.optim.AdamW(
            opt_params,
            lr=cfg_optim.lr,
            weight_decay=wd_for_opt,
            betas=cfg_optim.betas,
        )

    if scheduler_name == "none":
        return opt
    elif scheduler_name == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=cfg_scheduler.step_size,
            gamma=cfg_scheduler.gamma,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
    elif scheduler_name == "multistep":
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=list(cfg_scheduler.milestones),
            gamma=cfg_scheduler.gamma,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
    elif scheduler_name == "cosine":
        t_max = int(cfg_scheduler.t_max)
        warmup_epochs = int(getattr(cfg_scheduler, "warmup_epochs", 0))
        warmup_start_lr = float(getattr(cfg_scheduler, "warmup_start_lr", 0.0))
        base_lr = float(cfg_optim.lr)
        if warmup_epochs > 0:

            def lr_lambda(epoch: int) -> float:
                """Warm up, then cosine decay learning rate."""
                if epoch < warmup_epochs:
                    if warmup_epochs == 1:
                        return 1.0
                    warmup_lr = warmup_start_lr + (epoch + 1) / warmup_epochs * (base_lr - warmup_start_lr)
                    return warmup_lr / base_lr
                e = epoch - warmup_epochs
                T = max(t_max - warmup_epochs, 1)
                return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * e / T))).item()
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def wandb_init(cfg: DictConfig, job_type: str) -> WandbLogger:
    """Initializes a Weights & Biases run and returns a WandbLogger."""
    wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        group=cfg.run.experiment_id,
        job_type=job_type,
        name=cfg.logging.name,
        tags=list(cfg.logging.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        mode="offline" if cfg.logging.offline else "online",
        dir=cfg.paths.outputs_dir,
    )
    logger = WandbLogger(experiment=wandb.run, save_dir=cfg.paths.outputs_dir)
    return logger


def wandb_log(results: Dict[str, Any], prefix: Optional[str] = None, summary: bool = False) -> None:
    """Logs a dictionary of results to the current W&B run."""
    results = {f"{prefix}/{k}": v for k, v in results.items()} if prefix is not None else results
    wandb.log(results)
    if summary:
        wandb.run.summary.update(results)


def wandb_finish() -> None:
    """Safely closes the current W&B run."""
    if wandb.run is not None:
        wandb.finish()

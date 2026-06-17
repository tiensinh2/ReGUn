import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
from omegaconf import DictConfig
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import Callback
from models import load_model
from utils import resolve_checkpoint_path
from evaluation.metrics_aggregation import average_metric_gap
from evaluation.metrics_performance import accuracy
from evaluation.metrics_privacy import calc_reference_probs, membership_inference_attacks
from evaluation.metrics_divergence import prediction_divergence, entropy


class MULEvaluator:
    """Evaluation helper that caches immutable reference metrics across epochs."""

    def __init__(
        self,
        cfg: DictConfig,
        datamodule: LightningDataModule,
    ) -> None:
        """Initialize evaluator with config, datamodule, and reference settings."""
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.retain_loader = datamodule.retain_eval_dataloader()
        self.forget_loader = datamodule.forget_eval_dataloader()
        self.eval_loader = datamodule.val_dataloader()
        self.test_loader = datamodule.test_dataloader()

        self._reference_metrics_path = Path(self.cfg.mul_eval.cache)
        self.model_retrained = load_model(self.cfg, self.cfg.run.retrain_weights)

        self._ensure_reference_metrics_cached()


    def _ensure_reference_metrics_cached(self) -> None:
        """Persist reference metrics for the default reference if missing."""
        cached: Optional[Dict[str, Any]] = None
        reference_probs: Dict[str, Any] = {}
        required_keys = ("forget", "test", "eval")
        missing_keys = list(required_keys)
        needs_retrained_metrics = True

        if self._reference_metrics_path.exists():
            cached = self._get_cached_reference_metrics()
            reference_probs = dict(cached.get("reference_probs", {}))
            missing_keys = [k for k in required_keys if k not in reference_probs]
            retrained_metrics = cached.get("model_retrained", {})
            needs_retrained_metrics = any(
                key not in retrained_metrics for key in ("acc_eval", "rmia_eval_auc")
            )
            if not missing_keys and not needs_retrained_metrics:
                print(f"[MULEvaluator] Found cached reference metrics at {self._reference_metrics_path}.")
                return
            print(f"[MULEvaluator] Updating cached reference metrics at {self._reference_metrics_path}...")
        else:
            print(f"[MULEvaluator] Caching reference metrics at {self._reference_metrics_path}...")

        self._reference_metrics_path.parent.mkdir(parents=True, exist_ok=True)

        if missing_keys:
            loader_map = {
                "forget": self.forget_loader,
                "test": self.test_loader,
                "eval": self.eval_loader,
            }
            for key in missing_keys:
                reference_probs[key] = []

            for idx in range(1, self.cfg.mul_eval.mia_num_models + 1):
                ref_name = f"reference_{idx:03d}"
                ref_path = resolve_checkpoint_path(self.cfg, retrain=True, reference_run=idx)
                if not ref_path.exists():
                    raise FileNotFoundError(f"[MULEvaluator] Reference checkpoint missing at {ref_path} (name={ref_name}).")
                model = load_model(self.cfg, str(ref_path))
                for key in missing_keys:
                    reference_probs[key].append(calc_reference_probs(model, loader_map[key], self.device))

        metrics: Dict[str, Any] = {
            "reference_probs": reference_probs,
        }
        if needs_retrained_metrics or missing_keys:
            metrics["model_retrained"] = self._build_reference_metrics(reference_probs)
        elif cached is not None:
            metrics["model_retrained"] = cached.get("model_retrained", {})

        torch.save(metrics, str(self._reference_metrics_path))

    
    def _get_cached_reference_metrics(self) -> Dict[str, Any]:
        """Load cached metrics for the default reference model."""
        return torch.load(str(self._reference_metrics_path), map_location="cpu", weights_only=False)

    def _compute_mia_metrics(
        self,
        model: LightningModule,
        reference_probs: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute MIA metrics for test and eval splits."""
        mia_results: Dict[str, float] = {}
        for suffix, split_key, loader in [
            ("", "test", self.test_loader),
            ("_eval", "eval", self.eval_loader),
        ]:
            ref_probs = {
                "forget": reference_probs["forget"],
                "test": reference_probs[split_key],
            }
            mia_results.update(
                membership_inference_attacks(
                    model=model,
                    forget_loader=self.forget_loader,
                    nonmember_loader=loader,
                    ref_probs=ref_probs,
                    device=self.device,
                    suffix=suffix,
                )
            )
        return mia_results

    def _compute_gap_metrics(
    self,
    unlearned_metrics: Dict[str, Any],
    retrained_metrics: Dict[str, Any],
    split: str,
) -> Dict[str, float]:
    acc_key = f"acc_{split}"
    rmia_acc_key = "rmia_acc" if split == "test" else "rmia_eval_acc"
    rmia_auc_key = "rmia_auc" if split == "test" else "rmia_eval_auc"
    suffix = "" if split == "test" else f"_{split}"

    unlearned_gap_metrics = {
        "acc_retain": unlearned_metrics["acc_retain"],
        acc_key: unlearned_metrics[acc_key],
        "acc_forget": unlearned_metrics["acc_forget"],
        "mia_acc": unlearned_metrics[rmia_acc_key],
        "mia_auc": unlearned_metrics[rmia_auc_key],
    }
    retrained_gap_metrics = {
        "acc_retain": retrained_metrics["acc_retain"],
        acc_key: retrained_metrics[acc_key],
        "acc_forget": retrained_metrics["acc_forget"],
        "mia_acc": retrained_metrics[rmia_acc_key],
        "mia_auc": retrained_metrics[rmia_auc_key],
    }

    # --- THÊM VÀO: tính từng gap thành phần ---
    component_gaps = {
        f"gap_retain{suffix}":     abs(unlearned_gap_metrics["acc_retain"] - retrained_gap_metrics["acc_retain"]),
        f"gap_forget{suffix}":     abs(unlearned_gap_metrics["acc_forget"] - retrained_gap_metrics["acc_forget"]),
        f"gap_{acc_key}{suffix}":  abs(unlearned_gap_metrics[acc_key]      - retrained_gap_metrics[acc_key]),
        f"gap_mia_auc{suffix}":    abs(unlearned_gap_metrics["mia_auc"]    - retrained_gap_metrics["mia_auc"]),
    }

    return {
        f"average_gap{suffix}":      average_metric_gap(unlearned_gap_metrics, retrained_gap_metrics, unlearned_gap_metrics.keys()),
        f"average_gap{suffix}_auc":  average_metric_gap(unlearned_gap_metrics, retrained_gap_metrics, unlearned_gap_metrics.keys() - {"mia_acc"}),
        f"average_gap{suffix}_test": average_metric_gap(unlearned_gap_metrics, retrained_gap_metrics, ["mia_auc", acc_key]),
        **component_gaps,  # unpack 4 metrics thành phần vào cùng dict
    }

    def run(
        self,
        model_unlearned: LightningModule,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """Run the evaluation suite on the provided model."""

        self.device = torch.device(device) if device is not None else self.device

        model_unlearned = model_unlearned.to(self.device)
        model_unlearned.eval()

        results = self._compute_metrics(model_unlearned)

        return results

    def _compute_metrics(self, model_unlearned: LightningModule) -> Dict[str, Any]:
        """Compute all metrics for an unlearned model."""

        reference_metrics = self._get_cached_reference_metrics()
        retrained_model_metrics = reference_metrics["model_retrained"]
        self.model_retrained = self.model_retrained.to(self.device)
        self.model_retrained.eval()
    
        results: Dict[str, Any] = {}
        results["acc_retain"] = accuracy(model_unlearned, self.retain_loader, device=self.device)
        results["acc_eval"] =   accuracy(model_unlearned, self.eval_loader, device=self.device)
        results["acc_test"] =   accuracy(model_unlearned, self.test_loader, device=self.device)
        results["acc_forget"] = accuracy(model_unlearned, self.forget_loader, device=self.device)

        results["forget_entropy"] = entropy(model_unlearned, self.forget_loader, device=self.device)
        results["forget_entropy_gap_retrained"] = results["forget_entropy"]  - retrained_model_metrics["forget_entropy"]
        results["divergence_retain"] =  prediction_divergence(model_unlearned, self.model_retrained, self.retain_loader, device=self.device)
        results["divergence_eval"] =    prediction_divergence(model_unlearned, self.model_retrained, self.eval_loader, device=self.device)
        results["divergence_test"] =    prediction_divergence(model_unlearned, self.model_retrained, self.test_loader, device=self.device)
        results["divergence_forget"] =  prediction_divergence(model_unlearned, self.model_retrained, self.forget_loader, device=self.device)

        mia_results = self._compute_mia_metrics(model_unlearned, reference_metrics["reference_probs"])
        results.update(mia_results)

        results.update(self._compute_gap_metrics(results, retrained_model_metrics, split="test"))
        results.update(self._compute_gap_metrics(results, retrained_model_metrics, split="eval"))

        self.model_retrained.cpu()
        return results

    def _build_reference_metrics(self, ref_probs: Dict[str, Any]) -> Dict[str, float]:
        """Compute metrics for a reference model and return them as a dictionary."""
        self.model_retrained = self.model_retrained.to(self.device)
        self.model_retrained.eval()
        metrics: Dict[str, float] = {}
        metrics["acc_retain"] = accuracy(self.model_retrained, self.retain_loader, device=self.device)
        metrics["acc_eval"] = accuracy(self.model_retrained, self.eval_loader, device=self.device)
        metrics["acc_test"] = accuracy(self.model_retrained, self.test_loader, device=self.device)
        metrics["acc_forget"] = accuracy(self.model_retrained, self.forget_loader, device=self.device)
        metrics["forget_entropy"] = entropy(self.model_retrained, self.forget_loader, device=self.device)
        mia_results = self._compute_mia_metrics(self.model_retrained, ref_probs)
        metrics.update(mia_results)
        self.model_retrained.cpu()
        return metrics




class UnlearningEpochEvaluationCallback(Callback):
    """Lightning callback to run full MUL evaluation at the end of each unlearning epoch."""

    def __init__(
        self,
        evaluator: MULEvaluator,
        prefix: str = "unlearning",
    ) -> None:
        """Store evaluator reference and logging prefix."""
        super().__init__()
        self.evaluator = evaluator
        self.prefix = prefix

    def on_train_epoch_end(self, trainer: Any, pl_module: LightningModule) -> None:
        """Trigger evaluation at the end of each training epoch."""

        current_epoch = trainer.current_epoch + 1
        model_unlearned = getattr(pl_module, "model", pl_module)
        device = pl_module.device

        results = self.evaluator.run(
            model_unlearned=model_unlearned,
            device=device,
        )
        pl_module.train()
        model_unlearned.train()

        metrics = {f"{self.prefix}/{k}": v for k, v in results.items()}
        metrics[f"{self.prefix}/epoch"] = current_epoch

        trainer.logger.log_metrics(metrics, step=trainer.global_step)

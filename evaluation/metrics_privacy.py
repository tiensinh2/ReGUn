from typing import Dict, Sequence, Tuple
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader



def calc_reference_probs(
    model: LightningModule,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Compute true-label probabilities for RMIA on a loader."""
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            idx = torch.arange(len(y), device=y.device)
            true_probs = probs[idx, y] # p(y|x)
            all_probs.append(true_probs.cpu().numpy())

    return np.concatenate(all_probs)


def _extract_features(
    model: LightningModule,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Extract logits-derived features for MIA scoring."""

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    losses = F.cross_entropy(torch.tensor(logits), torch.tensor(labels), reduction="none").numpy()
    confidence = probs[np.arange(len(labels)), labels]
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1)), axis=1)

    reverse_probs = 1 - probs
    modified_probs = probs.copy()
    modified_probs[np.arange(len(labels)), labels] = reverse_probs[np.arange(len(labels)), labels]
    modified_entropy = -np.sum(modified_probs * np.log(np.clip(modified_probs, 1e-12, 1)), axis=1)

    predictions = np.argmax(logits, axis=1)
    correctness = (predictions == labels).astype(float)

    return {
        "loss": losses,
        "confidence": confidence,
        "entropy": entropy,
        "modified_entropy": modified_entropy,
        "correctness": correctness,
        "probs": probs,
        "logits": logits,
    }


def _mia(
    name: str,
    scores_forget: np.ndarray,
    scores_test: np.ndarray,
) -> Dict[str, float]:
    """Compute accuracy and AUC for the attack-defined scores."""
    y_true = np.concatenate([np.ones(len(scores_forget)), np.zeros(len(scores_test))])
    scores = np.concatenate([scores_forget, scores_test])

    finite_mask = np.isfinite(scores)
    if not np.all(finite_mask):
        warnings.warn(f"[MIA] Non-finite scores in {name}; filtering before ROC metrics.")
        y_true = y_true[finite_mask]
        scores = scores[finite_mask]

    if scores.size == 0 or np.unique(y_true).size < 2:
        warnings.warn(f"[MIA] Insufficient data for ROC in {name}; returning NaNs.")
        return {
            f"{name}_acc": float("nan"),
            f"{name}_auc": float("nan"),
            f"{name}_jmax_abs": float("nan"),
        }

    auc = roc_auc_score(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    jmax_abs = np.max(np.abs(j))
    best_idx = int(np.argmax(j))
    threshold = thresholds[best_idx]
    acc = np.mean((scores >= threshold) == y_true)

    return {
        f"{name}_acc": float(acc),
        f"{name}_auc": float(auc),
        f"{name}_jmax_abs": float(jmax_abs), #"attack advantage" (max abs diff TPR-FPR)
    }

def _robust_mia_scores(
    target_probs_forget: np.ndarray,              # (N_f,)
    target_probs_test: np.ndarray,                # (N_t,)
    ref_probs_forget_list: Sequence[np.ndarray],  # list of (N_f,)
    ref_probs_test_list: Sequence[np.ndarray],    # list of (N_t,)
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RMIA-style log-likelihood ratios for forget/test splits."""

    eps = 1e-12

    ref_forget = np.vstack(ref_probs_forget_list)
    ref_test   = np.vstack(ref_probs_test_list)

    def log_lr_split(target_probs: np.ndarray, ref_probs: np.ndarray) -> np.ndarray:
        """Compute per-sample log-likelihood ratios for a split."""
        # mean_k p_ref_k(y|x)
        ref_mean = ref_probs.mean(axis=0)  # (N,)

        # log q(x) = log p_target(y|x) - log(mean_k p_ref_k(y|x))
        log_q = np.log(np.clip(target_probs, eps, 1.0)) - np.log(
            np.clip(ref_mean, eps, 1.0)
        )
        return log_q

    log_q_forget = log_lr_split(target_probs_forget, ref_forget)
    log_q_test   = log_lr_split(target_probs_test,   ref_test)

    return log_q_forget, log_q_test

def _with_suffix(name: str, suffix: str) -> str:
    """Append a suffix to metric names when provided."""
    return f"{name}{suffix}" if suffix else name


def membership_inference_attacks(
    model: LightningModule,
    forget_loader: DataLoader,
    nonmember_loader: DataLoader,
    ref_probs: Dict[str, Sequence[np.ndarray]],
    device: torch.device,
    suffix: str = "",
) -> Dict[str, float]:
    """Collect per-example features and execute all registered attacks."""

    # Simple MIAs
    forget_features = _extract_features(model, forget_loader, device)
    nonmember_features = _extract_features(model, nonmember_loader, device)
    results: Dict[str, float] = {}
    results.update(_mia(_with_suffix("smia_loss", suffix), -forget_features["loss"], -nonmember_features["loss"]))
    results.update(_mia(_with_suffix("smia_confidence", suffix), forget_features["confidence"], nonmember_features["confidence"]))
    results.update(_mia(_with_suffix("smia_entropy", suffix), -forget_features["entropy"], -nonmember_features["entropy"]))
    results.update(_mia(_with_suffix("smia_modified_entropy", suffix), -forget_features["modified_entropy"], -nonmember_features["modified_entropy"]))
    results.update(_mia(_with_suffix("smia_correctness", suffix), forget_features["correctness"], nonmember_features["correctness"]))

    # RMIA
    rmia_forget, rmia_test = _robust_mia_scores(
        target_probs_forget=forget_features["confidence"] ,
        target_probs_test=nonmember_features["confidence"],
        ref_probs_forget_list=ref_probs["forget"],
        ref_probs_test_list=ref_probs["test"],
    )
    results.update(_mia(_with_suffix("rmia", suffix), rmia_forget, rmia_test))

    return results

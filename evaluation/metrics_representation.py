"""Sample-level representation analysis.

Adds three capabilities on top of the existing metrics_*.py suite:
  1. Extract a penultimate-layer embedding for every sample in a split (for
     representation-space analyses, e.g. neighbor selection, clustering).
  2. Find samples whose predicted class flips between two models (typically
     the retrain-from-scratch model and a post-unlearning model).
  3. For flipped samples, measure how far their representation moved between
     the two models (L2 distance / cosine similarity of normalized features).

Reuses the same penultimate-feature hook technique already used in regun.py
(_find_classifier_layer / _forward_with_features), generalized to any model.

Usage sketch:

    from evaluation.metrics_representation import (
        extract_all_splits, compare_models_on_split, flips_to_dataframe, summarize_flips,
    )

    device = torch.device("cuda")
    forget_loader = dm.forget_eval_dataloader()   # MUST be shuffle=False
    flip_result = compare_models_on_split(
        model_retrained, model_unlearned, forget_loader, device,
        label_a="retrain", label_b="unlearned",
    )
    print(summarize_flips(flip_result))
    df = flips_to_dataframe(flip_result, split_name="forget")
    df[df.flipped].sort_values("embed_l2_dist", ascending=False).head(20)

IMPORTANT ASSUMPTION: any dataloader passed to `extract_embeddings` /
`compare_models_on_split` must be non-shuffling (e.g. the `*_eval_dataloader()`
methods already used in evaluator.py / regun.py), otherwise the two models'
outputs won't be sample-aligned and the comparison is meaningless.
"""
from typing import Any, Dict, Iterable, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm


def find_classifier_layer(model: nn.Module) -> nn.Module:
    """Locate the final linear classifier layer (same heuristic ReGUn uses)."""
    layer = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == model.num_classes:
            layer = m
    if layer is None:
        raise ValueError("Classifier layer not found for feature extraction.")
    return layer


def get_dataset_indices(dataloader: Any) -> np.ndarray:
    """Return original dataset indices, in dataloader iteration order.

    Unwraps nested `Subset`s the same way ReGUn's `_unwrap_subset` does, so
    flipped/high-distance samples can be traced back to the original dataset
    (e.g. for the per-example / cluster-based neighbor-selection direction).
    Only meaningful if `dataloader` has `shuffle=False`.
    """
    ds = dataloader.dataset
    indices = None
    while isinstance(ds, Subset):
        cur = ds.indices
        indices = cur if indices is None else [cur[i] for i in indices]
        ds = ds.dataset
    if indices is None:
        indices = list(range(len(ds)))
    return np.asarray(indices)


def as_deterministic(dataloader: DataLoader) -> DataLoader:
    """Return a shuffle=False view of `dataloader`, rebuilding it if necessary.

    Necessary because cifar.py's `heldout_eval_dataloader()` is shuffle=True
    despite the `_eval_` name (unlike every other `*_eval_dataloader()`), which
    would otherwise misalign embeddings against the dataset indices returned by
    `get_dataset_indices` (which follow Subset order, not iteration order).
    Same technique as ReGUn's `_build_cache_loader`.
    """
    sampler = getattr(dataloader, "sampler", None)
    is_shuffled = isinstance(sampler, RandomSampler)
    if not is_shuffled:
        return dataloader
    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        persistent_workers=dataloader.persistent_workers,
        prefetch_factor=dataloader.prefetch_factor,
    )


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    dataloader: Iterable,
    device: torch.device,
    cls_layer: Optional[nn.Module] = None,
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """Extract penultimate embeddings, logits, predictions and labels for one dataloader."""
    model = model.to(device)
    was_training = model.training
    model.eval()

    dataloader = as_deterministic(dataloader)

    if cls_layer is None:
        cls_layer = find_classifier_layer(model)

    buf = []

    def _hook(_m, inp, _out):
        buf.append(inp[0].detach())

    handle = cls_layer.register_forward_hook(_hook)

    embeds, logits_all, labels_all = [], [], []
    try:
        for x, y in tqdm(dataloader, desc="Extracting embeddings", leave=False):
            x = x.to(device)
            buf.clear()
            logits = model(x)
            feat = buf[0]
            if normalize:
                feat = F.normalize(feat, dim=1)
            embeds.append(feat.cpu().numpy())
            logits_all.append(logits.detach().cpu().numpy())
            labels_all.append(y.numpy())
    finally:
        handle.remove()
        if was_training:
            model.train()

    embeddings = np.concatenate(embeds, axis=0)
    logits = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)

    return {
        "embeddings": embeddings,
        "logits": logits,
        "preds": logits.argmax(axis=1),
        "labels": labels,
        "indices": get_dataset_indices(dataloader),
    }


def extract_all_splits(
    model: nn.Module,
    dm: Any,
    device: torch.device,
    splits: Optional[Dict[str, Iterable]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract embeddings for every standard split (retain/forget/heldout/val/test).

    Pass `splits` explicitly if your datamodule uses different accessor names.
    Shuffled loaders (notably `heldout_eval_dataloader()`) are converted to
    deterministic ones internally, so embeddings stay aligned with `indices`.
    """
    if splits is None:
        splits = {
            "retain": dm.retain_eval_dataloader(),
            "forget": dm.forget_eval_dataloader(),
            "heldout": dm.heldout_eval_dataloader(),
            "val": dm.val_dataloader(),
            "test": dm.test_dataloader(),
        }
    cls_layer = find_classifier_layer(model)
    return {
        name: extract_embeddings(model, loader, device, cls_layer=cls_layer)
        for name, loader in splits.items()
    }


def save_embeddings(results: Dict[str, Dict[str, np.ndarray]], path: str) -> None:
    """Persist `extract_all_splits` output to a single .npz file (Kaggle-session-friendly cache)."""
    flat = {}
    for split, d in results.items():
        for key, arr in d.items():
            flat[f"{split}__{key}"] = arr
    np.savez_compressed(path, **flat)


def load_embeddings(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load embeddings saved by `save_embeddings`."""
    data = np.load(path)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for key in data.files:
        split, field = key.split("__", 1)
        out.setdefault(split, {})[field] = data[key]
    return out


def compare_models_on_split(
    model_a: nn.Module,
    model_b: nn.Module,
    dataloader: Iterable,
    device: torch.device,
    label_a: str = "model_a",
    label_b: str = "model_b",
) -> Dict[str, np.ndarray]:
    """Compare per-sample predictions and representations of two models on one split.

    Typical call: model_a = retrain-from-scratch model, model_b = unlearned model.
    `dataloader` MUST be non-shuffling so the two passes are sample-aligned.
    """
    out_a = extract_embeddings(model_a, dataloader, device, normalize=True)
    out_b = extract_embeddings(model_b, dataloader, device, normalize=True)

    if not np.array_equal(out_a["labels"], out_b["labels"]):
        raise ValueError(
            "Label mismatch between the two passes -- make sure `dataloader` has "
            "shuffle=False so both models see samples in the same order."
        )

    flipped = out_a["preds"] != out_b["preds"]
    embed_l2 = np.linalg.norm(out_a["embeddings"] - out_b["embeddings"], axis=1)
    embed_cos = np.sum(out_a["embeddings"] * out_b["embeddings"], axis=1)  # already normalized

    return {
        "indices": out_a["indices"],
        "labels": out_a["labels"],
        f"preds_{label_a}": out_a["preds"],
        f"preds_{label_b}": out_b["preds"],
        "flipped": flipped,
        "embed_l2_dist": embed_l2,
        "embed_cos_sim": embed_cos,
    }


def summarize_flips(flip_result: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Aggregate stats: flip rate, and whether flipped samples moved further in
    representation space than stable ones (expected if the flip reflects a
    genuine representation shift rather than a boundary-line coin-flip)."""
    flipped = flip_result["flipped"]
    dist = flip_result["embed_l2_dist"]
    return {
        "n_total": int(len(flipped)),
        "n_flipped": int(flipped.sum()),
        "flip_rate": float(flipped.mean()) if len(flipped) else float("nan"),
        "mean_dist_flipped": float(dist[flipped].mean()) if flipped.any() else float("nan"),
        "mean_dist_stable": float(dist[~flipped].mean()) if (~flipped).any() else float("nan"),
    }


def flips_to_dataframe(flip_result: Dict[str, np.ndarray], split_name: str = ""):
    """Convert `compare_models_on_split` output into a pandas DataFrame for inspection."""
    import pandas as pd

    pred_keys = [k for k in flip_result if k.startswith("preds_")]
    data = {
        "dataset_index": flip_result["indices"],
        "split": split_name,
        "label": flip_result["labels"],
        "flipped": flip_result["flipped"],
        "embed_l2_dist": flip_result["embed_l2_dist"],
        "embed_cos_sim": flip_result["embed_cos_sim"],
    }
    for k in pred_keys:
        data[k] = flip_result[k]
    return pd.DataFrame(data)

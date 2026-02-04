import random
from dataclasses import dataclass
from typing import List


@dataclass
class SplitIdx:
    """Index splits for retain/forget/held-out/val."""
    retain: List[int]
    forget: List[int]
    heldout: List[int]
    val: List[int]


def split_random(indices: List[int], seed: int) -> List[int]:
    """Return a shuffled copy of indices."""
    rng = random.Random(seed)
    idx = indices[:]
    rng.shuffle(idx)
    return idx


def stratified_split(
    labels: List[int],
    seed: int,
    val_frac: float,
    retain_frac: float,
    forget_frac: float,
    heldout_frac: float,
) -> SplitIdx:
    """Stratify by class and split into heldout/val/forget/retain (heldout first)."""
    per_class = {}
    for i, y in enumerate(labels):
        per_class.setdefault(y, []).append(i)
    rng = random.Random(seed)
    for c in per_class:
        rng.shuffle(per_class[c])

    val, retain, forget, held = [], [], [], []
    for _, idxs in per_class.items():
        n = len(idxs)
        # heldout taken first from total
        n_held = int(round(heldout_frac * n))
        held += idxs[:n_held]
        rem = idxs[n_held:]
        n_rem = len(rem)
        # val/forget/retain applied to remainder
        n_val = int(round(val_frac * n_rem))
        n_forget = int(round(forget_frac * n_rem))
        n_retain = int(round(retain_frac * n_rem))
        # trim if needed
        take = n_val + n_forget + n_retain
        if take > n_rem:
            n_retain = max(0, n_retain - (take - n_rem))
        cur = 0
        val += rem[cur:cur + n_val]; cur += n_val
        forget += rem[cur:cur + n_forget]; cur += n_forget
        retain += rem[cur:cur + n_retain]; cur += n_retain
        if cur < n_rem:
            retain += rem[cur:]
    return SplitIdx(retain=retain, forget=forget, heldout=held, val=val)


def stratified_split_random(
    labels: List[int],
    seed: int,
    val_frac: float,
    retain_frac: float,
    forget_frac: float,
    heldout_frac: float,
) -> SplitIdx:
    """Split by fractions without using class labels (heldout first)."""
    all_idx = list(range(len(labels)))
    all_idx = split_random(all_idx, seed)
    n = len(all_idx)
    # heldout taken FIRST from total
    n_held = int(round(heldout_frac * n))
    held = all_idx[:n_held]
    rem = all_idx[n_held:]
    n_rem = len(rem)
    # val/forget/retain applied to remainder
    n_val = int(round(val_frac * n_rem))
    n_forget = int(round(forget_frac * n_rem))
    n_retain = int(round(retain_frac * n_rem))
    take = n_val + n_forget + n_retain
    if take > n_rem:
        n_retain = max(0, n_retain - (take - n_rem))
    cur = 0
    val_idx = rem[cur:cur + n_val]; cur += n_val
    forget = rem[cur:cur + n_forget]; cur += n_forget
    retain = rem[cur:cur + n_retain]; cur += n_retain
    if cur < n_rem:
        retain += rem[cur:]
    return SplitIdx(retain=retain, forget=forget, heldout=held, val=val_idx)


def class_split(
    labels: List[int],
    seed: int,
    val_frac: float,
    retain_frac: float,
    heldout_frac: float,
    forget_classes: List[int],
    forget_frac: float,
) -> SplitIdx:
    """Forget a fraction of selected classes, then stratify the rest (heldout first)."""
    per_class = {}
    for i, y in enumerate(labels):
        per_class.setdefault(y, []).append(i)

    rng = random.Random(seed)
    # heldout taken FIRST from total (stratified)
    held, post_heldout = [], {}
    for cls, idxs in per_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_held = int(round(heldout_frac * len(idxs)))
        held += idxs[:n_held]
        post_heldout[cls] = idxs[n_held:]

    # For each forget class move forget_frac samples to forget
    forget, remaining = [], []
    for cls, idxs in post_heldout.items():
        if cls in forget_classes:
            n_forget = int(round(forget_frac * len(idxs)))
            n_forget = min(len(idxs), max(0, n_forget))
            forget += idxs[:n_forget]
            remaining += idxs[n_forget:]
        else:
            remaining += idxs

    # Stratified split over remaining (val + retain only)
    rem_labels = [labels[i] for i in remaining]
    tmp = stratified_split(
        labels=rem_labels,
        seed=seed,
        val_frac=val_frac,
        retain_frac=retain_frac,
        forget_frac=0.0,
        heldout_frac=0.0,
    )
    retain = [remaining[i] for i in tmp.retain]
    val = [remaining[i] for i in tmp.val]

    return SplitIdx(retain=retain, forget=forget, heldout=held, val=val)

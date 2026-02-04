from typing import Dict, Sequence
import numpy as np


def average_metric_gap(
    metrics_unlearned: Dict[str, float],
    metrics_retrained: Dict[str, float],
    metric_keys: Sequence[str],
) -> float:
    """Average absolute gap between two metric dictionaries over selected keys."""
    gaps = []
    for key in metric_keys:
        gaps.append(abs(metrics_unlearned[key] - metrics_retrained[key]))
    return float(np.mean(gaps))

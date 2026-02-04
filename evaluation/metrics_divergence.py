import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader

def prediction_divergence(
    model_unlearned: LightningModule,
    model_retrained: LightningModule,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Jensen-Shannon divergence between predictions of unlearned and retrained models."""
    model_unlearned = model_unlearned.to(device)
    model_unlearned.eval()
    model_retrained = model_retrained.to(device)
    model_retrained.eval()

    js_divs = []
    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            x = x.to(device)
            logits_unl = model_unlearned(x)
            logits_ret = model_retrained(x)
            probs_unl = F.softmax(logits_unl, dim=1)
            probs_ret = F.softmax(logits_ret, dim=1)
            m = 0.5 * (probs_unl + probs_ret)
            kl1 = F.kl_div(m.log(), probs_unl, reduction="none").sum(dim=1)
            kl2 = F.kl_div(m.log(), probs_ret, reduction="none").sum(dim=1)
            js = 0.5 * (kl1 + kl2)
            js_divs.extend(js.cpu().numpy())

    return float(np.mean(js_divs))


def entropy(
    model: LightningModule,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Average predictive entropy on the provided dataloader."""
    model = model.to(device)
    model.eval()
    entropies = []

    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            entropies.extend(entropy.cpu().numpy())

    return float(np.mean(entropies))

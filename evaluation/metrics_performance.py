import torch
from lightning.pytorch import LightningModule
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader


def accuracy(
    model: LightningModule,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Compute classification accuracy for a model on the provided dataloader."""
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    return accuracy_score(all_labels, all_preds)

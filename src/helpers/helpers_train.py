import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim

from constants import LABEL
from .constants import LABEL_MAP


def compute_class_weights(meta, label_map, device):
    """Compute class weights for imbalanced datasets."""
    counts = (
        meta[LABEL]
        .map(label_map)
        .value_counts()
        .reindex([0, 1, 2])
        .fillna(0)
        .values.astype(float)
    )
    weights = np.where(counts > 0, 1.0 / (counts + 1e-12), 0.0)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def _get_criterion(train_meta, device):
    """Return weighted CrossEntropyLoss.
    @TODO: Extend this function to support configurable loss functions such as FocalLoss, LabelSmoothingCrossEntropy
    """
    weights = compute_class_weights(train_meta, LABEL_MAP, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    return criterion


def _get_optimizer(model, lr):
    """Return Adam Optimizer.
    @TODO: Add flexibility to choose between different optimizers such as Adagrad, RMSprop etc
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer


def compute_val_metrics_and_loss(model, loader, device, criterion):
    """
    Run model on loader and compute:
      ys, preds, probs, val_loss (average), val_accuracy
    """
    model.eval()
    ys, preds, probs = [], [], []
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device).float()
            y = y.to(device).long()
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            loss = criterion(logits, y)
            batch_n = x.size(0)
            total_loss += loss.item() * batch_n
            total_samples += batch_n
            p = torch.argmax(prob, dim=1).cpu().numpy()
            pr = prob.cpu().numpy()
            preds.extend(p.tolist())
            ys.extend(y.cpu().numpy().tolist())
            probs.extend(pr.tolist())
    val_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
    val_acc = accuracy_score(ys, preds) if len(ys) > 0 else 0.0
    return ys, preds, probs, float(val_loss), float(val_acc)

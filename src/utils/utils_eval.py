import json
import math
from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import itertools
import warnings

from helpers.constants import LABELS, LABEL_IDX

# Expected Calibration Error
def compute_ece_per_class(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Compute per-class Expected Calibration Error (ECE) and overall ECE.
    probs: shape [N, C]
    y_true: shape [N] ints in [0..C-1]
    Returns dict: {"per_class": {class_idx: ece}, "overall": ece}
    """
    if probs is None or len(probs) == 0:
        return {"per_class": {}, "overall": float("nan")}
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    N, C = probs.shape
    ece_per = {}
    # overall: use max probabilities and correctness
    probabilities = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    overall_ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        mask = (probabilities > bins[i]) & (probabilities <= bins[i+1])
        if not mask.any():
            continue
        acc = (preds[mask] == y_true[mask]).mean()
        probability = probabilities[mask].mean()
        overall_ece += (mask.sum() / N) * abs(acc - probability)
    # per-class ECE (one-vs-rest)
    for c in range(C):
        prob_c = probs[:, c]
        true_bin = (y_true == c).astype(int)
        ece_c = 0.0
        for i in range(n_bins):
            mask = (prob_c > bins[i]) & (prob_c <= bins[i+1])
            if not mask.any():
                continue
            acc = (true_bin[mask] == 1).mean()
            probability = prob_c[mask].mean()
            ece_c += (mask.sum() / N) * abs(acc - probability)
        ece_per[c] = float(ece_c)
    return {"per_class": ece_per, "overall": float(overall_ece)}

# ---------- confusion matrix plotting ----------
def save_confusion_matrix(y_true: List[int], y_pred: List[int], classes: List[str], out_path: str):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes, ylabel="True label", xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2. if cm.size else 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

# ---------- PR curves (one-vs-rest) ----------
def save_pr_curves(probs: List[List[float]], y_true: List[int], classes: List[str], out_path: str):
    """
    Saves a PR curve figure for each class (one-vs-rest) as a single PNG.
    """
    if len(probs) == 0:
        return None
    probs = np.asarray(probs)
    y_true_arr = np.array(y_true)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, lab in enumerate(classes):
        true_bin = (y_true_arr == i).astype(int)
        try:
            precision, recall, _ = precision_recall_curve(true_bin, probs[:, i])
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, label=f"{lab} (AUC={pr_auc:.3f})")
        except Exception:
            continue
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

# ---------- main metrics aggregator ----------
def compute_metrics(y_true: List[int], y_pred: List[int], probs: Optional[List[List[float]]] = None) -> Dict:
    out = {}
    if len(y_true) == 0:
        return {"error": "no samples"}
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0)
    out["per_class"] = {}
    for i, lab in enumerate(LABELS):
        out["per_class"][lab] = {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(sup[i])}
    out["macro_f1"] = float(f1_score(y_true, y_pred, average='macro'))
    out["micro_f1"] = float(f1_score(y_true, y_pred, average='micro'))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    # ECE and PR AUCs if probs provided
    if probs is not None and len(probs) == len(y_true):
        try:
            ece_summary = compute_ece_per_class(probs, y_true)
            out["ece"] = ece_summary["overall"]
        except Exception:
            out["ece"] = None
        # per-class ROC AUC if possible
        try:
            pr_aucs = {}
            probs_arr = np.array(probs)
            y_true_arr = np.array(y_true)
            for i, lab in enumerate(LABELS):
                try:
                    precision, recall, _ = precision_recall_curve((y_true_arr == i).astype(int), probs_arr[:, i])
                    pr_aucs[lab] = float(auc(recall, precision))
                except Exception:
                    pr_aucs[lab] = None
            out["pr_auc_per_class"] = pr_aucs
        except Exception:
            out["pr_auc_per_class"] = None
    return out

# ---------- evaluate and optionally log to tracker ----------
def evaluate_and_log(y_true: List[int],
                     y_pred: List[int],
                     probs: Optional[List[List[float]]] = None,
                     out_dir: Optional[str] = None,
                     report_name: str = "eval_report.json",
                     tracker = None) -> Dict:
    """
    Compute metrics, save JSON report and plots to out_dir, and optionally log them to tracker.
    tracker: an object implementing .log_metric(name, value, step), .log_artifact(path, name)
             (i.e. Tracker from src/tracker.py)
    Returns metrics dict.
    """
    metrics = compute_metrics(y_true, y_pred, probs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # write JSON
        report_path = os.path.join(out_dir, report_name)
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        # confusion matrix
        try:
            cm_path = os.path.join(out_dir, "confusion.png")
            save_confusion_matrix(y_true, y_pred, classes=LABELS, out_path=cm_path)
            if tracker:
                tracker.log_artifact(cm_path, name="confusion.png")
        except Exception as e:
            warnings.warn(f"Failed produce confusion matrix: {e}")
        # PR curves
        if probs is not None:
            try:
                pr_path = os.path.join(out_dir, "pr_curves.png")
                save_pr_curves(probs, y_true, classes=LABELS, out_path=pr_path)
                if tracker:
                    tracker.log_artifact(pr_path, name="pr_curves.png")
            except Exception as e:
                warnings.warn(f"Failed produce PR curves: {e}")
        # save report artifact
        if tracker:
            tracker.log_artifact(report_path, name=report_name)
    # log scalar metrics to tracker
    if tracker:
        try:
            # top-level scalars
            if "macro_f1" in metrics:
                tracker.log_metric("macro_f1", metrics["macro_f1"])
            if "accuracy" in metrics:
                tracker.log_metric("accuracy", metrics["accuracy"])
            # per-class f1
            for lab, v in metrics.get("per_class", {}).items():
                tracker.log_metric(f"f1_{lab}", v.get("f1", 0.0))
            # ece
            if "ece" in metrics and metrics["ece"] is not None:
                tracker.log_metric("ece", metrics["ece"])
        except Exception:
            pass
    return metrics


# ---------------------------
# utils for matching windows to segments
# ---------------------------
def interval_intersection(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return intersection duration between two intervals (seconds)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def weighted_average_probs_for_segment(segment: Dict[str, Any], windows: List[Dict[str, Any]]) -> List[float]:
    """
    Given a groundtruth segment dict {'start_seconds','end_seconds'} and a list of predicted windows:
      windows: list of {"start","end","probs":[...]}
    Compute overlap-weighted average of probs. If no overlapping windows found, pick nearest window
    (closest center) and return its probs.
    """
    s0 = float(segment["start_seconds"])
    s1 = float(segment["end_seconds"])
    weights = []
    probs_list = []
    for w in windows:
        w0 = float(w["start_seconds"])
        w1 = float(w["end_seconds"])
        inter = interval_intersection(s0, s1, w0, w1)
        if inter > 0:
            weights.append(inter)
            probs_list.append(np.array(w["probabilities"], dtype=float))
    if len(weights) == 0:
        # fallback: pick nearest window by center distance
        if len(windows) == 0:
            # no windows at all: return uniform tiny vector
            return [1.0/len(LABELS)] * len(LABELS)
        seg_center = 0.5 * (s0 + s1)
        dists = [abs((0.5*(w["start_seconds"]+w["end_seconds"])) - seg_center) for w in windows]
        idx = int(np.argmin(dists))
        return list(np.array(windows[idx]["probabilities"], dtype=float))
    weights = np.array(weights)
    probs_stack = np.stack(probs_list, axis=0)  # [K, C]
    # normalize weights
    weights = weights / (weights.sum() + 1e-12)
    avg = (weights[:, None] * probs_stack).sum(axis=0)
    # normalize in case of numerical issues
    s = float(max(avg.sum(), 1e-12))
    avg = avg / s
    return avg.tolist()
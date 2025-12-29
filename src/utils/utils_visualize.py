import csv
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import warnings

from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

from constants import (
    PR_AUC,
    ACCURACY,
    MACRO_F1,
    MICRO_F1,
    F1,
    PRECISION,
    RECALL,
    SUPPORT,
    ACCURACY,
    PER_CLASS,
    ECE,
    OVERALL,
    METRICS,
    SUMMARY,
)
from .common import _safe_mkdir, _num
from .utils_eval import compute_ece_per_class


def generate_pr_and_calibration(
    y_true: List[int],
    probs: List[List[float]],
    classes: List[str],
    out_dir: str = "artifacts/eval_plots",
    prefix: str = "pr_cal",
    n_bins: int = 10,
    tracker=None,
) -> Dict[str, Any]:
    """Generate PR curves and calibration diagrams from per-sample predictions."""
    _safe_mkdir(out_dir)
    results: Dict[str, Any] = {}
    try:
        probs_arr = np.asarray(probs, dtype=float)
        y_arr = np.asarray(y_true, dtype=int)
    except Exception as e:
        raise ValueError(f"Invalid probs/y_true: {e}")
    if probs_arr.ndim != 2:
        raise ValueError("probs must be 2D array-like [N, C]")
    N, C = probs_arr.shape
    if len(classes) != C:
        warnings.warn(
            f"classes length {len(classes)} != probs shape {C}; adjusting classes to indices"
        )
        classes = [str(i) for i in range(C)]

    # --- PR curves + PR AUC per class ---
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        pr_aucs = {}
        for i, cname in enumerate(classes):
            y_bin = (y_arr == i).astype(int)
            # skip if no positive examples
            if y_bin.sum() == 0:
                pr_aucs[cname] = float("nan")
                continue
            precision, recall, _ = precision_recall_curve(y_bin, probs_arr[:, i])
            pr_auc = auc(recall, precision)
            pr_aucs[cname] = float(pr_auc)
            ax.plot(recall, precision, label=f"{cname} (AUC={pr_auc:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall (one-vs-rest)")
        ax.legend(loc="best")
        path_pr = os.path.join(out_dir, f"{prefix}_pr_curves.png")
        fig.tight_layout()
        fig.savefig(path_pr)
        plt.close(fig)
        results["pr_curves_png"] = path_pr
        results[f"{PR_AUC}_{PER_CLASS}"] = pr_aucs
        if tracker:
            try:
                tracker.log_artifact(path_pr, name=os.path.basename(path_pr))
            except Exception:
                pass
    except Exception as e:
        warnings.warn(f"Failed PR curves: {e}")

    # --- Reliability diagram (per-class calibration curve) ---
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, cname in enumerate(classes):
            y_bin = (y_arr == i).astype(int)
            if y_bin.sum() == 0:
                continue
            prob_pos = probs_arr[:, i]
            frac_pos, mean_pred = calibration_curve(
                y_bin, prob_pos, n_bins=n_bins, strategy="uniform"
            )
            ax.plot(mean_pred, frac_pos, marker="o", label=cname)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Reliability diagram (per-class)")
        ax.legend(loc="best")
        path_rel = os.path.join(out_dir, f"{prefix}_reliability.png")
        fig.tight_layout()
        fig.savefig(path_rel)
        plt.close(fig)
        results["reliability_png"] = path_rel
        if tracker:
            try:
                tracker.log_artifact(path_rel, name=os.path.basename(path_rel))
            except Exception:
                pass
    except Exception as e:
        warnings.warn(f"Failed reliability diagram: {e}")

    # --- Calibration histogram: per-class predicted probabilities distribution ---
    try:
        rows = math.ceil(C / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))
        axes = axes.flatten()
        for i, cname in enumerate(classes):
            axes[i].hist(probs_arr[:, i], bins=20, range=(0.0, 1.0))
            axes[i].set_title(f"Predicted prob: {cname}")
            axes[i].set_xlabel("Probability")
            axes[i].set_ylabel("Count")
        for j in range(C, len(axes)):
            axes[j].axis("off")
        path_hist = os.path.join(out_dir, f"{prefix}_calibration_hist.png")
        fig.tight_layout()
        fig.savefig(path_hist)
        plt.close(fig)
        results["calibration_hist_png"] = path_hist
        if tracker:
            try:
                tracker.log_artifact(path_hist, name=os.path.basename(path_hist))
            except Exception:
                pass
    except Exception as e:
        warnings.warn(f"Failed calibration hist: {e}")

    # --- Compute ECE per-class and overall ---
    try:
        ece_info = compute_ece_per_class(probs_arr, y_arr, n_bins=n_bins)
        # map class indices to names
        ece_named = {PER_CLASS: {}, OVERALL: ece_info.get(OVERALL, float("nan"))}
        for idx, v in ece_info.get(PER_CLASS, {}).items():
            name = classes[int(idx)] if int(idx) < len(classes) else str(idx)
            ece_named[PER_CLASS][name] = float(v)
        results[ECE] = ece_named
    except Exception as e:
        warnings.warn(f"Failed compute ECE: {e}")
        results[ECE] = {PER_CLASS: {}, OVERALL: float("nan")}

    return results


def generate_plots_from_summary(
    metrics_summary: Dict[str, Any],
    out_dir: str = "artifacts/eval_plots",
    prefix: str = "summary",
    tracker=None,
) -> Dict[str, str]:
    """
    Generate summary-level plots from a metrics dictionary
    """

    _safe_mkdir(out_dir)
    saved = {}

    # Normalize input
    ms = metrics_summary or {}
    per_class = ms.get(PER_CLASS, {})
    classes = sorted(list(per_class.keys()))
    if not classes:
        raise ValueError("metrics_summary missing 'per_class' keys")

    # Extract arrays
    precisions = [_num(per_class[c].get(PRECISION, float("nan"))) for c in classes]
    recalls = [_num(per_class[c].get(RECALL, float("nan"))) for c in classes]
    f1s = [_num(per_class[c].get(F1, float("nan"))) for c in classes]
    supports = [int(per_class[c].get(SUPPORT, 0)) for c in classes]

    # PR AUC per class (may be missing)
    pr_aucs_map = ms.get(f"{PR_AUC}_{PER_CLASS}", {})
    pr_aucs = [_num(pr_aucs_map.get(c, float("nan"))) for c in classes]

    # Top-level metrics
    macro_f1 = _num(ms.get(MACRO_F1, float("nan")))
    micro_f1 = _num(ms.get(MICRO_F1, float("nan")))
    accuracy = _num(ms.get(ACCURACY, float("nan")))
    ece = _num(ms.get(ECE, float("nan")))

    # -----------------------
    # 1) Per-class grouped bars (precision, recall, f1)
    # -----------------------
    try:
        fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.5), 4))
        x = np.arange(len(classes))
        width = 0.22
        ax.bar(x - width, precisions, width=width, label=PRECISION)
        ax.bar(x, recalls, width=width, label=RECALL)
        ax.bar(x + width, f1s, width=width, label=F1)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylabel("Score")
        ax.set_title("Per-class precision / recall / f1")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, f"{prefix}_{PER_CLASS}_{METRICS}.png")
        fig.savefig(p)
        plt.close(fig)
        saved[f"{PER_CLASS}_{METRICS}"] = p
        if tracker:
            try:
                tracker.log_artifact(p, name=os.path.basename(p))
            except Exception:
                pass
    except Exception as e:
        warnings.warn(f"Failed per_class_metrics plot: {e}")

    # -----------------------
    # 2) Support (counts) bar chart
    # -----------------------
    try:
        fig, ax = plt.subplots(figsize=(max(5, len(classes) * 1.2), 4))
        x = np.arange(len(classes))
        ax.bar(x, supports, tick_label=classes)
        ax.set_ylabel("Support (count)")
        ax.set_title("Per-class support")
        fig.tight_layout()
        p = os.path.join(out_dir, f"{prefix}_class_{SUPPORT}.png")
        fig.savefig(p)
        plt.close(fig)
        saved[f"class_{SUPPORT}"] = p
        if tracker:
            try:
                tracker.log_artifact(p, name=os.path.basename(p))
            except Exception:
                pass
    except Exception as e:
        warnings.warn(f"Failed class_{SUPPORT} plot: {e}")

    # -----------------------
    # 3) PR-AUC per-class bar chart (if available)
    # -----------------------
    try:
        # show NaNs as gray bars
        fig, ax = plt.subplots(figsize=(max(5, len(classes) * 1.2), 4))
        x = np.arange(len(classes))
        values = pr_aucs
        colors = ["C0" if not math.isnan(v) else "0.85" for v in values]
        ax.bar(
            x,
            [0 if math.isnan(v) else v for v in values],
            tick_label=classes,
            color=colors,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("PR AUC")
        ax.set_title("PR-AUC per class (from summary)")
        fig.tight_layout()
        p = os.path.join(out_dir, f"{prefix}_{PR_AUC}_{PER_CLASS}.png")
        fig.savefig(p)
        plt.close(fig)
        saved[f"{PR_AUC}_{PER_CLASS}"] = p
        if tracker:
            try:
                tracker.log_artifact(p, name=os.path.basename(p))
            except Exception:
                pass
    except Exception as e:
        warnings.warn(f"Failed pr_auc_per_class plot: {e}")

    # -----------------------
    # 4) Global summary bar chart (accuracy / macro / micro / ece)
    # -----------------------
    try:
        labels = [ACCURACY, MACRO_F1, MICRO_F1, ECE]
        vals = [accuracy, macro_f1, micro_f1, ece]
        # clamp ece to [0,1] for plotting sanity
        vals_plot = [(v if not (v is None or math.isnan(v)) else 0.0) for v in vals]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, vals_plot)
        ax.set_title("Global summary metrics")
        ax.set_ylim(0.0, 1.0)
        for i, v in enumerate(vals):
            txt = "nan" if (v is None or math.isnan(v)) else f"{v:.3f}"
            ax.text(i, vals_plot[i] + 0.02, txt, ha="center")
        fig.tight_layout()
        p = os.path.join(out_dir, f"{prefix}_{SUMMARY}_{METRICS}.png")
        fig.savefig(p)
        plt.close(fig)
        saved[f"{SUMMARY}_{METRICS}"] = p
        if tracker:
            try:
                tracker.log_artifact(p, name=os.path.basename(p))
            except Exception:
                pass
    except Exception as e:
        warnings.warn(f"Failed {SUMMARY}_{METRICS} plot: {e}")

    # -----------------------
    # 5) Radar / spider chart for per-class precision/recall/f1 (normalized)
    # -----------------------
    try:
        # Radar for f1/precision/recall averaged per class or stacked
        # We'll plot f1 values on radar
        labels_radar = classes
        stats = f1s  # use f1 for radar
        N = len(labels_radar)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        stats_cycle = stats + stats[:1]
        angles_cycle = angles + angles[:1]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles_cycle, stats_cycle, "o-", linewidth=2)
        ax.fill(angles_cycle, stats_cycle, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles), labels_radar)
        ax.set_title("Per-class F1 (radar)")
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()
        p = os.path.join(out_dir, f"{prefix}_{PER_CLASS}_{F1}_radar.png")
        fig.savefig(p)
        plt.close(fig)
        saved[f"{PER_CLASS}_{F1}_radar"] = p
        if tracker:
            try:
                tracker.log_artifact(p, name=os.path.basename(p))
            except Exception:
                pass
    except Exception as e:
        warnings.warn(f"Failed radar plot: {e}")

    # -----------------------
    # 6) Export CSV summary (one row per class and overall metrics)
    # -----------------------
    try:
        csv_path = os.path.join(out_dir, f"{prefix}_{SUMMARY}_table.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["class", PRECISION, RECALL, F1, SUPPORT, PR_AUC])
            for i, c in enumerate(classes):
                writer.writerow(
                    [
                        c,
                        f"{precisions[i]:.6f}" if not math.isnan(precisions[i]) else "",
                        f"{recalls[i]:.6f}" if not math.isnan(recalls[i]) else "",
                        f"{f1s[i]:.6f}" if not math.isnan(f1s[i]) else "",
                        supports[i],
                        f"{pr_aucs[i]:.6f}" if not math.isnan(pr_aucs[i]) else "",
                    ]
                )
            # top-level row
            writer.writerow([])
            writer.writerow(["metric", "value"])
            writer.writerow(
                [ACCURACY, f"{accuracy:.6f}" if not math.isnan(accuracy) else ""]
            )
            writer.writerow(
                [MACRO_F1, f"{macro_f1:.6f}" if not math.isnan(macro_f1) else ""]
            )
            writer.writerow(
                [MICRO_F1, f"{micro_f1:.6f}" if not math.isnan(micro_f1) else ""]
            )
            writer.writerow([ECE, f"{ece:.6f}" if not math.isnan(ece) else ""])
        saved[f"{SUMMARY}_csv"] = csv_path
        if tracker:
            try:
                tracker.log_artifact(csv_path, name=os.path.basename(csv_path))
            except Exception:
                pass
    except Exception as e:
        warnings.warn(f"Failed writing CSV summary: {e}")
    return saved

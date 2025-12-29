import csv
import os
import json
import math
import argparse
import sys
from typing import List, Optional, Dict, Any
from datetime import datetime

import numpy as np
import logging

from constants import (
    AUDIO_FILE,
    START_SECONDS,
    END_SECONDS,
    LABEL,
    PROBABILITIES,
    PROBABILITY,
    PER_CLASS,
    METRICS,
    ACCURACY,
    ECE,
    MACRO_F1,
    F1,
    PLOTS,
    TRUE_LABEL,
    PRED_LABEL,
    N_SEGEMENTS,
    CREATED_AT,
    MODEL,
    DATA_DIR,
    REPORT_PATH,
    CSV_PATH,
)

from helpers.common import load_model
from helpers.constants import FULL, LABEL_MAP, INV_LABEL_MAP, LABELS, SPLIT_CHOICES
from helpers.dataset import build_meta_from_dir
from helpers.helpers_eval import filter_meta_by_split

from utils.utils_eval import compute_metrics, weighted_average_probs_for_segment
from utils.utils_predict import predict
from utils.utils_visualize import (
    generate_plots_from_summary,
    generate_pr_and_calibration,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate_cli(
    model: str,
    data_dir: str,
    out_dir: str,
    tracker="none",
    tracker_project=None,
    tracker_task=None,
    split: str = FULL,
    split_map: Optional[str] = None,
    device: str = "cpu",
    window_seconds: Optional[float] = None,
    hop_seconds: Optional[float] = None,
    sr: int = 22050,
    n_mels: int = 64,
    batch_size: int = 64,
    top_k: int = 1,
    overlap_weighting: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model on a labeled audio dataset and save results.
    Performs segment-level predictions, computes metrics, generates plots,
    and saves outputs. Optionally logs results to a tracker.
    """

    os.makedirs(out_dir, exist_ok=True)
    meta = build_meta_from_dir(data_dir)
    if meta.empty:
        raise SystemExit("No label files found in data-dir")

    split_map = load_split_map(split_map) if split_map else {}
    meta = filter_meta_by_split(meta, split_map, split)

    # group segments by audio file
    files = sorted(meta[AUDIO_FILE].unique().tolist())

    all_segment_results = []  # rows for CSV
    y_true = []
    y_pred = []
    y_probs = []

    # Load Model
    _model, _metadata = load_model(model, device)

    # iterate files and run predict per file (so we can match by file)
    for audio in files:
        audio_path = os.path.join(data_dir, audio)
        if not os.path.exists(audio_path):
            logger.warning(f"audio missing, skipping: {audio_path}")
            continue
        # get segments for this file
        segs = meta[meta[AUDIO_FILE] == audio].to_dict(orient="records")

        # run predictions for this single file
        try:
            preds = predict(
                _model,
                _metadata,
                audio_path,
                tracker_type="none",  # don't use tracker for fetching here (model should resolve)
                tracker_project=None,
                tracker_task=None,
                device_str=device,
                window_seconds=window_seconds if window_seconds is not None else None,
                sr=sr,
                n_mels=n_mels,
                batch_size=batch_size,
                top_k=top_k,
            )
        except Exception as e:
            logger.warning(f"Prediction failed for {audio}: {e}")
            continue

        # preds is a list of window results (maybe multiple rows per window due to top_k expansion)
        # collapse preds into per-window unique windows, keeping probs vector
        # we only need one probs vector per window (per start,end)
        window_map = {}
        for p in preds:
            # ensure audio_file matches
            if os.path.basename(p.get(AUDIO_FILE, audio)) != os.path.basename(audio):
                # skip windows not for this audio (if predict returned many files)
                continue
            key = (float(p[START_SECONDS]), float(p[END_SECONDS]))
            # if multiple rows exist for same window (top_k), keep the full probs vector
            if key not in window_map:
                window_map[key] = np.array(p[PROBABILITIES], dtype=float)
            else:
                # keep the max probs (elementwise) as a conservative merge (unlikely necessary)
                window_map[key] = np.maximum(
                    window_map[key], np.array(p[PROBABILITIES], dtype=float)
                )
        # convert to list of windows with probs
        windows = []
        for (s, e), probs in sorted(window_map.items()):
            # normalize
            ssum = float(max(probs.sum(), 1e-12))
            probs = (probs / ssum).tolist()
            windows.append(
                {
                    START_SECONDS: float(s),
                    END_SECONDS: float(e),
                    PROBABILITIES: probs,
                }
            )

        # for each labeled segment compute avg probs via overlap weighting (or nearest window)
        for seg in segs:
            avg_probs = weighted_average_probs_for_segment(seg, windows)
            pred_idx = int(np.argmax(avg_probs))
            pred_label = INV_LABEL_MAP[pred_idx]
            probability = float(avg_probs[pred_idx])
            # store row
            all_segment_results.append(
                {
                    AUDIO_FILE: os.path.basename(audio),
                    START_SECONDS: float(seg[START_SECONDS]),
                    END_SECONDS: float(seg[END_SECONDS]),
                    TRUE_LABEL: seg[LABEL],
                    PRED_LABEL: pred_label,
                    PROBABILITY: probability,
                    f"{PER_CLASS}_{PROBABILITY}": {
                        INV_LABEL_MAP[i]: float(avg_probs[i])
                        for i in range(len(avg_probs))
                    },
                }
            )
            y_true.append(LABEL_MAP.get(seg[LABEL]))
            y_pred.append(pred_idx)
            y_probs.append(avg_probs)

    # compute metrics
    metrics = compute_metrics(y_true, y_pred, y_probs if len(y_probs) > 0 else None)

    # save CSV of segment-level results
    csv_path = os.path.join(out_dir, "per_segment_predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fo:
        writer = csv.writer(fo)
        writer.writerow(
            [
                AUDIO_FILE,
                START_SECONDS,
                END_SECONDS,
                TRUE_LABEL,
                PRED_LABEL,
                PROBABILITY,
            ]
        )
        for r in all_segment_results:
            writer.writerow(
                [
                    r[AUDIO_FILE],
                    f"{r[START_SECONDS]:.6f}",
                    f"{r[END_SECONDS]:.6f}",
                    r[TRUE_LABEL],
                    r[PRED_LABEL],
                    f"{r[PROBABILITY]:.6f}",
                ]
            )

    # save metrics JSON (include provenance)
    report = {
        CREATED_AT: datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        MODEL: model,
        DATA_DIR: os.path.abspath(data_dir),
        N_SEGEMENTS: len(all_segment_results),
        METRICS: metrics,
    }
    report_path = os.path.join(out_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # optionally log to tracker
    if tracker is not None:
        try:
            tracker.log_artifact(csv_path, name=os.path.basename(csv_path))
            tracker.log_artifact(report_path, name=os.path.basename(report_path))
            # log scalar metrics
            if MACRO_F1 in metrics:
                tracker.log_metric(MACRO_F1, metrics[MACRO_F1])
            if ACCURACY in metrics:
                tracker.log_metric(ACCURACY, metrics[ACCURACY])
            for lab, v in metrics.get(PER_CLASS, {}).items():
                tracker.log_metric(f"{F1}_{lab}", v.get(F1, 0.0))
            if ECE in metrics and metrics[ECE] is not None:
                tracker.log_metric(ECE, metrics[ECE])
        except Exception:
            pass

    evaluation_results = {
        REPORT_PATH: report_path,
        CSV_PATH: csv_path,
        METRICS: metrics,
    }

    # generate evaluation plots
    summary_metrics = generate_plots_from_summary(
        evaluation_results[METRICS],
        out_dir=os.path.join(out_dir, PLOTS),
        tracker=tracker,
    )

    pr_calibration_summary = generate_pr_and_calibration(
        y_true, y_probs, LABELS, out_dir=os.path.join(out_dir, PLOTS), tracker=tracker
    )
    return evaluation_results


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="Evaluate model on labeled directory",
        prog="evaluate",
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "--model", required=True, help="model directory or name (if using tracker)"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="directory with audio files and label .txt files",
    )
    parser.add_argument(
        "--out-dir", default="artifacts/eval", help="where to write outputs"
    )
    parser.add_argument(
        "--tracker", default="none", choices=["none", "clearml", "mlflow"]
    )
    parser.add_argument("--tracker-project", default="Burst_Classifier_POC")
    parser.add_argument("--tracker-task", default="evaluation_task")
    parser.add_argument(
        "--split",
        "-s",
        default=FULL,
        choices=SPLIT_CHOICES,
        help="the split to test the model on",
    )
    parser.add_argument(
        "--split-map",
        default=None,
        help="optional JSON file mapping audio filename -> split (TRAINING/VALIDATION/TEST)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--top-k", type=int, default=1, help="top-K predictions per window"
    )
    args = parser.parse_args(sys_argv)

    res = evaluate_cli(**vars(args))

    logger.info(f"Saved CSV: {res["csv_path"]}")
    logger.info(f"Saved report: {res["report_path"]}")


if __name__ == "__main__":
    cli(sys.argv[1:])

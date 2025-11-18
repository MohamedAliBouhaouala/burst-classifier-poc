import csv
import os
import json
import math
import argparse
import sys
from typing import List, Optional, Dict, Any
from datetime import datetime

import numpy as np
import warnings

from helpers.common import load_model
from helpers.constants import FULL, LABEL_MAP, INV_LABEL_MAP, LABELS, SPLIT_CHOICES
from helpers.dataset import build_meta_from_dir
from helpers.helpers_eval import filter_meta_by_split

from utils.utils_eval import compute_metrics, weighted_average_probs_for_segment
from utils.utils_predict import predict
from utils.utils_visualize import generate_plots_from_summary, generate_pr_and_calibration

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
    Runs evaluation and writes outputs to out_dir. Returns metrics dict.

    - model: local model dir or model name (if using tracker)
    - data_dir: contains .wav and .txt label files (build_meta_from_dir will detect)
    - split: TRAINING/VALIDATION/TEST/FULL
    - split_map: optional path to JSON file mapping audio filename -> split label
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = build_meta_from_dir(data_dir)
    if meta.empty:
        raise SystemExit("No label files found in data-dir")

    split_map = load_split_map(split_map) if split_map else {}
    meta = filter_meta_by_split(meta, split_map, split)

    # group segments by audio file
    files = sorted(meta['audio_file'].unique().tolist())

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
            print(f"[WARN] audio missing, skipping: {audio_path}")
            continue
        # get segments for this file
        segs = meta[meta['audio_file'] == audio].to_dict(orient='records')

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
            print(f"[ERROR] prediction failed for {audio}: {e}")
            continue

        # preds is a list of window results (maybe multiple rows per window due to top_k expansion)
        # collapse preds into per-window unique windows, keeping probs vector
        # we only need one probs vector per window (per start,end)
        window_map = {}  # (start,end) -> probs
        for p in preds:
            # ensure audio_file matches
            if os.path.basename(p.get("audio_file", audio)) != os.path.basename(audio):
                # skip windows not for this audio (if predict returned many files)
                continue
            key = (float(p["start_seconds"]), float(p["end_seconds"]))
            # if multiple rows exist for same window (top_k), keep the full probs vector
            if key not in window_map:
                window_map[key] = np.array(p["probabilities"], dtype=float)
            else:
                # keep the max probs (elementwise) as a conservative merge (unlikely necessary)
                window_map[key] = np.maximum(window_map[key], np.array(p["probabilities"], dtype=float))
        # convert to list of windows with probs
        windows = []
        for (s, e), probs in sorted(window_map.items()):
            # normalize
            ssum = float(max(probs.sum(), 1e-12))
            probs = (probs / ssum).tolist()
            windows.append({"start_seconds": float(s), "end_seconds": float(e), "probabilities": probs})

        # for each labeled segment compute avg probs via overlap weighting (or nearest window)
        for seg in segs:
            avg_probs = weighted_average_probs_for_segment(seg, windows)
            pred_idx = int(np.argmax(avg_probs))
            pred_label = INV_LABEL_MAP[pred_idx]
            probability = float(avg_probs[pred_idx])
            # store row
            all_segment_results.append({
                "audio_file": os.path.basename(audio),
                "start_seconds": float(seg["start_seconds"]),
                "end_seconds": float(seg["end_seconds"]),
                "true_label": seg["label"],
                "pred_label": pred_label,
                "probability": probability,
                "per_class_probability": {INV_LABEL_MAP[i]: float(avg_probs[i]) for i in range(len(avg_probs))}
            })
            y_true.append(LABEL_MAP.get(seg["label"]))
            y_pred.append(pred_idx)
            y_probs.append(avg_probs)

    # compute metrics
    metrics = compute_metrics(y_true, y_pred, y_probs if len(y_probs) > 0 else None)

    # save CSV of segment-level results
    csv_path = os.path.join(out_dir, "per_segment_predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fo:
        writer = csv.writer(fo)
        writer.writerow(["audio_file", "start_seconds", "end_seconds", "true_label", "pred_label", "probability"])
        for r in all_segment_results:
            writer.writerow([r["audio_file"], f"{r['start_seconds']:.6f}", f"{r['end_seconds']:.6f}", r["true_label"], r["pred_label"], f"{r['probability']:.6f}"])

    # save metrics JSON (include provenance)
    report = {
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "data_dir": os.path.abspath(data_dir),
        "n_segments": len(all_segment_results),
        "metrics": metrics
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
            if "macro_f1" in metrics:
                tracker.log_metric("macro_f1", metrics["macro_f1"])
            if "accuracy" in metrics:
                tracker.log_metric("accuracy", metrics["accuracy"])
            for lab, v in metrics.get("per_class", {}).items():
                tracker.log_metric(f"f1_{lab}", v.get("f1", 0.0))
            if "ece" in metrics and metrics["ece"] is not None:
                tracker.log_metric("ece", metrics["ece"])
        except Exception:
            pass

    evaluation_results = {"report_path": report_path, "csv_path": csv_path, "metrics": metrics}

    # generate evaluation plots
    summary_metrics = generate_plots_from_summary(
        evaluation_results["metrics"],
        out_dir=os.path.join(out_dir,"plots"), 
        tracker=tracker
    )

    pr_calibration_summary = generate_pr_and_calibration(
        y_true, 
        y_probs,
        LABELS,
        out_dir=os.path.join(out_dir,"plots"), 
        tracker=tracker
    )
    return evaluation_results

def cli(sys_argv):
    parser = argparse.ArgumentParser(description="Evaluate model on labeled directory", prog="evaluate", usage="%(prog)s [options]")
    parser.add_argument("--model", required=True, help="model directory or name (if using tracker)")
    parser.add_argument("--data-dir", required=True, help="directory with audio files and label .txt files")
    parser.add_argument("--out-dir", default="artifacts/eval", help="where to write outputs")
    parser.add_argument("--tracker", default="none", choices=["none", "clearml", "mlflow"])
    parser.add_argument("--tracker-project", default="Burst_Classifier_POC")
    parser.add_argument("--tracker-task", default="evaluation_task")
    parser.add_argument("--split", "-s", default=FULL, choices=SPLIT_CHOICES, help="the split to test the model on")
    parser.add_argument("--split-map", default=None, help="optional JSON file mapping audio filename -> split (TRAINING/VALIDATION/TEST)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=1, help="top-K predictions per window")
    args = parser.parse_args(sys_argv)

    res = evaluate_cli(**vars(args))

    print(json.dumps(res["metrics"], indent=2))
    print("Saved CSV:", res["csv_path"])
    print("Saved report:", res["report_path"])

if __name__ == "__main__":
    cli(sys.argv[1:])

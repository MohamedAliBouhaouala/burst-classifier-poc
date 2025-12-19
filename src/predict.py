import argparse
import sys
import json
import csv
import os
from typing import List, Dict, Any

import logging

from constants import (
    AUDIO_FILE,
    AUDIO_FILES,
    START_SECONDS,
    END_SECONDS,
    LABEL,
    PROBABILITY,
    PREDICTIONS,
)
from helpers.constants import AUDIO_EXTENSIONS
from helpers.common import load_model
from utils.utils_predict import predict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _gather_audio_paths(inputs: List[str], batch: bool = False) -> List[str]:
    """
    Accept a list of file or directory paths. Return a flattened list of audio files.
    - directories are scanned (non-recursive unless batch=True)
    - files are included if extension looks like audio
    """
    out = []
    for p in inputs:
        if os.path.isdir(p):
            if batch:
                for root, _, files in os.walk(p):
                    for fn in files:
                        if os.path.splitext(fn)[1].lower() in AUDIO_EXTENSIONS:
                            out.append(os.path.join(root, fn))
            else:
                for fn in os.listdir(p):
                    if os.path.splitext(fn)[1].lower() in AUDIO_EXTENSIONS:
                        out.append(os.path.join(p, fn))
        elif os.path.isfile(p):
            if os.path.splitext(p)[1].lower() in AUDIO_EXTENSIONS:
                out.append(p)
            else:
                # allow non-audio files to be passed but skip silently
                logger.warn("Skipping non-audio file: {p}")
        else:
            logger.warn(f"Path not found, skipping: {p}")
    # keep deterministic ordering
    out = sorted(list(dict.fromkeys(out)))
    return out


def predict_cli(
    model: str,
    audio: List[str],
    tracker: str,
    tracker_project: str,
    tracker_task: str,
    device: str,
    window_seconds: float,
    sr: int,
    n_mels: int,
    batch_size: int,
    top_k: int,
    out: str = None,
    batch: bool = False,
) -> List[Dict[str, Any]]:

    # expand audio inputs into proper list of audio files
    audio_paths = _gather_audio_paths(audio, batch=batch)
    if len(audio_paths) == 0:
        raise SystemExit("No audio files found from the provided --audio inputs")

    _model, _metadata = load_model(model, device)

    return predict(
        _model,
        _metadata,
        audio_paths,
        tracker,
        tracker_project,
        tracker_task,
        device,
        window_seconds,
        sr,
        n_mels,
        batch_size,
        top_k=top_k,
    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script performs prediction",
        prog="predict",
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "--model", required=True, help="local path or model name (if using tracker)"
    )
    # accept one or more files or directories
    parser.add_argument(
        "--audio",
        required=True,
        nargs="+",
        help="one or more audio files or directories",
    )
    parser.add_argument(
        "--tracker", default="none", choices=["none", "clearml", "mlflow"]
    )
    parser.add_argument("--tracker-project", default="Burst_Classifier_POC")
    parser.add_argument("--tracker-task", default="inference_task")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--window-seconds", type=float, default=0.5)
    parser.add_argument("--sr", type=int, default=22250)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--top-k", type=int, default=1, help="Return top-K predictions per window"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="write JSON or CSV output to this path (if .csv then CSV)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="when --audio contains directories, scan them recursively",
    )
    args = parser.parse_args(sys_argv)

    res = predict_cli(**vars(args))

    # If out is provided and ends with .csv, save CSV with header:
    # audio_file,start_seconds,end_seconds,label,probability
    if args.out:
        if args.out.lower().endswith(".csv"):
            # if out is a dir, write to out/preds.csv
            out_path = args.out
            if os.path.isdir(out_path):
                out_path = os.path.join(out_path, f"{PREDICTIONS}.csv")
            # ensure parent dir exists
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as fo:
                writer = csv.writer(fo)
                writer.writerow(
                    [
                        AUDIO_FILE,
                        START_SECONDS,
                        END_SECONDS,
                        LABEL,
                        PROBABILITY,
                    ]
                )
                for r in res:
                    writer.writerow(
                        [
                            r.get(AUDIO_FILE),
                            f"{r.get(START_SECONDS):.6f}",
                            f"{r.get(END_SECONDS):.6f}",
                            r.get(LABEL),
                            f"{r.get(PROBABILITY):.6f}",
                        ]
                    )
            logger.info(f"Wrote {len(res)} rows to {out_path}")
        else:
            # treat out as JSON file path (if dir provided, create preds.json inside)
            out_path = args.out
            if os.path.isdir(out_path):
                out_path = os.path.join(out_path, f"{PREDICTIONS}.json")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fo:
                json.dump({AUDIO_FILES: args.audio, PREDICTIONS: res}, fo, indent=2)
            logger.info(f"Wrote JSON with {len(res)} predictions to {out_path}")
    else:
        # print JSON to stdout
        logger.info(json.dumps(res, indent=2))


if __name__ == "__main__":
    cli(sys.argv[1:])

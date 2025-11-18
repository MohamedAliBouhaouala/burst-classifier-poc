"""
Evaluation script that:
- reads labelled segments from a data directory (wav + .txt label files),
- selects a split (TRAINING / VALIDATION / TEST / FULL) using an optional split-map,
- runs model inference (via helpers_predict.predict) over the audio files,
- matches predicted windows to each ground-truth labelled segment using overlap-weighted averaging of probs,
- computes metrics and saves JSON report + CSV of per-segment results.

Usage examples:
  python -m helpers.evaluate --model artifacts/best_model --data-dir data --out-dir results --split FULL
  python -m helpers.evaluate --model artifacts/best_model --data-dir data --out-dir results --split TEST --split-map data/split_map.json
"""
import os
import json
import math
from typing import List, Optional, Dict, Any
from datetime import datetime

import numpy as np

from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score, precision_recall_curve, auc
import warnings

from helpers.dataset import build_meta_from_dir, LABEL_MAP, INV_LABEL_MAP
from helpers.constants import LABELS, LABEL_IDX, FULL

def load_split_map(split_map_path: str) -> Dict[str, str]:
    """
    Load a JSON mapping audio_filename -> split_name (one of TRAINING/VALIDATION/TEST).
    """
    if not split_map_path or not os.path.exists(split_map_path):
        return {}
    with open(split_map_path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    # normalize keys (filenames) and values
    out = {}
    for k, v in d.items():
        out[os.path.basename(k)] = (v.upper() if isinstance(v, str) else v)
    return out

def filter_meta_by_split(meta_df, split_map: Dict[str, str], split_choice: str):
    """
    meta_df is a pandas DataFrame with column 'audio_file'.
    If split_choice == FULL -> return unchanged.
    Else filter meta_df to only rows whose audio_file basename maps to split_choice in split_map.
    If split_map is empty and split_choice != FULL -> raise error.
    """
    if split_choice == FULL:
        return meta_df
    if not split_map:
        raise SystemExit("split choice requested but no split_map provided (use --split-map or include split_map.json in data dir)")
    # filter where audio_file basename maps to split_choice
    allowed = {fname for fname, s in split_map.items() if s == split_choice}
    if not allowed:
        raise SystemExit(f"No files in split_map for split={split_choice}")
    # filter
    import pandas as pd
    return meta_df[meta_df['audio_file'].map(os.path.basename).isin(allowed)].reset_index(drop=True)

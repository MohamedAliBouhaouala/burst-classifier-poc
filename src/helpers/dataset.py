"""
Dataset utilities for Burst Classifier POC.
"""

from typing import Tuple, List, Dict, Any
import os
import json
import hashlib
import time
import random

import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np

from constants import (
    AUDIO_FILE,
    START_SECONDS,
    END_SECONDS,
    LABEL,
    LABELS,
    DATASET,
    MANIFEST,
    PATH,
    SIZE,
    SHA,
    SHA256,
    ERROR,
    CREATED_AT,
    DATA_DIR,
    AUDIO_FILES,
    N_SEGEMENTS,
    COUNTS,
)
from .constants import LABEL_MAP, INV_LABEL_MAP
from utils.common import dataset_hash, sha256_file


# ---------------------------
# Label parsing / metadata
# ---------------------------
def read_label_file(path: str, audio_filename: str) -> pd.DataFrame:
    """
    Read a label file with whitespace-separated rows:
      <start_seconds> <end_seconds> <label>

    Returns DataFrame with columns: audio_file, start_seconds, end_seconds, label

    NOTE: unknown labels (not in LABEL_MAP) are SILENTLY SKIPPED.
    """
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # normalize commas in decimals e.g. "12,351" -> "12.351"
                line = line.replace(",", ".")
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    label = parts[2].strip().lower()
                except ValueError:
                    # malformed numeric values — skip silently
                    continue
                if label not in LABEL_MAP:
                    # SILENT SKIP unknown labels for now (POC mode)
                    continue
                rows.append(
                    {
                        AUDIO_FILE: audio_filename,
                        START_SECONDS: float(start),
                        END_SECONDS: float(end),
                        LABEL: label,
                    }
                )
    except FileNotFoundError:
        # If the label file is missing/unreadable, return empty df
        return pd.DataFrame(columns=[AUDIO_FILE, START_SECONDS, END_SECONDS, LABEL])

    if len(rows) == 0:
        return pd.DataFrame(columns=[AUDIO_FILE, START_SECONDS, END_SECONDS, LABEL])
    return pd.DataFrame(rows)


def build_meta_from_dir(data_dir: str, label_suffixes=(".txt", ".csv")) -> pd.DataFrame:
    """
    Scans data_dir for label files and builds a single DataFrame of segments.
    Tries to pair label files with their corresponding audio file by basename.
    Unknown labels are skipped by read_label_file.
    """
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        return pd.DataFrame(columns=[AUDIO_FILE, START_SECONDS, END_SECONDS, LABEL])

    rows: List[pd.DataFrame] = []
    files = os.listdir(data_dir)
    for fname in files:
        if fname.lower().endswith(label_suffixes):
            base = fname.rsplit(".", 1)[0]
            # candidate audio names (common heuristics)
            candidates = [
                base + ".wav",
                base + ".flac",
                base + ".mp3",
                base.replace(f"_{LABELS}", "") + ".wav",
                base.replace(f"-{LABELS}", "") + ".wav",
            ]
            audio_file = None
            for c in candidates:
                if c in files:
                    audio_file = c
                    break
            if audio_file is None:
                # fallback: pick first wav if present
                wavs = [f for f in files if f.lower().endswith(".wav")]
                if wavs:
                    audio_file = wavs[0]
                else:
                    # no audio found -> skip this label file
                    continue
            df = read_label_file(os.path.join(data_dir, fname), audio_file)
            if len(df) > 0:
                rows.append(df)
    if len(rows) == 0:
        return pd.DataFrame(columns=[AUDIO_FILE, START_SECONDS, END_SECONDS, LABEL])
    return pd.concat(rows, ignore_index=True)


def build_and_write_dataset_manifest(
    meta_df: pd.DataFrame,
    data_dir: str,
    out_dir: str,
    prefix: str = f"{DATASET}_{MANIFEST}",
) -> Tuple[str, Dict[str, Any]]:
    """
    Builds and writes a JSON manifest with per-file sha256, size, mtime and label summary.
    Returns (manifest_path, manifest_dict).
    """
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.abspath(data_dir)

    audio_files = sorted(meta_df[AUDIO_FILE].unique().tolist())
    files_info: List[Dict[str, Any]] = []
    for rel in audio_files:
        p = os.path.join(data_dir, rel)
        if os.path.exists(p) and os.path.isfile(p):
            try:
                st = os.stat(p)
                sha, size = sha256_file(p)
                files_info.append(
                    {
                        f"rel{PATH}": os.path.relpath(p, start=data_dir),
                        SHA256: sha,
                        SIZE: st.st_size,
                        "mtime": int(st.st_mtime),
                    }
                )
            except Exception as e:
                files_info.append({f"rel{PATH}": rel, ERROR: str(e)})
        else:
            files_info.append({f"rel{PATH}": rel, "missing": True})

    # include .txt label files under data_dir
    for root, _, fnames in os.walk(data_dir):
        for fn in fnames:
            if fn.lower().endswith(".txt"):
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, start=data_dir)
                if not any(f.get(f"rel{PATH}") == rel for f in files_info):
                    try:
                        st = os.stat(p)
                        sha = sha256_file(p)
                        files_info.append(
                            {
                                f"rel{PATH}": rel,
                                SHA256: sha,
                                SIZE: st.st_size,
                                "mtime": int(st.st_mtime),
                            }
                        )
                    except Exception as e:
                        files_info.append({f"rel{PATH}": rel, ERROR: str(e)})

    label_counts = meta_df[LABEL].value_counts().to_dict()
    label_map_counts = {str(k): int(v) for k, v in label_counts.items()}

    manifest = {
        CREATED_AT: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        DATA_DIR: data_dir,
        f"n_{AUDIO_FILES}": len(audio_files),
        N_SEGEMENTS: int(len(meta_df)),
        f"{LABEL}_{COUNTS}": label_map_counts,
        "files": files_info,
    }

    name = f"{prefix}_{int(time.time())}.json"
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return path, manifest

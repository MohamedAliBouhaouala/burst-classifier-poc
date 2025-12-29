import hashlib
import os
import subprocess
import json
import pandas as pd
import sys

from constants import (
    AUDIO_FILE,
    AUDIO_FILES,
    START_SECONDS,
    END_SECONDS,
    LABEL,
    COUNTS,
    DATA_DIR,
    PATH,
    SHA256,
    SIZE,
    GIT,
    LOCAL,
    N_SEGEMENTS,
    CREATED_AT,
)


def _safe_mkdir(path: str):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def sha256_file(path: str) -> dict:
    """Compute the SHA-256 hash and file size of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest(), os.path.getsize(path)


def dataset_hash(meta_df: pd.DataFrame, data_dir: str) -> str:
    """Compute a deterministic SHA-256 hash of a dataset."""
    h = hashlib.sha256()
    for _, r in meta_df.sort_values(
        [AUDIO_FILE, START_SECONDS, END_SECONDS]
    ).iterrows():
        h.update(
            f"{r[AUDIO_FILE]},{r[START_SECONDS]},{r[END_SECONDS]},{r[LABEL]}\n".encode()
        )
        audio_path = os.path.join(data_dir, r[AUDIO_FILE])
        try:
            st = os.stat(audio_path)
            h.update(str(st.st_size).encode())
        except FileNotFoundError:
            pass
    return h.hexdigest()


def get_git_sha():
    """Get the current Git commit SHA."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return sha
    except Exception:
        return LOCAL


def save_json(path, obj):
    """Save an object to a JSON file"""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _num(x):
    """Convert a value to float, returning NaN on failure."""
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_manifest_fingerprint(manifest: dict) -> str:
    """
    Deterministic fingerprint: build a canonical string from manifest fields and files,
    and return its sha256 hex digest.
    """
    h = hashlib.sha256()
    # include top-level deterministic fields if present
    for k in (DATA_DIR, CREATED_AT, f"n_{AUDIO_FILES}", N_SEGEMENTS):
        if k in manifest:
            h.update(f"{k}={manifest[k]}\n".encode())
    # include label_counts in sorted order
    label_counts = manifest.get(f"{LABEL}_{COUNTS}", {})
    for k in sorted(label_counts.keys()):
        h.update(f"{LABEL}{COUNTS}:{k}={label_counts[k]}\n".encode())
    # include file entries sorted by relpath
    files = sorted(manifest.get("files", []), key=lambda x: x.get(f"rel{PATH}", ""))
    for f in files:
        rel = f.get(f"rel{PATH}", "")
        sha = f.get(SHA256, "")
        size = f.get(SIZE, "")
        h.update(f"{rel},{sha},{size}\n".encode())
    return h.hexdigest()


def get_env_info() -> dict:
    """Gather versions of key environment packages."""
    env_info = {"python": sys.version.split()[0]}
    try:
        import torch as _torch

        env_info["torch"] = _torch.__version__
    except Exception:
        env_info["torch"] = None
    try:
        import torchaudio as _ta

        env_info["torchaudio"] = _ta.__version__
    except Exception:
        env_info["torchaudio"] = None
    try:
        import numpy as _np

        env_info["numpy"] = _np.__version__
    except Exception:
        env_info["numpy"] = None
    return env_info

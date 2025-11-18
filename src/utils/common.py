import hashlib
import os
import subprocess
import json
import pandas as pd
import sys

def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

# def sha256_file(path: str) -> str:
#     h = hashlib.sha256()
#     with open(path, "rb") as f:
#         for chunk in iter(lambda: f.read(8192), b""):
#             h.update(chunk)
#     return h.hexdigest()

def sha256_file(path) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest(), os.path.getsize(path)

def dataset_hash(meta_df: pd.DataFrame, data_dir: str) -> str:
    h = hashlib.sha256()
    for _, r in meta_df.sort_values(['audio_file','start_seconds','end_seconds']).iterrows():
        h.update(f"{r['audio_file']},{r['start_seconds']},{r['end_seconds']},{r['label']}\n".encode())
        audio_path = os.path.join(data_dir, r['audio_file'])
        try:
            st = os.stat(audio_path)
            h.update(str(st.st_size).encode())
        except FileNotFoundError:
            pass
    return h.hexdigest()


def get_git_sha():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return sha
    except Exception:
        return "local"

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _num(x):
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
    for k in ("data_dir", "created_at", "n_audio_files", "n_segments"):
        if k in manifest:
            h.update(f"{k}={manifest[k]}\n".encode())
    # include label_counts in sorted order
    label_counts = manifest.get("label_counts", {})
    for k in sorted(label_counts.keys()):
        h.update(f"labelcount:{k}={label_counts[k]}\n".encode())
    # include file entries sorted by relpath
    files = sorted(manifest.get("files", []), key=lambda x: x.get("relpath", ""))
    for f in files:
        rel = f.get("relpath", "")
        sha = f.get("sha256", "")
        size = f.get("size", "")
        h.update(f"{rel},{sha},{size}\n".encode())
    return h.hexdigest()

def get_env_info() -> dict :
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
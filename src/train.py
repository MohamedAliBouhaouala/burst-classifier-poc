import argparse
import logging
import sys
import json
import socket
import platform
from datetime import datetime
from types import SimpleNamespace
import os
import time

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import numpy as np

from helpers.constants import LABEL_MAP
from helpers.dataset import build_and_write_dataset_manifest, build_meta_from_dir
from models.model import SmallCNN
from helpers.helpers_preprocess import set_seed
from tracker import TrackerFactory

from constants import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    VALIDATION,
    TEST,
    SPLIT,
    N_MELS,
    FIXED_SECONDS,
    AUDIO_FILE,
    LABEL,
    ARTIFACT,
    ARTIFACTS,
    CHECKSUMS,
    DATASET,
    ENV,
    GIT,
    HASH,
    MANIFEST,
    SHA,
    SHA256,
    SIZE,
    PATH,
    MODEL,
    PARAMETERS,
    METADATA,
    CREATED_AT,
    LOCAL,
    COUNTS,
    HOST,
    PLATFORM,
    TRACKER,
    N_SEGEMENTS,
)

from utils.common import (
    compute_manifest_fingerprint,
    dataset_hash,
    get_git_sha,
    save_json,
    sha256_file,
    get_env_info,
)
from utils.utils_train import train
from utils.utils_eval import save_confusion_matrix


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_cli(
    data_dir: str,
    artifacts_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    test_split: float,
    fixed_seconds: float,
    n_mels: int,
    tracker: str,
    tracker_project: str,
    tracker_task: str,
    save_epoch_checkpoints: bool,
) -> None:
    """
    Train a model on labeled audio data and save artifacts.
    - Loads labeled audio from a data directory.
    - Prepares dataset manifests, sets seeds.
    - Trains a model with the specified hyperparameters.
    - Saves checkpoints, results, and metadata and computes artifact checksums.
    - Optionally logs metrics, parameters, and artifacts to a tracker (ClearML, MLflow, or none).
    """
    set_seed()
    os.makedirs(artifacts_dir, exist_ok=True)

    # tracker object and serializable tracker metadata
    tracker_obj = TrackerFactory(
        tracker, project=tracker_project, task_name=tracker_task
    )
    tracker_meta = {"type": tracker}

    meta = build_meta_from_dir(data_dir)
    manifest_path, manifest = build_and_write_dataset_manifest(
        meta, data_dir, artifacts_dir
    )
    try:
        tracker_obj.log_artifact(manifest_path, name=f"{DATASET}_{MANIFEST}")
    except Exception:
        pass

    if meta.empty:
        raise SystemExit("No label files found in data-dir")

    logger.info(
        f"Loaded {len(meta)} labeled segments from {meta[AUDIO_FILE].nunique()} audio files"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_hash = dataset_hash(meta, data_dir)
    git_sha = get_git_sha()

    artifacts_acc = []

    # prepare args-like namespace for loader helpers
    args_ns = SimpleNamespace(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_split=val_split,
        test_split=test_split,
        fixed_seconds=fixed_seconds,
        n_mels=n_mels,
        tracker=tracker,
        tracker_project=tracker_project,
        tracker_task=tracker_task,
        save_epoch_checkpoints=save_epoch_checkpoints,
    )
    params = {
        EPOCHS: epochs,
        BATCH_SIZE: batch_size,
        LEARNING_RATE: lr,
        FIXED_SECONDS: fixed_seconds,
        N_MELS: n_mels,
        f"{VALIDATION}_{SPLIT}": val_split,
        f"{TEST}_{SPLIT}": test_split,
    }

    artifacts_acc.extend(
        train(
            meta, args_ns, device, artifacts_dir, tracker_obj, params, ds_hash, git_sha
        )
    )

    new_params = {
        EPOCHS: args_ns.epochs,
        BATCH_SIZE: args_ns.batch_size,
        LEARNING_RATE: args_ns.lr,
        FIXED_SECONDS: args_ns.fixed_seconds,
        N_MELS: args_ns.n_mels,
        f"{VALIDATION}_{SPLIT}": args_ns.val_split,
        f"{TEST}_{SPLIT}": args_ns.test_split,
    }

    # --- add artifact checksums (artifact_sha256, artifact_size) for each artifact entry if possible ---
    artifacts_checksums = {}
    for a in artifacts_acc:
        model_path = a.get(f"{MODEL}_{PATH}")
        if not model_path:
            continue
        # try to resolve model file: absolute, relative to artifacts_dir, or literal path
        candidates = []
        if os.path.isabs(model_path):
            candidates.append(model_path)
        candidates.append(os.path.join(artifacts_dir, model_path))
        candidates.append(os.path.join(artifacts_dir, os.path.basename(model_path)))
        candidates.append(model_path)
        found = None
        for c in candidates:
            if c and os.path.exists(c) and os.path.isfile(c):
                found = c
                break
        if found:
            try:
                sha, size = sha256_file(found)
                a[f"{ARTIFACT}_{SHA256}"] = sha
                a[f"{ARTIFACT}_{SIZE}"] = size
                artifacts_checksums[os.path.basename(found)] = {
                    SHA256: sha,
                    SIZE: size,
                    PATH: os.path.abspath(found),
                }
            except Exception as e:
                logger.error("Failed to compute checksum for %s: %s", found, e)
        else:
            logger.warning(
                "Model artifact listed but not found on disk (tried): %s", candidates
            )

    # --- FINALIZE ---
    results_path = os.path.join(artifacts_dir, "train_results.json")
    save_json(results_path, artifacts_acc)
    # log params
    try:
        tracker_obj.log_params(new_params)
    except Exception:
        pass
    # compute dataset manifest fingerprint (dataset_manifest_sha) for stronger linking
    dataset_manifest_sha = None
    try:
        if manifest:
            dataset_manifest_sha = compute_manifest_fingerprint(manifest)
    except Exception as e:
        logger.error("Failed to compute manifest fingerprint: %s", e)

    env_info = get_env_info()

    metadata = {
        f"{GIT}_{SHA}": git_sha,
        f"{GIT}_dirty_hint": False if git_sha != LOCAL else True,
        f"{DATASET}_{HASH}": ds_hash,
        f"{DATASET}_{MANIFEST}": (
            os.path.basename(manifest_path) if manifest_path else None
        ),
        f"{DATASET}_{MANIFEST}_{SHA}": dataset_manifest_sha,
        ARTIFACTS: artifacts_acc,
        f"{ARTIFACTS}_{CHECKSUMS}": artifacts_checksums,
        CREATED_AT: datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        PARAMETERS: new_params,
        N_SEGEMENTS: int(len(meta)) if meta is not None else None,
        f"{LABEL}_{COUNTS}": manifest.get(f"{LABEL}_{COUNTS}") if manifest else None,
        ENV: env_info,
        TRACKER: tracker_meta,
        HOST: socket.gethostname(),
        PLATFORM: platform.platform(),
    }
    save_json(os.path.join(artifacts_dir, f"{METADATA}.json"), metadata)
    try:
        tracker_obj.log_artifact(
            os.path.join(artifacts_dir, f"{METADATA}.json"), name=f"{METADATA}.json"
        )
        tracker_obj.log_artifact(results_path, name=os.path.basename(results_path))
    except Exception:
        pass
    try:
        tracker_obj.finalize()
    except Exception:
        pass

    logger.info("Training finished. Artifacts stored in %s", artifacts_dir)


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script trains a model",
        prog="train",
        usage="%(prog)s [options]",
    )

    # ----------------------------
    # Training parameters
    # ----------------------------
    parser.add_argument(
        "--data-dir", default="data", help="directory with .wav and .txt label files"
    )
    parser.add_argument(
        "--artifacts-dir", default="artifacts", help="where to save models and reports"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="validation split when cv-mode is none",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="optional held-out test split when cv-mode is none",
    )
    parser.add_argument("--fixed-seconds", type=float, default=0.5)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument(
        "--tracker",
        choices=["none", "clearml", "mlflow"],
        default="none",
        help="which experiment tracker to use",
    )
    parser.add_argument(
        "--tracker-project", default="Burst_Classifier_POC", help="tracker project name"
    )
    parser.add_argument(
        "--tracker-task", default="train_task", help="tracker task name"
    )
    parser.add_argument(
        "--save-epoch-checkpoints",
        action="store_true",
        help="save a checkpoint for every epoch (useful for debugging/POC)",
    )
    args = parser.parse_args(sys_argv)

    train_cli(**vars(args))


if __name__ == "__main__":
    cli(sys.argv[1:])

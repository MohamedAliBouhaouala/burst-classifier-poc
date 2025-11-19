import logging
import numpy as np
import pandas as pd
import os
import time
from types import SimpleNamespace

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from helpers.constants import LABEL_MAP, SEED
from helpers.helpers_preprocess import prepare_loader_from_meta
from helpers.helpers_train import (
    compute_val_metrics_and_loss,
    _get_criterion,
    _get_optimizer,
)
from models.model import SmallCNN
from .utils_eval import save_confusion_matrix


def train_one_epoch_with_criterion(model, loader, opt, device, criterion):
    """
    Train one epoch and return (avg_loss, train_accuracy)
    """
    model.train()
    running = 0.0
    total = 0
    correct = 0
    for x, y, _ in loader:
        x = x.to(device).float()
        y = y.to(device).long()
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        running += loss.item() * x.size(0)
        total += x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
    train_loss = running / total if total > 0 else 0.0
    train_acc = correct / total if total > 0 else 0.0
    return float(train_loss), float(train_acc)


def run_training(
    train_meta,
    val_meta,
    fold_tag: str,
    args: SimpleNamespace,
    device: torch.device,
    artifacts_dir: str,
    tracker_obj,
    params: dict,
    ds_hash: str,
    git_sha: str,
):
    """
    Train one fold (or single train/val run).
    Returns a dict describing the fold/artifact.
    """
    # loaders
    train_loader = prepare_loader_from_meta(
        train_meta, args.data_dir, args, augment=True
    )
    val_loader = prepare_loader_from_meta(val_meta, args.data_dir, args, augment=False)
    model = SmallCNN(in_channels=1, n_classes=3).to(device)
    opt = _get_optimizer(model, args.lr)
    criterion = _get_criterion(train_meta, device)

    best_path = os.path.join(artifacts_dir, f"best_model_{fold_tag}.pt")
    epoch_logs = []
    best_acc = -1.0
    best_epoch = -1
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_accuracy = train_one_epoch_with_criterion(
            model, train_loader, opt, device, criterion
        )
        ys, preds, probs, val_loss, val_acc = compute_val_metrics_and_loss(
            model, val_loader, device, criterion
        )
        val_macro = f1_score(ys, preds, average="macro") if len(ys) > 0 else 0.0
        rep = classification_report(
            ys, preds, target_names=["b", "mb", "h"], output_dict=True, zero_division=0
        )
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        epochs_left = max(0, args.epochs - epoch)
        eta_seconds = avg_epoch_time * epochs_left

        epoch_log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_loss),
            "val_macro_f1": float(val_macro),
            "val_accuracy": float(val_acc),
            "per_class_f1": {
                lab: float(rep.get(lab, {}).get("f1", 0.0)) for lab in ["b", "mb", "h"]
            },
            "epoch_time_seconds": float(epoch_time),
            "eta_seconds": float(eta_seconds),
        }
        epoch_logs.append(epoch_log)

        logging.info(
            f"[{fold_tag}] epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} val_macro_f1={val_macro:.4f} val_acc={val_acc:.4f} epoch_time={epoch_time:.2f}s"
        )

        # tracker logging
        try:
            # try to convert fold_tag that is numeric-like into an int for step derivation
            try:
                fold_num = int(fold_tag)
                step = fold_num * 1000 + epoch
            except Exception:
                step = epoch
            tracker_obj.log_metric("train_loss", float(train_loss), step=step)
            tracker_obj.log_metric("train_accuracy", float(train_accuracy), step=step)
            tracker_obj.log_metric("val_loss", float(val_loss), step=step)
            tracker_obj.log_metric("val_macro_f1", float(val_macro), step=step)
            tracker_obj.log_metric("val_accuracy", float(val_acc), step=step)
            tracker_obj.log_metric("epoch_time_seconds", float(epoch_time), step=step)
            tracker_obj.log_metric("eta_seconds", float(eta_seconds), step=step)
            for lab in ["b", "mb", "h"]:
                tracker_obj.log_metric(
                    f"f1_{lab}", float(epoch_log["per_class_f1"][lab]), step=step
                )
        except Exception:
            pass

        # optionally save epoch checkpoint
        if args.save_epoch_checkpoints:
            p = os.path.join(artifacts_dir, f"model_{fold_tag}_epoch_{epoch}.pt")
            ckpt_meta = {
                "epoch": epoch,
                "params": params,
                "git_sha": git_sha,
                "dataset_hash": ds_hash,
                "fold": fold_tag,
            }
            torch.save({"model_state": model.state_dict(), "metadata": ckpt_meta}, p)
            try:
                tracker_obj.log_artifact(p, name=os.path.basename(p))
            except Exception:
                pass

        # selection by validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            ckpt_meta = {
                "epoch": epoch,
                "params": params,
                "git_sha": git_sha,
                "dataset_hash": ds_hash,
                "fold": fold_tag,
                "train_accuracy": float(train_accuracy),
                "val_accuracy": float(val_acc),
            }
            torch.save(
                {"model_state": model.state_dict(), "metadata": ckpt_meta}, best_path
            )

    # final evaluation using best model
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model_state"])
    ys, preds, probs, val_loss, val_acc = compute_val_metrics_and_loss(
        model, val_loader, device, criterion
    )
    rep = classification_report(
        ys, preds, target_names=["b", "mb", "h"], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(ys, preds).tolist()
    cm_path = os.path.join(artifacts_dir, f"confusion_{fold_tag}.png")
    save_confusion_matrix(ys, preds, classes=["b", "mb", "h"], out_path=cm_path)

    try:
        tracker_obj.log_artifact(best_path, name=f"best_model_{fold_tag}")
        tracker_obj.log_artifact(cm_path, name=f"confusion_{fold_tag}")
    except Exception:
        pass

    fold_art = {
        "fold": fold_tag,
        "best_val_accuracy": best_acc,
        "best_epoch": best_epoch,
        "report": rep,
        "confusion_matrix": cm,
        "model_path": best_path,
        "confusion_png": cm_path,
        "epoch_logs": epoch_logs,
    }
    return fold_art


def train(meta, args_ns, device, artifacts_dir, tracker_obj, params, ds_hash, git_sha):
    """
    Train with a simple train/val (and optional test) split.
    Returns list of artifacts to append.
    """
    from sklearn.model_selection import train_test_split

    labels = meta["label"].map(LABEL_MAP).values
    stratify_arg = labels if len(np.unique(labels)) > 1 else None

    test_split = getattr(args_ns, "test_split", 0.0) or 0.0
    val_split = getattr(args_ns, "val_split", 0.2)

    if test_split and test_split > 0.0:
        remaining, test_meta = train_test_split(
            meta, test_size=test_split, stratify=stratify_arg, random_state=SEED
        )
        stratify_arg_rem = (
            remaining["label"].map(LABEL_MAP).values
            if len(np.unique(remaining["label"].map(LABEL_MAP).values)) > 1
            else None
        )
        train_meta, val_meta = train_test_split(
            remaining,
            test_size=val_split / (1.0 - test_split),
            stratify=stratify_arg_rem,
            random_state=SEED,
        )
        test_meta = test_meta.reset_index(drop=True)
    else:
        train_meta, val_meta = train_test_split(
            meta, test_size=val_split, stratify=stratify_arg, random_state=SEED
        )
        test_meta = None

    train_meta = train_meta.reset_index(drop=True)
    val_meta = val_meta.reset_index(drop=True)

    logging.info(
        f"Training on {len(train_meta)} samples, validating on {len(val_meta)} samples"
        + (f", testing on {len(test_meta)} samples" if test_meta is not None else "")
    )

    fold_art = run_training(
        train_meta,
        val_meta,
        fold_tag="single",
        args=args_ns,
        device=device,
        artifacts_dir=artifacts_dir,
        tracker_obj=tracker_obj,
        params=params,
        ds_hash=ds_hash,
        git_sha=git_sha,
    )
    artifacts_acc = [fold_art]

    # test evaluation
    if test_meta is not None:
        best_path = fold_art["model_path"]
        state = torch.load(best_path, map_location=device)
        model = SmallCNN(in_channels=1, n_classes=3).to(device)
        model.load_state_dict(state["model_state"])
        criterion = _get_criterion(train_meta, device)
        test_loader = prepare_loader_from_meta(
            test_meta, args_ns.data_dir, args_ns, augment=False
        )
        ys, preds, probs, test_loss, test_acc = compute_val_metrics_and_loss(
            model, test_loader, device, criterion
        )
        rep = classification_report(
            ys, preds, target_names=["b", "mb", "h"], output_dict=True, zero_division=0
        )
        cm = confusion_matrix(ys, preds).tolist()
        cm_path = os.path.join(artifacts_dir, f"confusion_test_single.png")
        save_confusion_matrix(ys, preds, classes=["b", "mb", "h"], out_path=cm_path)
        try:
            tracker_obj.log_artifact(cm_path, name="confusion_test_single")
        except Exception:
            pass
        test_art = {
            "fold": "test_single",
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "report": rep,
            "confusion_matrix": cm,
            "confusion_png": cm_path,
        }
        artifacts_acc.append(test_art)
    return artifacts_acc

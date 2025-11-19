import glob
import json
import torch
import os

from typing import Optional
import logging

from models.model import SmallCNN
from tracker import TrackerFactory

logger = logging.getLogger(__name__)


def resolve_settings(metadata: dict, args):
    """
    Resolver that prefers metadata['params'] values first, then CLI args (if metadata lacks a value),
    then hard defaults.

    Returns a dict with keys:
      window_seconds, sr, n_mels, batch_size
    """
    params = {}
    if metadata and isinstance(metadata, dict):
        params = metadata.get("params", {}) or {}

    # helper: prefer params[param_key] if present & valid, else CLI cli_attr if provided & valid, else default
    def pick_float(cli_attr, param_key, default):
        # try metadata params first
        val = params.get(param_key, None)
        if val is not None:
            try:
                return float(val)
            except Exception:
                # fall through to CLI/default on conversion error
                pass
        # then CLI
        cli_val = getattr(args, cli_attr, None)
        if cli_val is not None:
            try:
                return float(cli_val)
            except Exception:
                pass
        return float(default)

    def pick_int(cli_attr, param_key, default):
        # try metadata params first
        val = params.get(param_key, None)
        if val is not None:
            try:
                return int(val)
            except Exception:
                pass
        # then CLI
        cli_val = getattr(args, cli_attr, None)
        if cli_val is not None:
            try:
                return int(cli_val)
            except Exception:
                pass
        return int(default)

    # window_seconds: prefer metadata.fixed_seconds then CLI then default
    window_seconds = pick_float("window_seconds", "fixed_seconds", 0.5)

    # sample rate: try multiple possible names in metadata.params first, then CLI, then fallback 22050
    sr = None
    # try metadata keys
    for key in ("sample_rate", "sr", "sampling_rate"):
        v = params.get(key, None)
        if v is not None:
            try:
                sr = int(v)
                break
            except Exception:
                sr = None
    # if metadata didn't provide sr, try CLI
    if sr is None and hasattr(args, "sr") and getattr(args, "sr") is not None:
        try:
            sr = int(getattr(args, "sr"))
        except Exception:
            sr = None
    if sr is None:
        sr = 22050

    n_mels = pick_int("n_mels", "n_mels", 64)
    batch_size = pick_int("batch_size", "batch_size", 64)

    resolved = {
        "window_seconds": float(window_seconds),
        "sr": int(sr),
        "n_mels": int(n_mels),
        "batch_size": int(batch_size),
    }
    logger.info(
        f"prediction settings resolved (metadata.params preferred):"
        f"window_seconds={resolved['window_seconds']},"
        f"sr={resolved['sr']}, n_mels={resolved['n_mels']}, batch_size={resolved['batch_size']}"
    )
    return resolved


def load_model_from_path(path: str, device: torch.device):
    """
    Load a model and metadata from a directory.

    Expectations:
      - `path` MUST be a directory.
      - Directory should contain one .pt checkpoint (preferred names: *best*.pt, *model*.pt).
      - Directory may contain metadata.json (recommended). If present, its keys are loaded.
      - If the checkpoint contains a "metadata" dict, those keys override metadata.json keys.

    Returns:
      (model, metadata_dict)

    Raises:
      FileNotFoundError if no .pt checkpoint is found in directory.
      RuntimeError on state_dict load failures.
    """

    if not os.path.isdir(path):
        raise ValueError(
            f"load_model_from_path: expected a directory path, got: {path}"
        )

    # find checkpoint .pt inside directory
    def _find_ckpt(d):
        patterns = ["*best*.pt", "*model*.pt", "*.pt"]
        for pat in patterns:
            candidates = sorted(glob.glob(os.path.join(d, pat)))
            if candidates:
                return candidates[0]
        return None

    ckpt_path = _find_ckpt(path)
    if ckpt_path is None:
        raise FileNotFoundError(f"No .pt checkpoint found in directory: {path}")

    # try to load metadata.json if present
    meta_from_file = {}
    meta_path = os.path.join(path, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta_from_file = json.load(fh)
        except Exception:
            # best-effort: ignore malformed metadata.json but continue
            meta_from_file = {}

    # load checkpoint (map to CPU first)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # locate state dict inside checkpoint
    state = None
    if isinstance(ckpt, dict):
        for k in (
            "model_state",
            "state_dict",
            "model",
            "model_state_dict",
            "net_state_dict",
        ):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        if state is None:
            # heuristic: if ckpt looks like a state dict (tensor values), use it
            try_keys = list(ckpt.keys())[:6]
            if try_keys and all(isinstance(ckpt[k], (torch.Tensor,)) for k in try_keys):
                state = ckpt
            else:
                # no obvious state found; fail with helpful message
                raise RuntimeError(
                    f"Checkpoint {ckpt_path} does not contain a recognized state_dict key."
                )
    else:
        # ckpt is not a dict -> assume it is a state_dict
        state = ckpt

    # instantiate model and load weights
    model = SmallCNN(in_channels=1, n_classes=3)
    try:
        model.load_state_dict(state)
    except Exception as e:
        # try stripping common 'module.' prefix (DataParallel)
        try:
            stripped = {
                (k.replace("module.", "") if k.startswith("module.") else k): v
                for k, v in state.items()
            }
            model.load_state_dict(stripped)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load state_dict from {ckpt_path}: {e}; retry also failed: {e2}"
            )

    # assemble metadata: checkpoint metadata (if any) overrides metadata.json
    meta = {}
    if isinstance(ckpt, dict):
        # Try to get metadata from checkpoint if available
        meta_ckpt = ckpt.get("metadata") or ckpt.get("meta") or {}
        if isinstance(meta_ckpt, dict):
            # Give priority to metadata.json over checkpoint metadata
            meta.update(meta_ckpt or {})
            meta.update(meta_from_file or {})
        else:
            # Only metadata.json available
            meta.update(meta_from_file or {})
    else:
        # If checkpoint is not a dict, rely solely on metadata.json
        meta.update(meta_from_file or {})

    # Always attach traceability info
    meta.setdefault("checkpoint_path", os.path.abspath(ckpt_path))
    if meta_from_file:
        meta.setdefault("metadata_file", os.path.abspath(meta_path))

    # move model and set eval
    model.to(device)
    model.eval()

    return model, meta


def try_fetch_model_via_tracker(
    tracker: TrackerFactory, model_name: str
) -> Optional[str]:
    """
    Try common tracker download methods. Return local path or None.
    """
    # try typical method names, best-effort
    methods = ["download_model", "get_model_path", "download_artifact", "download"]
    for m in methods:
        if hasattr(tracker, m):
            try:
                fn = getattr(tracker, m)
                # try a couple of calling conventions
                try:
                    candidate = fn(model_name)
                except TypeError:
                    try:
                        candidate = fn(model_name=model_name, artifact_path=model_name)
                    except TypeError:
                        candidate = fn(model_name=model_name)
                if candidate:
                    return candidate
            except Exception:
                # ignore method errors, try next
                continue
    return None


def load_model(
    model_or_name: str,
    tracker_type: str = "none",
    tracker_project: Optional[str] = None,
    tracker_task: Optional[str] = None,
    device_str: str = "cpu",
):
    # resolve candidate: local path or try tracker
    if os.path.exists(model_or_name):
        candidate = os.path.abspath(model_or_name)
        if os.path.isfile(candidate) and candidate.lower().endswith(".pt"):
            candidate = os.path.dirname(candidate)
    else:
        candidate = try_fetch_model_via_tracker(tracker_type, model_or_name)
        if (
            candidate
            and os.path.isfile(candidate)
            and candidate.lower().endswith(".pt")
        ):
            candidate = os.path.dirname(candidate)
        if not candidate:
            raise RuntimeError(
                f"Model '{model_or_name}' not found locally and tracker could not fetch it."
            )

    device = torch.device(device_str)
    model, metadata = load_model_from_path(candidate, device=device)
    return model, metadata

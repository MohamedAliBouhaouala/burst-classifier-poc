import os
from typing import List, Dict, Any, Optional

import torch
import torchaudio
import numpy as np
from torchaudio.transforms import MelSpectrogram

from types import SimpleNamespace

from helpers.common import resolve_settings, load_model_from_path
from helpers.constants import INV_LABEL_MAP
from helpers.helpers_predict import batch_infer_specs, seg_to_spec_tensor

def sliding_window_infer_model(
    model: torch.nn.Module,
    audio_path,                        # str or List[str]
    device_str: str = "cpu",
    window_seconds: float = 0.5,
    sr: int = 22050,
    n_mels: int = 64,
    batch_size: int = 64,
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    """
    Sliding-window inference.

    - Accepts audio_path as either a single path (str/Path) OR a list/tuple of paths.
    - If a list is passed, the function will iterate each file and concatenate results.
    - Returns a flat list of result dicts (may contain multiple rows per window if top_k>1).
    """
    # handle list of audio files by delegating to single-file calls
    if isinstance(audio_path, (list, tuple)):
        all_results: List[Dict[str, Any]] = []
        for p in audio_path:
            try:
                res = sliding_window_infer_model(
                    model=model,
                    audio_path=p,
                    device_str=device_str,
                    window_seconds=window_seconds,
                    sr=sr,
                    n_mels=n_mels,
                    batch_size=batch_size,
                    top_k=top_k
                )
                all_results.extend(res)
            except Exception as e:
                print(f"[WARN] failed to infer on {p}: {e}")
                continue
        return all_results

    # From here on, audio_path is a single file (string/Path)
    if not isinstance(audio_path, (str, os.PathLike)):
        raise ValueError("audio_path must be a string path or a list of paths")

    device = torch.device(device_str)

    # load audio once (wrap in try/except to provide a useful message)
    try:
        waveform, fs = torchaudio.load(str(audio_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {audio_path}: {e}") from e

    if fs != sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=fs, new_freq=sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    total_seconds = waveform.shape[1] / sr
    if total_seconds <= 0:
        raise RuntimeError(f"Audio has zero length: {audio_path}")

    # prepare mel transform (reuse)
    mel_transform = MelSpectrogram(sample_rate=sr, n_mels=n_mels)

    results: List[Dict[str, Any]] = []
    specs_batch: List[torch.Tensor] = []
    metas_batch: List[Dict[str, Any]] = []

    # audio file basename (traceability)
    audio_basename = os.path.basename(str(audio_path))

    start = 0.0
    # iterate windows (allow last window to be padded)
    while start < total_seconds:
        s_frame = int(round(start * sr))
        e_frame = int(round((start + window_seconds) * sr))
        seg = waveform[:, s_frame:e_frame]
        spec = seg_to_spec_tensor(seg, sr, window_seconds, n_mels, mel_transform)  # [1,n_mels,time]
        specs_batch.append(spec)
        metas_batch.append({"start_seconds": float(start), "end_seconds": float(min(start + window_seconds, total_seconds)), "audio_file": audio_basename})

        # flush batch
        if len(specs_batch) >= batch_size:
            probs_batch = batch_infer_specs(model, device, specs_batch)  # [B,C]
            for i in range(len(specs_batch)):
                p = probs_batch[i]  # numpy array length C
                per_class = {INV_LABEL_MAP[idx]: float(p[idx]) for idx in range(len(p))}
                topk_idx = np.argsort(p)[-top_k:][::-1] if top_k > 0 else np.argsort(p)[::-1]
                for idx in topk_idx:
                    label = INV_LABEL_MAP[int(idx)]
                    probability = float(p[int(idx)])
                    results.append({
                        "audio_file": metas_batch[i]["audio_file"],
                        "start_seconds": metas_batch[i]["start_seconds"],
                        "end_seconds": metas_batch[i]["end_seconds"],
                        "label": label,
                        "probability": probability,
                        "probabilities": p.tolist(),
                        "per_class_probability": per_class
                    })
            specs_batch = []
            metas_batch = []
        # @TODO: The hop size is currently fixed at 25% of window_seconds.
        # Make this ratio (hop_seconds / window_seconds) configurable in the future,
        # allowing flexible overlap control between consecutive windows.
        start += window_seconds * 0.25

    # flush remaining
    if len(specs_batch) > 0:
        probs_batch = batch_infer_specs(model, device, specs_batch)
        for i in range(len(specs_batch)):
            p = probs_batch[i]
            per_class = {INV_LABEL_MAP[idx]: float(p[idx]) for idx in range(len(p))}
            topk_idx = np.argsort(p)[-top_k:][::-1] if top_k > 0 else np.argsort(p)[::-1]
            for idx in topk_idx:
                label = INV_LABEL_MAP[int(idx)]
                probability = float(p[int(idx)])
                results.append({
                    "audio_file": metas_batch[i]["audio_file"],
                    "start_seconds": metas_batch[i]["start_seconds"],
                    "end_seconds": metas_batch[i]["end_seconds"],
                    "label": label,
                    "probability": probability,
                    "probabilities": p.tolist(),
                    "per_class_probability": per_class
                })

    return results

def predict(
    model: torch.nn.Module,
    metadata: Dict[str, Any],
    audio_path: str,
    tracker_type: str = "none",
    tracker_project: Optional[str] = None,
    tracker_task: Optional[str] = None,
    device_str: str = "cpu",
    window_seconds: float = 0.5,
    sr: int = 22050,
    n_mels: int = 64,
    batch_size: int = 64,
    top_k: int = 1,
) -> List[Dict[str, Any]]:

    fake_args = SimpleNamespace(
        window_seconds=window_seconds,
        sr=sr,
        n_mels=n_mels,
        batch_size=batch_size
    )

    print("METADATA")
    print(metadata["params"])
    resolved_settings = resolve_settings(metadata, fake_args)

    results = sliding_window_infer_model(
        model=model,
        audio_path=audio_path,
        device_str=device_str,
        window_seconds=resolved_settings["window_seconds"],
        sr=resolved_settings["sr"],
        n_mels=resolved_settings["n_mels"],
        batch_size=resolved_settings["batch_size"],
        top_k=top_k
    )

    return results

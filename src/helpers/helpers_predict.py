"""
Prediction helpers for Burst Classifier POC (fixed + optimized).
"""
import os
import json
from typing import List, Dict, Any, Optional

import torch
import torchaudio
import numpy as np
from torchaudio.transforms import MelSpectrogram

from models.model import SmallCNN
from dataset import INV_LABEL_MAP
from tracker import TrackerFactory

def seg_to_spec_tensor(
    waveform: torch.Tensor,
    sr: int,
    fixed_seconds: float,
    n_mels: int,
    mel_transform: MelSpectrogram
) -> torch.Tensor:
    """
    Convert waveform segment (shape [1, T]) to spectrogram tensor shape [1, n_mels, time_frames].
    Pads/trims to fixed_seconds.
    """
    target_len = int(round(fixed_seconds * sr))
    if waveform.shape[1] < target_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
    else:
        waveform = waveform[:, :target_len]
    spec = mel_transform(waveform)  # shape [channel, n_mels, time] (channel==1)
    # ensure single-channel
    if spec.dim() == 3 and spec.shape[0] > 1:
        spec = spec.mean(dim=0, keepdim=True)
    # log + normalize per-sample
    spec = torch.log1p(spec)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec  # dtype float32


def batch_infer_specs(
    model: torch.nn.Module,
    device: torch.device,
    specs: List[torch.Tensor]
) -> np.ndarray:
    """
    Run model on a batch of spec tensors (list of tensors shape [1, n_mels, time]).
    Returns numpy array shape [B, C] of softmax probs.
    """
    if len(specs) == 0:
        return np.zeros((0, 3), dtype=float)
    # stack -> [B, 1, n_mels, time]
    x = torch.stack(specs, dim=0).to(device).float()
    with torch.no_grad():
        logits = model(x)  # [B, C]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

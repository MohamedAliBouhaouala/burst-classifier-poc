from typing import Dict
import os
import random

import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np

from constants import AUDIO_FILE, START_SECONDS, END_SECONDS, LABEL
from helpers.constants import LABEL_MAP, INV_LABEL_MAP
from utils.common import dataset_hash, sha256_file
from helpers.dataset import (
    read_label_file,
    build_meta_from_dir,
    build_and_write_dataset_manifest,
)


class SegmentDataset(Dataset):
    """
    Extract labeled segments from audio, return log-mel spectrograms.
    """

    def __init__(
        self,
        meta_df: pd.DataFrame,
        audio_root: str,
        sr: int = 22050,
        n_mels: int = 64,
        fixed_seconds: float = 0.5,
        transform=None,
        augment: bool = False,
        seed: int = 42,
    ):
        self.meta = meta_df.reset_index(drop=True).copy()
        self.audio_root = os.path.abspath(audio_root)
        self.sr = int(sr)
        self.n_mels = int(n_mels)
        self.fixed_seconds = float(fixed_seconds)
        self.transform = transform
        self.augment = augment
        self.rng = random.Random(seed)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_mels=self.n_mels
        )
        # cache durations to reduce repeated IO
        self._dur_cache: Dict[str, float] = {}

    def __len__(self) -> int:
        return len(self.meta)

    def _load_waveform(self, file_path: str) -> torch.Tensor:
        """
        Loads waveform and returns mono waveform tensor shape [1, T] at self.sr.
        """
        waveform, sr = torchaudio.load(file_path)  # shape [channels, T]
        if sr != self.sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sr
            )
        # to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform  # [1, T]

    def _apply_augmentation(self, seg: torch.Tensor) -> torch.Tensor:
        # seg: [1, T]
        if self.rng.random() < 0.5:
            shift = int(self.rng.uniform(-0.1, 0.1) * seg.shape[1])
            seg = torch.roll(seg, shifts=shift, dims=1)
        if self.rng.random() < 0.5:
            noise = torch.randn_like(seg) * 0.005
            seg = seg + noise
        return seg

    def __getitem__(self, idx: int):
        row = self.meta.loc[idx]
        rel_audio = row[AUDIO_FILE]
        audio_path = os.path.join(self.audio_root, rel_audio)
        start_frame = int(row[START_SECONDS] * self.sr)
        end_frame = int(row[END_SECONDS] * self.sr)
        target_len = int(self.fixed_seconds * self.sr)

        try:
            waveform = self._load_waveform(audio_path)  # [1, T]
            # extract segment
            if start_frame < 0:
                start_frame = 0
            if end_frame <= start_frame:
                # fallback: center small window at start_frame
                end_frame = start_frame + 1
            seg = waveform[:, start_frame:end_frame]  # may be shorter/longer
            # pad or trim to target_len
            if seg.shape[1] < target_len:
                pad = target_len - seg.shape[1]
                seg = torch.nn.functional.pad(seg, (0, pad))
            else:
                seg = seg[:, :target_len]
            if self.augment:
                seg = self._apply_augmentation(seg)
            # mel spectrogram
            spec = self.melspec(seg)  # shape [channel, n_mels, time]
            # ensure spec shape [1, n_mels, time]
            if spec.dim() == 3 and spec.shape[0] > 1:
                spec = spec.mean(dim=0, keepdim=True)
            # log1p and normalize per-sample
            spec = torch.log1p(spec)
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)
            label = LABEL_MAP.get(row[LABEL], -1)
            meta = {
                AUDIO_FILE: rel_audio,
                START_SECONDS: float(row[START_SECONDS]),
                END_SECONDS: float(row[END_SECONDS]),
                LABEL: row[LABEL],
            }
            return spec.float(), int(label), meta
        except FileNotFoundError:
            # Return zero tensor placeholder if missing file (training will continue but manifest will flag missing)
            approx_frames = max(1, int(np.ceil(target_len / 512)))
            spec = torch.zeros((1, self.n_mels, approx_frames), dtype=torch.float32)
            label = LABEL_MAP.get(row[LABEL], -1)
            meta = {
                AUDIO_FILE: rel_audio,
                START_SECONDS: float(row[START_SECONDS]),
                END_SECONDS: float(row[END_SECONDS]),
                LABEL: row[LABEL],
                "missing": True,
            }
            return spec, int(label), meta
        except Exception:
            approx_frames = max(1, int(np.ceil(target_len / 512)))
            spec = torch.zeros((1, self.n_mels, approx_frames), dtype=torch.float32)
            label = LABEL_MAP.get(row[LABEL], -1)
            meta = {
                AUDIO_FILE: rel_audio,
                START_SECONDS: float(row[START_SECONDS]),
                END_SECONDS: float(row[END_SECONDS]),
                LABEL: row[LABEL],
                ERROR: True,
            }
            return spec, int(label), meta


__all__ = [
    "read_label_file",
    "build_meta_from_dir",
    "SegmentDataset",
    "LABEL_MAP",
    "INV_LABEL_MAP",
    "sha256_file",
    "dataset_hash",
    "build_and_write_dataset_manifest",
]

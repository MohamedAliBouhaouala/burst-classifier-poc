import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from .constants import SEED
from dataset import SegmentDataset


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_loader_from_meta(meta_df, data_dir, args, augment):
    durations = (
        (meta_df["end_seconds"] - meta_df["start_seconds"])
        .astype(float)
        .clip(lower=1e-6)
    )
    if durations.empty:
        computed_fixed_seconds = float(args.fixed_seconds)  # fallback
    else:
        max_dur = float(durations.max())
        # choose the larger of CLI-provided fixed_seconds and observed max_dur
        computed_fixed_seconds = max(float(args.fixed_seconds), max_dur)
    args.fixed_seconds = computed_fixed_seconds
    print(
        f"Computed fixed_seconds for dataset (max): {computed_fixed_seconds:.3f}s (label max={max_dur:.3f}s, cli={args.fixed_seconds})"
    )

    ds = SegmentDataset(
        meta_df,
        audio_root=data_dir,
        fixed_seconds=computed_fixed_seconds,
        n_mels=args.n_mels,
        augment=augment,
        seed=SEED,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=augment)
    return loader

"""PyTorch dataset over the stored log-mel spectrograms.

Reads one part (train, val or test) of a split file such as data/splits/fold_0.csv, loads
those clips' 128 x 400 spectrograms into RAM once (float16, about 100 KB per clip), and
returns for each clip:
    x : float32 tensor [1, 128, 400], standardised as (dB - mean) / std with the split's
        TRAIN-only global mean and std from norm_stats_<split>.json (written by preprocessing)
    y : label id 0..6

Optional (used by experiments, defaults reproduce the baseline exactly):
    split_df  a split table to use instead of the fold's CSV (e.g. one training actor held out as val)
    stats     {"mean", "std"} to use instead of norm_stats_<split>.json (must come from that split's TRAIN)
    augment   src.data.augment.LogMelAugment; only ever passed for the train part
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils import logmel_dir as logmel_path


def split_paths(cfg: dict, fold: int | str):
    s = cfg["paths"]["splits_dir"]
    if fold == "random":
        return s / "random_split.csv", s / "norm_stats_random.json"
    return s / f"fold_{fold}.csv", s / f"norm_stats_fold_{fold}.json"


class SpectrogramDataset(Dataset):
    def __init__(self, cfg: dict, fold: int | str, part: str, split_df: pd.DataFrame | None = None,
                 stats: dict | None = None, augment=None):
        split_csv, stats_json = split_paths(cfg, fold)
        f = split_df if split_df is not None else pd.read_csv(split_csv)
        f = f[f["split"] == part].sort_values("clip_id").reset_index(drop=True)
        logmel_dir = logmel_path(cfg)
        index = pd.read_csv(logmel_dir / "index.csv")
        items = f.merge(index[["clip_id", "file", "row", "n_frames_valid"]], on="clip_id", how="left",
                        validate="one_to_one")
        assert items["row"].notna().all(), "some clips have no stored spectrogram"
        if stats is None:
            stats = json.loads(stats_json.read_text(encoding="utf-8"))
        self.mean, self.std = float(stats["mean"]), float(stats["std"])
        assert augment is None or part == "train", "augmentation is for training clips only"
        self.augment = augment

        arrays = {fn: np.load(logmel_dir / fn, mmap_mode="r") for fn in items["file"].unique()}
        first = next(iter(arrays.values()))
        self.x = np.empty((len(items),) + first.shape[1:], dtype=first.dtype)  # RAM copy, float16
        for fn, g in items.groupby("file"):
            self.x[g.index.to_numpy()] = arrays[fn][g["row"].to_numpy().astype(int)]
        self.y = items["label"].to_numpy(dtype=np.int64)
        self.n_valid = items["n_frames_valid"].to_numpy(dtype=np.int64)
        self.clip_ids = items["clip_id"].tolist()
        self.items = items

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.x[i].astype(np.float32)
        if self.augment is None:
            x = (x - self.mean) / self.std
        else:
            x, start, end = self.augment.db_stage(x, int(self.n_valid[i]))
            x = self.augment.mask_stage((x - self.mean) / self.std, start, end)
        return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0), int(self.y[i])

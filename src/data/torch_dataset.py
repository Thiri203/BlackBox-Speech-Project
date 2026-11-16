"""PyTorch dataset over the stored log-mel spectrograms.

Reads one part (train, val or test) of a split file such as data/splits/fold_0.csv, loads
those clips' 128 x 400 spectrograms into RAM once (float16, about 100 KB per clip), and
returns for each clip:
    x : float32 tensor [1, 128, 400], standardised as (dB - mean) / std with the split's
        TRAIN-only global mean and std from norm_stats_<split>.json (written by preprocessing)
    y : label id 0..6
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def split_paths(cfg: dict, fold: int | str):
    s = cfg["paths"]["splits_dir"]
    if fold == "random":
        return s / "random_split.csv", s / "norm_stats_random.json"
    return s / f"fold_{fold}.csv", s / f"norm_stats_fold_{fold}.json"


class SpectrogramDataset(Dataset):
    def __init__(self, cfg: dict, fold: int | str, part: str):
        split_csv, stats_json = split_paths(cfg, fold)
        f = pd.read_csv(split_csv)
        f = f[f["split"] == part].sort_values("clip_id").reset_index(drop=True)
        logmel_dir = cfg["paths"]["processed_dir"] / "logmel"
        index = pd.read_csv(logmel_dir / "index.csv")
        items = f.merge(index[["clip_id", "file", "row"]], on="clip_id", how="left", validate="one_to_one")
        assert items["row"].notna().all(), "some clips have no stored spectrogram"
        stats = json.loads(stats_json.read_text(encoding="utf-8"))
        self.mean, self.std = float(stats["mean"]), float(stats["std"])

        arrays = {fn: np.load(logmel_dir / fn, mmap_mode="r") for fn in items["file"].unique()}
        first = next(iter(arrays.values()))
        self.x = np.empty((len(items),) + first.shape[1:], dtype=first.dtype)  # RAM copy, float16
        for fn, g in items.groupby("file"):
            self.x[g.index.to_numpy()] = arrays[fn][g["row"].to_numpy().astype(int)]
        self.y = items["label"].to_numpy(dtype=np.int64)
        self.clip_ids = items["clip_id"].tolist()
        self.items = items

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = (self.x[i].astype(np.float32) - self.mean) / self.std
        return torch.from_numpy(x).unsqueeze(0), int(self.y[i])

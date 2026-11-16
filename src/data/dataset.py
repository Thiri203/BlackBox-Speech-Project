"""Dataset class stub for the ResNet stage (torch is NOT installed yet).

It already reads the real preprocessing outputs, so the model code only has to wrap it in
a torch DataLoader later. It returns numpy arrays; convert with torch.from_numpy(...) then.

Usage later:
    ds = LogMelDataset(cfg, fold=0, split="train", normalise=True)       # or fold="random"
    x, y, n_valid, clip_id = ds[0]     # x: [n_mels, n_frames] = [128, 400] float32
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


class LogMelDataset:
    def __init__(self, cfg: dict, fold: int | str, split: str, normalise: bool = True):
        """fold: 0..3 for the leave-one-actor-out folds, or "random" for the random split."""
        splits_dir = cfg["paths"]["splits_dir"]
        logmel_dir = cfg["paths"]["processed_dir"] / "logmel"
        csv, tag = ("random_split.csv", "random") if fold == "random" else (f"fold_{fold}.csv", f"fold_{fold}")
        f = pd.read_csv(splits_dir / csv)
        f = f[f["split"] == split]
        index = pd.read_csv(logmel_dir / "index.csv")
        self.items = f.merge(index, on=["clip_id", "actor"], how="left").reset_index(drop=True)
        assert self.items["row"].notna().all(), "some split clips have no spectrogram"
        self._arrays = {fn: np.load(logmel_dir / fn, mmap_mode="r") for fn in self.items["file"].unique()}
        self.normalise = normalise
        if normalise:
            stats = json.loads((splits_dir / f"norm_stats_{tag}.json").read_text(encoding="utf-8"))
            # Normalisation uses TRAIN statistics of this fold only (global mean/std in dB).
            self.mean, self.std = stats["mean"], stats["std"]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        r = self.items.iloc[i]
        x = np.asarray(self._arrays[r["file"]][int(r["row"])], dtype=np.float32)
        if self.normalise:
            x = (x - self.mean) / self.std
        return x, int(r["label"]), int(r["n_frames_valid"]), r["clip_id"]

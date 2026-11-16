"""Training-only augmentation on cached log-mel spectrograms (Change A).

Applied to TRAIN clips only, never to validation or test. Works on one clip [n_mels, n_frames] in dB,
with the number of real (non-padding) frames known from logmel/index.csv. Order:

  in dB (before standardisation)
    1. frequency shift: move the mel axis by k bins, k uniform in [-max, +max] (small pitch shift).
       Edge bands are repeated into the vacated rows (no wrap-around: top bands never jump to the bottom).
    2. time shift: move the clip by s frames, s uniform in [-max, +max] (0.25 s = 25 frames at 10 ms hop).
       Vacated frames get the padding value (floor, -100 dB); real-frame region moves with the audio.
    3. random gain: add g dB, g uniform in [-max, +max], to every real value above the floor
       (like turning the volume up/down; padding stays at the floor), then clamp at the floor.
    4. light Gaussian noise: add N(0, sigma) dB to real values above the floor, then clamp at the floor.
  after standardisation
    5. SpecAugment: 1 or 2 frequency masks (width 0..max bins) and 1 or 2 time masks (width 0..max frames,
       placed inside the real-frame region); masked values are set to 0 = the training mean.

Randomness comes from one numpy Generator seeded with the run seed (data loading runs in the main
process, num_workers = 0), so runs are repeatable.
"""
from __future__ import annotations

import numpy as np


class LogMelAugment:
    def __init__(self, params: dict, floor_db: float, seed: int):
        self.p = params
        self.floor = float(floor_db)
        self.rng = np.random.default_rng(seed)

    def db_stage(self, x: np.ndarray, n_valid: int) -> tuple[np.ndarray, int, int]:
        """x: float32 [n_mels, T] in dB. Returns (x, real_start, real_end)."""
        r, p, floor = self.rng, self.p, self.floor
        n_mels, T = x.shape
        start, end = 0, int(n_valid)

        k = int(r.integers(-p["freq_shift_bins"], p["freq_shift_bins"] + 1))
        if k > 0:
            x = np.concatenate([np.repeat(x[:1], k, axis=0), x[:-k]], axis=0)
        elif k < 0:
            x = np.concatenate([x[-k:], np.repeat(x[-1:], -k, axis=0)], axis=0)

        s = int(r.integers(-p["time_shift_frames"], p["time_shift_frames"] + 1))
        if s > 0:
            x = np.concatenate([np.full((n_mels, s), floor, np.float32), x[:, :-s]], axis=1)
            start, end = s, min(end + s, T)
        elif s < 0:
            x = np.concatenate([x[:, -s:], np.full((n_mels, -s), floor, np.float32)], axis=1)
            start, end = 0, max(end + s, 0)

        real = np.zeros_like(x, dtype=bool)
        real[:, start:end] = True
        real &= x > floor
        g = r.uniform(-p["gain_db"], p["gain_db"])
        noise = r.normal(0.0, p["noise_std_db"], size=x.shape).astype(np.float32)
        x = np.where(real, np.maximum(x + g + noise, floor), x).astype(np.float32)
        return x, start, end

    def mask_stage(self, z: np.ndarray, start: int, end: int) -> np.ndarray:
        """z: standardised [n_mels, T]; masks are filled with 0 (= training mean)."""
        r, p = self.rng, self.p
        n_mels, _ = z.shape
        z = z.copy()
        for _ in range(int(r.integers(1, p["freq_masks_max"] + 1))):
            w = int(r.integers(0, p["freq_mask_bins"] + 1))
            f0 = int(r.integers(0, n_mels - w + 1))
            z[f0:f0 + w, :] = 0.0
        real_len = max(end - start, 0)
        for _ in range(int(r.integers(1, p["time_masks_max"] + 1))):
            w = int(r.integers(0, min(p["time_mask_frames"], real_len) + 1))
            t0 = start + int(r.integers(0, real_len - w + 1))
            z[:, t0:t0 + w] = 0.0
        return z

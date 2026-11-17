"""Shared helpers for the time-segment explainers (same code for LRfS and the ResNet).

All explainers work on the model's real input: a standardised log-mel [N, 1, 128, 400],
z = (dB - g_mean) / g_std with the fold's train-only global statistics. Deleting a segment means
writing a per-band fill (dB, train-only) into all 128 x 25 values of that segment; padding segments
are never touched (they stay at -100 dB) and are never ranked.
"""
from __future__ import annotations

import numpy as np
import torch

N_SEG = 16
SEG_FRAMES = 25


def fill_segments(x_db: np.ndarray, off: np.ndarray, band_fill: np.ndarray) -> np.ndarray:
    """x_db [128, 400]; off [16] bool (segments to delete) -> copy with those segments filled."""
    out = x_db.astype(np.float32, copy=True)
    for j in np.flatnonzero(off):
        out[:, j * SEG_FRAMES:(j + 1) * SEG_FRAMES] = band_fill[:, None]
    return out


def fill_batch(x_db: np.ndarray, keep: np.ndarray, band_fill: np.ndarray) -> np.ndarray:
    """x_db [128, 400]; keep [S, 16] bool (True = segment kept) -> [S, 128, 400] float32."""
    frame_keep = np.repeat(keep, SEG_FRAMES, axis=1)                        # [S, 400]
    return np.where(frame_keep[:, None, :], x_db[None].astype(np.float32),
                    band_fill.astype(np.float32)[None, :, None])


def to_input(x_db: np.ndarray, g_mean: float, g_std: float, device) -> torch.Tensor:
    """[S, 128, 400] dB -> standardised [S, 1, 128, 400] float32 tensor on device."""
    return ((torch.from_numpy(np.ascontiguousarray(x_db, dtype=np.float32)) - g_mean) / g_std).unsqueeze(1).to(device)


def pixels_to_segments(attr: np.ndarray) -> np.ndarray:
    """[128, 400] pixel attributions -> [16] sums per segment."""
    return attr.reshape(attr.shape[0], N_SEG, SEG_FRAMES).sum(axis=(0, 2))


def make_predict_fn(model: torch.nn.Module, g_mean: float, g_std: float, device, batch: int = 128):
    """Returns f(x_db [S, 128, 400]) -> logits [S, 7] numpy (float32, eval mode, no autocast)."""
    model.eval()

    @torch.no_grad()
    def f(x_db: np.ndarray) -> np.ndarray:
        out = []
        for a in range(0, len(x_db), batch):
            out.append(model(to_input(x_db[a:a + batch], g_mean, g_std, device)).float().cpu().numpy())
        return np.concatenate(out)

    return f

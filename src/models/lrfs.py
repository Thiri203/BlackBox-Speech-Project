"""LRfS: logistic regression on the 16 segments of 250 ms, a white-box check for the explainers.

Features. For a clip's stored log-mel spectrogram X (128 mel bands x 400 frames, dB, padding at
-100 dB) the 16 features are

    x_j = mean of X over all 128 bands and all 25 frames of segment j      (j = 0..15)

Padding frames count at -100 dB, so x is an exact linear function of the same 128 x 400 input the
ResNet gets. Segment j covers frames 25 j .. 25 j + 24 (250 ms).

Linear form. With the train-only StandardScaler (mu_j, s_j) and the fitted weights w, b:

    logit_c(X) = b0_c + sum_j V_cj x_j,   V_cj = w_cj / s_j,   b0_c = b_c - sum_j w_cj mu_j / s_j

and, as a weight map on the raw dB spectrogram, W_db[c, m, t] = V_c,(t // 25) / (128 * 25).
The ResNet input is standardised, z = (X - g_mean) / g_std, so on z the same model is

    logit_c(z) = (b0_c + g_mean * sum_j V_cj) + sum_{m,t} (g_std * W_db[c, m, t]) z[m, t]

`LRfSTorch` implements this, so explainer code written for the ResNet runs unchanged on LRfS.

Exact segment contributions (for the PREDICTED class's logit, c = argmax):

    phi_j = V_cj (x_j - f) for real segments,   phi_j = 0 for pure padding segments

f is the scalar fill: the mean over the 128 bands of a per-band fill. Replacing every real segment
of the spectrogram with the per-band fill sets each such x_j to exactly f, so

    sum over real j of phi_j = logit_c(X) - logit_c(X with every real segment filled)

The identity holds over real segments only; padding segments stay at -100 dB on both sides.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

N_SEG = 16
SEG_FRAMES = 25


def segment_features(X: np.ndarray, chunk: int = 512) -> np.ndarray:
    """X: [N, 128, 400] dB (any float dtype) -> [N, 16] float64 means over bands and frames."""
    n, n_mels, n_frames = X.shape
    assert n_frames == N_SEG * SEG_FRAMES, f"expected {N_SEG * SEG_FRAMES} frames, got {n_frames}"
    out = np.empty((n, N_SEG), dtype=np.float64)
    for a in range(0, n, chunk):
        b = X[a:a + chunk].astype(np.float64).reshape(-1, n_mels, N_SEG, SEG_FRAMES)
        out[a:a + chunk] = b.mean(axis=(1, 3))
    return out


def real_segment_mask(n_frames_valid: np.ndarray) -> np.ndarray:
    """[N] real frames per clip -> [N, 16] bool: segment holds at least one real frame."""
    starts = np.arange(N_SEG) * SEG_FRAMES
    return starts[None, :] < np.asarray(n_frames_valid)[:, None]


def linear_form(coef: np.ndarray, intercept: np.ndarray, mu: np.ndarray, scale: np.ndarray):
    """Fold the StandardScaler into the weights: returns V [7, 16] (per dB) and b0 [7]."""
    V = coef / scale[None, :]
    b0 = intercept - coef @ (mu / scale)
    return V, b0


def logits_from_features(x: np.ndarray, V: np.ndarray, b0: np.ndarray) -> np.ndarray:
    return x @ V.T + b0[None, :]


def pixel_weights_db(V: np.ndarray, n_mels: int = 128) -> np.ndarray:
    """[7, 16] -> [7, 128, 400]: weight of every spectrogram value (raw dB) in each class logit."""
    w = np.repeat(V, SEG_FRAMES, axis=1) / (n_mels * SEG_FRAMES)       # [7, 400]
    return np.broadcast_to(w[:, None, :], (V.shape[0], n_mels, w.shape[1])).copy()


def contributions(x: np.ndarray, real: np.ndarray, V: np.ndarray, cls: np.ndarray, fill: float) -> np.ndarray:
    """phi [N, 16] for class cls[n] of each clip; 0 on padding segments."""
    return V[cls] * (x - fill) * real


class LRfSTorch(nn.Module):
    """LRfS on the ResNet's standardised input [N, 1, 128, 400] -> logits [N, 7]. Exact, no training."""

    def __init__(self, V: np.ndarray, b0: np.ndarray, g_mean: float, g_std: float, n_mels: int = 128):
        super().__init__()
        W = pixel_weights_db(V, n_mels)                                     # raw-dB weight map
        self.register_buffer("weight", torch.from_numpy(g_std * W).float())  # weight on z
        self.register_buffer("bias", torch.from_numpy(b0 + g_mean * V.sum(axis=1)).float())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nmt,cmt->nc", z[:, 0].to(self.weight.dtype), self.weight) + self.bias

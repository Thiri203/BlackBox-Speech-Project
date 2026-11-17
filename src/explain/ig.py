"""Integrated Gradients on the standardised log-mel input, aggregated to 16 time segments.

attr = (x - x') * mean over alpha of dF/dx at x' + alpha (x - x'), midpoint Riemann sum with
`steps` points (alpha = (k + 0.5) / steps). F is the target class's logit. The baseline x' is the
clip with every real segment replaced by the fill (padding unchanged), so attributions on padding
segments are exactly 0 and the segment sums are directly comparable with the exact contributions.
"""
from __future__ import annotations

import numpy as np
import torch

from src.explain.segments import fill_segments, pixels_to_segments, to_input


def integrated_gradients(model: torch.nn.Module, x_db: np.ndarray, real: np.ndarray, band_fill: np.ndarray,
                         target: int, g_mean: float, g_std: float, device, steps: int = 50,
                         batch: int = 50, baseline_noise: np.ndarray | None = None,
                         alpha_jitter: np.ndarray | None = None) -> dict:
    """Stability options (both None by default = the main-result IG):
    baseline_noise [128, 400] dB: added to the FILLED (real) segments of the baseline only;
    alpha_jitter [steps] in [0, 1): integration points alpha_k = (k + u_k) / steps instead of (k + 0.5) / steps."""
    model.eval()
    x = to_input(x_db[None], g_mean, g_std, device)                              # [1, 1, 128, 400]
    base_db = fill_segments(x_db, real, band_fill)
    if baseline_noise is not None:
        frame_real = np.repeat(real, base_db.shape[1] // len(real))
        base_db = base_db + np.where(frame_real[None, :], baseline_noise, 0.0).astype(np.float32)
    xb = to_input(base_db[None], g_mean, g_std, device)
    diff = x - xb
    offs = torch.full((steps,), 0.5) if alpha_jitter is None else torch.as_tensor(alpha_jitter, dtype=torch.float32)
    alphas = ((torch.arange(steps, dtype=torch.float32) + offs) / steps).to(device)
    grad_sum = torch.zeros_like(x)
    for a in range(0, steps, batch):
        al = alphas[a:a + batch].view(-1, 1, 1, 1)
        inp = (xb + al * diff).detach().requires_grad_(True)
        out = model(inp)[:, target].sum()
        (g,) = torch.autograd.grad(out, inp)
        grad_sum += g.sum(0, keepdim=True)
    attr = (diff * grad_sum / steps)[0, 0].detach().cpu().numpy().astype(np.float64)
    with torch.no_grad():
        f_x = float(model(x)[0, target])
        f_b = float(model(xb)[0, target])
    seg = pixels_to_segments(attr)
    return {"segments": seg, "f_input": f_x, "f_baseline": f_b,
            "completeness_gap": float(attr.sum() - (f_x - f_b))}

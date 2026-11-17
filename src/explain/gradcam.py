"""Grad-CAM on a convolutional layer, turned into 16 time-segment scores.

cam = ReLU( sum_k alpha_k A_k ),  alpha_k = spatial mean of d logit_target / d A_k,
upsampled bilinearly to 128 x 400 and summed per segment. No fill is involved. For the ResNet-18 on
a 128 x 400 input, layer4 gives a 4 x 13 map, so the time resolution is about 31 frames (0.31 s),
coarser than one 250 ms segment. Grad-CAM needs a convolutional layer, so it does not apply to LRfS.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.explain.segments import pixels_to_segments, to_input


def grad_cam(model: torch.nn.Module, layer: torch.nn.Module, x_db: np.ndarray, target: int,
             g_mean: float, g_std: float, device) -> dict:
    model.eval()
    store = {}
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        x = to_input(x_db[None], g_mean, g_std, device)
        logit = model(x)[0, target]
        a = store["a"]
        (g,) = torch.autograd.grad(logit, a)
    finally:
        h.remove()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy().astype(np.float64)
    return {"segments": pixels_to_segments(cam), "map_shape": tuple(a.shape[-2:])}

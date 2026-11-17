"""LIME on time segments: switch real 250 ms segments off (write the fill) and fit a local linear model.

Follows the defaults of lime_image / lime_base (lime 0.2), re-implemented so it runs on our input:
- interpretable features: the clip's REAL segments (padding segments are never switched, score 0);
- samples: first row all on (the original clip), the rest uniform on/off (p = 0.5 each);
- distance: cosine distance of each on/off row to the all-on row; kernel sqrt(exp(-d^2 / 0.25^2));
- surrogate: weighted Ridge(alpha = 1) with intercept; the coefficient of a segment is its score.
The explained quantity is the target class's logit (the same quantity as the exact contributions).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances

from src.explain.segments import N_SEG, fill_batch


def lime_segments(predict_fn, x_db: np.ndarray, real: np.ndarray, band_fill: np.ndarray, target: int,
                  rng: np.random.Generator, n_samples: int = 500, kernel_width: float = 0.25,
                  alpha: float = 1.0) -> dict:
    idx = np.flatnonzero(real)
    Z = rng.integers(0, 2, size=(n_samples, len(idx))).astype(np.float64)
    Z[0] = 1.0
    keep = np.ones((n_samples, N_SEG), dtype=bool)          # padding segments stay as they are
    keep[:, idx] = Z.astype(bool)
    y = predict_fn(fill_batch(x_db, keep, band_fill))[:, target].astype(np.float64)
    d = pairwise_distances(Z, Z[:1], metric="cosine").ravel()
    w = np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    reg = Ridge(alpha=alpha, fit_intercept=True).fit(Z, y, sample_weight=w)
    seg = np.zeros(N_SEG)
    seg[idx] = reg.coef_
    return {"segments": seg, "r2": float(reg.score(Z, y, sample_weight=w)), "intercept": float(reg.intercept_)}

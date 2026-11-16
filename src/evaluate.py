"""Test-set evaluation helpers: metrics, confusions, shortcut baselines.

Used by src/train.py once, at the end of a run, on the best checkpoint's test predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def classification_metrics(y_true, y_pred, styles: list[str]) -> dict:
    labels = list(range(len(styles)))
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {s: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(n[i])}
                      for i, s in enumerate(styles)},
        "confusion_counts": cm.tolist(),
        "confusion_row_normalised": (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).round(4).tolist(),
    }


def top_confusions(cm: list[list[int]], styles: list[str], k: int = 8) -> list[dict]:
    cm = np.asarray(cm)
    rows = []
    for i in range(len(styles)):
        for j in range(len(styles)):
            if i != j and cm[i, j] > 0:
                rows.append({"true": styles[i], "predicted": styles[j], "count": int(cm[i, j]),
                             "share_of_true_class": round(float(cm[i, j] / cm[i].sum()), 4)})
    return sorted(rows, key=lambda r: (-r["count"], r["true"], r["predicted"]))[:k]


def accuracy_by(pred: pd.DataFrame, col: str) -> dict:
    """pred has columns y_true, y_pred and `col`."""
    out = {}
    for v, g in pred.groupby(col, sort=True):
        out[str(v)] = {"n": int(len(g)), "accuracy": round(float((g["y_true"] == g["y_pred"]).mean()), 4)}
    return out


def length_baseline(manifest: pd.DataFrame, split_df: pd.DataFrame, styles: list[str], seed: int) -> dict:
    """Shortcut check: logistic regression on duration_s and was_cropped only (train -> test)."""
    m = manifest.set_index("clip_id")
    d = split_df.join(m[["duration_s", "was_cropped"]], on="clip_id")
    X = lambda g: np.c_[g["duration_s"].to_numpy(float), g["was_cropped"].astype(float).to_numpy()]  # noqa: E731
    tr, te = d[d["split"] == "train"], d[d["split"] == "test"]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    clf.fit(X(tr), tr["label"])
    pred = clf.predict(X(te))
    labels = list(range(len(styles)))
    majority = int(tr["label"].value_counts().sort_index().idxmax())
    maj_pred = np.full(len(te), majority)
    return {
        "length_only_logreg": {
            "features": ["duration_s (real audio in the 4 s window)", "was_cropped"],
            "accuracy": float(accuracy_score(te["label"], pred)),
            "macro_f1": float(f1_score(te["label"], pred, labels=labels, average="macro", zero_division=0)),
            "per_class_recall": dict(zip(styles, precision_recall_fscore_support(
                te["label"], pred, labels=labels, zero_division=0)[1].round(4).tolist())),
        },
        "majority_class": {
            "class": styles[majority],
            "accuracy": float(accuracy_score(te["label"], maj_pred)),
            "macro_f1": float(f1_score(te["label"], maj_pred, labels=labels, average="macro", zero_division=0)),
        },
        "chance": {"accuracy": 1 / len(styles), "macro_f1_note": "1/7 expected for uniform random guessing"},
    }

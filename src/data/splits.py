"""Step 5: data splits and per-split statistics.

Grouping unit: the "id group" = actor + id number (manifest column group_id). The number at
the end of an id (00260 in ex01_happy_00260) marks the same prompt in all 7 styles of one
actor, even when a transcript has a small wording variant ("I'm" / "I am"). All clips of one
id group always land on the same side of a split, so the model cannot score by recognising a
sentence it trained on in another style.

1) Leave-one-actor-out (LOAO), 4 folds. In fold k:
   - test  = every clip of actor k (the model never hears this voice during training),
   - train / val = the other 3 actors; val is about 20%, stratified by actor x style and
     grouped by id group.
2) Random split (one split, all 4 actors in every part): about 60% train, 15% val, 25% test,
   stratified by actor x style (hence also by style) and grouped by id group. Only the
   speaker can "leak": the same voice is heard in train and test, but never the same prompt.

Output files (simple CSV, used by the ResNet and by the logistic regression baseline):
  data/splits/fold_{k}.csv, random_split.csv   columns: clip_id, actor, style, label, split
                                               (split = train | val | test)
  data/splits/folds.json, random_split.json    what each split file contains
  data/splits/norm_stats_{fold_k|random}.json  mean / std of the TRAIN spectrograms (not applied)
  data/splits/fill_values_{fold_k|random}.json per-band mean and 10th-percentile dB from TRAIN,
                                               fills for deletion tests
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils import logmel_dir

SPLIT_COLS = ["clip_id", "actor", "style", "label", "split"]


def _one_group_fold(n_splits: int, seed: int, strat, groups) -> np.ndarray:
    """Indices of the first held-out part of a stratified, grouped K-fold split."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held = next(sgkf.split(np.zeros(len(groups)), strat, groups))
    return held


def _n_splits_for(fraction: float) -> int:
    n = int(round(1.0 / fraction))
    assert abs(1.0 / n - fraction) < 1e-9, f"fraction {fraction} must be 1/n for a whole number n"
    return n


def make_folds(cfg: dict, manifest: pd.DataFrame) -> dict[int, pd.DataFrame]:
    seed = cfg["seed"]
    n_splits = _n_splits_for(cfg["splits"]["val_fraction"])
    actors = sorted(manifest["actor"].unique())
    folds = {}
    for k, test_actor in enumerate(actors):
        df = manifest.sort_values("clip_id").reset_index(drop=True)
        split = np.where(df["actor"] == test_actor, "test", "train").astype(object)
        pool = np.flatnonzero(df["actor"] != test_actor)
        strat = (df.loc[pool, "actor"] + "|" + df.loc[pool, "style"]).to_numpy()
        val_local = _one_group_fold(n_splits, seed, strat, df.loc[pool, "group_id"].to_numpy())
        split[pool[val_local]] = "val"
        df["split"] = split
        check_split(df)
        assert (df.loc[df["split"] == "test", "actor"] == test_actor).all(), "non-test actor in test"
        assert not (df.loc[df["split"] != "test", "actor"] == test_actor).any(), "test actor in train/val"
        folds[k] = df[SPLIT_COLS].copy()
    return folds


def make_random_split(cfg: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    """One random split: test first, then val from the rest; both grouped by id group."""
    r = cfg["splits"]["random"]
    assert abs(r["train"] + r["val"] + r["test"] - 1) < 1e-9, "random split fractions must sum to 1"
    seed = cfg["seed"]
    df = manifest.sort_values("clip_id").reset_index(drop=True)
    strat = (df["actor"] + "|" + df["style"]).to_numpy()
    split = np.full(len(df), "train", dtype=object)
    test = _one_group_fold(_n_splits_for(r["test"]), seed, strat, df["group_id"].to_numpy())
    split[test] = "test"
    rest = np.flatnonzero(split == "train")
    val_local = _one_group_fold(_n_splits_for(r["val"] / (1 - r["test"])), seed, strat[rest],
                                df.loc[rest, "group_id"].to_numpy())
    split[rest[val_local]] = "val"
    df["split"] = split
    check_split(df)
    for part, g in df.groupby("split"):
        assert set(g["actor"]) == set(df["actor"]), f"random split: not every actor in {part}"
        assert set(g["style"]) == set(df["style"]), f"random split: not every style in {part}"
    return df[SPLIT_COLS].copy()


def check_split(df: pd.DataFrame) -> None:
    """Hard checks shared by every split; any failure stops the pipeline."""
    assert df["clip_id"].is_unique, "a clip appears twice in a split"
    assert set(df["split"]) == {"train", "val", "test"}
    parts = {p: set(g["group_id"]) for p, g in df.groupby("split")}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = parts[a] & parts[b]
        assert not shared, f"{len(shared)} id groups shared between {a} and {b}"


def overlap_report(manifest: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Report only (nothing changed): sentences (normalised text) found on both sides of a split.

    The splits are grouped by id group, so id groups are never shared (checked by assertion).
    The same normalised text can still occur in two different id groups, e.g. a sentence that
    two actors both read, or one actor read under two numbers.
    """
    m = manifest.set_index("clip_id")
    rows = []
    for name, f in splits.items():
        t = f.join(m[["text_norm", "group_id"]], on="clip_id")
        tr = t[t["split"] == "train"]
        for other in ("val", "test"):
            o = t[t["split"] == other]
            hit = o["text_norm"].isin(set(tr["text_norm"]))
            rows.append({
                "split_file": name,
                "test_actor": ",".join(sorted(t.loc[t["split"] == "test", "actor"].unique())),
                "compared": f"{other} vs train",
                "clips": len(o),
                "unique_sentences": o["text_norm"].nunique(),
                "sentences_also_in_train": len(set(o["text_norm"]) & set(tr["text_norm"])),
                "clips_whose_sentence_is_in_train": int(hit.sum()),
                "share_of_clips": round(float(hit.mean()), 4),
                "id_groups_also_in_train": len(set(o["group_id"]) & set(tr["group_id"])),
            })
    return pd.DataFrame(rows)


_NB = 1 << 16  # number of distinct float16 bit patterns
_F16_VALUES = np.arange(_NB, dtype=np.uint16).view(np.float16).astype(np.float64)


def train_value_counts(cfg: dict, fold_df: pd.DataFrame, index: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    """Count every stored dB value per mel band over TRAIN clips, real frames only.

    The spectrograms are stored as float16, which has only 65,536 possible values. Counting
    how often each value occurs per band gives EXACT means, standard deviations and
    percentiles without holding all values in memory at once.
    Returns (counts [n_mels, 65536], n_train_clips, n_frames).
    """
    assert np.dtype(cfg["spectrogram"]["dtype"]) == np.float16, "value counting assumes float16 storage"
    proc = logmel_dir(cfg)
    train_ids = set(fold_df.loc[fold_df["split"] == "train", "clip_id"])
    idx = index[index["clip_id"].isin(train_ids)]
    n_mels = cfg["spectrogram"]["n_mels"]
    offsets = (np.arange(n_mels, dtype=np.int64) * _NB)[:, None]
    counts = np.zeros(n_mels * _NB, dtype=np.int64)
    n_frames = 0
    for fname, g in idx.groupby("file", sort=True):
        arr = np.load(proc / fname, mmap_mode="r")
        buf, size = [], 0
        for row, nv in zip(g["row"].to_numpy(), g["n_frames_valid"].to_numpy()):
            bits = np.asarray(arr[row, :, :nv]).view(np.uint16).astype(np.int64) + offsets
            buf.append(bits.ravel())
            size += bits.size
            n_frames += int(nv)
            if size > 20_000_000:  # flush in chunks to keep memory low
                counts += np.bincount(np.concatenate(buf), minlength=counts.size)
                buf, size = [], 0
        if buf:
            counts += np.bincount(np.concatenate(buf), minlength=counts.size)
    return counts.reshape(n_mels, _NB), int(len(idx)), n_frames


def _band_stats(c: np.ndarray, q: float) -> tuple[float, float, float]:
    """Mean, std and q-quantile (inverted CDF: smallest value with at least q of the mass at or below it)."""
    nz = c > 0
    v, w = _F16_VALUES[nz], c[nz]
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    n = w.sum()
    mean = float((v * w).sum() / n)
    std = float(np.sqrt(max((np.square(v - mean) * w).sum() / n, 0.0)))
    qv = float(v[np.searchsorted(np.cumsum(w), q * n, side="left")])
    return mean, std, qv


def fold_statistics(cfg: dict, fold_df: pd.DataFrame, index: pd.DataFrame) -> tuple[dict, dict]:
    """Returns (norm_stats, fill_values) for one fold, both from TRAIN clips, real frames only."""
    counts, n_clips, n_frames = train_value_counts(cfg, fold_df, index)
    q = cfg["splits"]["fill_percentile"] / 100.0
    per_band = np.array([_band_stats(c, q) for c in counts])  # [n_mels, (mean, std, q)]
    g_mean, g_std, _ = _band_stats(counts.sum(axis=0), q)
    common = {"computed_on": "train clips only, real (non-padding) frames only, values in dB",
              "n_train_clips": n_clips, "n_frames": n_frames}
    norm = {**common, "mean": g_mean, "std": g_std,
            "per_mel_mean": [round(float(v), 6) for v in per_band[:, 0]],
            "per_mel_std": [round(float(v), 6) for v in per_band[:, 1]]}
    floor = float(cfg["spectrogram"]["floor_db"])
    gap = per_band[:, 2] - floor
    pct = cfg["splits"]["fill_percentile"]
    fill = {
        **common,
        "description": (
            "Fill values for deletion tests (e.g. replacing a time segment of a spectrogram). "
            "mean_db_per_mel = neutral 'average' fill; "
            f"p{pct}_db_per_mel = 'room noise' fill (the level a band sits at or below {pct}% of the time). "
            "Percentiles are exact over the stored float16 values (inverted-CDF definition)."
        ),
        "mean_db_per_mel": [round(float(v), 6) for v in per_band[:, 0]],
        f"p{pct}_db_per_mel": [round(float(v), 6) for v in per_band[:, 2]],
        "padding_value_db": floor,
        f"p{pct}_minus_padding_db": {
            "per_mel": [round(float(v), 4) for v in gap],
            "min": round(float(gap.min()), 4), "median": round(float(np.median(gap)), 4),
            "mean": round(float(gap.mean()), 4), "max": round(float(gap.max()), 4),
        },
    }
    return norm, fill


def _write_stats(cfg: dict, out: Path, tag: str, split_df: pd.DataFrame, index: pd.DataFrame) -> None:
    stats, fill = fold_statistics(cfg, split_df, index)
    (out / f"norm_stats_{tag}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out / f"fill_values_{tag}.json").write_text(json.dumps(fill, indent=2), encoding="utf-8")


def write_splits(cfg: dict, manifest: pd.DataFrame, index: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Write all split files and their statistics. Returns ({name: split_df}, overlap report)."""
    out: Path = cfg["paths"]["splits_dir"]
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):  # start clean so no stale split files survive a rerun
        if old.is_file() and old.name != ".gitkeep":
            old.unlink()

    folds = make_folds(cfg, manifest)
    meta = {}
    for k, f in folds.items():
        f.to_csv(out / f"fold_{k}.csv", index=False)
        test_actor = f.loc[f["split"] == "test", "actor"].iloc[0]
        meta[str(k)] = {"test_actor": test_actor, "file": f"fold_{k}.csv",
                        "norm_stats": f"norm_stats_fold_{k}.json", "fill_values": f"fill_values_fold_{k}.json"}
        _write_stats(cfg, out, f"fold_{k}", f, index)
    (out / "folds.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rnd = make_random_split(cfg, manifest)
    rnd.to_csv(out / "random_split.csv", index=False)
    _write_stats(cfg, out, "random", rnd, index)
    shares = rnd["split"].value_counts(normalize=True).round(4).to_dict()
    (out / "random_split.json").write_text(json.dumps({
        "file": "random_split.csv", "norm_stats": "norm_stats_random.json", "fill_values": "fill_values_random.json",
        "target_fractions": cfg["splits"]["random"], "actual_fractions": shares,
        "grouped_by": "group_id (actor + id number)", "stratified_by": "actor x style",
    }, indent=2), encoding="utf-8")

    all_splits = {f"fold_{k}": f for k, f in folds.items()} | {"random_split": rnd}
    return all_splits, overlap_report(manifest, all_splits)

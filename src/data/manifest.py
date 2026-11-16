"""Step 3: build the clip manifest (one row per kept clip).

What happens here, in order:
1. Every decoded row gets a "subset" tag. In this dataset the `style` column holds 11 values:
   the 7 speaking styles, plus "emphasis" and "essentials" (both read in the default voice,
   their ids look like ex01_default_emphasis_00001), "longform" (long paragraphs; these reuse
   the id ex0X_default_00001 of a normal default clip) and "singing".
2. Only the 7 styles in the config are kept. Everything else is excluded with a reason.
   (configs: labels.styles and labels.merge_into_default)
3. Exact duplicates are removed: rows with the same id, and rows whose decoded audio
   samples are byte-identical (same hash). The first occurrence (sorted by id) is kept.
4. Clips longer than audio.max_seconds are either cropped to their first max_seconds
   (audio.long_clip_policy: crop, the current setting, nothing is dropped) or dropped ("drop").
5. Label ids are assigned alphabetically from the style list, and every clip gets a
   group_id (actor + id number) used by the splits.

Outputs:
  data/processed/manifest.csv       kept clips (the file every model reads)
  data/interim/excluded_clips.csv   every dropped row and why
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import normalise_text


def subset_of(row) -> str:
    """Which part of the read corpus a row comes from (from the style column and the id)."""
    style, cid = row["style"], row["id"]
    if style in ("emphasis", "essentials", "longform", "singing"):
        return style
    if "_emphasis_" in cid:
        return "emphasis"
    if "_essentials_" in cid:
        return "essentials"
    return "base"


def build_manifest(cfg: dict, raw_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    styles = list(cfg["labels"]["styles"])
    assert styles == sorted(styles), "labels.styles must be in alphabetical order"
    merge = set(cfg["labels"].get("merge_into_default") or [])
    df = raw_index.copy()
    df["subset"] = df.apply(subset_of, axis=1)
    # Optional: treat emphasis / essentials clips as "default" style (they are read in the default voice).
    df["label_style"] = np.where(df["subset"].isin(merge), "default", df["style"])
    df["text_norm"] = df["text"].map(lambda t: normalise_text(t, cfg["splits"]))
    df = df.sort_values(["id", "row_index"]).reset_index(drop=True)

    reason = pd.Series("", index=df.index, dtype=object)

    # 1) style filter
    reason[~df["label_style"].isin(styles)] = "style_not_in_7"

    # 2) duplicate ids among the remaining rows (keep the first)
    alive = reason == ""
    reason[alive & df["id"].where(alive).duplicated(keep="first")] = "duplicate_id"

    # 3) identical audio among the remaining rows (keep the first)
    alive = reason == ""
    reason[alive & df["audio_sha256"].where(alive).duplicated(keep="first")] = "duplicate_audio"

    # 4) clips longer than the window: drop them, or keep only their first max_seconds (crop)
    max_s = cfg["audio"]["max_seconds"]
    assert max_s is not None, "set audio.max_seconds in the config (see reports/duration_coverage.csv)"
    policy = cfg["audio"]["long_clip_policy"]
    assert policy in ("drop", "crop"), "audio.long_clip_policy must be 'drop' or 'crop'"
    alive = reason == ""
    too_long = alive & (df["duration_s"] > max_s)
    if policy == "drop":
        coverage = 1 - too_long.sum() / alive.sum()
        assert coverage >= cfg["audio"]["min_coverage"], (
            f"max_seconds={max_s} keeps only {coverage:.3%} of clips, below audio.min_coverage"
        )
        reason[too_long] = f"longer_than_{max_s}s"

    df["exclude_reason"] = reason
    excluded = df[reason != ""].copy()
    kept = df[reason == ""].copy()

    # Length bookkeeping. original_* = the whole recording; duration_s / n_samples_16k = the real
    # audio inside the max_seconds window (equal to the original for clips that fit).
    max_samples = int(max_s * cfg["audio"]["target_sr"])
    kept["original_duration"] = kept["duration_s"]
    kept["original_n_samples_16k"] = kept["n_samples_16k"]
    kept["was_cropped"] = kept["n_samples_16k"] > max_samples
    kept["n_samples_16k"] = kept["n_samples_16k"].clip(upper=max_samples)
    kept["duration_s"] = kept["n_samples_16k"] / cfg["audio"]["target_sr"]

    kept["label"] = kept["label_style"].map({s: i for i, s in enumerate(styles)}).astype(int)
    kept = kept.rename(columns={"id": "clip_id", "speaker_id": "actor"})
    kept = kept.drop(columns=["style"]).rename(columns={"label_style": "style"})
    kept["group_id"] = kept["clip_id"].map(group_id_of)
    cols = [
        "clip_id", "actor", "style", "label", "group_id", "duration_s", "n_samples_16k", "was_cropped",
        "original_duration", "original_n_samples_16k", "subset", "text", "text_norm",
        "orig_sr", "channels", "peak", "rms_dbfs", "n_over_full_scale_16k", "audio_sha256", "audio16k_file",
    ]
    kept = kept[cols].sort_values("clip_id").reset_index(drop=True)
    assert kept["clip_id"].is_unique
    return kept, excluded


def group_id_of(clip_id: str) -> str:
    """Actor + id number, the unit that must never be split between train and validation.

    The number at the end of an id marks the same prompt in every style of one actor, so
    ex01_happy_00260 and ex01_sad_00260 both get the group "ex01_00260". Other id shapes keep
    their subset name so numbers from different subsets never collide
    (ex01_default_emphasis_00001 -> "ex01_emphasis_00001").
    """
    parts = clip_id.split("_")
    return "_".join([parts[0]] + parts[2:])


def duration_coverage(durations: pd.Series, candidates=range(2, 31)) -> pd.DataFrame:
    """Share of clips kept for each whole-second maximum length."""
    return pd.DataFrame(
        {"max_seconds": list(candidates), "share_kept": [float((durations <= c).mean()) for c in candidates]}
    )

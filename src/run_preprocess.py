"""Run all preprocessing end to end:   python -m src.run_preprocess

Stages:
  1. download   parquet shards from Hugging Face (skipped when already complete)
  2. decode     every clip -> description table + 16 kHz FLAC copy   (data/interim)
  3. manifest   filter styles, remove duplicates, crop long clips to 4 s (data/processed/manifest.csv)
  4. logmel     log-mel spectrograms 128 x 400, one .npy per actor     (data/processed/logmel)
  5. segments   16 x 250 ms segments per clip                          (data/processed/segments.csv)
  6. splits     4 leave-one-actor-out folds + 1 random split, with
                train-only norm stats and fill values                  (data/splits)
  7. reports    tables and figures                                     (reports/)
  8. checksums  SHA-256 of every output, to prove reruns are identical (reports/output_checksums.json)

Options:
  --config PATH     use another config file
  --reuse-audio     skip stage 2 if data/interim already holds a complete decode (faster reruns)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from src.data import audio, download, manifest, parallel_text, reports, spectrograms, splits
from src.utils import REPO_ROOT, load_config, set_seed, sha256_file


def stage(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def write_checksums(cfg: dict) -> Path:
    p = cfg["paths"]
    files = sorted(
        [p["interim_dir"] / "raw_index.csv", p["interim_dir"] / "excluded_clips.csv"]
        + list(p["processed_dir"].glob("*.csv"))
        + list((p["processed_dir"] / "logmel").glob("*"))
        + list(p["splits_dir"].glob("*.csv"))
        + list(p["splits_dir"].glob("*.json"))
        + [f for f in p["reports_dir"].glob("*") if f.is_file() and f.name != "output_checksums.json"]
    )
    out = {f.relative_to(REPO_ROOT).as_posix(): sha256_file(f) for f in files}
    # The ~11k FLAC files are summarised as one combined hash.
    h = hashlib.sha256()
    flacs = sorted((p["interim_dir"] / "audio16k").glob("*.flac"))
    for f in flacs:
        h.update(f.name.encode() + sha256_file(f).encode())
    out[f"data/interim/audio16k/*.flac ({len(flacs)} files, combined)"] = h.hexdigest()
    path = p["reports_dir"] / "output_checksums.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--reuse-audio", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    P = cfg["paths"]
    for k in ("interim_dir", "processed_dir", "splits_dir", "reports_dir"):
        P[k].mkdir(parents=True, exist_ok=True)

    stage("1/8 download")
    raw_dir = download.download(cfg)
    shards = download.parquet_files(raw_dir)

    stage("2/8 decode audio")
    raw_index_path = P["interim_dir"] / "raw_index.csv"
    if args.reuse_audio and raw_index_path.exists():
        print(f"[decode] reusing {raw_index_path}")
    else:
        audio.decode_all(cfg, shards).to_csv(raw_index_path, index=False)
    # Always read back from the CSV so a fresh run and a reused run see identical values.
    raw = pd.read_csv(raw_index_path, keep_default_na=False)
    info = reports.inspection_report(cfg, raw, P["reports_dir"])
    print(f"[decode] {info['rows']} rows, sample rates {info['decoded_sample_rates']}, "
          f"channels {info['decoded_channels']}")

    # Duration check BEFORE choosing max_seconds: coverage among the 7-style clips.
    seven = raw[raw["style"].isin(cfg["labels"]["styles"])]
    cov = manifest.duration_coverage(seven["duration_s"])
    cov.to_csv(P["reports_dir"] / "duration_coverage.csv", index=False)
    if cfg["audio"]["max_seconds"] is None:
        print(cov.to_string(index=False))
        sys.exit("audio.max_seconds is not set. Pick it from the table above "
                 "(reports/duration_coverage.csv) and run again.")

    stage("3/8 manifest")
    kept, excluded = manifest.build_manifest(cfg, raw)
    kept.to_csv(P["processed_dir"] / "manifest.csv", index=False)
    excluded.to_csv(P["interim_dir"] / "excluded_clips.csv", index=False)
    print(f"[manifest] kept {len(kept)} clips, excluded {len(excluded)}:")
    print(excluded["exclude_reason"].value_counts().to_string())

    print(f"[manifest] {int(kept['was_cropped'].sum())} clips cropped to the first "
          f"{cfg['audio']['max_seconds']} s")

    stage("4/8 log-mel spectrograms")
    index = spectrograms.build_spectrograms(cfg, kept)

    stage("5/8 segment table")
    seg = spectrograms.build_segments(cfg, kept, index)
    print(f"[segments] {len(seg)} rows, {int(seg['is_real_audio'].sum())} segments with real audio")

    stage("6/8 splits")
    all_splits, overlap = splits.write_splits(cfg, kept, index)
    overlap.to_csv(P["reports_dir"] / "sentence_overlap.csv", index=False)
    print("[splits] all split assertions passed")
    print(overlap.to_string(index=False))

    stage("7/8 reports")
    reports.counts_tables(kept, P["reports_dir"])
    reports.cropped_table(kept, P["reports_dir"])
    reports.excluded_summary(excluded, P["reports_dir"])
    warn = reports.audio_warnings(cfg, kept, P["reports_dir"])
    print(f"[reports] {len(warn)} clips flagged in reports/audio_warnings.csv")
    long_reason = f"longer_than_{cfg['audio']['max_seconds']}s"
    dropped_long = excluded.loc[excluded["exclude_reason"] == long_reason, "duration_s"]
    reports.duration_histogram(cfg, pd.concat([kept["original_duration"], dropped_long]), P["reports_dir"])
    ex = reports.spectrogram_per_style(cfg, kept, index, P["reports_dir"])
    (P["reports_dir"] / "spectrogram_examples.json").write_text(json.dumps(ex, indent=2), encoding="utf-8")
    reports.split_summary(all_splits, P["reports_dir"])

    # Parallel text check over all 7-style clips (plus the kept clips if any were dropped for length).
    cols = ["clip_id", "actor", "style", "text", "text_norm"]
    scopes = {"all_7_style_clips": kept[cols]}
    if len(dropped_long):
        too_long = excluded[excluded["exclude_reason"] == long_reason].rename(
            columns={"id": "clip_id", "speaker_id": "actor", "style": "orig_style"}
        ).rename(columns={"label_style": "style"})
        scopes = {"all_7_style_clips": pd.concat([kept[cols], too_long[cols]]).sort_values("clip_id"),
                  "kept_clips": kept[cols]}
    parallel_text.run(cfg, scopes, kept[cols], all_splits, P["reports_dir"])
    print("[reports] parallel text check written to reports/parallel_text_check.md")

    stage("8/8 checksums")
    print(f"[checksums] {write_checksums(cfg)}")
    print("\nPreprocessing finished.")


if __name__ == "__main__":
    main()

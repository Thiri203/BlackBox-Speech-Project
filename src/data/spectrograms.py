"""Step 4: turn each 16 kHz clip into a log-mel spectrogram.

Plain-words background:
- A spectrogram is a picture of sound: time runs left to right, pitch (frequency) runs
  bottom to top, and the brightness of each spot says how much energy the sound has at
  that pitch and that moment. We cut the audio into short overlapping windows (25 ms long,
  one every 10 ms) and measure the energy at each frequency inside each window.
- The "mel scale" squeezes the frequency axis the way human hearing does: we tell apart
  100 Hz and 200 Hz easily, but 7100 Hz and 7200 Hz sound almost the same. So the 128 mel
  bands are narrow at low pitch and wide at high pitch.
- The "dB (decibel) scale" is a logarithm of energy. Sound energy spans a huge range, and
  our ears hear ratios, not differences, so the log makes quiet and loud parts both visible.
  We use one FIXED reference for every clip, so a quieter clip really gets lower numbers.
  (librosa's common default ref=np.max would make every clip's loudest spot 0 dB and erase
  loudness differences between clips; we do not do that.)

Every clip is cut to the first audio.max_seconds (4 s) if longer, and padded with silence to
that length if shorter, so every spectrogram has the same width (400 frames of 10 ms).
The real (unpadded) frame count is stored next to it so padding can be ignored.

Storage: one .npy file per actor holding an array [n_clips, n_mels, n_frames] in float16,
plus index.csv saying which row belongs to which clip. Load with np.load(..., mmap_mode="r")
to read single clips without loading the whole file.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.audio import load_audio16k


def n_frames_window(cfg: dict) -> int:
    """Frames per stored spectrogram: exactly window length / hop (4 s / 10 ms = 400).

    Frame t is centred at t x hop. librosa (center=True) also returns one extra frame centred
    exactly at the end of the window; we drop it so the window splits evenly into segments.
    """
    sp = cfg["spectrogram"]
    max_samples = int(cfg["audio"]["max_seconds"] * sp["sr"])
    assert max_samples % sp["hop_length"] == 0, "max_seconds must be a whole number of hops"
    return max_samples // sp["hop_length"]


def n_real_frames(n_samples: int, sp: dict) -> int:
    """Frames whose centre lies inside the real audio (sample index < n_samples); the rest is padding."""
    return -(-int(n_samples) // sp["hop_length"])  # ceil(n_samples / hop)


def logmel(y: np.ndarray, sp: dict) -> np.ndarray:
    """Waveform -> log-mel spectrogram in dB with a fixed reference and a fixed floor."""
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sp["sr"],
        n_fft=sp["n_fft"],
        hop_length=sp["hop_length"],
        win_length=sp["win_length"],
        window=sp["window"],
        center=sp["center"],
        pad_mode=sp["pad_mode"],
        power=sp["power"],
        n_mels=sp["n_mels"],
        fmin=sp["fmin"],
        fmax=sp["fmax"],
    )
    db = 10.0 * np.log10(np.maximum(S, sp["amin"]) / sp["ref"])
    return np.maximum(db, sp["floor_db"]).astype(np.float32)


def build_spectrograms(cfg: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    sp = cfg["spectrogram"]
    assert sp["sr"] == cfg["audio"]["target_sr"], "spectrogram sr must match the audio target_sr"
    max_samples = int(cfg["audio"]["max_seconds"] * sp["sr"])
    T = n_frames_window(cfg)
    out_dir: Path = cfg["paths"]["processed_dir"] / "logmel"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*"):  # start clean so no stale actor files survive a rerun
        old.unlink()
    dtype = np.dtype(sp["dtype"])

    index_rows = []
    for actor, g in manifest.groupby("actor", sort=True):
        g = g.sort_values("clip_id")
        fname = f"{actor}.npy"
        arr = np.lib.format.open_memmap(out_dir / fname, mode="w+", dtype=dtype, shape=(len(g), sp["n_mels"], T))
        for row, rec in enumerate(tqdm(g.itertuples(index=False), total=len(g), desc=f"logmel {actor}")):
            y = load_audio16k(cfg, rec.audio16k_file)[:max_samples]  # crop: keep the first max_seconds
            assert len(y) == rec.n_samples_16k, f"{rec.clip_id}: length does not match the manifest"
            y = np.pad(y, (0, max_samples - len(y)))  # pad with silence (zeros) at the end
            arr[row] = logmel(y, sp)[:, :T].astype(dtype)
            index_rows.append(
                {
                    "clip_id": rec.clip_id,
                    "actor": actor,
                    "file": fname,
                    "row": row,
                    # frames that contain real audio; the rest is padding
                    "n_frames_valid": min(n_real_frames(rec.n_samples_16k, sp), T),
                    "n_frames_total": T,
                }
            )
        arr.flush()
        del arr
    index = pd.DataFrame(index_rows)
    index.to_csv(out_dir / "index.csv", index=False)
    return index


def build_segments(cfg: dict, manifest: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Cut every 4 s window into equal time segments (16 x 250 ms) and describe each one.

    Columns: clip_id, segment (0..15), start_s, end_s, is_real_audio, n_real_frames, mean_db.
    - is_real_audio: True if the segment contains any real audio (it starts before the end of
      the clip's real audio inside the window); a partly real segment counts as real.
    - mean_db: mean of the stored log-mel values over all mel bands and the segment's REAL
      frames only; empty for segments that are pure padding.
    No speech / pause threshold is applied here.
    """
    sp = cfg["spectrogram"]
    T = n_frames_window(cfg)
    seg_ms = cfg["segments"]["length_ms"]
    fps = sp["sr"] // sp["hop_length"]  # frames per second (100)
    per_seg = seg_ms * fps // 1000      # frames per segment (25)
    assert per_seg * 1000 == seg_ms * fps and T % per_seg == 0, "segments must tile the window exactly"
    n_seg = T // per_seg
    proc = cfg["paths"]["processed_dir"] / "logmel"
    dur = manifest.set_index("clip_id")["duration_s"]
    rows = []
    for fname, g in index.groupby("file", sort=True):
        arr = np.load(proc / fname, mmap_mode="r")
        for cid, row, nv in zip(g["clip_id"], g["row"], g["n_frames_valid"]):
            x = np.asarray(arr[int(row)], dtype=np.float64)
            for s in range(n_seg):
                a, b = s * per_seg, (s + 1) * per_seg
                n_real = max(0, min(b, int(nv)) - a)
                start = s * seg_ms / 1000
                is_real = bool(start < dur[cid])
                assert is_real == (n_real > 0), f"{cid} segment {s}: duration and frame count disagree"
                rows.append({
                    "clip_id": cid, "segment": s, "start_s": start, "end_s": (s + 1) * seg_ms / 1000,
                    "is_real_audio": is_real, "n_real_frames": n_real,
                    "mean_db": round(float(x[:, a:a + n_real].mean()), 4) if n_real else np.nan,
                })
    seg = pd.DataFrame(rows).sort_values(["clip_id", "segment"]).reset_index(drop=True)
    seg.to_csv(cfg["paths"]["processed_dir"] / "segments.csv", index=False)
    return seg


def load_logmel(cfg: dict, index_row) -> np.ndarray:
    """Load one stored spectrogram [n_mels, n_frames] (float16) by its index.csv row."""
    path = cfg["paths"]["processed_dir"] / "logmel" / index_row["file"]
    return np.load(path, mmap_mode="r")[int(index_row["row"])]

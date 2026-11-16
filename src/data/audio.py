"""Step 2: decode every clip, describe it, and save a 16 kHz copy.

Plain-words background:
- A digital recording is a long list of numbers ("samples"), each one the air pressure
  at one instant. The "sample rate" says how many numbers per second: 48 kHz means
  48,000 numbers per second. We convert everything to 16 kHz (16,000 per second), which
  still keeps all frequencies up to 8 kHz, enough for speech.
- "Mono" means one channel (one microphone signal). If a file had two channels we would
  average them into one.

For every row of the parquet shards this step:
1. decodes the stored audio bytes with soundfile (we never use the "datasets" library),
2. records the true sample rate, number of channels, length, peak and loudness,
3. hashes the decoded samples, so exact duplicate recordings can be found,
4. converts to mono, resamples to 16 kHz and writes a lossless 24 bit FLAC file.

No loudness normalisation happens here (or anywhere): loudness is information we want
the model to keep.
"""
from __future__ import annotations

import hashlib
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

META_COLS = ["id", "speaker_id", "style", "text"]


def decode_bytes(b: bytes) -> tuple[np.ndarray, int]:
    """Decode encoded audio bytes. Returns (samples with shape [n, channels], sample_rate)."""
    data, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    return data, sr


def to_mono(x: np.ndarray) -> np.ndarray:
    """[n, channels] -> [n] by averaging the channels."""
    return x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]


def resample(y: np.ndarray, sr_in: int, sr_out: int, res_type: str) -> np.ndarray:
    import librosa

    if sr_in == sr_out:
        return y.astype(np.float32)
    return librosa.resample(y, orig_sr=sr_in, target_sr=sr_out, res_type=res_type).astype(np.float32)


def rms_dbfs(y: np.ndarray) -> float:
    """Overall loudness in dB relative to full scale (0 dBFS = the loudest possible steady signal)."""
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) if len(y) else 0.0
    return 20 * np.log10(max(rms, 1e-12))


def _process_shard(args) -> list[dict]:
    shard, out_dir, target_sr, res_type, subtype = args
    rows = []
    pf = pq.ParquetFile(shard)
    for batch in pf.iter_batches(batch_size=64, columns=META_COLS + ["audio"]):
        tbl = batch.to_pydict()
        for i in range(batch.num_rows):
            audio = tbl["audio"][i]
            x, sr = decode_bytes(audio["bytes"])
            n, ch = x.shape
            # Hash of the exact decoded samples (24 bit values fit exactly in float32).
            h = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            y = to_mono(x)
            y16 = resample(y, sr, target_sr, res_type)
            clip_id = tbl["id"][i]
            # Files are named by the row's position too, so a duplicated id cannot overwrite.
            rel = f"{Path(shard).stem}_{len(rows):05d}_{clip_id}.flac"
            # Values above 1.0 would be clipped by the 24 bit writer; count them first.
            n_over = int(np.sum(np.abs(y16) > 1.0))
            sf.write(out_dir / rel, np.clip(y16, -1.0, 1.0), target_sr, subtype=subtype, format="FLAC")
            rows.append(
                {
                    "id": clip_id,
                    "speaker_id": tbl["speaker_id"][i],
                    "style": tbl["style"][i],
                    "text": tbl["text"][i],
                    "shard": Path(shard).name,
                    "orig_sr": sr,
                    "channels": ch,
                    "orig_n_samples": n,
                    "duration_s": n / sr,
                    "n_samples_16k": len(y16),
                    "peak": float(np.max(np.abs(y))) if n else 0.0,
                    "rms_dbfs": rms_dbfs(y),
                    "n_over_full_scale_16k": n_over,
                    "audio_sha256": h,
                    "audio16k_file": rel,
                    "src_audio_path": audio.get("path"),
                }
            )
    return rows


def decode_all(cfg: dict, shards: list[Path]) -> pd.DataFrame:
    """Decode every shard in parallel (one worker per shard), keep a fixed row order."""
    import os

    from tqdm import tqdm

    out_dir = cfg["paths"]["interim_dir"] / "audio16k"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.flac"):  # start clean so no stale files survive a rerun
        old.unlink()
    a = cfg["audio"]
    jobs = [(str(s), out_dir, a["target_sr"], a["resample_type"], a["audio16k_subtype"]) for s in shards]
    workers = max(1, min(len(jobs), a.get("num_workers", 4), (os.cpu_count() or 2) - 1))
    results: list[list[dict]] = [None] * len(jobs)  # type: ignore[list-item]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_shard, j): k for k, j in enumerate(jobs)}
        for f in tqdm(futs, total=len(futs), desc="decode shards"):
            results[futs[f]] = f.result()
    df = pd.DataFrame([r for shard_rows in results for r in shard_rows])
    df.insert(0, "row_index", np.arange(len(df)))
    return df


def load_audio16k(cfg: dict, rel: str) -> np.ndarray:
    """Load one stored 16 kHz clip as float32 (true length, no padding)."""
    y, sr = sf.read(cfg["paths"]["interim_dir"] / "audio16k" / rel, dtype="float32")
    assert sr == cfg["audio"]["target_sr"], f"unexpected sample rate {sr} in {rel}"
    return y

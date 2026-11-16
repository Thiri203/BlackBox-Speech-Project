"""Step 1: download the "read" part of Expresso from Hugging Face.

The files are parquet shards (a compact table format). Each row holds one audio clip
(as encoded bytes) plus its text, speaker and style.

The download is resumable (huggingface_hub continues partial files) and is skipped when
a completion marker exists and every listed file is on disk with the right size.
That skip check needs no internet connection.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Windows cannot always create symlinks; this silences the warning about it.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MARKER = "DOWNLOAD_COMPLETE.json"


def _is_complete(raw_dir: Path) -> bool:
    marker = raw_dir / MARKER
    if not marker.exists():
        return False
    info = json.loads(marker.read_text(encoding="utf-8"))
    for rel, size in info["files"].items():
        p = raw_dir / rel
        if not p.exists() or p.stat().st_size != size:
            return False
    return True


def parquet_files(raw_dir: Path) -> list[Path]:
    """All parquet shards of the read config, in a fixed (sorted) order."""
    return sorted((raw_dir / "read").glob("*.parquet"))


def download(cfg: dict) -> Path:
    raw_dir: Path = cfg["paths"]["raw_dir"]
    if _is_complete(raw_dir):
        print(f"[download] data already complete in {raw_dir}, skipping download.")
        return raw_dir

    from huggingface_hub import HfApi, snapshot_download

    ds = cfg["dataset"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(ds["repo_id"], revision=ds.get("revision"))
    commit = info.sha
    print(f"[download] {ds['repo_id']} at commit {commit} -> {raw_dir}")
    snapshot_download(
        repo_id=ds["repo_id"],
        repo_type=ds["repo_type"],
        revision=commit,
        allow_patterns=ds["allow_patterns"],
        local_dir=str(raw_dir),
    )
    files = {p.relative_to(raw_dir).as_posix(): p.stat().st_size for p in parquet_files(raw_dir)}
    if not files:
        raise RuntimeError("Download finished but no parquet files were found under read/.")
    (raw_dir / MARKER).write_text(
        json.dumps({"repo_id": ds["repo_id"], "revision": commit, "files": files}, indent=2),
        encoding="utf-8",
    )
    total_gb = sum(files.values()) / 1e9
    print(f"[download] done: {len(files)} files, {total_gb:.2f} GB")
    return raw_dir


if __name__ == "__main__":
    from src.utils import load_config

    download(load_config())

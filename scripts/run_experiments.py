"""Experiments A (augmentation) and B (validation on a held-out training actor), seed 1, folds 0 and 2.

Each run uses its own config and its own folder; earlier runs are never touched:
    A: configs/train_aug.yaml       -> models/fold{k}_seed1_aug/
    B: configs/train_valactor.yaml  -> models/fold{k}_seed1_valactor/
A run is skipped if its folder already contains DONE; src/train.py also refuses to overwrite one.

Start from the repository root:
    .\\.venv\\Scripts\\python.exe scripts\\run_experiments.py
Measured on an RTX 4060: A about 30 s per epoch, B about 23 s per epoch, at most 30 epochs each.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS = ["configs/train_aug.yaml", "configs/train_valactor.yaml"]
FOLDS, SEED = [0, 2], 1


def main() -> int:
    failed = []
    for cfg in CONFIGS:
        tag = yaml.safe_load((REPO / cfg).read_text(encoding="utf-8"))["run_tag"]
        for fold in FOLDS:
            out = REPO / "models" / f"fold{fold}_seed{SEED}_{tag}"
            if (out / "DONE").exists():
                print(f"skip {out.name}: already finished")
                continue
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "src.train", "--fold", str(fold), "--seed", str(SEED),
                   "--pretrained", "true", "--config", cfg]
            print(f"start {out.name} ({cfg})", flush=True)
            t = time.time()
            with open(out / "train.log", "w", encoding="utf-8") as log:
                code = subprocess.call(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            status = "ok" if code == 0 and (out / "DONE").exists() else f"FAILED (exit code {code})"
            print(f"  {status} after {(time.time() - t) / 60:.1f} min", flush=True)
            if status != "ok":
                failed.append(out.name)
    print("all runs finished" if not failed else f"failed runs: {failed}; see their train.log")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

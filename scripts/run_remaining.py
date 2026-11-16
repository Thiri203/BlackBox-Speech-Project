"""Train the remaining ResNet-18 runs one after another (pretrained only).

Order: seed 1 on folds 1, 2, 3, then seed 2 on folds 0, 1, 2, 3.
Each run goes to models/fold{k}_seed{s}_pretrained/ with its own train.log.
A run is skipped if its folder already contains DONE (written at the very end of a
successful run). An unfinished run (folder without DONE) is started again from scratch.

Start from the repository root:
    .\\.venv\\Scripts\\python.exe scripts\\run_remaining.py
Each run takes about 10 to 15 minutes on an RTX 4060 (about 27 s per epoch, at most 30 epochs).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = [(1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2)]  # (fold, seed)


def main() -> int:
    failed = []
    for fold, seed in RUNS:
        out = REPO / "models" / f"fold{fold}_seed{seed}_pretrained"
        if (out / "DONE").exists():
            print(f"skip fold {fold} seed {seed}: already finished")
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "src.train", "--fold", str(fold), "--seed", str(seed),
               "--pretrained", "true", "--out", str(out)]
        print(f"start fold {fold} seed {seed} -> {out.relative_to(REPO)}", flush=True)
        t = time.time()
        with open(out / "train.log", "w", encoding="utf-8") as log:
            code = subprocess.call(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        status = "ok" if code == 0 and (out / "DONE").exists() else f"FAILED (exit code {code})"
        print(f"  {status} after {(time.time() - t) / 60:.1f} min", flush=True)
        if status != "ok":
            failed.append((fold, seed))
    print("all runs finished" if not failed else f"failed runs (fold, seed): {failed}; see their train.log")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

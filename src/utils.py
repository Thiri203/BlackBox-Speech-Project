"""Small shared helpers: find the repo root, load the config, set seeds, hash files."""
from __future__ import annotations

import hashlib
import os
import random
import re
from pathlib import Path

import numpy as np
import yaml

# The repo root is two levels above this file (src/utils.py -> src -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "preprocess.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML config and turn every entry under `paths` into an absolute Path."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: (REPO_ROOT / v).resolve() for k, v in cfg["paths"].items()}
    cfg["_config_path"] = path.resolve()
    return cfg


def set_seed(seed: int) -> None:
    """Fix the random seed for Python and NumPy (torch gets added later, in the model code)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file on disk, used to check that reruns give identical outputs."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


_TAG = re.compile(r"<[^>]+>")          # transcript tags such as <laugh> or <breath>
_PUNCT = re.compile(r"[^\w\s']")        # punctuation (apostrophes are kept: "it's" stays one word)


def normalise_text(text: str, opts: dict) -> str:
    """Make two readings of the same sentence give the same string.

    Steps (each switchable in the config under `splits`): lowercase, remove the emphasis
    asterisks, remove tags like <laugh>, remove punctuation, collapse whitespace.
    Example: "<laugh> *Patience*, come with me!"  ->  "patience come with me"
    """
    t = (text or "").lower()
    for ch in opts.get("text_strip_chars", "*"):
        t = t.replace(ch, "")
    if opts.get("text_remove_tags", True):
        t = _TAG.sub(" ", t)
    if opts.get("text_remove_punctuation", True):
        t = _PUNCT.sub(" ", t)
    return " ".join(t.split())

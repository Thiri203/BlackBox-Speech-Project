"""Extract the 88 eGeMAPS features for every clip in the manifest (NOT run yet).

eGeMAPS ("extended Geneva Minimalistic Acoustic Parameter Set") is a standard list of 88
hand-designed voice measurements: pitch, loudness, voice quality (jitter, shimmer),
spectral shape, speaking-rate proxies and so on, each summarised over the whole clip
(mean, standard deviation, percentiles). They feed the logistic regression baseline.

It reads the same manifest as the ResNet pipeline and uses the same 16 kHz audio files,
so both models see identical clips. Clips longer than the model window (audio.max_seconds,
4 s) are cut to their first 4 s, exactly like the spectrograms. Padding is NOT added here:
shorter clips are measured at their true length.

Setup (not done yet):   pip install opensmile==<pinned version>
Run:                    python -m src.features.egemaps
Output:                 data/processed/egemaps_functionals.csv  (one row per clip_id, 88 feature columns)
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.data.audio import load_audio16k
from src.utils import load_config


def main() -> None:
    import opensmile  # imported here so the rest of the repo works without it

    cfg = load_config()
    manifest = pd.read_csv(cfg["paths"]["processed_dir"] / "manifest.csv")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    sr = cfg["audio"]["target_sr"]
    rows = []
    for rec in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="eGeMAPS"):
        # Same audio as the ResNet: only the part inside the model window (first max_seconds).
        y = load_audio16k(cfg, rec.audio16k_file)[: rec.n_samples_16k]
        feats = smile.process_signal(y, sr)  # one row of 88 functionals
        feats = feats.reset_index(drop=True)
        feats.insert(0, "clip_id", rec.clip_id)
        rows.append(feats)
    out = pd.concat(rows, ignore_index=True)
    assert out.shape[1] == 1 + 88, f"expected 88 features, got {out.shape[1] - 1}"
    path = cfg["paths"]["processed_dir"] / "egemaps_functionals.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path} with {len(out)} rows")


if __name__ == "__main__":
    main()

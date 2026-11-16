# Black Box: explainable speech style classification on Expresso

Group project for the course *Explainable Data Science* (AIT).

We classify **how** a sentence is spoken (7 styles: confused, default, enunciated, happy,
laughing, sad, whisper) from the read part of the Expresso dataset, then explain the
classifier and test the explanations.

- **Main model:** ResNet-18 on log-mel spectrograms (`src/models/`, not built yet).
- **Baseline:** logistic regression on the 88 eGeMAPS features (`baseline/`, by a teammate).
- **Explanations:** Grad-CAM, Integrated Gradients, LIME on time segments (`src/explain/`,
  later), tested for faithfulness, stability, a human study and counterfactuals on
  speaking rate, loudness and pauses.

Status: **preprocessing done**; ResNet-18 trained on fold 0, seed 1 (pretrained and from scratch),
see `reports/training_fold0/report.html`. The other 7 runs are scripted, not run yet.

## Setup (Windows, Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

All versions are pinned in `requirements.txt`, including torch 2.11.0 and torchvision 0.26.0
as CUDA 12.8 builds from the official PyTorch index (the file points pip to it). opensmile is
added later, at the baseline stage. The development machine has an NVIDIA GeForce RTX 4060
(8 GB, driver supports CUDA up to 13.1).

> **Windows Smart App Control.** If an import fails with "An Application Control policy
> has blocked this file", Windows blocked a freshly released compiled package. This is why
> pandas and pyarrow are pinned to older releases. Rerunning sometimes works because the
> decision depends on an online lookup; otherwise pin an older release of the package.

## Run

```powershell
.\.venv\Scripts\python.exe -m src.run_preprocess
```

One command does everything: downloads the data if it is not on disk (about 5.8 GB, no
login needed), decodes it, builds the manifest, spectrograms, splits and every report
figure. After the download (5.8 GB, once) a full run takes a few minutes on the development
machine; the download is skipped on reruns.
`--reuse-audio` additionally skips re-decoding the audio. Every choice lives in
[`configs/preprocess.yaml`](configs/preprocess.yaml); seed = 42.

Reproducibility check: `reports/output_checksums.json` holds SHA-256 fingerprints of every
output. A rerun must give the same file.

## Train

Needs an NVIDIA GPU (developed on an RTX 4060, 8 GB; about 27 s per epoch). Settings live in
[`configs/train.yaml`](configs/train.yaml).

```powershell
# quick check: a few batches, shapes, loss going down, time per epoch (writes nothing)
.\.venv\Scripts\python.exe -m src.train --fold 0 --seed 1 --pretrained true --smoke
# one run (writes models/fold0_seed1_pretrained/)
.\.venv\Scripts\python.exe -m src.train --fold 0 --seed 1 --pretrained true
# from-scratch pilot (writes models/fold0_seed1_scratch/)
.\.venv\Scripts\python.exe -m src.train --fold 0 --seed 1 --pretrained false
# fold-0 report (reports/training_fold0/report.html and report.md)
.\.venv\Scripts\python.exe -m src.report_training
```

**Remaining runs** (pretrained only: seed 1 on folds 1, 2, 3, then seed 2 on folds 0 to 3;
about 10 to 15 minutes each, so roughly 1.5 hours in total). Runs that already finished (their
folder contains `DONE`) are skipped, so the script can be restarted after an interruption:

```powershell
.\.venv\Scripts\python.exe scripts\run_remaining.py
```

Each run folder `models/fold{k}_seed{s}_pretrained/` holds `best.pt`, `settings.json`,
`epochs.csv`, `test_predictions.csv`, `metrics.json` and `train.log`.

## Folder map

| path | content |
|---|---|
| `configs/preprocess.yaml` | every preprocessing setting |
| `src/run_preprocess.py` | the one entry point |
| `src/data/` | download, decoding, manifest, spectrograms, splits, reports, dataset class stub |
| `src/features/egemaps.py` | eGeMAPS extraction (written, not run; needs `pip install opensmile`) |
| `configs/train.yaml` | every training setting |
| `src/models/resnet.py` | ResNet-18 with a 1-channel input and 7 outputs |
| `src/data/torch_dataset.py` | PyTorch dataset: spectrograms in RAM, standardised with train-only statistics |
| `src/train.py`, `src/evaluate.py` | training, early stopping, one-time test evaluation, shortcut baselines |
| `src/report_training.py` | builds `reports/training_fold0/report.html` |
| `scripts/run_remaining.py` | trains the remaining folds and seeds one after another (not run yet) |
| `models/` | run outputs: checkpoints, logs, predictions, metrics (not in git) |
| `src/explain/` | explainers go here (later, teammate) |
| `baseline/` | logistic regression baseline (teammate) |
| `data/raw/expresso_hf/` | downloaded parquet shards (not in git) |
| `data/interim/` | `raw_index.csv` (every decoded row), `excluded_clips.csv`, `audio16k/*.flac` (not in git) |
| `data/processed/` | `manifest.csv`, `logmel/<actor>.npy` (128 x 400 per clip, 4 s window) + `logmel/index.csv`, `segments.csv` (16 x 250 ms segments per clip) (not in git) |
| `data/splits/` | `fold_{0..3}.csv` + `folds.json` (leave-one-actor-out), `random_split.csv` + `random_split.json` (60/15/25, all actors), `norm_stats_*.json`, `fill_values_*.json` (not in git) |
| `reports/` | summary tables and sanity-check figures (in git) |
| `docs/preprocessing.md` | plain-language explanation of every step |
| `docs/*.pdf` | project task description |
| `DECISIONS.md` | every choice, its reason and the config key that changes it |

## For the baseline teammate

Put your code in `baseline/`. Use the same splits as the ResNet:

- `data/splits/fold_{k}.csv` with columns `clip_id, actor, style, label, split`
  (`split` is `train`, `val` or `test`; fold k tests on actor `folds.json[k].test_actor`).
- `data/splits/random_split.csv`, same columns: one split with all 4 actors in every part.
- Every split keeps all readings of one prompt (`group_id` = actor + id number in the
  manifest) on the same side, so no extra grouping is needed on your side.
- Features: `python -m src.features.egemaps` (after `pip install opensmile`) writes
  `data/processed/egemaps_functionals.csv`, one row per `clip_id`. Join on `clip_id`.
  It measures the same first 4 s of each clip that the ResNet sees.
- Fit any scaler on `split == "train"` rows only.

## Dataset and license

Expresso (Nguyen et al., 2023, "EXPRESSO: A Benchmark and Analysis of Discrete Expressive
Speech Resynthesis"), Hugging Face mirror
[`ylacombe/expresso`](https://huggingface.co/datasets/ylacombe/expresso), config `read`,
pinned at commit `9fb79a189698de3255eff48edd2bc0d9e487adc0`.
License: **CC BY-NC 4.0** (attribution required, non-commercial use only). The data is
not redistributed in this repository; the pipeline downloads it.

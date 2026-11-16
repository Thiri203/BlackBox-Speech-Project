"""Re-evaluate saved best checkpoints on their test actor (no training).

    python -m src.eval_checkpoints

For every finished run models/fold{k}_seed{s}_pretrained/ (folder has DONE):
  1. rebuild ResNet-18, load best.pt, predict the fold's test split (same AMP / channels_last
     settings as training);
  2. compare the predictions with the run's saved test_predictions.csv (sanity check);
  3. compute the confusion matrix (rows = true style, columns = predicted style) and
     per-style precision, recall and F1.

Writes to reports/training_all/:
  checkpoint_test_metrics.json   per run: accuracy, macro-F1, confusion counts, per-style metrics,
                                 agreement with the saved predictions
  confusion_fold{k}.png          one image per fold, the seeds side by side
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.torch_dataset import SpectrogramDataset  # noqa: E402
from src.evaluate import classification_metrics  # noqa: E402
from src.models.resnet import build_resnet18  # noqa: E402
from src.train import predict  # noqa: E402
from src.utils import REPO_ROOT, load_config  # noqa: E402

OUT = REPO_ROOT / "reports" / "training_all"
INK = "#222222"
PNG_META = {"Software": None}  # no version string in the PNG, so reruns give identical files


def plot_confusion(ax, cm: np.ndarray, styles: list[str], title: str) -> None:
    """Rows = true style, columns = predicted style; each box: share of its row and clip count."""
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(styles)):
        for j in range(len(styles)):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=7.5,
                    color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(styles)), styles, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(styles)), styles, fontsize=9)
    ax.set_xlabel("predicted style", color=INK)
    ax.set_ylabel("true style", color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK)


def main() -> None:
    tcfg = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    pcfg = load_config(REPO_ROOT / tcfg["preprocess_config"])
    styles = pcfg["labels"]["styles"]
    amp, cl = bool(tcfg["train"]["mixed_precision"]), bool(tcfg["train"]["channels_last"])
    assert torch.cuda.is_available(), "CUDA GPU not available"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    runs = []
    for d in sorted((REPO_ROOT / "models").glob("fold*_seed*_pretrained")):
        m = re.fullmatch(r"fold(\d+)_seed(\d+)_pretrained", d.name)
        if m and (d / "DONE").exists() and (d / "best.pt").exists():
            runs.append((int(m[1]), int(m[2]), d))

    results, test_sets = {}, {}
    for fold, seed, d in runs:
        if fold not in test_sets:
            test_sets[fold] = SpectrogramDataset(pcfg, fold, "test")
        ds = test_sets[fold]
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        model = build_resnet18(tcfg["model"]["n_classes"], tcfg["model"]["in_channels"], pretrained=False)
        model.load_state_dict(torch.load(d / "best.pt", map_location="cpu", weights_only=True))
        model = model.to(device)
        if cl:
            model = model.to(memory_format=torch.channels_last)
        logits, y, _ = predict(model, loader, device, amp, cl)
        yp = logits.argmax(1).numpy()

        saved = pd.read_csv(d / "test_predictions.csv").set_index("clip_id").loc[ds.clip_ids]
        assert (saved["y_true"].to_numpy() == y).all(), f"{d.name}: test labels differ from the saved run"
        agree = float((saved["y_pred"].to_numpy() == yp).mean())

        met = classification_metrics(y, yp, styles)
        actor = sorted(ds.items["clip_id"].str.split("_").str[0].unique())
        results[d.name] = {"fold": fold, "seed": seed, "test_actor": actor, "n_test": int(len(y)),
                           "accuracy": met["accuracy"], "macro_f1": met["macro_f1"],
                           "per_class": met["per_class"], "confusion_counts": met["confusion_counts"],
                           "agreement_with_saved_predictions": agree}
        print(f"{d.name}: n={len(y)} acc {met['accuracy']:.4f} macro-F1 {met['macro_f1']:.4f} "
              f"| same prediction as saved run for {agree:.2%} of clips", flush=True)
        del model
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoint_test_metrics.json").write_text(
        json.dumps({"styles": styles, "runs": results}, indent=2), encoding="utf-8")

    for fold in sorted({r["fold"] for r in results.values()}):
        fr = [r for r in results.values() if r["fold"] == fold]
        fr.sort(key=lambda r: r["seed"])
        fig, axes = plt.subplots(1, len(fr), figsize=(6.2 * len(fr), 5.6))
        for ax, r in zip(np.atleast_1d(axes), fr):
            plot_confusion(ax, np.asarray(r["confusion_counts"]), styles,
                           f"Seed {r['seed']}: accuracy {r['accuracy']:.1%}, macro-F1 {r['macro_f1']:.3f}")
        fig.suptitle(f"Fold {fold}, test actor {', '.join(fr[0]['test_actor'])} ({fr[0]['n_test']} clips): "
                     "share of each true style (row) and clip count", x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"confusion_fold{fold}.png", dpi=130, metadata=PNG_META)
        plt.close(fig)
        print(f"wrote {OUT / f'confusion_fold{fold}.png'}")


if __name__ == "__main__":
    main()

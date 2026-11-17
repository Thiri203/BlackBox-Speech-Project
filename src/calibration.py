"""Calibration of the 8 baseline ResNet-18 runs (2 seeds x 4 folds) on their test actor. No retraining, no tuning,
no temperature scaling: the saved best checkpoints' softmax probabilities are evaluated as they are.

    python -m src.calibration

- Inference exactly as the original test evaluation (src.train.predict: float16 autocast, channels_last);
  predictions are checked against each run's saved test_predictions.csv.
- Brier score, same definition as Aung's LR report: mean over clips of sum over the 7 classes of
  (p_c - 1[c = true class])^2. Uniform guess (p = 1/7) = 6/7 = 0.857; 0 = perfect.
- Confidence = highest predicted probability; mean confidence vs accuracy per actor (gap > 0 = over-confident).
- Reliability: 10 equal-width confidence bins on [0, 1], all test clips pooled over the 4 actors, per seed.
  ECE (sum over bins of bin share x |accuracy - mean confidence|) is given as a one-number summary of the table.
- Aung's LR Brier per actor (his report of 12 Nov 2026, his fold k = our fold k-1), joined on the ACTOR id.
- CLAUDE.md: test confusion matrices and per-style precision, recall, F1 of every run.

Writes reports/calibration/: report.md, metrics.json, reliability.png, confusion_fold{k}.png.
"""
from __future__ import annotations

import json
import os
import re

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
from src.eval_checkpoints import PNG_META, plot_confusion  # noqa: E402
from src.evaluate import classification_metrics  # noqa: E402
from src.models.resnet import build_resnet18  # noqa: E402
from src.train import predict  # noqa: E402
from src.utils import REPO_ROOT, load_config  # noqa: E402

OUT = REPO_ROOT / "reports" / "calibration"
N_BINS = 10
# Aung's LR (88 eGeMAPS features), Brier per test actor, from his report of 12 Nov 2026 (not checked in this repo).
AUNG_LR_BRIER = {"ex01": 0.604, "ex02": 0.462, "ex03": 0.859, "ex04": 0.495}
AUNG_LR_MEAN = 0.605
SEED_COLOURS = {1: "#2a78d6", 2: "#eb6834"}
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-clip Brier: sum over classes of (p_c - onehot_c)^2."""
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return ((p - oh) ** 2).sum(1)


def reliability(conf: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    edges = np.linspace(0, 1, N_BINS + 1)
    b = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, N_BINS - 1)   # bin i = (edge_i, edge_i+1]
    rows = []
    for i in range(N_BINS):
        m = b == i
        rows.append({"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": int(m.sum()),
                     "share": float(m.mean()),
                     "mean_confidence": float(conf[m].mean()) if m.any() else float("nan"),
                     "accuracy": float(correct[m].mean()) if m.any() else float("nan")})
    return pd.DataFrame(rows)


def ece(rel: pd.DataFrame) -> float:
    r = rel[rel.n > 0]
    return float((r.share * (r.accuracy - r.mean_confidence).abs()).sum())


def main() -> None:
    tcfg = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    pcfg = load_config(REPO_ROOT / tcfg["preprocess_config"])
    styles = pcfg["labels"]["styles"]
    amp, cl = bool(tcfg["train"]["mixed_precision"]), bool(tcfg["train"]["channels_last"])
    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    OUT.mkdir(parents=True, exist_ok=True)

    runs = []
    for d in sorted((REPO_ROOT / "models").glob("fold*_seed*_pretrained")):
        m = re.fullmatch(r"fold(\d+)_seed(\d+)_pretrained", d.name)
        if m and (d / "DONE").exists():
            runs.append((int(m[1]), int(m[2]), d))
    assert len(runs) == 8, f"expected 8 finished baseline runs, found {len(runs)}"

    per_run, clips, test_sets = [], [], {}
    for fold, seed, d in runs:
        if fold not in test_sets:
            test_sets[fold] = SpectrogramDataset(pcfg, fold, "test")
        ds = test_sets[fold]
        model = build_resnet18(7, 1, pretrained=False)
        model.load_state_dict(torch.load(d / "best.pt", map_location="cpu", weights_only=True))
        model = model.to(device)
        if cl:
            model = model.to(memory_format=torch.channels_last)
        logits, y, _ = predict(model, DataLoader(ds, batch_size=128, shuffle=False), device, amp, cl)
        p = torch.softmax(logits.double(), 1).numpy()
        yp = p.argmax(1)
        saved = pd.read_csv(d / "test_predictions.csv").set_index("clip_id").loc[ds.clip_ids]
        agree = float((saved["y_pred"].to_numpy() == yp).mean())
        prob_diff = float(np.abs(saved[[f"prob_{s}" for s in styles]].to_numpy() - p).max())
        actor = ds.items["actor"].iloc[0]
        assert (ds.items["actor"] == actor).all()
        conf = p.max(1)
        correct = yp == y
        b = brier(p, y)
        met = classification_metrics(y, yp, styles)
        per_run.append({"fold": fold, "seed": seed, "actor": actor, "n": int(len(y)),
                        "brier": float(b.mean()), "accuracy": float(correct.mean()), "macro_f1": met["macro_f1"],
                        "mean_confidence": float(conf.mean()), "gap_conf_minus_acc": float(conf.mean() - correct.mean()),
                        "mean_conf_correct": float(conf[correct].mean()), "mean_conf_wrong": float(conf[~correct].mean()),
                        "share_conf_over_0_9": float((conf > 0.9).mean()),
                        "acc_when_conf_over_0_9": float(correct[conf > 0.9].mean()) if (conf > 0.9).any() else None,
                        "agreement_with_saved_predictions": agree, "max_abs_prob_diff_vs_saved": prob_diff,
                        "per_class": met["per_class"], "confusion_counts": met["confusion_counts"]})
        clips.append(pd.DataFrame({"fold": fold, "seed": seed, "actor": actor, "clip_id": ds.clip_ids, "y": y,
                                   "y_pred": yp, "confidence": conf, "correct": correct, "brier": b}))
        print(f"{d.name} ({actor}): Brier {b.mean():.3f}, acc {correct.mean():.3f}, mean conf {conf.mean():.3f}, "
              f"same predictions as saved {agree:.2%}, max prob diff {prob_diff:.1e}", flush=True)
        del model
        torch.cuda.empty_cache()

    R = pd.DataFrame(per_run)
    C = pd.concat(clips, ignore_index=True)

    # ---- summaries ----
    by_seed = R.groupby("seed")[["brier", "accuracy", "mean_confidence", "gap_conf_minus_acc"]].mean()
    by_actor = R.groupby("actor")[["brier", "accuracy", "mean_confidence", "gap_conf_minus_acc"]].mean()
    rel = {s: reliability(C[C.seed == s].confidence.to_numpy(), C[C.seed == s].correct.to_numpy())
           for s in sorted(C.seed.unique())}
    rel_both = reliability(C.confidence.to_numpy(), C.correct.to_numpy())

    plot_reliability(rel)
    for fold in sorted(R.fold.unique()):
        fr = R[R.fold == fold].sort_values("seed")
        fig, axes = plt.subplots(1, len(fr), figsize=(6.2 * len(fr), 5.6))
        for ax, (_, r) in zip(np.atleast_1d(axes), fr.iterrows()):
            plot_confusion(ax, np.asarray(r["confusion_counts"]), styles,
                           f"Seed {r['seed']}: accuracy {r['accuracy']:.1%}, macro-F1 {r['macro_f1']:.3f}, "
                           f"Brier {r['brier']:.3f}")
        fig.suptitle(f"Fold {fold}, test actor {fr.actor.iloc[0]} ({fr.n.iloc[0]} clips): share of each true style "
                     "(row) and clip count", x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"confusion_fold{fold}.png", dpi=130, metadata=PNG_META)
        plt.close(fig)

    metrics = {"runs": R.to_dict(orient="records"),
               "mean_of_folds_by_seed": by_seed.reset_index().to_dict(orient="records"),
               "mean_over_seeds_by_actor": by_actor.reset_index().to_dict(orient="records"),
               "mean_of_folds_over_seeds": {k: float(by_seed[k].mean()) for k in by_seed.columns},
               "reliability_by_seed": {int(s): r.to_dict(orient="records") for s, r in rel.items()},
               "reliability_both_seeds": rel_both.to_dict(orient="records"),
               "ece_by_seed": {int(s): ece(r) for s, r in rel.items()}, "ece_both_seeds": ece(rel_both),
               "aung_lr_brier": AUNG_LR_BRIER, "aung_lr_brier_mean": AUNG_LR_MEAN, "uniform_brier": 6 / 7}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(R, by_seed, by_actor, rel, rel_both, metrics, styles)
    print(f"wrote {OUT / 'report.md'}")


def plot_reliability(rel: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color="#b9b8b2", linewidth=1.2, linestyle="--")
    ax.text(0.62, 0.55, "perfect calibration", rotation=40, fontsize=8, color=INK2)
    for s, r in rel.items():
        r = r[r.n >= 20]                                   # very small bins hidden in the plot (all bins in the table)
        ax.plot(r.mean_confidence, r.accuracy, color=SEED_COLOURS[s], linewidth=2, marker="o", markersize=5,
                label=f"seed {s}")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("mean confidence in the bin (highest probability)", color=INK2)
    ax.set_ylabel("accuracy in the bin", color=INK2)
    ax.grid(color="#e4e3df", linewidth=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b9b8b2")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    ax.set_title("Reliability, ResNet-18, all 4 test actors pooled (10 bins)\nbelow the diagonal = over-confident",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "reliability.png", dpi=130, metadata=PNG_META, facecolor=SURFACE)
    plt.close(fig)


def write_report(R, by_seed, by_actor, rel, rel_both, m, styles) -> None:
    L = ["# Calibration of the ResNet-18 (8 baseline runs: 2 seeds x 4 folds)", "",
         "Saved best checkpoints, each run's test actor, inference exactly as in the original test evaluation (float16 "
         "autocast). No retraining, no tuning and **no temperature scaling or other calibration**: these are the "
         "model's raw softmax probabilities. Predictions match each run's saved `test_predictions.csv` "
         f"({min(r['agreement_with_saved_predictions'] for r in m['runs']):.0%} of clips in every run; largest "
         f"probability difference {max(r['max_abs_prob_diff_vs_saved'] for r in m['runs']):.1e}, from the 6-decimal "
         "rounding of the saved file).", "",
         "**Brier score** (same definition as Aung's LR report): mean over clips of the sum over the 7 classes of "
         "(p_c - 1[c = true class])^2. 0 = perfect, **uniform guess (1/7 each) = 0.857**, worse than uniform is "
         "possible (a confident wrong answer scores up to 2). **Confidence** = the highest predicted probability. "
         "Gap = mean confidence - accuracy; a positive gap means over-confident.", "",
         "## Brier score per test actor", "",
         "| fold | test actor | Brier seed 1 | Brier seed 2 | Brier, mean of seeds | Aung's LR Brier (same actor) |",
         "|---|---|---|---|---|---|"]
    for a, g in R.groupby("actor"):
        b1 = g[g.seed == 1].brier.iloc[0]
        b2 = g[g.seed == 2].brier.iloc[0]
        L.append(f"| {g.fold.iloc[0]} | {a} | {b1:.3f} | {b2:.3f} | {by_actor.loc[a, 'brier']:.3f} | "
                 f"{AUNG_LR_BRIER[a]:.3f} |")
    L.append(f"| **mean of the 4 folds** | | **{by_seed.loc[1, 'brier']:.3f}** | **{by_seed.loc[2, 'brier']:.3f}** | "
             f"**{m['mean_of_folds_over_seeds']['brier']:.3f}** | **{AUNG_LR_MEAN:.3f}** |")
    L += ["", "Aung's numbers are from his report of 12 Nov 2026 and are **not checked in this repository**. They are "
          "joined on the test actor: his fold k is our fold k-1 (his fold 2 = ex02 = our fold 1). Lower is better.", ""]

    L += ["## Confidence against accuracy per test actor", "",
          "| fold | test actor | seed | accuracy | mean confidence | gap (conf - acc) | mean conf. when correct | "
          "mean conf. when wrong | clips with conf. > 0.9 | accuracy on those |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.sort_values(["fold", "seed"]).iterrows():
        a9 = "" if r.acc_when_conf_over_0_9 is None else f"{r.acc_when_conf_over_0_9:.1%}"
        L.append(f"| {r.fold} | {r.actor} | {r.seed} | {r.accuracy:.1%} | {r.mean_confidence:.1%} | "
                 f"{r.gap_conf_minus_acc * 100:+.1f} pts | {r.mean_conf_correct:.1%} | {r.mean_conf_wrong:.1%} | "
                 f"{r.share_conf_over_0_9:.1%} | {a9} |")
    for s in (1, 2):
        L.append(f"| **mean of folds** | | {s} | {by_seed.loc[s, 'accuracy']:.1%} | {by_seed.loc[s, 'mean_confidence']:.1%} | "
                 f"{by_seed.loc[s, 'gap_conf_minus_acc'] * 100:+.1f} pts | | | | |")
    L.append("")

    L += ["## Reliability (10 bins, all test clips pooled over the 4 actors)", "",
          "![Reliability](reliability.png)", "",
          "Bins by the highest predicted probability. With 7 classes the highest probability is at least 1/7, so the "
          "lowest bin is always empty. Bins with fewer than 20 clips are left out of the plot but kept in the table.", "",
          "| confidence bin | seed 1: clips | mean conf. | accuracy | seed 2: clips | mean conf. | accuracy |",
          "|---|---|---|---|---|---|---|"]
    r1, r2 = rel[1], rel[2]
    f = lambda v: "" if np.isnan(v) else f"{v:.1%}"  # noqa: E731
    for i in range(N_BINS):
        L.append(f"| {r1.bin[i]} | {r1.n[i]} | {f(r1.mean_confidence[i])} | {f(r1.accuracy[i])} | {r2.n[i]} | "
                 f"{f(r2.mean_confidence[i])} | {f(r2.accuracy[i])} |")
    L += ["", f"Expected calibration error (summary of the table: sum over bins of share x |accuracy - mean "
          f"confidence|): seed 1 {m['ece_by_seed'][1]:.3f}, seed 2 {m['ece_by_seed'][2]:.3f}, both seeds pooled "
          f"{m['ece_both_seeds']:.3f}.", ""]

    L += ["## Test confusion matrices and per-style precision, recall, F1 (CLAUDE.md rule)", "",
          "From the same predictions as above (identical to `reports/training_all/`). Rows = true style, columns = "
          "predicted style; each box the share of its row and the clip count.", ""]
    for fold in sorted(R.fold.unique()):
        fr = R[R.fold == fold].sort_values("seed")
        L += [f"### Fold {fold} (test actor {fr.actor.iloc[0]})", "", f"![Fold {fold}](confusion_fold{fold}.png)", "",
              "| style | test clips | seed 1 precision | seed 1 recall | seed 1 F1 | seed 2 precision | seed 2 recall | "
              "seed 2 F1 |", "|---|---|---|---|---|---|---|---|"]
        p1, p2 = fr.per_class.iloc[0], fr.per_class.iloc[1]
        for s in styles:
            L.append(f"| {s} | {p1[s]['support']} | {p1[s]['precision']:.2f} | {p1[s]['recall']:.2f} | "
                     f"{p1[s]['f1']:.2f} | {p2[s]['precision']:.2f} | {p2[s]['recall']:.2f} | {p2[s]['f1']:.2f} |")
        mp = lambda p, k: np.mean([p[s][k] for s in styles])  # noqa: E731
        L += [f"| **macro average** | {sum(p1[s]['support'] for s in styles)} | **{mp(p1, 'precision'):.2f}** | "
              f"**{mp(p1, 'recall'):.2f}** | **{fr.macro_f1.iloc[0]:.2f}** | **{mp(p2, 'precision'):.2f}** | "
              f"**{mp(p2, 'recall'):.2f}** | **{fr.macro_f1.iloc[1]:.2f}** |", ""]
    L += ["Files: `metrics.json` (every number above, per run), `reliability.png`, `confusion_fold{k}.png`.", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

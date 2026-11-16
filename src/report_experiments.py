"""Compare the baseline with Change A (augmentation) and Change B (validation on a held-out actor).

    python -m src.report_experiments

Reads models/fold{k}_seed1_{pretrained|aug|valactor}/metrics.json (+ settings.json, epochs.csv) for
folds 0 and 2 and writes reports/experiments/:
    report.md              summary table, per-run details, per-fold confusion matrices and
                           per-style precision / recall / F1 (CLAUDE.md rule), then notes.md if present
    confusion_fold{k}.png  baseline, A and B side by side
All test numbers come from each run's single test evaluation with its best-validation checkpoint.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval_checkpoints import PNG_META, plot_confusion  # noqa: E402
from src.report_all import md_table  # noqa: E402
from src.utils import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "reports" / "experiments"
FOLDS, SEED = [0, 2], 1
RUNS = [("baseline", "pretrained", "configs/train.yaml"),
        ("A: augmentation", "aug", "configs/train_aug.yaml"),
        ("B: val on new voice", "valactor", "configs/train_valactor.yaml")]


def load(fold: int, tag: str, seed: int = SEED) -> dict | None:
    d = REPO_ROOT / "models" / f"fold{fold}_seed{seed}_{tag}"
    if not (d / "DONE").exists():
        return None
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    s = json.loads((d / "settings.json").read_text(encoding="utf-8"))
    return {"dir": d.name, "metrics": m, "settings": s}


def prf_table(named: list[tuple[str, dict]], styles: list[str]) -> pd.DataFrame:
    """Per-style precision / recall / F1 for several runs side by side, plus the macro average."""
    rows = []
    for k in styles + ["macro average"]:
        row = {"style": k if k in styles else "**macro average**",
               "test clips": named[0][1]["per_class"][k]["support"] if k in styles
               else sum(p["support"] for p in named[0][1]["per_class"].values())}
        for short, m in named:
            for q, lab in (("precision", "P"), ("recall", "R"), ("f1", "F1")):
                v = m["per_class"][k][q] if k in styles else np.mean([p[q] for p in m["per_class"].values()])
                row[f"{short} {lab}"] = f"{v:.2f}" if k in styles else f"**{v:.2f}**"
        rows.append(row)
    return pd.DataFrame(rows)


def seed2_sections(styles: list[str]) -> list[str]:
    """Change A with seed 2 (same config as seed 1), and baseline vs A over both seeds."""
    seeds = [1, 2]
    runs = {(f, s, t): load(f, t, s) for f in FOLDS for s in seeds for t in ("pretrained", "aug")}
    missing = [f"fold{f}_seed{s}_{t}" for (f, s, t), r in runs.items() if r is None]
    if missing:
        raise SystemExit(f"unfinished runs: {missing}")
    for f in FOLDS:  # seed 2 must use exactly the seed-1 A config
        c1, c2 = (runs[(f, s, "aug")]["settings"]["train_config"] for s in seeds)
        assert c1 == c2, f"fold {f}: seed 2 A config differs from seed 1"
        assert runs[(f, 2, "aug")]["settings"]["seed"] == 2

    md = ["## Change A with seed 2", "",
          "Same config as the seed-1 A runs (`configs/train_aug.yaml`, checked identical in `settings.json`), "
          "only the seed changed. Baseline seed 2 = `models/fold{k}_seed2_pretrained`. Test set used once per run.", ""]
    det = []
    for f in FOLDS:
        for s in seeds:
            for t, name in (("pretrained", "baseline"), ("aug", "A")):
                m = runs[(f, s, t)]["metrics"]
                det.append({"fold": f, "seed": s, "run": name, "folder": f"`models/{runs[(f, s, t)]['dir']}`",
                            "epochs (best)": f"{m['epochs_run']} ({m['best_epoch']})",
                            "best val macro-F1": f"{m['best_val_macro_f1']:.3f}",
                            "minutes": f"{m['timing']['train_seconds_total'] / 60:.1f}"})
    md += [md_table(pd.DataFrame(det)), ""]

    for f in FOLDS:
        b2, a2 = runs[(f, 2, "pretrained")]["metrics"], runs[(f, 2, "aug")]["metrics"]
        actor = ", ".join(a2["test_actor"])
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8))
        for ax, (name, m) in zip(axes, (("baseline", b2), ("A: augmentation", a2))):
            plot_confusion(ax, np.asarray(m["confusion_counts"]), styles,
                           f"{name}: acc {m['accuracy']:.1%}, macro-F1 {m['macro_f1']:.3f}")
        fig.suptitle(f"Fold {f}, test actor {actor}, seed 2: share of each true style (row) and clip count",
                     x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"confusion_fold{f}_seed2.png", dpi=130, metadata=PNG_META)
        plt.close(fig)
        md += [f"### Fold {f} (test actor {actor}), seed 2", "",
               f"![Test confusion matrices, fold {f}, seed 2](confusion_fold{f}_seed2.png)", "",
               "P = precision, R = recall, F1 = F1 score; baseline / A.", "",
               md_table(prf_table([("baseline", b2), ("A", a2)], styles)), ""]

    md += ["## Baseline vs A on both seeds", "",
           "Test set. Change = A minus baseline with the same seed (accuracy in points).", ""]
    for f in FOLDS:
        actor = ", ".join(runs[(f, 1, "aug")]["metrics"]["test_actor"])
        cols = [(s, t) for s in seeds for t in ("pretrained", "aug")]
        rows = []
        for lab, get, fmt in ([("accuracy", lambda m: m["accuracy"], "pct"), ("macro-F1", lambda m: m["macro_f1"], "f3")]
                              + [(f"recall {k}", (lambda k: lambda m: m["per_class"][k]["recall"])(k), "f2")
                                 for k in styles]):
            row = {"metric": lab}
            for s, t in cols:
                v = get(runs[(f, s, t)]["metrics"])
                row[f"seed {s} {'baseline' if t == 'pretrained' else 'A'}"] = (
                    f"{100 * v:.1f}%" if fmt == "pct" else f"{v:.3f}" if fmt == "f3" else f"{v:.2f}")
            for s in seeds:
                d = get(runs[(f, s, "aug")]["metrics"]) - get(runs[(f, s, "pretrained")]["metrics"])
                row[f"change seed {s}"] = (f"{100 * d:+.1f}" if fmt == "pct" else f"{d:+.3f}" if fmt == "f3"
                                           else f"{d:+.2f}")
            rows.append(row)
        md += [f"### Fold {f} (test actor {actor})", "", md_table(pd.DataFrame(rows)), ""]
    return md


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {(f, name): load(f, tag) for f in FOLDS for name, tag, _ in RUNS}
    missing = [f"fold {f} {n}" for (f, n), r in runs.items() if r is None]
    if missing:
        raise SystemExit(f"unfinished runs: {missing}")
    styles = list(next(iter(runs.values()))["metrics"]["per_class"])

    summary, details = [], []
    for f in FOLDS:
        for name, tag, cfg in RUNS:
            r = runs[(f, name)]
            m, s = r["metrics"], r["settings"]
            row = {"fold (test actor)": f"{f} ({', '.join(m['test_actor'])})", "run": name,
                   "accuracy": f"{100 * m['accuracy']:.1f}%", "macro-F1": f"{m['macro_f1']:.3f}"}
            row.update({f"recall {k}": f"{m['per_class'][k]['recall']:.2f}" for k in styles})
            summary.append(row)
            v = s.get("validation") or {}
            acts = v.get("actors") or {}
            if not acts:  # runs made before settings.json recorded the actors: read the fold's split file
                sp = pd.read_csv(REPO_ROOT / "data" / "splits" / f"fold_{f}.csv")
                acts = {"train": sorted(sp.loc[sp["split"] == "train", "actor"].unique())}
            details.append({
                "fold": f, "run": name, "folder": f"`models/{r['dir']}`", "config": f"`{cfg}`",
                "train actors (clips)": f"{', '.join(acts['train'])} ({s['n_clips']['train']})",
                "val (clips)": (f"{v['val_actor']}, all clips ({s['n_clips']['val']})" if v.get("val_actor")
                                else f"same 3 actors, 20% ({s['n_clips']['val']})"),
                "epochs (best)": f"{m['epochs_run']} ({m['best_epoch']})",
                "best val macro-F1": f"{m['best_val_macro_f1']:.3f}",
                "minutes": f"{m['timing']['train_seconds_total'] / 60:.1f}"})

    md = ["# Training changes compared with the baseline (ResNet-18 pretrained, seed 1)", "",
          "Baseline: `configs/train.yaml`. Change A: training-only augmentation on the log-mel "
          "(`configs/train_aug.yaml`). Change B: one training actor held out completely as the validation set, "
          "early stopping and checkpoint choice on its macro-F1 (`configs/train_valactor.yaml`). Everything else "
          "is identical. Each run used its test actor once, with the checkpoint chosen on validation; nothing was "
          "chosen with the test set. One seed per setting.", "",
          "## Summary (test set)", "", md_table(pd.DataFrame(summary)), "",
          "## Runs", "", md_table(pd.DataFrame(details)), ""]

    md += ["## Test confusion matrices and per-style precision, recall, F1", "",
           "Rows = true style, columns = predicted style; each box shows the share of its row and the clip count.", ""]
    for f in FOLDS:
        fr = [(name, runs[(f, name)]["metrics"]) for name, _, _ in RUNS]
        fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.8))
        for ax, (name, m) in zip(axes, fr):
            plot_confusion(ax, np.asarray(m["confusion_counts"]), styles,
                           f"{name}: acc {m['accuracy']:.1%}, macro-F1 {m['macro_f1']:.3f}")
        actor = ", ".join(fr[0][1]["test_actor"])
        fig.suptitle(f"Fold {f}, test actor {actor}, seed {SEED}: share of each true style (row) and clip count",
                     x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"confusion_fold{f}.png", dpi=130, metadata=PNG_META)
        plt.close(fig)

        rows = []
        for k in styles + ["macro average"]:
            row = {"style": k if k != "macro average" else "**macro average**",
                   "test clips": fr[0][1]["per_class"][k]["support"] if k in styles
                   else sum(p["support"] for p in fr[0][1]["per_class"].values())}
            for name, m in fr:
                short = name.split(":")[0]
                for q, lab in (("precision", "P"), ("recall", "R"), ("f1", "F1")):
                    val = (m["per_class"][k][q] if k in styles
                           else np.mean([p[q] for p in m["per_class"].values()]))
                    row[f"{short} {lab}"] = f"{val:.2f}" if k in styles else f"**{val:.2f}**"
            rows.append(row)
        md += [f"### Fold {f} (test actor {actor})", "", f"![Test confusion matrices, fold {f}](confusion_fold{f}.png)",
               "", "P = precision, R = recall, F1 = F1 score; baseline / A / B.", "", md_table(pd.DataFrame(rows)), ""]

    md += seed2_sections(styles)

    notes = OUT / "notes.md"
    if notes.exists():
        md += [notes.read_text(encoding="utf-8")]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()

"""Results table over all folds and seeds (pretrained runs only).

    python -m src.report_all

Reads models/fold{k}_seed{s}_pretrained/metrics.json for every finished run (folder has DONE)
and writes reports/training_all/report.md. The from-scratch pilot is not included.
Spread = sample standard deviation (ddof=1) and min to max.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src.utils import REPO_ROOT

OUT = REPO_ROOT / "reports" / "training_all"


def md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def collect() -> pd.DataFrame:
    rows = []
    for d in sorted((REPO_ROOT / "models").glob("fold*_seed*_pretrained")):
        m = re.fullmatch(r"fold(\d+)_seed(\d+)_pretrained", d.name)
        if not m or not (d / "DONE").exists():
            continue
        x = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"fold": int(m[1]), "seed": int(m[2]), "test_actor": ",".join(x["test_actor"]),
                     "accuracy": x["accuracy"], "macro_f1": x["macro_f1"],
                     "best_val_macro_f1": x["best_val_macro_f1"], "best_epoch": x["best_epoch"],
                     "epochs_run": x["epochs_run"], "train_min": x["timing"]["train_seconds_total"] / 60,
                     "length_only_acc": x["baselines"]["length_only_logreg"]["accuracy"],
                     "length_only_f1": x["baselines"]["length_only_logreg"]["macro_f1"]})
    return pd.DataFrame(rows).sort_values(["fold", "seed"]).reset_index(drop=True)


def spread(s: pd.Series, pct: bool) -> str:
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.3f}")
    sd = s.std(ddof=1) if len(s) > 1 else float("nan")
    sd_txt = (f"{100 * sd:.1f} pts" if pct else f"{sd:.3f}") if len(s) > 1 else "n/a"
    return f"{f(s.mean())} (sd {sd_txt}; {f(s.min())} to {f(s.max())})"


def per_fold_confusion_sections(df: pd.DataFrame) -> list[str]:
    """Confusion-matrix image and per-style precision / recall / F1 table for every fold.

    Source: reports/training_all/checkpoint_test_metrics.json, written by `python -m src.eval_checkpoints`
    (best.pt of every run re-evaluated on its test actor). Required for every finished run (see CLAUDE.md).
    """
    src = OUT / "checkpoint_test_metrics.json"
    if not src.exists():
        raise SystemExit(f"{src} missing: run `python -m src.eval_checkpoints` first")
    ev = json.loads(src.read_text(encoding="utf-8"))
    styles, er = ev["styles"], ev["runs"]
    missing = [f"fold{f}_seed{s}_pretrained" for f, s in zip(df["fold"], df["seed"])
               if f"fold{f}_seed{s}_pretrained" not in er]
    if missing:
        raise SystemExit(f"no checkpoint test evaluation for {missing}: run `python -m src.eval_checkpoints`")

    md = ["## Test confusion matrices and per-style precision, recall, F1", "",
          "Each run's best checkpoint (`best.pt`) loaded again and applied to its test actor; no retraining "
          "(`python -m src.eval_checkpoints`). The predictions match each run's saved `test_predictions.csv` "
          "clip for clip. Matrix: rows = true style, columns = predicted style; each box shows the share of "
          "its row and the clip count. Precision: of the clips predicted as this style, the share that "
          "really are. Recall: of the clips of this style, the share found. F1: their harmonic mean.", ""]
    for fold, g in df.groupby("fold"):
        g = g.sort_values("seed")
        rs = [er[f"fold{fold}_seed{s}_pretrained"] for s in g["seed"]]
        agree = min(r["agreement_with_saved_predictions"] for r in rs)
        md += [f"### Fold {fold} (test actor {', '.join(rs[0]['test_actor'])}, {rs[0]['n_test']} clips)", "",
               f"![Test confusion matrices, fold {fold}](confusion_fold{fold}.png)", ""]
        if agree < 1:
            md += [f"Note: re-evaluated predictions match the saved ones for only {agree:.2%} of clips.", ""]
        rows = []
        for s in styles:
            row = {"style": s, "test clips": rs[0]["per_class"][s]["support"]}
            for r in rs:
                pc = r["per_class"][s]
                for k in ("precision", "recall", "f1"):
                    row[f"seed {r['seed']} {'F1' if k == 'f1' else k}"] = f"{pc[k]:.2f}"
            rows.append(row)
        row = {"style": "**macro average**", "test clips": rs[0]["n_test"]}
        for r in rs:
            pcs = r["per_class"].values()
            for k in ("precision", "recall", "f1"):
                row[f"seed {r['seed']} {'F1' if k == 'f1' else k}"] = (
                    f"**{sum(p[k] for p in pcs) / len(styles):.2f}**")
        rows.append(row)
        md += [md_table(pd.DataFrame(rows)), ""]
    return md


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    runs = df.assign(**{
        "test accuracy": df["accuracy"].map(lambda v: f"{100 * v:.1f}%"),
        "test macro-F1": df["macro_f1"].map(lambda v: f"{v:.3f}"),
        "best val macro-F1": df["best_val_macro_f1"].map(lambda v: f"{v:.3f}"),
        "length-only acc / F1": [f"{100 * a:.1f}% / {f:.3f}" for a, f in zip(df["length_only_acc"], df["length_only_f1"])],
        "epochs (best)": [f"{e} ({b})" for e, b in zip(df["epochs_run"], df["best_epoch"])],
        "minutes": df["train_min"].map(lambda v: f"{v:.1f}"),
    })[["fold", "test_actor", "seed", "test accuracy", "test macro-F1", "best val macro-F1",
        "length-only acc / F1", "epochs (best)", "minutes"]]
    per_fold = df.groupby(["fold", "test_actor"]).agg(
        runs=("seed", "size"), acc=("accuracy", lambda s: spread(s, True)),
        f1=("macro_f1", lambda s: spread(s, False))).reset_index().rename(
        columns={"acc": "test accuracy: mean (sd; min to max)", "f1": "test macro-F1: mean (sd; min to max)"})
    per_seed = df.groupby("seed").agg(
        runs=("fold", "size"), acc=("accuracy", lambda s: spread(s, True)),
        f1=("macro_f1", lambda s: spread(s, False))).reset_index().rename(
        columns={"acc": "test accuracy: mean (sd; min to max)", "f1": "test macro-F1: mean (sd; min to max)"})
    n_expected = 8
    md = [
        "# ResNet-18 (ImageNet-pretrained): results over all folds and seeds",
        "",
        "Leave-one-actor-out: in fold k the model is tested on one actor whose voice it never heard in "
        "training. Settings: `configs/train.yaml` (unchanged for all runs). Test set used once per run, with "
        "the checkpoint that had the best validation macro-F1. Spread: sample standard deviation (sd) and "
        "the range.",
        "",
        f"Finished runs: {len(df)} of {n_expected}.",
        "",
        "## Overall",
        "",
        f"- Test accuracy over all {len(df)} runs: {spread(df['accuracy'], True)}",
        f"- Test macro-F1 over all {len(df)} runs: {spread(df['macro_f1'], False)}",
        f"- Best validation macro-F1 (same voices as training): {spread(df['best_val_macro_f1'], False)}",
        f"- Length-only baseline accuracy: {spread(df['length_only_acc'], True)}; chance is 14.3%",
        "",
        "## Per fold (test actor), over seeds",
        "",
        md_table(per_fold),
        "",
        "## Per seed, over folds",
        "",
        md_table(per_seed),
        "",
        "## Every run",
        "",
        md_table(runs),
        "",
    ]
    md += per_fold_confusion_sections(df)
    notes = OUT / "notes.md"  # plain-language notes written by a person after looking at the numbers
    if notes.exists():
        md += [notes.read_text(encoding="utf-8")]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT / 'report.md'} ({len(df)} runs)")


if __name__ == "__main__":
    build()

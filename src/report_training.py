"""Build the fold-0 training report: reports/training_fold0/report.html (+ report.md, PNGs).

    python -m src.report_training

Reads models/fold0_seed1_pretrained/ and models/fold0_seed1_scratch/. The plain-language
"reading" paragraphs live in reports/training_fold0/notes.yaml (written by a person after
looking at the numbers), so the generated numbers and the interpretation stay separate.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.utils import REPO_ROOT, load_config  # noqa: E402

RUNS = {"A": ("pretrained", "models/fold0_seed1_pretrained"), "B": ("from scratch", "models/fold0_seed1_scratch")}
COL = {"A": "#2a78d6", "B": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"
OUT = REPO_ROOT / "reports" / "training_fold0"
PNG_META = {"Software": None}


def load_run(path: str) -> dict:
    p = REPO_ROOT / path
    return {"metrics": json.loads((p / "metrics.json").read_text(encoding="utf-8")),
            "settings": json.loads((p / "settings.json").read_text(encoding="utf-8")),
            "epochs": pd.read_csv(p / "epochs.csv")}


def _axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def fig_curves(runs: dict) -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for k, r in runs.items():
        e = r["epochs"]
        name = f"Run {k} ({RUNS[k][0]})"
        a1.plot(e["epoch"], e["train_loss"], color=COL[k], lw=2, ls="--", label=f"{name}, training")
        a1.plot(e["epoch"], e["val_loss"], color=COL[k], lw=2, label=f"{name}, validation")
        a2.plot(e["epoch"], e["val_macro_f1"], color=COL[k], lw=2, marker="o", ms=4, label=name)
        b = r["metrics"]["best_epoch"]
        a2.plot([b], [e.loc[e["epoch"] == b, "val_macro_f1"].iloc[0]], marker="*", ms=14, color=COL[k],
                markeredgecolor="white")
    a1.set_title("Loss (lower is better)", loc="left", fontsize=11, color=INK)
    a2.set_title("Validation macro-F1 (higher is better); star = kept checkpoint", loc="left", fontsize=11,
                 color=INK)
    for a in (a1, a2):
        a.set_xlabel("epoch", color=INK)
        a.legend(frameon=False, fontsize=8)
        _axes(a)
    fig.tight_layout()
    path = OUT / "learning_curves.png"
    fig.savefig(path, dpi=130, metadata=PNG_META)
    plt.close(fig)
    return path


def fig_confusion(runs: dict, styles: list[str]) -> Path:
    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 5.4))
    for ax, (k, r) in zip(np.atleast_1d(axes), runs.items()):
        cmn = np.asarray(r["metrics"]["confusion_row_normalised"])
        cm = np.asarray(r["metrics"]["confusion_counts"])
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(styles)):
            for j in range(len(styles)):
                ax.text(j, i, f"{cmn[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=7.5,
                        color="white" if cmn[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(styles)), styles, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(styles)), styles, fontsize=9)
        ax.set_xlabel("predicted style", color=INK)
        ax.set_ylabel("true style", color=INK)
        ax.set_title(f"Run {k} ({RUNS[k][0]})", loc="left", fontsize=11, color=INK)
    fig.suptitle("Test confusion matrix, actor ex01: share of each true style (row) and clip count",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    path = OUT / "confusion_matrices.png"
    fig.savefig(path, dpi=130, metadata=PNG_META)
    plt.close(fig)
    return path


def pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def tables(runs: dict, styles: list[str]) -> dict[str, pd.DataFrame]:
    A = runs["A"]["metrics"]
    T = {}
    T["data"] = pd.DataFrame([{"part": p, "clips": n, "actors": a} for p, n, a in [
        ("train", runs["A"]["settings"]["n_clips"]["train"], "ex02, ex03, ex04"),
        ("validation", runs["A"]["settings"]["n_clips"]["val"], "ex02, ex03, ex04"),
        ("test", runs["A"]["settings"]["n_clips"]["test"], "ex01 only")]])
    T["time"] = pd.DataFrame([{
        "run": f"{k} ({RUNS[k][0]})", "epochs run": r["metrics"]["epochs_run"],
        "best epoch": r["metrics"]["best_epoch"],
        "seconds per epoch": f"{r['metrics']['timing']['seconds_per_epoch_mean']:.1f}",
        "total training time": f"{r['metrics']['timing']['train_seconds_total'] / 60:.1f} min"} for k, r in runs.items()])
    T["overall"] = pd.DataFrame([{
        "run": f"{k} ({RUNS[k][0]})", "test accuracy": pct(r["metrics"]["accuracy"]),
        "test macro-F1": f"{r['metrics']['macro_f1']:.3f}",
        "best validation macro-F1": f"{r['metrics']['best_val_macro_f1']:.3f}"} for k, r in runs.items()])
    rows = []
    for s in styles:
        row = {"style": s, "test clips": A["per_class"][s]["support"]}
        for k, r in runs.items():
            pc = r["metrics"]["per_class"][s]
            row[f"{k} precision"] = f"{pc['precision']:.2f}"
            row[f"{k} recall"] = f"{pc['recall']:.2f}"
            row[f"{k} F1"] = f"{pc['f1']:.2f}"
        rows.append(row)
    T["per_class"] = pd.DataFrame(rows)
    T["confusions"] = pd.DataFrame([{
        "true style": c["true"], "predicted as": c["predicted"], "clips": c["count"],
        "share of that style": pct(c["share_of_true_class"])} for c in A["top_confusions"]])
    b = A["baselines"]
    T["shortcut"] = pd.DataFrame([
        {"model": "Run A: ResNet-18 pretrained", "test accuracy": pct(A["accuracy"]),
         "test macro-F1": f"{A['macro_f1']:.3f}"},
        {"model": "Run B: ResNet-18 from scratch", "test accuracy": pct(runs["B"]["metrics"]["accuracy"]),
         "test macro-F1": f"{runs['B']['metrics']['macro_f1']:.3f}"},
        {"model": "Length only (logistic regression on duration and was_cropped)",
         "test accuracy": pct(b["length_only_logreg"]["accuracy"]),
         "test macro-F1": f"{b['length_only_logreg']['macro_f1']:.3f}"},
        {"model": f"Always predict the most common training style ({b['majority_class']['class']})",
         "test accuracy": pct(b["majority_class"]["accuracy"]), "test macro-F1": f"{b['majority_class']['macro_f1']:.3f}"},
        {"model": "Chance (random guess among 7 styles)", "test accuracy": pct(b["chance"]["accuracy"]),
         "test macro-F1": "about 0.143"},
    ])
    T["cropped"] = pd.DataFrame([{
        "clips": "cropped (longer than 4 s)" if v == "True" else "not cropped (4 s or shorter)",
        "test clips": runs["A"]["metrics"]["accuracy_by_was_cropped"][v]["n"],
        **{f"Run {k} accuracy": pct(r["metrics"]["accuracy_by_was_cropped"][v]["accuracy"]) for k, r in runs.items()}}
        for v in ("False", "True")])
    T["length_recall"] = pd.DataFrame([{"style": s, "recall of length-only model": f"{v:.2f}"}
                                       for s, v in b["length_only_logreg"]["per_class_recall"].items()])
    return T


def to_html_table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
                   for r in df.itertuples(index=False))
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def to_md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pcfg = load_config()
    styles = pcfg["labels"]["styles"]
    runs = {k: load_run(p) for k, (_, p) in RUNS.items()}
    notes = yaml.safe_load((OUT / "notes.yaml").read_text(encoding="utf-8"))
    curves, conf = fig_curves(runs), fig_confusion(runs, styles)
    T = tables(runs, styles)
    s = runs["A"]["settings"]
    tc = s["train_config"]
    settings_list = [
        "Model: ResNet-18 (an image-recognition network) reading the spectrogram as a 1-channel picture "
        "of 128 frequency bands x 400 time steps (4 seconds).",
        "Run A starts from weights pre-trained on ImageNet (photos); run B starts from random weights. "
        "Everything else is identical.",
        f"Optimiser AdamW, learning rate {tc['optim']['lr_pretrained']} (A) and {tc['optim']['lr_scratch']} (B), "
        f"weight decay {tc['optim']['weight_decay']}, cosine schedule, batch size {tc['train']['batch_size']}, "
        f"at most {tc['train']['max_epochs']} epochs, early stopping after "
        f"{tc['train']['early_stopping_patience']} epochs without improvement, no data augmentation, "
        "mixed precision.",
        "The checkpoint with the best validation macro-F1 is kept; the test set was used once, at the end.",
        f"Fold 0, seed {s['seed']}. Test actor: ex01, whose voice the model never heard during training.",
    ]
    gpu = s["gpu"]
    img = lambda p: base64.b64encode(p.read_bytes()).decode()  # noqa: E731
    secs = [
        ("1. What we trained", "list", settings_list, T["data"], notes.get("trained", "")),
        ("2. How long it took", "text", f"GPU: {gpu}.", T["time"], notes.get("time", "")),
        ("3. Learning curves", "img", curves, None, notes.get("curves", "")),
        ("4. Test results (actor ex01)", "img", conf, [T["overall"], T["per_class"]], notes.get("results", "")),
        ("5. What it got wrong (run A)", "text", "The most frequent mistakes of the pretrained model:",
         T["confusions"], notes.get("errors", "")),
        ("6. Shortcut check: can clip length alone do this?", "text",
         "Slow styles fill more of the 4 second window, so length carries some information about the style. "
         "A simple model that sees only the clip length (seconds of real audio in the window, and whether the "
         "clip was cut at 4 s) shows how far that alone gets.",
         [T["shortcut"], T["length_recall"], T["cropped"]], notes.get("shortcut", "")),
        ("7. Caveats", "list", notes.get("caveats", []), None, ""),
    ]
    explain = notes.get("glossary", "")

    # HTML
    h = ["<!doctype html>", "<html lang='en'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width, initial-scale=1'>",
         "<title>Fold 0 training report</title>", """<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e3e2de;--card:#fff}
body{background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,Segoe UI,sans-serif;margin:0;padding:24px 16px}
main{max-width:1000px;margin:0 auto} h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:34px 0 8px}
.sub{color:var(--muted)} .tw{overflow-x:auto;margin:10px 0} table{border-collapse:collapse;font-size:13.5px}
th,td{border-bottom:1px solid var(--line);padding:5px 10px;text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600} img{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
.note{background:var(--card);border-left:3px solid #2a78d6;padding:10px 14px;margin:12px 0;border-radius:4px}
</style></head><body>""", "<main>", "<h1>Black Box: first ResNet-18 results (fold 0, seed 1)</h1>",
         f"<p class='sub'>{html.escape(notes.get('subtitle', ''))}</p>"]
    if explain:
        h.append(f"<div class='note'>{html.escape(explain)}</div>")
    md = ["# Black Box: first ResNet-18 results (fold 0, seed 1)", "", notes.get("subtitle", ""), ""]
    if explain:
        md += [f"> {explain}", ""]
    for title, kind, body, tbl, note in secs:
        h.append(f"<h2>{html.escape(title)}</h2>")
        md += [f"## {title}", ""]
        if kind == "list":
            h.append("<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in body) + "</ul>")
            md += [f"- {x}" for x in body] + [""]
        elif kind == "text":
            h.append(f"<p>{html.escape(body)}</p>")
            md += [body, ""]
        tbls = tbl if isinstance(tbl, list) else ([tbl] if tbl is not None else [])
        for t in tbls:
            h.append(to_html_table(t))
            md += [to_md_table(t), ""]
        if kind == "img":
            h.append(f"<img alt='{html.escape(title)}' src='data:image/png;base64,{img(body)}'>")
            md += [f"![{title}]({body.name})", ""]
        if note:
            h.append(f"<div class='note'>{html.escape(note)}</div>")
            md += [f"**Reading:** {note}", ""]
    h.append("</main></body></html>")
    (OUT / "report.html").write_text("\n".join(h), encoding="utf-8")
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'report.html'} and report.md")


if __name__ == "__main__":
    build()

"""Sanity check (model-parameter randomisation) for Grad-CAM, IG and LIME: ResNet-18 fold 1 seed 1, 100 clips.

    python -m src.explain.run_sanity --stages 0            # control (a) + timing
    python -m src.explain.run_sanity --stages 1 2 3 4 5 6 control43
    python -m src.explain.run_sanity --report              # rebuild the report from the saved stage files

Question: do the explanations depend on what the model learned? The learned weights are destroyed from the top down
(cascading) and each stage's explanation is compared with the saved original one.

Stages: 0 original; 1 fc; 2 + layer4; 3 + layer3; 4 + layer2; 5 + layer1; 6 + conv1 and bn1 (fully random).
Re-initialisation: PyTorch default reset_parameters() for Conv2d / Linear / BatchNorm2d (batch-norm weight, bias
AND running statistics), torch.manual_seed(42) once, blocks in the fixed order fc, layer4, layer3, layer2, layer1,
conv1+bn1, so every stage keeps the previous stage's random blocks (true cascade). Control (b): stage 6 with seed 43.

Everything else as in the main run (src/explain/run_resnet.py): same 100 clips, same fills, same explainer code and
settings and LIME seeds, float32 eval mode, real segments only. Target = the ORIGINAL model's predicted class.
best.pt is loaded into memory and never written. Existing reports and scores are not changed.

Writes reports/sanity_fold1/: scores_stage{k}.csv, scores_control43.csv, report.md, metrics.json, sanity_tau.png,
confusion_original.png, confusion_random.png.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
from torch import nn  # noqa: E402

from src.data.torch_dataset import SpectrogramDataset  # noqa: E402
from src.eval_checkpoints import PNG_META, plot_confusion  # noqa: E402
from src.evaluate import classification_metrics  # noqa: E402
from src.explain.gradcam import grad_cam  # noqa: E402
from src.explain.ig import integrated_gradients  # noqa: E402
from src.explain.lime_segments import lime_segments  # noqa: E402
from src.explain.run_resnet import FILL_ORDER, FOLD, OUT as EXPLAIN_DIR, RUN  # noqa: E402
from src.explain.run_whitebox import FILLS, IG_STEPS, LIME_SAMPLES, SEED, top_k_overlap  # noqa: E402
from src.explain.segments import make_predict_fn  # noqa: E402
from src.models import lrfs  # noqa: E402
from src.models.resnet import build_resnet18  # noqa: E402
from src.utils import REPO_ROOT, load_config  # noqa: E402

OUT = REPO_ROOT / "reports" / f"sanity_fold{FOLD}"
STAGE_NAMES = ["original", "fc", "+ layer4", "+ layer3", "+ layer2", "+ layer1", "+ conv1, bn1 (all)"]
INIT_SEED, CONTROL_SEED = 42, 43
EXPL = [("GradCAM", "none")] + [(m, f) for m in ("IG", "LIME") for f in FILL_ORDER]
COLS = [f"s{j:02d}" for j in range(16)]
COLOURS = {"GradCAM": "#2a78d6", "IG": "#eb6834", "LIME": "#1baf7a"}
DASH = {"none": "-", "silence": "-", "average": "--", "room_noise": ":"}
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def randomise(model: nn.Module, stage: int, seed: int) -> nn.Module:
    """Copy of `model` with the top `stage` blocks re-initialised (cascading, fixed seed and order). CPU."""
    m = copy.deepcopy(model).cpu()
    blocks = [[m.fc], [m.layer4], [m.layer3], [m.layer2], [m.layer1], [m.conv1, m.bn1]]
    torch.manual_seed(seed)
    for blk in blocks[:stage]:
        for top in blk:
            for mod in top.modules():
                if isinstance(mod, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    mod.reset_parameters()          # BatchNorm2d: also resets running mean / var
    return m


def wording(t: float) -> str:
    if np.isnan(t):
        return "undefined"
    return "changes strongly" if t < 0.3 else ("partly changes" if t <= 0.7 else "barely changes")


class Ctx:
    def __init__(self):
        tcfg = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
        self.pcfg = load_config(REPO_ROOT / tcfg["preprocess_config"])
        self.styles = self.pcfg["labels"]["styles"]
        splits = self.pcfg["paths"]["splits_dir"]
        self.device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        norm = json.loads((splits / f"norm_stats_fold_{FOLD}.json").read_text(encoding="utf-8"))
        fv = json.loads((splits / f"fill_values_fold_{FOLD}.json").read_text(encoding="utf-8"))
        self.g_mean, self.g_std = float(norm["mean"]), float(norm["std"])
        self.band = {k: (np.asarray(fv[key], dtype=np.float64) if key else
                         np.full(len(fv["mean_db_per_mel"]), float(fv["padding_value_db"])))
                     for k, (key, _) in FILLS.items()}
        base = build_resnet18(tcfg["model"]["n_classes"], tcfg["model"]["in_channels"], pretrained=False)
        base.load_state_dict(torch.load(REPO_ROOT / "models" / RUN / "best.pt", map_location="cpu", weights_only=True))
        self.base = base.eval()
        self.clips = pd.read_csv(EXPLAIN_DIR / f"clips_fold{FOLD}.csv")
        self.saved = pd.read_csv(EXPLAIN_DIR / "resnet_segment_scores.csv")
        info = self.saved.drop_duplicates("clip_id").set_index("clip_id").loc[self.clips["clip_id"]]
        self.target = info["y_pred"].to_numpy()          # the ORIGINAL model's predicted class
        ds = SpectrogramDataset(self.pcfg, FOLD, "test")
        rows = pd.Series(np.arange(len(ds.clip_ids)), index=ds.clip_ids).loc[self.clips["clip_id"]].to_numpy()
        self.X = ds.x[rows].astype(np.float32)
        self.y = ds.y[rows]
        self.real = lrfs.real_segment_mask(ds.n_valid[rows])


def explain(c: Ctx, model: nn.Module) -> tuple[pd.DataFrame, np.ndarray]:
    model = model.to(c.device).eval()
    predict = make_predict_fn(model, c.g_mean, c.g_std, c.device)
    logits = predict(c.X)
    rows = []
    for i, cid in enumerate(c.clips["clip_id"]):
        t = int(c.target[i])
        gc = grad_cam(model, model.layer4, c.X[i], t, c.g_mean, c.g_std, c.device)
        rows.append({"clip_id": cid, "method": "GradCAM", "fill": "none", **dict(zip(COLS, gc["segments"]))})
        for fname in FILL_ORDER:
            ig = integrated_gradients(model, c.X[i], c.real[i], c.band[fname], t, c.g_mean, c.g_std, c.device,
                                      steps=IG_STEPS, batch=25)
            rng = np.random.default_rng([SEED, i, list(FILLS).index(fname)])        # as in the main run
            li = lime_segments(predict, c.X[i], c.real[i], c.band[fname], t, rng, n_samples=LIME_SAMPLES)
            rows.append({"clip_id": cid, "method": "IG", "fill": fname, **dict(zip(COLS, ig["segments"]))})
            rows.append({"clip_id": cid, "method": "LIME", "fill": fname, **dict(zip(COLS, li["segments"]))})
    return pd.DataFrame(rows), logits


def run(c: Ctx, which: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for w in which:
        t0 = time.time()
        if w == "control43":
            model, tag = randomise(c.base, 6, CONTROL_SEED), "control43"
        else:
            k = int(w)
            model, tag = randomise(c.base, k, INIT_SEED), f"stage{k}"
        df, logits = explain(c, model)
        df.to_csv(OUT / f"scores_{tag}.csv", index=False, float_format="%.8g")
        np.save(OUT / f"logits_{tag}.npy", logits)
        dt = time.time() - t0
        timing = json.loads((OUT / "timing.json").read_text()) if (OUT / "timing.json").exists() else {}
        timing[tag] = dt
        (OUT / "timing.json").write_text(json.dumps(timing, indent=2))
        print(f"{tag}: {dt / 60:.1f} min", flush=True)
        del model
        torch.cuda.empty_cache()


def compare(c: Ctx, cur: pd.DataFrame, ref: pd.DataFrame) -> list[dict]:
    out = []
    for meth, fname in EXPL:
        A = cur[(cur.method == meth) & (cur.fill == fname)].set_index("clip_id").loc[c.clips["clip_id"], COLS].to_numpy(float)
        B = ref[(ref.method == meth) & (ref.fill == fname)].set_index("clip_id").loc[c.clips["clip_id"], COLS].to_numpy(float)
        taus, tabs, top3, top1 = [], [], [], []
        excluded = 0
        for i in range(len(A)):
            r = c.real[i]
            a, b = A[i][r], B[i][r]
            if np.ptp(a) == 0 or np.ptp(b) == 0:          # constant over real segments: tau undefined
                excluded += 1
                continue
            taus.append(kendalltau(a, b).statistic)
            tabs.append(kendalltau(np.abs(a), np.abs(b)).statistic)
            top3.append(top_k_overlap(a, b))
            top1.append(int(np.argmax(a) == np.argmax(b)))
        taus = np.asarray(taus)
        out.append({"method": meth, "fill": fname, "n_clips": int(len(taus)), "excluded_undefined": excluded,
                    "tau_mean": float(np.mean(taus)) if len(taus) else float("nan"),
                    "tau_median": float(np.median(taus)) if len(taus) else float("nan"),
                    "tau_min": float(np.min(taus)) if len(taus) else float("nan"),
                    "tau_abs_mean": float(np.mean(tabs)) if len(tabs) else float("nan"),
                    "top3_mean": float(np.mean(top3)) if len(top3) else float("nan"),
                    "same_top1": float(np.mean(top1)) if len(top1) else float("nan")})
    return out


def report(c: Ctx) -> None:
    stages = [k for k in range(7) if (OUT / f"scores_stage{k}.csv").exists()]
    res, acc = [], []
    for k in stages:
        cur = pd.read_csv(OUT / f"scores_stage{k}.csv")
        for r in compare(c, cur, c.saved):
            res.append({"stage": k, **r})
        lg = np.load(OUT / f"logits_stage{k}.npy")
        met = classification_metrics(c.y, lg.argmax(1), c.styles)
        acc.append({"stage": k, "accuracy": met["accuracy"], "macro_f1": met["macro_f1"],
                    "same_prediction_as_original": float((lg.argmax(1) == c.target).mean()),
                    "target_logit_mean": float(lg[np.arange(len(lg)), c.target].mean()), "metrics": met})
    R = pd.DataFrame(res)
    ctrl = []
    if (OUT / "scores_control43.csv").exists() and 6 in stages:
        ctrl = compare(c, pd.read_csv(OUT / "scores_control43.csv"), pd.read_csv(OUT / "scores_stage6.csv"))
        lg43 = np.load(OUT / "logits_control43.npy")
        acc43 = classification_metrics(c.y, lg43.argmax(1), c.styles)
    timing = json.loads((OUT / "timing.json").read_text()) if (OUT / "timing.json").exists() else {}

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(8.6, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color="#b9b8b2", linewidth=0.8)
    for meth, fname in EXPL:
        d = R[(R.method == meth) & (R.fill == fname)].sort_values("stage")
        lab = "Grad-CAM" if meth == "GradCAM" else f"{meth} {fname.replace('_', ' ')}"
        ax.plot(d.stage, d.tau_mean, color=COLOURS[meth], linestyle=DASH[fname], linewidth=2, marker="o",
                markersize=4, label=lab)
    for r in ctrl:
        ax.plot(6.25, r["tau_mean"], marker="x", color=COLOURS[r["method"]], markersize=7, linestyle="none")
    if ctrl:
        ax.text(6.32, max(r["tau_mean"] for r in ctrl) + 0.04, "x = two random\ninits (42 vs 43)", fontsize=7,
                color=INK2, va="bottom")
    ax.set_xticks(range(7), [f"{k}\n{n}" for k, n in enumerate(STAGE_NAMES)], fontsize=7.5, color=INK2)
    ax.set_xlim(-0.2, 7.1)
    ax.set_ylim(-0.3, 1.05)
    ax.set_ylabel("mean Kendall tau vs the original explanation", color=INK2)
    ax.set_xlabel("randomised blocks (cascading, from the output down)", color=INK2)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right", ncol=2)
    ax.set_title(f"Sanity check, ResNet-18 fold 1 seed 1, {len(c.clips)} clips: explanation vs original as the "
                 "weights are randomised", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "sanity_tau.png", dpi=130, metadata=PNG_META, facecolor=SURFACE)
    plt.close(fig)

    for k, fname, title in ((0, "confusion_original.png", "Original model"),
                            (6, "confusion_random.png", "Fully randomised model (stage 6, seed 42)")):
        a = next((x for x in acc if x["stage"] == k), None)
        if a is None:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 5.8))
        plot_confusion(ax, np.asarray(a["metrics"]["confusion_counts"]), c.styles,
                       f"{title}: accuracy {a['accuracy']:.0%}, macro-F1 {a['macro_f1']:.3f} (100 clips)")
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=130, metadata=PNG_META)
        plt.close(fig)

    metrics = {"model": RUN, "n_clips": len(c.clips), "stages": STAGE_NAMES, "init_seed": INIT_SEED,
               "control_seed": CONTROL_SEED, "comparisons": res, "control_random_vs_random": ctrl,
               "model_per_stage": [{k: v for k, v in a.items() if k != "metrics"} for a in acc],
               "timing_s": timing}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ---- report ----
    nm = lambda m, f: "Grad-CAM" if m == "GradCAM" else f"{m} {f.replace('_', ' ')}"  # noqa: E731
    L = ["# Sanity check: model-parameter randomisation, ResNet-18 fold 1 seed 1 (100 clips)", "",
         "**Question:** do the Grad-CAM, IG and LIME explanations depend on what the model learned? If the learned "
         "weights are destroyed and the explanation stays the same, the explanation is not about the model.", "",
         "## Design (fixed before the results; see DECISIONS.md)", "",
         "- Model: `models/fold1_seed1_pretrained/best.pt`, loaded as an in-memory copy, float32, eval mode; the file "
         "is never written.",
         "- Cascading randomisation from the top down: " + "; ".join(f"stage {k} = {n}" for k, n in enumerate(STAGE_NAMES))
         + ". Re-initialised with PyTorch's default init per layer type (`reset_parameters()`), "
         f"`torch.manual_seed({INIT_SEED})` once, blocks in that fixed order, so each stage keeps the previous "
         "stage's random blocks. Batch-norm layers in a randomised block: weight, bias and running statistics reset.",
         "- Same 100 fold-1 test clips (`clips_fold1.csv`), same fills (fold-1 train: room noise -78.23 dB, average "
         "-45.74 dB, silence -100 dB), same explainer code and settings as the main run (Grad-CAM on layer4, IG 50 "
         "steps, LIME 500 samples with the same seeds), real segments only.",
         "- **Target: the ORIGINAL model's predicted class for each clip, explained as its logit on every randomised "
         "model**, so every stage explains the same quantity.",
         "- Each stage's explanation is compared with the saved original (`reports/explain_fold1/resnet_segment_scores.csv`): "
         "Kendall tau-b on signed scores (main) and on absolute values, top-3 overlap (signed), share of clips with "
         "the same top-1 segment; mean, median and minimum over clips.",
         "- Undefined tau (scores constant over the real segments, e.g. Grad-CAM zero everywhere after its ReLU): "
         "the clip is left out of that mean and counted in *excluded*; it is never filled with 0.",
         "- Controls: (a) stage 0 rerun vs the saved original; (b) a second fully random model (seed "
         f"{CONTROL_SEED}) vs the seed-{INIT_SEED} one, to show how similar two meaningless explanations are by "
         "chance.",
         "- Wording fixed in advance, by mean signed tau at stage 6: below 0.3 *changes strongly*, 0.3 to 0.7 *partly "
         "changes*, above 0.7 *barely changes*. No pass/fail threshold.", "",
         "![Sanity check](sanity_tau.png)", "",
         "## The model at each stage (100 clips)", "",
         "| stage | randomised | accuracy | macro-F1 | same prediction as the original | mean logit of the target class |",
         "|---|---|---|---|---|---|"]
    for a in acc:
        L.append(f"| {a['stage']} | {STAGE_NAMES[a['stage']]} | {a['accuracy']:.1%} | {a['macro_f1']:.3f} | "
                 f"{a['same_prediction_as_original']:.0%} | {a['target_logit_mean']:.2f} |")
    if ctrl:
        L.append(f"| 6 (seed {CONTROL_SEED}) | all, second init | {acc43['accuracy']:.1%} | {acc43['macro_f1']:.3f} | "
                 f"{float((lg43.argmax(1) == c.target).mean()):.0%} | "
                 f"{float(lg43[np.arange(len(lg43)), c.target].mean()):.2f} |")
    L += ["", "Chance accuracy is 14.3% (1 of 7).", ""]

    if 6 in stages:
        L += ["## Result at stage 6 (fully random model), in the fixed wording", "",
              "| method | fill | mean tau at stage 6 | wording | random vs random (seed 42 vs 43): mean tau |",
              "|---|---|---|---|---|"]
        for meth, fname in EXPL:
            r6 = R[(R.stage == 6) & (R.method == meth) & (R.fill == fname)].iloc[0]
            cc = next((x for x in ctrl if x["method"] == meth and x["fill"] == fname), None)
            ctau = "" if cc is None else f"{cc['tau_mean']:.3f}"
            L.append(f"| {nm(meth, '')} | {fname.replace('_', ' ')} | {r6.tau_mean:.3f} | **{wording(r6.tau_mean)}** | "
                     f"{ctau} |")
        L.append("")

    L += ["## Per method and fill, at every stage (compared with the saved original)", ""]
    for meth, fname in EXPL:
        d = R[(R.method == meth) & (R.fill == fname)].sort_values("stage")
        L += [f"### {nm(meth, fname)}", "",
              "| stage | randomised | clips | excluded (undefined tau) | tau mean | tau median | tau min | tau on abs values: mean "
              "| top-3 overlap | same top-1 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            L.append(f"| {r.stage} | {STAGE_NAMES[r.stage]} | {r.n_clips} | {r.excluded_undefined} | {r.tau_mean:.3f} | "
                     f"{r.tau_median:.3f} | {r.tau_min:.3f} | {r.tau_abs_mean:.3f} | {r.top3_mean:.3f} | {r.same_top1:.0%} |")
        cc = next((x for x in ctrl if x["method"] == meth and x["fill"] == fname), None)
        if cc:
            L.append(f"| control | seed 42 vs seed 43, both fully random | {cc['n_clips']} | {cc['excluded_undefined']} | "
                     f"{cc['tau_mean']:.3f} | {cc['tau_median']:.3f} | {cc['tau_min']:.3f} | {cc['tau_abs_mean']:.3f} | "
                     f"{cc['top3_mean']:.3f} | {cc['same_top1']:.0%} |")
        L.append("")

    s0 = R[R.stage == 0]
    if len(s0):
        L += ["## Control (a): stage 0 rerun vs the saved original", "",
              f"Mean tau {s0.tau_mean.min():.3f} or higher for every method and fill, same top-1 segment in "
              f"{s0.same_top1.min():.0%} of clips; lowest single-clip tau {s0.tau_min.min():.3f} (IG). The design "
              "asked for 1.000. LIME and Grad-CAM reproduce the saved scores up to the 6-digit rounding of the saved "
              "file (relative differences about 1e-6); IG differs by up to 0.1% of its largest score because summed "
              "GPU gradients are not bit-reproducible (some GPU operations have no deterministic version; noted in "
              "DECISIONS.md). That is enough to swap nearly tied segments in a few clips. Nothing was changed to "
              "force 1.000; this is the noise floor of every comparison below, and it is far above every later "
              "stage.", ""]

    L += ["## What this shows and does not show", "",
          "- Result in the fixed wording: at stage 6 every method and fill **changes strongly** (mean tau 0.02 to 0.09). "
          "Randomising only the output layer (fc) already drops the mean tau to 0.03 to 0.19, so the explanations "
          "depend on the learned read-out, not only on the input. Grad-CAM keeps the most structure after fc alone "
          "(0.19), because its layer4 activations are still the learned ones at that stage.",
          "- Agreement with the original at stage 6 is **below** the agreement between two random models (seed 42 vs "
          "43: 0.10 to 0.25). Two random networks share some input-driven structure (for example the louder "
          "segments); the original explanation does not resemble that structure.",
          "- It shows whether each explanation **depends on the learned weights**: if the ranking of segments stays "
          "close to the original after the weights are destroyed, it is driven by the input (or the method) rather "
          "than by what the model learned.",
          "- The random-vs-random control shows how much agreement two meaningless models give by chance; agreement "
          "with the original at stage 6 should be compared with that level, not with 0.",
          "- It does **not** show that an explanation is faithful or correct. Passing (the explanation changes when "
          "the weights change) is necessary, not sufficient: an explanation can depend on the weights and still "
          "point at the wrong segments. Faithfulness is tested by the deletion curves, the white-box check and the "
          "planted-tone test.",
          "- Randomising from the top down measures dependence on the upper layers first; early stages change the "
          "target logit's meaning only through fc and layer4.", ""]

    for k, fname, title in ((0, "confusion_original.png", "Original model (stage 0)"),
                            (6, "confusion_random.png", "Fully randomised model (stage 6, seed 42)")):
        a = next((x for x in acc if x["stage"] == k), None)
        if a is None:
            continue
        pc = a["metrics"]["per_class"]
        L += [f"## {title} on the 100 clips: confusion matrix and per-style metrics (CLAUDE.md rule)", "",
              "Rows = true style, columns = predicted style; each box the share of its row and the clip count.", "",
              f"![{title}]({fname})", "", "| style | clips | precision | recall | F1 |", "|---|---|---|---|---|"]
        for s in c.styles:
            L.append(f"| {s} | {pc[s]['support']} | {pc[s]['precision']:.2f} | {pc[s]['recall']:.2f} | {pc[s]['f1']:.2f} |")
        L += [f"| **macro average** | {sum(pc[s]['support'] for s in c.styles)} | "
              f"**{np.mean([pc[s]['precision'] for s in c.styles]):.2f}** | "
              f"**{np.mean([pc[s]['recall'] for s in c.styles]):.2f}** | **{a['macro_f1']:.2f}** |", ""]

    tot = sum(timing.values()) / 60 if timing else float("nan")
    L += [f"Runtime: {tot:.0f} min in total (" + ", ".join(f"{k} {v / 60:.1f}" for k, v in timing.items()) + " min).", "",
          "## Files", "",
          "- `scores_stage{0..6}.csv`, `scores_control43.csv`: 16 segment scores per clip, method and fill at each stage.",
          "- `logits_stage{0..6}.npy`, `logits_control43.npy`: logits of each model on the 100 clips.",
          "- `metrics.json` (every number above), `timing.json`, `sanity_tau.png`, `confusion_original.png`, "
          "`confusion_random.png`.", "- Code: `src/explain/run_sanity.py`.", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT / 'report.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=[])
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    c = Ctx()
    if a.stages:
        run(c, a.stages)
    report(c)


if __name__ == "__main__":
    main()

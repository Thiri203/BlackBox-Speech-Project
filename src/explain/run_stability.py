"""Stability of the ResNet explanations (fold 1, seed 1) on 50 of the 100 clips.

    python -m src.explain.run_stability --parts repeats perturb      # LIME / Grad-CAM repeats, input changes
    python -m src.explain.run_stability --parts ig_repeats            # IG with a noisy baseline (needs OK)
    python -m src.explain.run_stability --parts report                # rebuild the report only

Clips: 50 of the 100 explanation clips, stratified by correct / wrong prediction, seed 42, saved to
reports/explain_fold1/stability_clips.csv (fixed once, reused).

1. Repeats (10 per method and fill). LIME: 10 new seeds, 500 samples. Grad-CAM: 10 runs, deterministic
   reference. IG: see IG_BASELINE_NOISE_DB below (only run when asked). Scores between repeats: mean over
   the 45 pairs of top-3 overlap and Kendall's tau-b over real segments, per clip, then over clips.
2. Small input changes, applied to the 16 kHz waveform and pushed through the same log-mel pipeline as
   the cache (rebuilt clean spectrograms equal the cache bit for bit):
   - noise 40 dB: white Gaussian noise on the REAL samples only (padding untouched), variance = mean power
     of the clip's real samples in the 4 s window / 10^4 (SNR 40 dB), seeded per clip;
   - gain +1 dB / -1 dB: waveform x 10^(+-1/20), no clipping;
   - clean rerun: no change; measures the floor from GPU numerics.
   Only clips whose predicted-class probability moves by less than 0.05 are kept. Each explanation of the
   original predicted class after the change is compared with the saved one (LIME with the same seed).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.data.audio import load_audio16k  # noqa: E402
from src.data.spectrograms import logmel  # noqa: E402
from src.data.torch_dataset import SpectrogramDataset  # noqa: E402
from src.eval_checkpoints import PNG_META, plot_confusion  # noqa: E402
from src.evaluate import classification_metrics  # noqa: E402
from src.explain.gradcam import grad_cam  # noqa: E402
from src.explain.ig import integrated_gradients  # noqa: E402
from src.explain.lime_segments import lime_segments  # noqa: E402
from src.explain.run_resnet import EXPL, FILL_ORDER, FOLD, OUT, RUN  # noqa: E402
from src.explain.run_whitebox import FILLS, IG_STEPS, LIME_SAMPLES, SEED, score  # noqa: E402
from src.explain.segments import make_predict_fn  # noqa: E402
from src.models import lrfs  # noqa: E402
from src.models.resnet import build_resnet18  # noqa: E402
from src.utils import REPO_ROOT, load_config  # noqa: E402

N_CLIPS, N_REPEATS, P_MAX = 50, 10, 0.05
SNR_DB, GAIN_DB = 40.0, 1.0
IG_BASELINE_NOISE_DB = 1.0       # proposal (a): sd of Gaussian noise added to the filled segments of the baseline
PERTURB = ["clean_rerun", "noise_40db", "gain_plus_1db", "gain_minus_1db", "noise_50db"]
ADDED_LATER = {"noise_50db": "added after seeing the 40 dB result (40 dB moved the prediction for most clips)"}
LIME_SAMPLES_LARGE = 1000
COLS = [f"s{j:02d}" for j in range(16)]


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


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
        m = build_resnet18(tcfg["model"]["n_classes"], tcfg["model"]["in_channels"], pretrained=False)
        m.load_state_dict(torch.load(REPO_ROOT / "models" / RUN / "best.pt", map_location="cpu", weights_only=True))
        self.model = m.to(self.device).eval()
        self.predict = make_predict_fn(self.model, self.g_mean, self.g_std, self.device)

        all100 = pd.read_csv(OUT / f"clips_fold{FOLD}.csv")
        self.scores = pd.read_csv(OUT / "resnet_segment_scores.csv")
        info = self.scores.drop_duplicates("clip_id").set_index("clip_id").loc[all100["clip_id"]]
        all100 = all100.assign(correct=info["correct"].to_numpy(), y_pred=info["y_pred"].to_numpy(),
                               idx100=np.arange(len(all100)))
        path = OUT / "stability_clips.csv"
        sel, _ = train_test_split(all100, train_size=N_CLIPS, stratify=all100["correct"], random_state=SEED)
        sel = sel.sort_values("clip_id").reset_index(drop=True)
        if path.exists():
            assert pd.read_csv(path)["clip_id"].tolist() == sel["clip_id"].tolist(), "stability clip list changed"
        else:
            sel[["clip_id", "actor", "style", "label", "y_pred", "correct", "idx100"]].to_csv(path, index=False)
        self.clips = sel

        ds = SpectrogramDataset(self.pcfg, FOLD, "test")
        rows = pd.Series(np.arange(len(ds.clip_ids)), index=ds.clip_ids).loc[sel["clip_id"]].to_numpy()
        self.X = ds.x[rows].astype(np.float32)
        self.real = lrfs.real_segment_mask(ds.n_valid[rows])
        self.manifest = pd.read_csv(self.pcfg["paths"]["processed_dir"] / "manifest.csv").set_index("clip_id")


def pairwise(group: pd.DataFrame, real: np.ndarray) -> dict:
    A = group.sort_values("repeat")[COLS].to_numpy(float)
    res = [score(A[a][real], A[b][real]) for a, b in itertools.combinations(range(len(A)), 2)]
    return {k: float(np.nanmean([r[k] for r in res])) for k in ("tau", "top3", "tau_abs", "top3_abs")} | \
           {"tau_nan_pairs": int(sum(np.isnan(r["tau"]) for r in res)), "identical": bool(np.ptp(A, axis=0).max() == 0)}


def run_repeats(c: Ctx) -> None:
    rows = []
    for i, r in c.clips.iterrows():
        cls = int(r.y_pred)
        for rep in range(N_REPEATS):
            gc = grad_cam(c.model, c.model.layer4, c.X[i], cls, c.g_mean, c.g_std, c.device)
            rows.append({"clip_id": r.clip_id, "method": "GradCAM", "fill": "none", "repeat": rep,
                         **dict(zip(COLS, gc["segments"]))})
        for fname in FILL_ORDER:
            for rep in range(N_REPEATS):
                rng = np.random.default_rng([SEED, int(r.idx100), list(FILLS).index(fname), 1000 + rep])
                li = lime_segments(c.predict, c.X[i], c.real[i], c.band[fname], cls, rng, n_samples=LIME_SAMPLES)
                rows.append({"clip_id": r.clip_id, "method": "LIME", "fill": fname, "repeat": rep,
                             **dict(zip(COLS, li["segments"]))})
        print(f"  repeats {i + 1}/{len(c.clips)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "stability_repeats.csv", index=False, float_format="%.6g")


def run_ig_repeats(c: Ctx, variant: str) -> None:
    """variant "a": baseline + N(0, 1 dB) on the filled segments; "b": jittered integration points."""
    rows = []
    for i, r in c.clips.iterrows():
        cls = int(r.y_pred)
        for fname in FILL_ORDER:
            for rep in range(N_REPEATS):
                off = 2000 if variant == "a" else 4000
                rng = np.random.default_rng([SEED, int(r.idx100), list(FILLS).index(fname), off + rep])
                kw = ({"baseline_noise": rng.normal(0.0, IG_BASELINE_NOISE_DB, size=c.X[i].shape)} if variant == "a"
                      else {"alpha_jitter": rng.random(IG_STEPS)})
                ig = integrated_gradients(c.model, c.X[i], c.real[i], c.band[fname], cls, c.g_mean, c.g_std,
                                          c.device, steps=IG_STEPS, batch=25, **kw)
                rows.append({"clip_id": r.clip_id, "method": "IG" if variant == "a" else "IG_jitter",
                             "fill": fname, "repeat": rep,
                             "completeness_gap": ig["completeness_gap"], **dict(zip(COLS, ig["segments"]))})
        print(f"  IG repeats ({variant}) {i + 1}/{len(c.clips)}", flush=True)
    name = "stability_ig_repeats.csv" if variant == "a" else "stability_ig_jitter_repeats.csv"
    pd.DataFrame(rows).to_csv(OUT / name, index=False, float_format="%.6g")


def run_lime_large(c: Ctx) -> None:
    rows = []
    for i, r in c.clips.iterrows():
        for fname in FILL_ORDER:
            for rep in range(N_REPEATS):
                rng = np.random.default_rng([SEED, int(r.idx100), list(FILLS).index(fname), 3000 + rep])
                li = lime_segments(c.predict, c.X[i], c.real[i], c.band[fname], int(r.y_pred), rng,
                                   n_samples=LIME_SAMPLES_LARGE)
                rows.append({"clip_id": r.clip_id, "method": "LIME_1000", "fill": fname, "repeat": rep,
                             **dict(zip(COLS, li["segments"]))})
        print(f"  LIME 1000 repeats {i + 1}/{len(c.clips)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "stability_lime1000_repeats.csv", index=False, float_format="%.6g")


def perturbed_logmel(c: Ctx, clip_id: str, kind: str, idx: int) -> np.ndarray:
    sp = c.pcfg["spectrogram"]
    ms = int(c.pcfg["audio"]["max_seconds"] * sp["sr"])
    y = load_audio16k(c.pcfg, c.manifest.loc[clip_id, "audio16k_file"])[:ms].astype(np.float64)
    if kind.startswith("noise_"):
        snr = float(kind[len("noise_"):-len("db")])
        rng = np.random.default_rng([SEED, idx, int(snr)])           # 40 dB: same seed as the first run
        y = y + rng.normal(0.0, np.sqrt(np.mean(y ** 2) / 10 ** (snr / 10)), size=len(y))
    elif kind == "gain_plus_1db":
        y = y * 10 ** (GAIN_DB / 20)
    elif kind == "gain_minus_1db":
        y = y * 10 ** (-GAIN_DB / 20)
    y = np.pad(y, (0, ms - len(y))).astype(np.float32)          # padding stays exact zeros
    return logmel(y, sp)[:, :400].astype(np.float16).astype(np.float32)


def run_perturb(c: Ctx, kinds: list[str]) -> None:
    """Runs the given changes and replaces only their rows in the CSVs (other changes are kept as they are)."""
    rows, probs = [], []
    p0 = softmax(c.predict(c.X).astype(np.float64))
    for kind in kinds:
        Xp = c.X.copy() if kind == "clean_rerun" else np.stack(
            [perturbed_logmel(c, cid, kind, int(c.clips.idx100[i])) for i, cid in enumerate(c.clips.clip_id)])
        if kind == "clean_rerun":
            Xc = np.stack([perturbed_logmel(c, cid, "clean_rerun", 0) for cid in c.clips.clip_id])
            assert np.array_equal(Xc, c.X), "rebuilt clean spectrograms differ from the cache"
        p1 = softmax(c.predict(Xp).astype(np.float64))
        for i, r in c.clips.iterrows():
            cls = int(r.y_pred)
            dp = float(p1[i, cls] - p0[i, cls])
            keep = abs(dp) < P_MAX
            probs.append({"clip_id": r.clip_id, "perturbation": kind, "p_before": float(p0[i, cls]),
                          "p_after": float(p1[i, cls]), "delta_p": dp, "kept": keep,
                          "pred_after": c.styles[int(p1[i].argmax())], "class_changed": bool(p1[i].argmax() != cls)})
            if not keep:
                continue
            gc = grad_cam(c.model, c.model.layer4, Xp[i], cls, c.g_mean, c.g_std, c.device)
            rows.append({"clip_id": r.clip_id, "perturbation": kind, "method": "GradCAM", "fill": "none",
                         **dict(zip(COLS, gc["segments"]))})
            for fname in FILL_ORDER:
                ig = integrated_gradients(c.model, Xp[i], c.real[i], c.band[fname], cls, c.g_mean, c.g_std,
                                          c.device, steps=IG_STEPS, batch=25)
                rng = np.random.default_rng([SEED, int(r.idx100), list(FILLS).index(fname)])   # saved run's seed
                li = lime_segments(c.predict, Xp[i], c.real[i], c.band[fname], cls, rng, n_samples=LIME_SAMPLES)
                for meth, a in (("IG", ig["segments"]), ("LIME", li["segments"])):
                    rows.append({"clip_id": r.clip_id, "perturbation": kind, "method": meth, "fill": fname,
                                 **dict(zip(COLS, a))})
        print(f"  perturbation {kind}: kept {sum(p['kept'] for p in probs if p['perturbation'] == kind)}/50",
              flush=True)
    for name, new in (("stability_perturbed_scores.csv", rows), ("stability_perturbed_probs.csv", probs)):
        new = pd.DataFrame(new)
        if (OUT / name).exists():
            old = pd.read_csv(OUT / name)
            new = pd.concat([old[~old.perturbation.isin(kinds)], new], ignore_index=True)
        new.to_csv(OUT / name, index=False, float_format="%.6g")


def summarise(c: Ctx) -> dict:
    real = dict(zip(c.clips.clip_id, c.real))
    corr = dict(zip(c.clips.clip_id, c.clips.correct))
    out = {"repeats": [], "perturb": [], "perturb_counts": []}

    def agg(per: pd.DataFrame, keys: dict, dest: list) -> None:
        for grp in ("all", "correct", "wrong"):
            g = per if grp == "all" else per[per.correct == (grp == "correct")]
            if not len(g):
                continue
            dest.append({**keys, "group": grp, "n_clips": int(len(g)),
                         **{f"{k}_mean": float(g[k].mean()) for k in ("tau", "top3", "tau_abs", "top3_abs")},
                         "tau_median": float(g["tau"].median()), "tau_min": float(g["tau"].min()),
                         "top3_min": float(g["top3"].min()),
                         "share_clips_identical": float(g["identical"].mean()) if "identical" in g else None,
                         "share_top3_all_same": float((g["top3"] == 1).mean())})

    rep = []
    for f in ("stability_repeats.csv", "stability_ig_repeats.csv", "stability_ig_jitter_repeats.csv",
              "stability_lime1000_repeats.csv"):
        if (OUT / f).exists():
            rep.append(pd.read_csv(OUT / f))
    if rep:
        rep = pd.concat(rep)
        for (m, fl), g in rep.groupby(["method", "fill"], sort=False):
            per = pd.DataFrame([{"clip_id": cid, "correct": corr[cid], **pairwise(gg, real[cid])}
                                for cid, gg in g.groupby("clip_id")])
            agg(per, {"method": m, "fill": fl}, out["repeats"])

    if (OUT / "stability_perturbed_scores.csv").exists():
        ps = pd.read_csv(OUT / "stability_perturbed_scores.csv")
        pr = pd.read_csv(OUT / "stability_perturbed_probs.csv")
        before = c.scores.set_index(["clip_id", "method", "fill"])
        for kind in PERTURB:
            q = pr[pr.perturbation == kind]
            out["perturb_counts"].append({"perturbation": kind, "kept": int(q.kept.sum()), "n": int(len(q)),
                                          "abs_delta_p_median": float(q.delta_p.abs().median()),
                                          "abs_delta_p_max": float(q.delta_p.abs().max()),
                                          "class_changed": int(q.class_changed.sum())})
            for (m, fl), g in ps[ps.perturbation == kind].groupby(["method", "fill"], sort=False):
                per = []
                for _, r in g.iterrows():
                    b = before.loc[(r.clip_id, m, fl), COLS].to_numpy(float)
                    a = r[COLS].to_numpy(float)
                    rr = real[r.clip_id]
                    per.append({"clip_id": r.clip_id, "correct": corr[r.clip_id], **score(a[rr], b[rr]),
                                "identical": bool(np.array_equal(a, b))})
                agg(pd.DataFrame(per), {"perturbation": kind, "method": m, "fill": fl}, out["perturb"])
    return out


def write_report(c: Ctx, s: dict) -> None:
    sel = c.clips
    met = classification_metrics(sel["label"].to_numpy(), sel["y_pred"].to_numpy(), c.styles)
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    plot_confusion(ax, np.asarray(met["confusion_counts"]), c.styles,
                   f"ResNet-18 seed 1 on the 50 stability clips: accuracy {met['accuracy']:.0%}, "
                   f"macro-F1 {met['macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_resnet_50clips.png", dpi=130, metadata=PNG_META)
    plt.close(fig)

    nm = {"GradCAM": "Grad-CAM", "IG": "IG (a) noisy baseline", "IG_jitter": "IG (b) jittered steps (side check)",
          "LIME": "LIME 500 samples", "LIME_1000": "LIME 1,000 samples"}
    pnm = {"GradCAM": "Grad-CAM", "IG": "IG", "LIME": "LIME"}
    n_c = int(sel.correct.sum())
    L = ["# Stability of the explanations: ResNet-18 (fold 1, seed 1, pretrained), 50 clips", "",
         "## Setup", "",
         f"- **Clips:** 50 of the 100 explanation clips, stratified by correct / wrong prediction (seed {SEED}): "
         f"{n_c} correct, {N_CLIPS - n_c} wrong (`stability_clips.csv`). Same model, same 3 fills, same explainer "
         "code, explaining the predicted class's logit. Scores over real segments only.",
         f"- **Repeats:** {N_REPEATS} per method and fill. LIME: {N_REPEATS} new seeds, {LIME_SAMPLES} samples each. "
         "Grad-CAM: run 10 times as a deterministic reference. IG has no randomness with a fixed fill, so it is "
         f"varied in two ways: **(a)** baseline = the filled clip plus Gaussian noise of sd {IG_BASELINE_NOISE_DB:g} dB "
         "on the filled segments only, a new noise seed per repeat (main IG stability result); **(b)** side check: "
         "the 50 integration points jittered, alpha_k = (k + u_k) / 50 with u_k uniform in [0, 1), which tests only "
         "the numerical integration. LIME was also repeated with 1,000 samples (the original plan) to see whether "
         "its instability comes from the sample count.",
         "- **Between repeats:** for each clip the mean over the 45 pairs of repeats of Kendall's tau-b and top-3 "
         "overlap (signed scores; absolute-value versions in the JSON), then the mean, median and minimum over "
         "clips. 1 = identical rankings.",
         "- **Small input changes**, applied to the 16 kHz waveform and pushed through the same log-mel pipeline as "
         "the cache (rebuilding the clean clips reproduces the cache bit for bit): *noise 40 dB* = white Gaussian "
         f"noise on the real samples only (padding untouched), variance = mean power of the clip's real samples in "
         f"the 4 s window / 10^4; *gain +1 dB / -1 dB* = waveform x 10^(+-1/20); *clean rerun* = no change "
         "(measures the floor from GPU numerics). A clip is kept only if the predicted-class probability moves by "
         f"less than {P_MAX}. **Noise at 50 dB SNR was {ADDED_LATER['noise_50db']}**; the 40 dB result is kept "
         "unchanged. The explanation after the change (same class, LIME with the same seed as the saved "
         "run) is compared with the saved one.", "",
         "Nothing was tuned to make a method look more stable: the settings are the ones used for the main results.",
         ""]

    if s["repeats"]:
        L += ["## Repeats: agreement between 10 runs of the same method", "",
              "| method | fill | clips | Kendall tau: mean | median | min | top-3 overlap: mean | min | clips with the "
              "same top 3 in all pairs |", "|---|---|---|---|---|---|---|---|---|"]
        for r in [r for r in s["repeats"] if r["group"] == "all"]:
            L.append(f"| {nm[r['method']]} | {r['fill'].replace('_', ' ')} | {r['n_clips']} | {r['tau_mean']:.3f} | "
                     f"{r['tau_median']:.3f} | {r['tau_min']:.3f} | {r['top3_mean']:.3f} | {r['top3_min']:.3f} | "
                     f"{r['share_top3_all_same']:.0%} |")
        L += ["", "Split by prediction (Kendall tau mean / top-3 overlap mean):", "",
              "| method | fill | correct | wrong |", "|---|---|---|---|"]
        for r in [r for r in s["repeats"] if r["group"] == "correct"]:
            w = next(x for x in s["repeats"] if x["group"] == "wrong" and x["method"] == r["method"]
                     and x["fill"] == r["fill"])
            L.append(f"| {nm[r['method']]} | {r['fill'].replace('_', ' ')} | {r['tau_mean']:.3f} / {r['top3_mean']:.3f} "
                     f"| {w['tau_mean']:.3f} / {w['top3_mean']:.3f} |")
        L.append("")

    if s["perturb"]:
        L += ["## Small input changes: explanation before vs after", "",
              "| change | clips kept (prob. change < 0.05) | median abs prob. change | max | predicted class changed |",
              "|---|---|---|---|---|"]
        for r in s["perturb_counts"]:
            L.append(f"| {r['perturbation'].replace('_', ' ')} | {r['kept']} of {r['n']} | {r['abs_delta_p_median']:.4f} "
                     f"| {r['abs_delta_p_max']:.4f} | {r['class_changed']} |")
        L += ["", "| change | method | fill | clips | Kendall tau: mean | median | min | top-3 overlap: mean | min | "
              "same top 3 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in [r for r in s["perturb"] if r["group"] == "all"]:
            L.append(f"| {r['perturbation'].replace('_', ' ')} | {pnm[r['method']]} | {r['fill'].replace('_', ' ')} | "
                     f"{r['n_clips']} | {r['tau_mean']:.3f} | {r['tau_median']:.3f} | {r['tau_min']:.3f} | "
                     f"{r['top3_mean']:.3f} | {r['top3_min']:.3f} | {r['share_top3_all_same']:.0%} |")
        L += ["", "Split by prediction (Kendall tau mean / top-3 overlap mean; noise and gain only):", "",
              "| change | method | fill | correct | wrong |", "|---|---|---|---|---|"]
        for r in [r for r in s["perturb"] if r["group"] == "correct" and r["perturbation"] != "clean_rerun"]:
            w = next((x for x in s["perturb"] if x["group"] == "wrong" and x["perturbation"] == r["perturbation"]
                      and x["method"] == r["method"] and x["fill"] == r["fill"]), None)
            wt = f"{w['tau_mean']:.3f} / {w['top3_mean']:.3f} (n={w['n_clips']})" if w else "no clips"
            L.append(f"| {r['perturbation'].replace('_', ' ')} | {pnm[r['method']]} | {r['fill'].replace('_', ' ')} | "
                     f"{r['tau_mean']:.3f} / {r['top3_mean']:.3f} (n={r['n_clips']}) | {wt} |")
        L.append("")

    pc = met["per_class"]
    L += ["## ResNet-18 on these 50 clips (CLAUDE.md rule)", "",
          f"Accuracy {met['accuracy']:.1%}, macro-F1 {met['macro_f1']:.3f}. Rows = true style, columns = predicted "
          "style; each box the share of its row and the clip count.", "",
          "![ResNet confusion on the 50 clips](confusion_resnet_50clips.png)", "",
          "| style | clips | precision | recall | F1 |", "|---|---|---|---|---|"]
    for st in c.styles:
        L.append(f"| {st} | {pc[st]['support']} | {pc[st]['precision']:.2f} | {pc[st]['recall']:.2f} | {pc[st]['f1']:.2f} |")
    L.append(f"| **macro average** | {sum(pc[st]['support'] for st in c.styles)} | "
             f"**{np.mean([pc[st]['precision'] for st in c.styles]):.2f}** | "
             f"**{np.mean([pc[st]['recall'] for st in c.styles]):.2f}** | **{met['macro_f1']:.2f}** |")
    L += ["", "Files: `stability_clips.csv`, `stability_repeats.csv`, `stability_ig_repeats.csv` (when run), "
          "`stability_perturbed_scores.csv`, `stability_perturbed_probs.csv`, `stability_metrics.json`.", ""]
    (OUT / "stability.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "stability_metrics.json").write_text(json.dumps({**s, "resnet_on_these_clips": met}, indent=2),
                                                encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["report"], choices=["repeats", "perturb", "noise50", "ig_repeats", "ig_jitter", "lime1000", "report"])
    args = ap.parse_args()
    c = Ctx()
    if "repeats" in args.parts:
        run_repeats(c)
    if "perturb" in args.parts:
        run_perturb(c, [k for k in PERTURB if k not in ADDED_LATER])
    if "noise50" in args.parts:
        run_perturb(c, ["noise_50db"])
    if "ig_repeats" in args.parts:
        run_ig_repeats(c, "a")
    if "ig_jitter" in args.parts:
        run_ig_repeats(c, "b")
    if "lime1000" in args.parts:
        run_lime_large(c)
    write_report(c, summarise(c))
    print(f"wrote {OUT / 'stability.md'}")


if __name__ == "__main__":
    main()

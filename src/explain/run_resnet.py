"""Explain the ResNet-18 (fold 1, seed 1, pretrained, no augmentation) on the 100 white-box clips.

    python -m src.explain.run_resnet

Same explainer code as the white-box check (src/explain/), same clips (clips_fold1.csv), same fills.
Explained quantity: the logit of the ResNet's predicted class (float32, eval mode, no autocast).
- Grad-CAM (layer4, no fill), IG (50 steps, baseline = real segments filled) and LIME (500 samples,
  real segments, off = filled) for the fills silence / average / room noise -> 16 segment scores.
- IG completeness gap per clip and fill; IG cross-checked against Captum (riemann_middle, 50 steps)
  on 10 clips. Captum is not used for the results.
- Energy label of each real segment: speech if its mean_db (segments.csv, real frames) is at or above
  an Otsu threshold computed on the fold-1 TRAIN clips' real segments, else pause.
- Report: top-segment agreement between methods, speech/pause share of the top-3 segments, and the
  ResNet's confusion matrix and per-style metrics on these clips (CLAUDE.md).

Writes reports/explain_fold1/: resnet_segment_scores.csv, resnet_ig_completeness.csv,
resnet_ig_vs_captum.csv, segments_100clips.csv, resnet_explain_metrics.json, resnet_explain.md,
confusion_resnet_100clips.png. Deletion curves are a later step.
"""
from __future__ import annotations

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
from scipy.stats import kendalltau  # noqa: E402

from src.data.torch_dataset import SpectrogramDataset  # noqa: E402
from src.eval_checkpoints import PNG_META, plot_confusion  # noqa: E402
from src.evaluate import classification_metrics  # noqa: E402
from src.explain.gradcam import grad_cam  # noqa: E402
from src.explain.ig import integrated_gradients  # noqa: E402
from src.explain.lime_segments import lime_segments  # noqa: E402
from src.explain.run_whitebox import FILLS, IG_STEPS, LIME_SAMPLES, SEED, top_k_overlap  # noqa: E402
from src.explain.segments import fill_segments, make_predict_fn, pixels_to_segments, to_input  # noqa: E402
from src.models import lrfs  # noqa: E402
from src.models.resnet import build_resnet18  # noqa: E402
from src.utils import REPO_ROOT, load_config  # noqa: E402

FOLD, RESNET_SEED = 1, 1
RUN = f"fold{FOLD}_seed{RESNET_SEED}_pretrained"
OUT = REPO_ROOT / "reports" / f"explain_fold{FOLD}"
N_CAPTUM = 10
FILL_ORDER = ["silence", "average", "room_noise"]


def otsu(values: np.ndarray, bins: int = 1000) -> float:
    h, e = np.histogram(values, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(e[1:][np.argmax(between)])     # upper edge of the best "low" bin


EXPL = [("GradCAM", "none")] + [("IG", f) for f in FILL_ORDER] + [("LIME", f) for f in FILL_ORDER]
CELLS = [(p, e) for p in ("first", "middle", "last") for e in ("speech", "pause")]


def top1_position(scores: pd.DataFrame, segs: pd.DataFrame) -> dict:
    """Where each explanation's top-1 real segment sits: first real / last real (next to the padding or the
    window end) / in between, and speech or pause; shares per group (all, correct, wrong). Also the
    expected shares for a segment drawn at random from each clip's real segments."""
    E = segs.pivot(index="clip_id", columns="segment", values="energy")
    cols = [f"s{j:02d}" for j in range(16)]

    def cell(idx, j, e):
        pos = "first" if j == idx[0] else ("last" if j == idx[-1] else "middle")
        return f"{pos}_{e}"

    out = {}
    base = {g: [] for g in ("all", "correct", "wrong")}
    for k, (meth, fill) in enumerate(EXPL):
        d = scores[(scores.method == meth) & (scores.fill == fill)]
        tally = {g: [] for g in ("all", "correct", "wrong")}
        for _, r in d.iterrows():
            en = E.loc[r.clip_id].to_numpy()
            idx = np.flatnonzero(en != "padding")
            a = r[cols].to_numpy(dtype=float)
            j = idx[np.argmax(a[idx])]
            c = cell(idx, j, en[j])
            g = "correct" if r.correct else "wrong"
            tally["all"].append(c)
            tally[g].append(c)
            if k == 0:   # random-segment reference, once per clip
                shares = pd.Series([cell(idx, jj, en[jj]) for jj in idx]).value_counts(normalize=True)
                base["all"].append(shares)
                base[g].append(shares)
        out[f"{meth}/{fill}"] = {g: {"n": len(v), **{f"{p}_{e}": float(np.mean(np.asarray(v) == f"{p}_{e}"))
                                                     for p, e in CELLS}} for g, v in tally.items()}
    out["random_segment"] = {g: {"n": len(v), **{f"{p}_{e}": float(pd.concat(v, axis=1).fillna(0)
                                                                  .reindex([f"{p}_{e}"]).fillna(0).mean(axis=1).iloc[0])
                                                 for p, e in CELLS}} for g, v in base.items()}
    return out


def report_only() -> None:
    m = json.loads((OUT / "resnet_explain_metrics.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(OUT / "resnet_segment_scores.csv")
    segs = pd.read_csv(OUT / "segments_100clips.csv")
    m["top1_position"] = top1_position(scores, segs)
    (OUT / "resnet_explain_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    styles = load_config(REPO_ROOT / "configs" / "preprocess.yaml")["labels"]["styles"]
    write_report(m, styles, None, None)
    print(f"rewrote {OUT / 'resnet_explain.md'}")


def main() -> None:
    import sys
    if "--report-only" in sys.argv:
        return report_only()
    tcfg = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    pcfg = load_config(REPO_ROOT / tcfg["preprocess_config"])
    styles = pcfg["labels"]["styles"]
    splits = pcfg["paths"]["splits_dir"]
    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)

    norm = json.loads((splits / f"norm_stats_fold_{FOLD}.json").read_text(encoding="utf-8"))
    fv = json.loads((splits / f"fill_values_fold_{FOLD}.json").read_text(encoding="utf-8"))
    g_mean, g_std = float(norm["mean"]), float(norm["std"])
    band = {k: (np.asarray(fv[key], dtype=np.float64) if key else
                np.full(len(fv["mean_db_per_mel"]), float(fv["padding_value_db"])))
            for k, (key, _) in FILLS.items()}

    model = build_resnet18(tcfg["model"]["n_classes"], tcfg["model"]["in_channels"], pretrained=False)
    model.load_state_dict(torch.load(REPO_ROOT / "models" / RUN / "best.pt", map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    predict = make_predict_fn(model, g_mean, g_std, device)

    ds = SpectrogramDataset(pcfg, FOLD, "test")
    clips = pd.read_csv(OUT / f"clips_fold{FOLD}.csv")
    rows = pd.Series(np.arange(len(ds.clip_ids)), index=ds.clip_ids).loc[clips["clip_id"]].to_numpy()
    X = ds.x[rows].astype(np.float32)
    y = ds.y[rows]
    assert (y == clips["label"].to_numpy()).all()
    real = lrfs.real_segment_mask(ds.n_valid[rows])

    logits = predict(X)
    pred = logits.argmax(1)
    saved = pd.read_csv(REPO_ROOT / "models" / RUN / "test_predictions.csv").set_index("clip_id")
    agree_saved = float((saved.loc[clips["clip_id"], "y_pred"].to_numpy() == pred).mean())
    print(f"float32 predictions agree with the saved (AMP) run for {agree_saved:.0%} of the 100 clips")

    # ---- energy labels (threshold from fold-1 TRAIN clips only) ----
    seg = pd.read_csv(pcfg["paths"]["processed_dir"] / "segments.csv")
    split = pd.read_csv(splits / f"fold_{FOLD}.csv")
    train_ids = set(split.loc[split["split"] == "train", "clip_id"])
    tr = seg[seg["is_real_audio"] & seg["clip_id"].isin(train_ids)]
    thr = otsu(tr["mean_db"].to_numpy())
    s100 = seg[seg["clip_id"].isin(set(clips["clip_id"]))].copy()
    s100["energy"] = np.where(~s100["is_real_audio"], "padding",
                              np.where(s100["mean_db"] >= thr, "speech", "pause"))
    s100.to_csv(OUT / "segments_100clips.csv", index=False)
    energy = s100.pivot(index="clip_id", columns="segment", values="energy").loc[clips["clip_id"]].to_numpy()
    assert ((energy != "padding") == real).all()
    print(f"energy threshold (Otsu, fold-1 train real segments): {thr:.2f} dB")

    # ---- explanations ----
    score_rows, comp_rows = [], []
    layer = model.layer4
    map_shape = None
    lime_r2 = {f: [] for f in FILL_ORDER}
    for i, cid in enumerate(clips["clip_id"]):
        c = int(pred[i])
        base = {"clip_id": cid, "style": styles[y[i]], "y_true": int(y[i]), "y_pred": c,
                "pred_style": styles[c], "correct": bool(c == y[i]), "n_real": int(real[i].sum())}
        gc = grad_cam(model, layer, X[i], c, g_mean, g_std, device)
        map_shape = gc["map_shape"]
        score_rows.append({**base, "method": "GradCAM", "fill": "none",
                           **{f"s{j:02d}": gc["segments"][j] for j in range(16)}})
        for fname in FILL_ORDER:
            ig = integrated_gradients(model, X[i], real[i], band[fname], c, g_mean, g_std, device,
                                      steps=IG_STEPS, batch=25)
            rng = np.random.default_rng([SEED, i, list(FILLS).index(fname)])
            li = lime_segments(predict, X[i], real[i], band[fname], c, rng, n_samples=LIME_SAMPLES)
            lime_r2[fname].append(li["r2"])
            delta = ig["f_input"] - ig["f_baseline"]
            comp_rows.append({"clip_id": cid, "fill": fname, "logit_x": ig["f_input"], "logit_fill": ig["f_baseline"],
                              "sum_attr": delta + ig["completeness_gap"], "gap": ig["completeness_gap"],
                              "rel_gap": abs(ig["completeness_gap"]) / max(abs(delta), 1e-12)})
            for meth, a in (("IG", ig["segments"]), ("LIME", li["segments"])):
                score_rows.append({**base, "method": meth, "fill": fname, **{f"s{j:02d}": a[j] for j in range(16)}})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/100 clips", flush=True)

    scores = pd.DataFrame(score_rows)
    scores.to_csv(OUT / "resnet_segment_scores.csv", index=False, float_format="%.6g")
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT / "resnet_ig_completeness.csv", index=False, float_format="%.6g")

    # ---- IG vs Captum on 10 clips (every 10th clip of the list) ----
    from captum.attr import IntegratedGradients
    cap = IntegratedGradients(model)
    cap_rows = []
    for i in range(0, len(clips), len(clips) // N_CAPTUM):
        c = int(pred[i])
        for fname in FILL_ORDER:
            x = to_input(X[i][None], g_mean, g_std, device)
            xb = to_input(fill_segments(X[i], real[i], band[fname])[None], g_mean, g_std, device)
            a = cap.attribute(x, baselines=xb, target=c, n_steps=IG_STEPS, method="riemann_middle",
                              internal_batch_size=25)
            ca = pixels_to_segments(a[0, 0].detach().cpu().numpy().astype(np.float64))
            mine = scores[(scores.clip_id == clips.clip_id[i]) & (scores.method == "IG") & (scores.fill == fname)]
            mine = mine[[f"s{j:02d}" for j in range(16)]].to_numpy()[0]
            r = real[i]
            cap_rows.append({"clip_id": clips.clip_id[i], "fill": fname,
                             "kendall_tau": float(kendalltau(mine[r], ca[r]).statistic),
                             "same_top_segment": bool(np.argmax(mine[r]) == np.argmax(ca[r])),
                             "top3_overlap": top_k_overlap(mine[r], ca[r]),
                             "max_abs_diff": float(np.abs(mine - ca).max()),
                             "max_abs_score": float(np.abs(mine).max())})
    capdf = pd.DataFrame(cap_rows)
    capdf.to_csv(OUT / "resnet_ig_vs_captum.csv", index=False, float_format="%.6g")
    print(f"captum: tau min {capdf.kendall_tau.min():.4f}, max abs diff {capdf.max_abs_diff.max():.2e}")

    # ---- agreement and speech/pause ----
    expl = [("GradCAM", "none")] + [("IG", f) for f in FILL_ORDER] + [("LIME", f) for f in FILL_ORDER]
    names = ["Grad-CAM"] + [f"IG {f.replace('_', ' ')}" for f in FILL_ORDER] + \
            [f"LIME {f.replace('_', ' ')}" for f in FILL_ORDER]
    S = {}
    for (meth, fname) in expl:
        d = scores[(scores.method == meth) & (scores.fill == fname)].set_index("clip_id").loc[clips["clip_id"]]
        S[(meth, fname)] = d[[f"s{j:02d}" for j in range(16)]].to_numpy()
    correct = (pred == y)

    def ranked(a, r, k):
        idx = np.flatnonzero(r)
        return idx[np.argsort(-a[idx], kind="stable")[:k]]

    n = len(expl)
    top1 = np.zeros((n, n))
    top3 = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            t1, t3 = [], []
            for i in range(len(clips)):
                a, b = S[expl[p]][i], S[expl[q]][i]
                t1.append(ranked(a, real[i], 1)[0] == ranked(b, real[i], 1)[0])
                t3.append(len(set(ranked(a, real[i], 3)) & set(ranked(b, real[i], 3))) / min(3, real[i].sum()))
            top1[p, q], top3[p, q] = np.mean(t1), np.mean(t3)
    gc_allzero = int(sum((S[("GradCAM", "none")][i][real[i]] <= 0).all() for i in range(len(clips))))

    sp = []
    for (meth, fname), nm in zip(expl, names):
        lab = {"all": [], "correct": [], "wrong": []}
        first = []
        for i in range(len(clips)):
            t = ranked(S[(meth, fname)][i], real[i], 3)
            e = energy[i, t]
            lab["all"].extend(e)
            lab["correct" if correct[i] else "wrong"].extend(e)
            first.append(energy[i, t[0]])
        sp.append({"explanation": nm, **{f"speech_share_{k}": float(np.mean(np.asarray(v) == "speech"))
                                        for k, v in lab.items()},
                   "top1_speech_share": float(np.mean(np.asarray(first) == "speech"))})
    base_speech = float(np.mean(energy[real] == "speech"))
    base_speech_c = float(np.mean(energy[correct][real[correct]] == "speech"))
    base_speech_w = float(np.mean(energy[~correct][real[~correct]] == "speech"))

    met = classification_metrics(y, pred, styles)
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    plot_confusion(ax, np.asarray(met["confusion_counts"]), styles,
                   f"ResNet-18 seed 1 on the 100 clips: accuracy {met['accuracy']:.0%}, macro-F1 {met['macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_resnet_100clips.png", dpi=130, metadata=PNG_META)
    plt.close(fig)

    comp_sum = {f: {"abs_gap_mean": float(g["gap"].abs().mean()), "abs_gap_max": float(g["gap"].abs().max()),
                    "rel_gap_median": float(g["rel_gap"].median()), "rel_gap_max": float(g["rel_gap"].max()),
                    "abs_logit_change_mean": float((g["logit_x"] - g["logit_fill"]).abs().mean())}
                for f, g in comp.groupby("fill", sort=False)}
    metrics = {"run": RUN, "n_clips": len(clips), "agree_with_saved_amp_predictions": agree_saved,
               "energy_threshold_db": thr, "gradcam_layer": "layer4", "gradcam_map_shape": list(map_shape),
               "gradcam_all_zero_clips": gc_allzero,
               "fills_db": {k: float(v.mean()) for k, v in band.items()},
               "ig_completeness": comp_sum,
               "ig_vs_captum": {"kendall_tau_min": float(capdf.kendall_tau.min()),
                                "kendall_tau_mean": float(capdf.kendall_tau.mean()),
                                "same_top_segment_share": float(capdf.same_top_segment.mean()),
                                "max_abs_diff": float(capdf.max_abs_diff.max())},
               "lime_r2_mean": {f: float(np.mean(v)) for f, v in lime_r2.items()},
               "names": names, "top1_agreement": top1.tolist(), "top3_overlap": top3.tolist(),
               "speech_pause": sp, "base_speech_share_real_segments": base_speech,
               "base_speech_share_correct": base_speech_c, "base_speech_share_wrong": base_speech_w,
               "n_correct": int(correct.sum()), "resnet_on_these_clips": met,
               "top1_position": top1_position(scores, s100)}
    (OUT / "resnet_explain_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics, styles, real, energy)
    print(f"wrote {OUT}")


def write_report(m: dict, styles, real, energy) -> None:
    met = m["resnet_on_these_clips"]
    names = m["names"]
    L = [f"# Explaining the ResNet-18 (fold 1, seed 1, pretrained): Grad-CAM, IG and LIME on 100 clips", "",
         "## Setup", "",
         "- **Model:** `models/fold1_seed1_pretrained/best.pt`, float32, eval mode. Its float32 predictions match "
         f"the saved (mixed-precision) test run for {m['agree_with_saved_amp_predictions']:.0%} of the clips.",
         "- **Clips:** the same 100 fold-1 test clips (ex02) as the white-box check (`clips_fold1.csv`).",
         "- **Explained quantity:** the logit of the ResNet's predicted class.",
         "- **Same explainer code as the white-box check** (`src/explain/`). IG: 50 steps, baseline = every real "
         "segment filled. LIME: 500 samples, the real segments are the features, off = filled. Fills: " +
         ", ".join(f"{k.replace('_', ' ')} {v:.2f} dB" for k, v in m["fills_db"].items()) + " (fold-1 train clips).",
         f"- **Grad-CAM:** layer4, no fill. Its map is {m['gradcam_map_shape'][0]} x {m['gradcam_map_shape'][1]} "
         "(mel x time), upsampled bilinearly to 128 x 400 and summed per segment, so its time resolution "
         "(about 0.31 s) is coarser than a segment. Grad-CAM keeps only positive evidence (ReLU). "
         f"Clips where Grad-CAM is zero on every real segment: {m['gradcam_all_zero_clips']}.",
         "- **Ranking:** real segments only, by signed score (highest = most support for the prediction).",
         f"- **Energy label:** a real segment is *speech* if its `mean_db` (segments.csv, mean over bands and its "
         f"real frames) is at least {m['energy_threshold_db']:.2f} dB, else *pause*. The threshold is Otsu's "
         "threshold on the real segments of the fold-1 training clips (their histogram has a low pause/room hump "
         "around -85 to -60 dB and a speech peak around -45 to -35 dB).", ""]

    L += ["## Checks", "", "### IG completeness (sum of attributions vs logit(x) - logit(fill)), per clip in "
          "`resnet_ig_completeness.csv`", "",
          "| fill | mean abs logit(x) - logit(fill) | mean abs gap | max abs gap | median relative gap | max relative gap |",
          "|---|---|---|---|---|---|"]
    for f, c in m["ig_completeness"].items():
        L.append(f"| {f.replace('_', ' ')} | {c['abs_logit_change_mean']:.2f} | {c['abs_gap_mean']:.3f} | "
                 f"{c['abs_gap_max']:.3f} | {c['rel_gap_median']:.1%} | {c['rel_gap_max']:.1%} |")
    cv = m["ig_vs_captum"]
    L += ["", f"### Our IG vs Captum `IntegratedGradients` (riemann_middle, 50 steps), 10 clips x 3 fills", "",
          f"Kendall's tau of the 16 segment scores (real segments): mean {cv['kendall_tau_mean']:.4f}, min "
          f"{cv['kendall_tau_min']:.4f}; same top segment in {cv['same_top_segment_share']:.0%} of cases; largest "
          f"absolute difference of a segment score {cv['max_abs_diff']:.1e}. Captum is used only for this check "
          "(`resnet_ig_vs_captum.csv`).", "",
          "LIME surrogate fit (weighted R2, mean over clips): " +
          ", ".join(f"{k.replace('_', ' ')} {v:.3f}" for k, v in m["lime_r2_mean"].items()) +
          ". Lower than on LRfS (0.9997) because the ResNet is not linear in the segments.", ""]

    L += ["## Agreement between methods", "",
          "Share of the 100 clips where the two explanations have the **same top segment**:", "",
          "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for i, nm in enumerate(names):
        L.append(f"| {nm} | " + " | ".join(f"{v:.0%}" for v in m["top1_agreement"][i]) + " |")
    L += ["", "Mean **top-3 overlap** (share of shared segments among the top 3):", "",
          "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for i, nm in enumerate(names):
        L.append(f"| {nm} | " + " | ".join(f"{v:.2f}" for v in m["top3_overlap"][i]) + " |")
    L.append("")

    L += ["## Speech or pause in the top-3 segments", "",
          f"Base rate: {m['base_speech_share_real_segments']:.0%} of the real segments of these clips are speech "
          f"({m['base_speech_share_correct']:.0%} in the {m['n_correct']} correctly classified clips, "
          f"{m['base_speech_share_wrong']:.0%} in the {100 - m['n_correct']} wrong ones). A share above the base "
          "rate means the method favours speech; below, pauses.", "",
          "| explanation | top-3 segments that are speech | in correct clips | in wrong clips | top-1 segment is speech |",
          "|---|---|---|---|---|"]
    for r in m["speech_pause"]:
        L.append(f"| {r['explanation']} | {r['speech_share_all']:.0%} | {r['speech_share_correct']:.0%} | "
                 f"{r['speech_share_wrong']:.0%} | {r['top1_speech_share']:.0%} |")
    L.append("")

    L += ["## Where is the top-1 segment?", "",
          "For each explanation, the position of its top-1 real segment: the **first** real segment, the **last** "
          "real segment (the one next to the padding, or the window end for clips that fill all 4 s), or in "
          "**between**, and whether it is speech or pause. The last row is the expected share for a segment drawn "
          "at random from each clip's real segments. A large *last, pause* share would point at the clip end, "
          "i.e. a length cue. Note that the last real segment is often only partly real (its remaining frames are "
          "padding), which also lowers its `mean_db`.", ""]
    tp = m["top1_position"]
    cell_names = [f"{p if p != 'middle' else 'between'}, {e}" for p, e in CELLS]
    row_names = names + ["random real segment"]
    keys = [f"{a}/{b}" for a, b in EXPL] + ["random_segment"]
    for g, title in (("all", "All 100 clips"), ("correct", "Correct predictions"), ("wrong", "Wrong predictions")):
        L += [f"**{title}** (n = {tp[keys[0]][g]['n']})", "",
              "| explanation | " + " | ".join(cell_names) + " |", "|---|" + "---|" * len(cell_names)]
        for nm, k in zip(row_names, keys):
            L.append(f"| {nm} | " + " | ".join(f"{tp[k][g][f'{p}_{e}']:.0%}" for p, e in CELLS) + " |")
        L.append("")

    L += ["## Caveat: Grad-CAM's time resolution", "",
          f"Grad-CAM is computed on layer4, whose map has {m['gradcam_map_shape'][1]} time cells for the 400 frames "
          f"(about {400 / m['gradcam_map_shape'][1]:.1f} frames = {4 / m['gradcam_map_shape'][1]:.2f} s per cell), "
          "while a segment is 25 frames (0.25 s). The 13 cells are upsampled bilinearly to 400 frames and summed per "
          "segment, so each segment score mixes one or two cells and neighbouring segments share cells: Grad-CAM "
          "cannot separate adjacent segments well, its scores for neighbours are correlated, and its top-1 segment is "
          "partly decided by where a cell boundary falls. Each layer4 cell also sees a wide receptive field (more "
          "than a segment). Low top-1 agreement between Grad-CAM and the segment-level methods (IG, LIME) is "
          "therefore partly a resolution effect, not only a disagreement about evidence.", ""]

    L += ["## ResNet-18 on these 100 clips (CLAUDE.md rule)", "",
          f"Accuracy {met['accuracy']:.1%}, macro-F1 {met['macro_f1']:.3f} (whole test actor: 68.6% / 0.670, "
          "`reports/training_all/report.md`). Rows = true style, columns = predicted style; each box the share of "
          "its row and the clip count.", "", "![ResNet confusion on the 100 clips](confusion_resnet_100clips.png)", "",
          "| style | clips | precision | recall | F1 |", "|---|---|---|---|---|"]
    pc = met["per_class"]
    for s in styles:
        L.append(f"| {s} | {pc[s]['support']} | {pc[s]['precision']:.2f} | {pc[s]['recall']:.2f} | {pc[s]['f1']:.2f} |")
    L.append(f"| **macro average** | {sum(pc[s]['support'] for s in styles)} | "
             f"**{np.mean([pc[s]['precision'] for s in styles]):.2f}** | "
             f"**{np.mean([pc[s]['recall'] for s in styles]):.2f}** | **{met['macro_f1']:.2f}** |")
    L += ["", "## Files (for the deletion curves)", "",
          "- `resnet_segment_scores.csv`: one row per clip x method x fill (Grad-CAM: fill `none`) with true and "
          "predicted style, `correct`, `n_real` and the 16 scores s00..s15. IG and LIME are 0 on padding "
          "segments; Grad-CAM keeps its raw value there, so rank real segments only.",
          "- `segments_100clips.csv`: per clip and segment, `is_real_audio`, `mean_db` and the energy label.",
          "- `resnet_ig_completeness.csv`, `resnet_ig_vs_captum.csv`, `resnet_explain_metrics.json`.", ""]
    (OUT / "resnet_explain.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

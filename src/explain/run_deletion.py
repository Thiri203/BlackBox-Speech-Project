"""Deletion curves for the ResNet explanations (fold 1, seed 1) on the 100 clips.

    python -m src.explain.run_deletion

Reads the saved segment scores (reports/explain_fold1/resnet_segment_scores.csv); no explainer is rerun.
For each clip, ordering and deletion fill, the top-k real segments (k = 0..K, K = min(8, real segments))
are replaced by the per-band fill and the ResNet's predicted class (the class that was explained) is
scored again. Orderings: the explanation's scores (highest first, real segments only), and 10 random
orderings of the real segments (the same 10 for every fill, seeded per clip).
Fills: IG and LIME are deleted with the fill they were computed with; Grad-CAM (no fill) with all 3.

Measured: drop of the predicted class's logit (logit(x) - logit(x with top-k deleted)) and of its
probability. Area per clip = mean drop over k = 1..K (so clips with fewer than 8 real segments are not
extrapolated; method and random use the same K). Higher area = the model's output falls faster when
the segments the method ranks first are deleted.

Writes reports/explain_fold1/: deletion_curves.csv (every clip x ordering x fill x k),
deletion_metrics.json, deletion.md, deletion_curves.png.
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

from src.data.torch_dataset import SpectrogramDataset  # noqa: E402
from src.eval_checkpoints import PNG_META  # noqa: E402
from src.explain.run_resnet import FILL_ORDER, FOLD, OUT, RUN  # noqa: E402
from src.explain.run_whitebox import FILLS, SEED  # noqa: E402
from src.explain.segments import N_SEG, fill_batch, make_predict_fn  # noqa: E402
from src.models import lrfs  # noqa: E402
from src.models.resnet import build_resnet18  # noqa: E402
from src.utils import REPO_ROOT, load_config  # noqa: E402

K_MAX, N_RANDOM, N_BOOT = 8, 10, 2000
# Method x deletion fill. Grad-CAM has no fill of its own, so it is deleted with each of the three.
CURVES = [("GradCAM", "none", f) for f in FILL_ORDER] + [("IG", f, f) for f in FILL_ORDER] + \
         [("LIME", f, f) for f in FILL_ORDER]
COLOURS = {"Grad-CAM": "#2a78d6", "IG": "#eb6834", "LIME": "#1baf7a", "random": "#8a8984"}
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def boot_ci(v: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    m = v[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main() -> None:
    tcfg = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    pcfg = load_config(REPO_ROOT / tcfg["preprocess_config"])
    splits = pcfg["paths"]["splits_dir"]
    styles = pcfg["labels"]["styles"]
    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    norm = json.loads((splits / f"norm_stats_fold_{FOLD}.json").read_text(encoding="utf-8"))
    fv = json.loads((splits / f"fill_values_fold_{FOLD}.json").read_text(encoding="utf-8"))
    g_mean, g_std = float(norm["mean"]), float(norm["std"])
    band = {k: (np.asarray(fv[key], dtype=np.float64) if key else
                np.full(len(fv["mean_db_per_mel"]), float(fv["padding_value_db"])))
            for k, (key, _) in FILLS.items()}

    model = build_resnet18(tcfg["model"]["n_classes"], tcfg["model"]["in_channels"], pretrained=False)
    model.load_state_dict(torch.load(REPO_ROOT / "models" / RUN / "best.pt", map_location="cpu", weights_only=True))
    predict = make_predict_fn(model.to(device).eval(), g_mean, g_std, device)

    clips = pd.read_csv(OUT / f"clips_fold{FOLD}.csv")
    scores = pd.read_csv(OUT / "resnet_segment_scores.csv")
    segs = pd.read_csv(OUT / "segments_100clips.csv")
    energy = segs.pivot(index="clip_id", columns="segment", values="energy").loc[clips["clip_id"]].to_numpy()
    ds = SpectrogramDataset(pcfg, FOLD, "test")
    rows = pd.Series(np.arange(len(ds.clip_ids)), index=ds.clip_ids).loc[clips["clip_id"]].to_numpy()
    X = ds.x[rows].astype(np.float32)
    real = lrfs.real_segment_mask(ds.n_valid[rows])
    assert ((energy != "padding") == real).all()
    cols = [f"s{j:02d}" for j in range(N_SEG)]
    S = {(m, f): scores[(scores.method == m) & (scores.fill == f)].set_index("clip_id").loc[clips["clip_id"]]
         for m, f, _ in CURVES}
    info = S[("IG", "room_noise")][["y_true", "y_pred", "correct", "n_real"]]

    recs = []
    for i, cid in enumerate(clips["clip_id"]):
        c = int(info["y_pred"].iloc[i])
        idx = np.flatnonzero(real[i])
        K = min(K_MAX, len(idx))
        rng = np.random.default_rng([SEED, i, 777])
        rand_orders = [rng.permutation(idx) for _ in range(N_RANDOM)]
        for fname in FILL_ORDER:
            orders = [(m, sf, idx[np.argsort(-S[(m, sf)].iloc[i][cols].to_numpy(float)[idx], kind="stable")])
                      for m, sf, df in CURVES if df == fname]
            orders += [("random", f"r{r}", o) for r, o in enumerate(rand_orders)]
            keep = []
            for _, _, o in orders:
                for k in range(K + 1):
                    kp = np.ones(N_SEG, dtype=bool)
                    kp[o[:k]] = False
                    keep.append(kp)
            logit = predict(fill_batch(X[i], np.asarray(keep), band[fname]))
            prob = softmax(logit.astype(np.float64))
            assert int(logit[0].argmax()) == c, f"{cid}: prediction changed"
            p = 0
            for m, sf, o in orders:
                l0, p0 = logit[p, c], prob[p, c]
                for k in range(K + 1):
                    recs.append({"clip_id": cid, "correct": bool(info["correct"].iloc[i]), "method": m,
                                 "score_fill": sf, "deletion_fill": fname, "k": k, "K": K,
                                 "deleted_segment": int(o[k - 1]) if k else -1,
                                 "deleted_energy": energy[i, o[k - 1]] if k else "",
                                 "logit_drop": float(l0 - logit[p + k, c]), "prob_drop": float(p0 - prob[p + k, c]),
                                 "still_predicted": bool(logit[p + k].argmax() == c)})
                p += K + 1
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/100 clips", flush=True)

    cur = pd.DataFrame(recs)
    cur.to_csv(OUT / "deletion_curves.csv", index=False, float_format="%.6g")

    # ---- areas per clip ----
    pos = cur[cur.k > 0]
    area = pos.groupby(["clip_id", "correct", "method", "score_fill", "deletion_fill"], as_index=False)[
        ["logit_drop", "prob_drop"]].mean()
    rnd = area[area.method == "random"]
    rnd_clip = rnd.groupby(["clip_id", "deletion_fill"])[["logit_drop", "prob_drop"]].mean()
    brng = np.random.default_rng(SEED)
    summary = []
    for m, sf, df in CURVES + [("random", None, f) for f in FILL_ORDER]:
        for grp in ("all", "correct", "wrong"):
            a = area[(area.method == m) & (area.deletion_fill == df)]
            if sf is not None:
                a = a[a.score_fill == sf]
            if grp != "all":
                a = a[a.correct == (grp == "correct")]
            row = {"method": m, "score_fill": sf or "", "deletion_fill": df, "group": grp,
                   "n_clips": int(a.clip_id.nunique())}
            for q in ("logit_drop", "prob_drop"):
                if m == "random":
                    per_order = a.groupby("score_fill")[q].mean()          # 10 values, one per random ordering
                    per_clip = a.groupby("clip_id")[q].mean()
                    row[f"{q}_area"] = float(per_clip.mean())
                    row[f"{q}_sd_over_orderings"] = float(per_order.std(ddof=1))
                    row[f"{q}_min_ordering"] = float(per_order.min())
                    row[f"{q}_max_ordering"] = float(per_order.max())
                else:
                    v = a.set_index("clip_id")[q]
                    r = rnd_clip.xs(df, level="deletion_fill")[q].loc[v.index]
                    d = (v - r).to_numpy()
                    lo, hi = boot_ci(d, brng)
                    row[f"{q}_area"] = float(v.mean())
                    row[f"{q}_minus_random"] = float(d.mean())
                    row[f"{q}_minus_random_ci95"] = [lo, hi]
                    row[f"{q}_beats_random_share"] = float((d > 0).mean())
            summary.append(row)

    # ---- what is deleted first ----
    first = []
    for m, sf, df in CURVES + [("random", None, f) for f in FILL_ORDER]:
        d = pos[(pos.method == m) & (pos.deletion_fill == df)]
        if sf is not None:
            d = d[d.score_fill == sf]
        for grp in ("all", "correct", "wrong"):
            g = d if grp == "all" else d[d.correct == (grp == "correct")]
            first.append({"method": m, "score_fill": sf or "", "deletion_fill": df, "group": grp,
                          "speech_share_k1": float((g[g.k == 1].deleted_energy == "speech").mean()),
                          "speech_share_k1to3": float((g[g.k <= 3].deleted_energy == "speech").mean())})

    # ---- mean curves for the figure / table ----
    mc = cur.copy()
    mc["label"] = np.where(mc.method == "random", "random", mc.method.replace({"GradCAM": "Grad-CAM"}))
    mean_curve = (mc.groupby(["label", "deletion_fill", "k"])[["logit_drop", "prob_drop"]].mean()
                  .reset_index())
    n_at_k = cur[(cur.method == "IG") & (cur.deletion_fill == "room_noise")].groupby("k").clip_id.nunique()
    rnd_order_curve = (cur[cur.method == "random"].groupby(["deletion_fill", "score_fill", "k"])["logit_drop"]
                       .mean().reset_index())

    plot(mean_curve, rnd_order_curve)
    met = json.loads((OUT / "resnet_explain_metrics.json").read_text(encoding="utf-8"))["resnet_on_these_clips"]
    metrics = {"run": RUN, "k_max": K_MAX, "n_random_orderings": N_RANDOM, "n_clips": len(clips),
               "clips_with_fewer_than_8_real": int((real.sum(1) < K_MAX).sum()),
               "clips_at_k": {int(k): int(v) for k, v in n_at_k.items()},
               "fills_db": {k: float(v.mean()) for k, v in band.items()},
               "summary": summary, "deleted_first": first,
               "mean_curve": mean_curve.to_dict(orient="records"), "resnet_on_these_clips": met}
    (OUT / "deletion_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics, styles)
    print(f"wrote {OUT / 'deletion.md'}")


def plot(mean_curve: pd.DataFrame, rnd_order_curve: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
    for ax, f in zip(axes, FILL_ORDER):
        ax.set_facecolor(SURFACE)
        ro = rnd_order_curve[rnd_order_curve.deletion_fill == f].pivot(index="k", columns="score_fill",
                                                                      values="logit_drop")
        ax.fill_between(ro.index, ro.min(1), ro.max(1), color=COLOURS["random"], alpha=0.18, linewidth=0,
                        label="random: range of 10 orderings")
        ends = {}
        for lab in ("random", "Grad-CAM", "IG", "LIME"):
            d = mean_curve[(mean_curve.label == lab) & (mean_curve.deletion_fill == f)]
            ax.plot(d.k, d.logit_drop, color=COLOURS[lab], linewidth=2, marker="o", markersize=4,
                    linestyle="--" if lab == "random" else "-", label=lab if lab != "random" else "random: mean")
            ends[lab] = float(d.logit_drop.iloc[-1])
        # end labels, pushed apart so they never overlap (min gap 0.32 logit units)
        placed = []
        for lab, yv in sorted(ends.items(), key=lambda t: t[1]):
            yv = max(yv, placed[-1] + 0.32) if placed else yv
            placed.append(yv)
            ax.text(K_MAX + 0.15, yv, lab, va="center", fontsize=8, color=INK2)
        ax.set_title(f"deletion fill: {f.replace('_', ' ')}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("segments deleted (k)", color=INK2)
        ax.set_xticks(range(0, K_MAX + 1))
        ax.set_xlim(0, K_MAX + 1.2)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b9b8b2")
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].set_ylabel("drop of the predicted class's logit", color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=5, frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Deletion curves, ResNet-18 fold 1 seed 1, 100 clips (mean over clips)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "deletion_curves.png", dpi=130, metadata=PNG_META, facecolor=SURFACE)
    plt.close(fig)


def write_report(m: dict, styles) -> None:
    name = {"GradCAM": "Grad-CAM", "IG": "IG", "LIME": "LIME", "random": "random (10 orderings)"}
    L = ["# Deletion curves: ResNet-18 (fold 1, seed 1, pretrained), 100 clips", "",
         "## What the curve measures", "",
         "We delete the k real segments an explanation ranks highest (k = 1 to 8, never more than the clip's real "
         "segments) by writing a fill into all 128 x 25 values of each, and track how much the logit (and the "
         "probability) of the predicted class falls. **The curve measures how much the model relies on that "
         "segment being present in its normal form, not what the segment contains:** a deleted segment also "
         "becomes a pause (silence, room noise or an average-sound block), so a drop can come from the content "
         "going away *or* from the new pause itself (for example a changed amount of speech in the window). A "
         "faithful ranking should make the logit fall faster than random orderings that use the same fill.", "",
         "**Caveat when comparing methods:** LIME was fitted on exactly this kind of perturbation (real segments "
         "switched off with the same fill, effect on the same logit), so the deletion test is close to what LIME "
         "optimises; IG (gradients along a path to the filled clip) and Grad-CAM (layer4 activations, no fill) "
         "are not. A higher LIME area is expected for that reason and is not by itself proof that LIME is more "
         "faithful in general. Grad-CAM uses one ranking for all three fills.", "",
         "## Setup", "",
         f"- Clips: the same 100 fold-1 test clips (`clips_fold1.csv`); {m['clips_with_fewer_than_8_real']} have "
         "fewer than 8 real segments, so their curves stop earlier (clips per k: " +
         ", ".join(f"k={k}: {v}" for k, v in m["clips_at_k"].items() if k > 0) + ").",
         "- Orderings: the saved segment scores (`resnet_segment_scores.csv`), highest first, real segments only. "
         "Random control: 10 random orderings of each clip's real segments, the same 10 for every fill.",
         "- Fills: IG and LIME are deleted with the fill they were computed with; Grad-CAM has no fill and is "
         "deleted with all 3. Fill values (fold-1 train clips): " +
         ", ".join(f"{k.replace('_', ' ')} {v:.2f} dB" for k, v in m["fills_db"].items()) + ".",
         "- Area = mean drop over k = 1..K per clip (K = min(8, real segments)), then averaged over clips. "
         "*Minus random* = the clip's area minus the mean area of its 10 random orderings; 95% interval by "
         "bootstrap over clips. *Beats random* = share of clips where that difference is positive.", "",
         "![Deletion curves](deletion_curves.png)", ""]

    for grp, title in (("all", "All 100 clips"), ("correct", "Correct predictions"), ("wrong", "Wrong predictions")):
        rows = [r for r in m["summary"] if r["group"] == grp]
        L += [f"## Area under the deletion curve: {title} (n = {rows[0]['n_clips']})", "",
              "| deletion fill | method (scores) | logit drop: area | minus random [95% CI] | beats random | "
              "probability drop: area | minus random [95% CI] |", "|---|---|---|---|---|---|---|"]
        for f in FILL_ORDER:
            for r in [r for r in rows if r["deletion_fill"] == f]:
                if r["method"] == "random":
                    L.append(f"| {f.replace('_', ' ')} | random (10 orderings) | {r['logit_drop_area']:.2f} "
                             f"(sd over orderings {r['logit_drop_sd_over_orderings']:.2f}; "
                             f"{r['logit_drop_min_ordering']:.2f} to {r['logit_drop_max_ordering']:.2f}) | | | "
                             f"{r['prob_drop_area']:.3f} (sd {r['prob_drop_sd_over_orderings']:.3f}; "
                             f"{r['prob_drop_min_ordering']:.3f} to {r['prob_drop_max_ordering']:.3f}) | |")
                else:
                    ci, cp = r["logit_drop_minus_random_ci95"], r["prob_drop_minus_random_ci95"]
                    sf = "" if r["score_fill"] in ("none", r["deletion_fill"]) else f" ({r['score_fill']})"
                    L.append(f"| {f.replace('_', ' ')} | {name[r['method']]}{sf} | {r['logit_drop_area']:.2f} | "
                             f"{r['logit_drop_minus_random']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | "
                             f"{r['logit_drop_beats_random_share']:.0%} | {r['prob_drop_area']:.3f} | "
                             f"{r['prob_drop_minus_random']:+.3f} [{cp[0]:+.3f}, {cp[1]:+.3f}] |")
        L.append("")

    L += ["## Mean curve (logit drop by k, all clips)", "",
          "| deletion fill | method | " + " | ".join(f"k={k}" for k in range(1, K_MAX + 1)) + " |",
          "|---|---|" + "---|" * K_MAX]
    mc = pd.DataFrame(m["mean_curve"])
    for f in FILL_ORDER:
        for lab in ("Grad-CAM", "IG", "LIME", "random"):
            d = mc[(mc.label == lab) & (mc.deletion_fill == f)].set_index("k")
            L.append(f"| {f.replace('_', ' ')} | {lab} | " +
                     " | ".join(f"{d.loc[k, 'logit_drop']:.2f}" for k in range(1, K_MAX + 1)) + " |")
    L += ["", "From k = 6 on, the mean covers fewer clips (see Setup).", ""]

    L += ["## Are the segments deleted first speech or pause?", "",
          "Share of deleted segments that are speech (energy label: `mean_db` >= -55.58 dB). For random orderings "
          "this is the base rate of the clips' real segments.", "",
          "| deletion fill | method | first deleted (k=1): all | correct | wrong | first 3 deleted: all | correct | wrong |",
          "|---|---|---|---|---|---|---|---|"]
    fr = pd.DataFrame(m["deleted_first"])
    for f in FILL_ORDER:
        for meth in ("GradCAM", "IG", "LIME", "random"):
            g = fr[(fr.method == meth) & (fr.deletion_fill == f)].set_index("group")
            L.append(f"| {f.replace('_', ' ')} | {name[meth]} | " +
                     " | ".join(f"{g.loc[x, 'speech_share_k1']:.0%}" for x in ("all", "correct", "wrong")) + " | " +
                     " | ".join(f"{g.loc[x, 'speech_share_k1to3']:.0%}" for x in ("all", "correct", "wrong")) + " |")
    L.append("")

    met = m["resnet_on_these_clips"]
    pc = met["per_class"]
    L += ["## ResNet-18 on these 100 clips (CLAUDE.md rule)", "",
          f"Accuracy {met['accuracy']:.1%}, macro-F1 {met['macro_f1']:.3f}. Rows = true style, columns = predicted "
          "style; each box the share of its row and the clip count.", "",
          "![ResNet confusion on the 100 clips](confusion_resnet_100clips.png)", "",
          "| style | clips | precision | recall | F1 |", "|---|---|---|---|---|"]
    for s in styles:
        L.append(f"| {s} | {pc[s]['support']} | {pc[s]['precision']:.2f} | {pc[s]['recall']:.2f} | {pc[s]['f1']:.2f} |")
    L.append(f"| **macro average** | {sum(pc[s]['support'] for s in styles)} | "
             f"**{np.mean([pc[s]['precision'] for s in styles]):.2f}** | "
             f"**{np.mean([pc[s]['recall'] for s in styles]):.2f}** | **{met['macro_f1']:.2f}** |")
    L += ["", "Files: `deletion_curves.csv` (every clip x ordering x fill x k, with the deleted segment, its energy "
          "label, logit and probability drop, and whether the prediction still holds), `deletion_metrics.json`.", ""]
    (OUT / "deletion.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

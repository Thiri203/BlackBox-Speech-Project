"""White-box check: IG and LIME on LRfS vs LRfS's exact segment contributions (fold 1).

    python -m src.explain.run_whitebox

- Clips: 100 fold-1 test clips (ex02), stratified by true style, seed 42, chosen without looking at
  any model's prediction; saved to reports/explain_fold1/clips_fold1.csv and reused for the ResNet.
- Target: LRfS's predicted class's logit (the quantity the exact contributions describe).
- For each fill (room noise = main, average, silence = -100 dB), the truth uses that same fill:
  phi_j = V_cj (x_j - f) on real segments. IG (50 steps, baseline = all real segments filled) and
  LIME (500 samples, real segments as features, off = filled) run through the generic explainer code
  in src/explain/, on LRfSTorch, i.e. on the standardised 128 x 400 input exactly like the ResNet.
- Scores over REAL segments only: Kendall's tau-b (signed values; also on |values|) and top-3
  overlap (|top 3 by explainer ∩ top 3 by truth| / 3, by signed value; also by |value|).

Writes reports/explain_fold1/: whitebox_lrfs.md, whitebox_lrfs_metrics.json,
whitebox_lrfs_per_clip.csv, whitebox_lrfs_attributions.csv, confusion_lrfs_100clips.png.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.data.torch_dataset import SpectrogramDataset  # noqa: E402
from src.eval_checkpoints import PNG_META, plot_confusion  # noqa: E402
from src.evaluate import classification_metrics  # noqa: E402
from src.explain.ig import integrated_gradients  # noqa: E402
from src.explain.lime_segments import lime_segments  # noqa: E402
from src.explain.segments import make_predict_fn  # noqa: E402
from src.models import lrfs  # noqa: E402
from src.utils import REPO_ROOT, load_config  # noqa: E402

FOLD, N_CLIPS, SEED = 1, 100, 42
IG_STEPS, LIME_SAMPLES = 50, 500
FILLS = {"room_noise": ("p10_db_per_mel", "main"), "average": ("mean_db_per_mel", "second"),
         "silence": (None, "comparison (= padding value -100 dB)")}
OUT = REPO_ROOT / "reports" / f"explain_fold{FOLD}"


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int = 3) -> float:
    k = min(k, len(a))
    ta = set(np.argsort(-a, kind="stable")[:k])
    tb = set(np.argsort(-b, kind="stable")[:k])
    return len(ta & tb) / k


def score(expl: np.ndarray, truth: np.ndarray) -> dict:
    t_s = kendalltau(expl, truth).statistic
    t_a = kendalltau(np.abs(expl), np.abs(truth)).statistic
    return {"tau": float(t_s), "tau_abs": float(t_a), "top3": top_k_overlap(expl, truth),
            "top3_abs": top_k_overlap(np.abs(expl), np.abs(truth))}


def select_clips(ds) -> pd.DataFrame:
    items = ds.items[["clip_id", "actor", "style", "label"]].copy()
    items["row"] = np.arange(len(items))
    sel, _ = train_test_split(items, train_size=N_CLIPS, stratify=items["label"], random_state=SEED)
    return sel.sort_values("clip_id").reset_index(drop=True)


def main() -> None:
    tcfg = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    pcfg = load_config(REPO_ROOT / tcfg["preprocess_config"])
    styles = pcfg["labels"]["styles"]
    splits = pcfg["paths"]["splits_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    OUT.mkdir(parents=True, exist_ok=True)

    norm = json.loads((splits / f"norm_stats_fold_{FOLD}.json").read_text(encoding="utf-8"))
    fv = json.loads((splits / f"fill_values_fold_{FOLD}.json").read_text(encoding="utf-8"))
    band = {k: (np.asarray(fv[key], dtype=np.float64) if key else
                np.full(len(fv["mean_db_per_mel"]), float(fv["padding_value_db"])))
            for k, (key, _) in FILLS.items()}
    g_mean, g_std = float(norm["mean"]), float(norm["std"])

    p = np.load(REPO_ROOT / "models" / f"lrfs_fold{FOLD}" / "lrfs.npz")
    V, b0 = p["V"], p["b0"]
    model = lrfs.LRfSTorch(V, b0, g_mean, g_std).to(device).eval()
    predict = make_predict_fn(model, g_mean, g_std, device)

    ds = SpectrogramDataset(pcfg, FOLD, "test")
    clips_path = OUT / f"clips_fold{FOLD}.csv"
    sel = select_clips(ds)
    if clips_path.exists():   # the list is fixed once; later runs (and the ResNet) must reuse it
        old = pd.read_csv(clips_path)
        assert old["clip_id"].tolist() == sel["clip_id"].tolist(), "clip selection changed"
    else:
        sel[["clip_id", "actor", "style", "label"]].to_csv(clips_path, index=False)

    X = ds.x[sel["row"].to_numpy()].astype(np.float64)               # [100, 128, 400] dB
    real = lrfs.real_segment_mask(ds.n_valid[sel["row"].to_numpy()])
    x16 = lrfs.segment_features(X)
    logit_np = lrfs.logits_from_features(x16, V, b0)
    pred = logit_np.argmax(1)
    assert (predict(X.astype(np.float32)).argmax(1) == pred).all(), "torch LRfS disagrees with the linear form"

    # Truth must agree with what train_lrfs saved for the same clips.
    saved = pd.read_csv(REPO_ROOT / "models" / f"lrfs_fold{FOLD}" / "contributions_test.csv")
    phi_cols = [f"phi_s{j:02d}" for j in range(16)]

    per_clip, attr_rows = [], []
    ig_gap = {}
    for fname, f_arr in band.items():
        f = float(f_arr.mean())
        truth_all = lrfs.contributions(x16, real, V, pred, f)
        sv = saved[saved["fill"] == fname].set_index("clip_id").loc[sel["clip_id"], phi_cols].to_numpy()
        assert np.allclose(sv, truth_all, rtol=1e-5, atol=1e-6), f"{fname}: truth differs from saved contributions"
        gaps = []
        for i, cid in enumerate(sel["clip_id"]):
            r = real[i]
            ig = integrated_gradients(model, X[i].astype(np.float32), r, f_arr, int(pred[i]), g_mean, g_std,
                                      device, steps=IG_STEPS)
            rng = np.random.default_rng([SEED, i, list(FILLS).index(fname)])
            li = lime_segments(predict, X[i].astype(np.float32), r, f_arr, int(pred[i]), rng,
                               n_samples=LIME_SAMPLES)
            gaps.append(abs(ig["completeness_gap"]))
            t = truth_all[i, r]
            for meth, a, extra in (("IG", ig["segments"], {"ig_completeness_gap": ig["completeness_gap"]}),
                                   ("LIME", li["segments"], {"lime_r2": li["r2"]})):
                s = score(a[r], t)
                per_clip.append({"clip_id": cid, "style": sel.loc[i, "style"], "pred": styles[pred[i]],
                                 "fill": fname, "method": meth, "n_real": int(r.sum()), **s, **extra})
                attr_rows.append({"clip_id": cid, "fill": fname, "method": meth,
                                  **{f"s{j:02d}": a[j] for j in range(16)}})
            attr_rows.append({"clip_id": cid, "fill": fname, "method": "truth",
                              **{f"s{j:02d}": truth_all[i, j] for j in range(16)}})
        ig_gap[fname] = float(np.max(gaps))
        print(f"fill {fname}: done ({len(sel)} clips), max IG completeness gap {ig_gap[fname]:.2e}", flush=True)

    pc = pd.DataFrame(per_clip)
    pc.to_csv(OUT / "whitebox_lrfs_per_clip.csv", index=False, float_format="%.6g")
    pd.DataFrame(attr_rows).to_csv(OUT / "whitebox_lrfs_attributions.csv", index=False, float_format="%.6g")

    summary = {}
    for (fname, meth), g in pc.groupby(["fill", "method"], sort=False):
        summary[f"{fname}/{meth}"] = {
            "fill": fname, "method": meth, "n_clips": int(len(g)),
            "tau_mean": float(g["tau"].mean()), "tau_median": float(g["tau"].median()),
            "tau_min": float(g["tau"].min()), "tau_nan": int(g["tau"].isna().sum()),
            "tau_abs_mean": float(g["tau_abs"].mean()),
            "top3_mean": float(g["top3"].mean()), "top3_abs_mean": float(g["top3_abs"].mean()),
            "top3_all_three_share": float((g["top3"] == 1).mean()),
            **({"lime_r2_mean": float(g["lime_r2"].mean())} if meth == "LIME" else
               {"ig_max_abs_completeness_gap": float(g["ig_completeness_gap"].abs().max())})}

    met = classification_metrics(sel["label"].to_numpy(), pred, styles)
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    plot_confusion(ax, np.asarray(met["confusion_counts"]), styles,
                   f"LRfS on the 100 white-box clips: accuracy {met['accuracy']:.0%}, macro-F1 {met['macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_lrfs_100clips.png", dpi=130, metadata=PNG_META)
    plt.close(fig)

    metrics = {"fold": FOLD, "n_clips": N_CLIPS, "seed": SEED, "ig_steps": IG_STEPS, "lime_samples": LIME_SAMPLES,
               "fills_db": {k: float(v.mean()) for k, v in band.items()}, "target": "LRfS predicted class logit",
               "n_real_segments": {"mean": float(real.sum(1).mean()), "min": int(real.sum(1).min()),
                                   "max": int(real.sum(1).max())},
               "summary": summary, "lrfs_on_these_clips": met}
    (OUT / "whitebox_lrfs_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics, styles, sel)
    print(f"wrote {OUT}")


def write_report(m: dict, styles: list[str], sel: pd.DataFrame) -> None:
    S, met = m["summary"], m["lrfs_on_these_clips"]
    L = [f"# White-box check: IG and LIME on LRfS, fold {m['fold']} (test actor ex02)", "",
         "LRfS (logistic regression on 16 segment means, `reports/lrfs_fold1/report.md`) has exact segment "
         "contributions, so we know the true answer. Integrated Gradients and LIME are run on it with the same "
         "code that will run on the ResNet (`src/explain/`), on the same standardised 128 x 400 input, and "
         "scored against that truth. Grad-CAM needs a convolutional layer and is covered by the sanity check "
         "instead.", "",
         "## Setup", "",
         f"- **Clips:** {m['n_clips']} fold-1 test clips, stratified by true style, seed {m['seed']}, chosen "
         "without looking at any model's prediction (`clips_fold1.csv`; the ResNet explainers reuse this list). "
         f"Real segments per clip: mean {m['n_real_segments']['mean']:.1f}, min {m['n_real_segments']['min']}, "
         f"max {m['n_real_segments']['max']}.",
         "- **Explained quantity:** the logit of LRfS's predicted class, as in the exact contributions.",
         "- **Truth:** phi_j = V_cj (x_j - f) on real segments, with the **same fill** f the explainer uses. "
         "The truth depends on the fill, so every result is reported per fill.",
         f"- **Integrated Gradients:** {m['ig_steps']} steps (midpoint Riemann sum), baseline = the clip with "
         "every real segment replaced by the fill (padding unchanged); pixel attributions summed per segment.",
         f"- **LIME:** {m['lime_samples']} samples; the features are the clip's real segments, a segment "
         "switched off gets the fill; lime_image defaults otherwise (first sample = original clip, uniform "
         "on/off, cosine distance, kernel width 0.25, weighted Ridge alpha 1). Implemented in "
         "`src/explain/lime_segments.py` (the `lime` package is not installed).",
         "- **Scores, over real segments only:** Kendall's tau-b between explainer and truth on signed values "
         "(main) and on absolute values; top-3 overlap = share of the truth's 3 highest segments that are also "
         "the explainer's 3 highest, on signed values (segments that most support the prediction) and on "
         "absolute values.", "",
         "Fills (train clips of fold 1 only, mean over the 128 bands): " +
         ", ".join(f"{k.replace('_', ' ')} {v:.2f} dB" for k, v in m["fills_db"].items()) + ".", "",
         "## Results (100 clips)", "",
         "| fill | method | Kendall tau: mean | median | min | tau on abs values: mean | top-3 overlap: mean | "
         "all 3 found | top-3 overlap, abs: mean | check |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in S.items():
        chk = (f"LIME R2 {s['lime_r2_mean']:.4f}" if s["method"] == "LIME"
               else f"IG completeness gap max {s['ig_max_abs_completeness_gap']:.1e}")
        L.append(f"| {s['fill'].replace('_', ' ')} | {s['method']} | {s['tau_mean']:.3f} | {s['tau_median']:.3f} | "
                 f"{s['tau_min']:.3f} | {s['tau_abs_mean']:.3f} | {s['top3_mean']:.3f} | "
                 f"{s['top3_all_three_share']:.0%} | {s['top3_abs_mean']:.3f} | {chk} |")
    L += ["", "IG completeness gap = sum of IG attributions minus (logit of the clip - logit of the baseline).", ""]
    L += ["## LRfS on these 100 clips (CLAUDE.md rule)", "",
          f"Accuracy {met['accuracy']:.1%}, macro-F1 {met['macro_f1']:.3f} (full test actor: "
          "`reports/lrfs_fold1/report.md`). Rows = true style, columns = predicted style; each box the share of "
          "its row and the clip count.", "", "![LRfS confusion on the 100 clips](confusion_lrfs_100clips.png)", "",
          "| style | clips | precision | recall | F1 |", "|---|---|---|---|---|"]
    pc = met["per_class"]
    for s in styles:
        L.append(f"| {s} | {pc[s]['support']} | {pc[s]['precision']:.2f} | {pc[s]['recall']:.2f} | {pc[s]['f1']:.2f} |")
    L.append(f"| **macro average** | {sum(pc[s]['support'] for s in styles)} | "
             f"**{np.mean([pc[s]['precision'] for s in styles]):.2f}** | "
             f"**{np.mean([pc[s]['recall'] for s in styles]):.2f}** | **{met['macro_f1']:.2f}** |")
    L += ["", "Files: `whitebox_lrfs_per_clip.csv` (scores per clip, fill and method), "
          "`whitebox_lrfs_attributions.csv` (16 segment values per clip, fill and method, plus the truth), "
          "`whitebox_lrfs_metrics.json`.", ""]
    (OUT / "whitebox_lrfs.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

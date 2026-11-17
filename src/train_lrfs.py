"""Train LRfS (logistic regression on 16 segment means) for one fold, test once, write the report.

    python -m src.train_lrfs            # fold 1 (test actor ex02), the explanation fold

1. Features: mean log-mel over all 128 bands x 25 frames of each 250 ms segment (padding at
   -100 dB), from the same cached 128 x 400 spectrograms the ResNet reads (src/models/lrfs.py).
2. StandardScaler on the TRAIN split; C chosen on the VAL split by macro-F1 (ties -> smaller C).
   The test actor is loaded only after C is fixed and is predicted once.
3. Exact contributions of every real segment to the predicted class's logit for 3 fills
   (room noise, average, silence = -100 dB), checked against a real deletion in the 128 x 400 input.
4. Report with the test confusion matrix and per-style precision, recall and F1 (CLAUDE.md).

Writes models/lrfs_fold{k}/ (model, predictions, contributions; refuses to overwrite a finished
run) and reports/lrfs_fold{k}/ (report.md, confusion.png, metrics.json).
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data.torch_dataset import SpectrogramDataset  # noqa: E402
from src.eval_checkpoints import PNG_META, plot_confusion  # noqa: E402
from src.evaluate import classification_metrics, length_baseline  # noqa: E402
from src.models import lrfs  # noqa: E402
from src.utils import REPO_ROOT, load_config  # noqa: E402

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 42
RESNET_SEED = 1


def load_part(pcfg: dict, fold: int, part: str, seg_table: pd.DataFrame):
    ds = SpectrogramDataset(pcfg, fold, part)
    x = lrfs.segment_features(ds.x)
    real = lrfs.real_segment_mask(ds.n_valid)
    # The rule must agree with the segment table's is_real_audio.
    tab = seg_table.loc[ds.clip_ids].to_numpy(dtype=bool)
    assert (tab == real).all(), f"{part}: real-segment mask disagrees with segments.csv"
    return ds, x, real


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    args = ap.parse_args()
    fold = args.fold

    tcfg = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    pcfg = load_config(REPO_ROOT / tcfg["preprocess_config"])
    styles = pcfg["labels"]["styles"]
    splits = pcfg["paths"]["splits_dir"]
    out_model = REPO_ROOT / "models" / f"lrfs_fold{fold}"
    out_rep = REPO_ROOT / "reports" / f"lrfs_fold{fold}"
    if (out_model / "DONE").exists():
        raise SystemExit(f"{out_model} is finished (DONE); delete it on purpose to retrain")
    out_model.mkdir(parents=True, exist_ok=True)
    out_rep.mkdir(parents=True, exist_ok=True)

    seg = pd.read_csv(pcfg["paths"]["processed_dir"] / "segments.csv")
    seg_real = seg.pivot(index="clip_id", columns="segment", values="is_real_audio")
    norm = json.loads((splits / f"norm_stats_fold_{fold}.json").read_text(encoding="utf-8"))
    fv = json.loads((splits / f"fill_values_fold_{fold}.json").read_text(encoding="utf-8"))
    band_fills = {
        "room_noise": np.asarray(fv["p10_db_per_mel"], dtype=np.float64),
        "average": np.asarray(fv["mean_db_per_mel"], dtype=np.float64),
        "silence": np.full(len(fv["mean_db_per_mel"]), float(fv["padding_value_db"])),  # = padding value
    }
    fills = {k: float(v.mean()) for k, v in band_fills.items()}   # scalar fill = mean over bands

    # ---- train and choose C (train + val only) ----
    ds_tr, x_tr, _ = load_part(pcfg, fold, "train", seg_real)
    y_tr = ds_tr.y
    tr_df = pd.read_csv(splits / f"fold_{fold}.csv")
    ds_va, x_va, _ = load_part(pcfg, fold, "val", seg_real)
    y_va = ds_va.y
    train_actors = sorted(tr_df.loc[tr_df["split"] == "train", "actor"].unique())
    assert len(y_tr) == len(x_tr)

    scaler = StandardScaler().fit(x_tr)
    grid = []
    fitted = {}
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=10000, random_state=SEED).fit(scaler.transform(x_tr), y_tr)
        pv = clf.predict(scaler.transform(x_va))
        ptr = clf.predict(scaler.transform(x_tr))
        grid.append({"C": C, "val_macro_f1": float(f1_score(y_va, pv, average="macro")),
                     "val_accuracy": float(accuracy_score(y_va, pv)),
                     "train_accuracy": float(accuracy_score(y_tr, ptr)), "n_iter": int(clf.n_iter_.max())})
        fitted[C] = clf
        print(f"C={C:g}: val macro-F1 {grid[-1]['val_macro_f1']:.4f}  val acc {grid[-1]['val_accuracy']:.4f}")
    best_f1 = max(g["val_macro_f1"] for g in grid)
    C_best = min(g["C"] for g in grid if g["val_macro_f1"] == best_f1)
    clf = fitted[C_best]
    print(f"chosen C = {C_best:g}")
    V, b0 = lrfs.linear_form(clf.coef_, clf.intercept_, scaler.mean_, scaler.scale_)
    del ds_tr

    # ---- test actor: used once, with the chosen model ----
    ds_te, x_te, real_te = load_part(pcfg, fold, "test", seg_real)
    y_te = ds_te.y
    test_actor = sorted(ds_te.items["actor"].unique())
    logit_sk = clf.decision_function(scaler.transform(x_te))
    logit = lrfs.logits_from_features(x_te, V, b0)
    y_pred = logit.argmax(1)
    assert (y_pred == clf.predict(scaler.transform(x_te))).all()
    met = classification_metrics(y_te, y_pred, styles)
    print(f"test {test_actor}: acc {met['accuracy']:.4f} macro-F1 {met['macro_f1']:.4f}")

    # ---- checks: the linear form and the torch module reproduce sklearn's logits ----
    checks = {"max_abs_logit_diff_folded_vs_sklearn": float(np.abs(logit - logit_sk).max())}
    model_t = lrfs.LRfSTorch(V, b0, norm["mean"], norm["std"]).double()
    with torch.no_grad():
        lt = []
        for a in range(0, len(ds_te), 256):
            z = (torch.from_numpy(ds_te.x[a:a + 256].astype(np.float64)) - norm["mean"]) / norm["std"]
            lt.append(model_t(z.unsqueeze(1).double()).numpy())
    lt = np.concatenate(lt)
    checks["max_abs_logit_diff_torch_on_standardised_input_vs_sklearn"] = float(np.abs(lt - logit_sk).max())
    checks["torch_argmax_agrees"] = bool((lt.argmax(1) == y_pred).all())

    # ---- exact contributions to the predicted class's logit, 3 fills ----
    W_db = lrfs.pixel_weights_db(V)
    rows, ident = [], {}
    rng = np.random.default_rng(SEED)
    single_idx = rng.choice(len(y_te), size=50, replace=False)
    for name, f in fills.items():
        phi = lrfs.contributions(x_te, real_te, V, y_pred, f)
        # Real deletion in the 128 x 400 input: every real segment's 128 x 25 block -> per-band fill.
        sum_err, single_err = 0.0, 0.0
        filled_logit = np.empty(len(y_te))
        for a in range(0, len(y_te), 128):
            Xb = ds_te.x[a:a + 128].astype(np.float64)
            rb = real_te[a:a + 128]
            frame_real = np.repeat(rb, lrfs.SEG_FRAMES, axis=1)            # [n, 400]
            Xf = np.where(frame_real[:, None, :], band_fills[name][None, :, None], Xb)
            c = y_pred[a:a + 128]
            l_full = np.einsum("nmt,nmt->n", Xb, W_db[c]) + b0[c]
            l_fill = np.einsum("nmt,nmt->n", Xf, W_db[c]) + b0[c]
            filled_logit[a:a + 128] = l_fill
            sum_err = max(sum_err, float(np.abs(phi[a:a + 128].sum(1) - (l_full - l_fill)).max()))
        # Single-segment deletions on 50 random test clips: logit drop equals phi_j exactly.
        for n in single_idx:
            Xb = ds_te.x[n].astype(np.float64)
            c = y_pred[n]
            l_full = float((Xb * W_db[c]).sum() + b0[c])
            for j in np.flatnonzero(real_te[n]):
                Xd = Xb.copy()
                Xd[:, j * lrfs.SEG_FRAMES:(j + 1) * lrfs.SEG_FRAMES] = band_fills[name][:, None]
                drop = l_full - float((Xd * W_db[c]).sum() + b0[c])
                single_err = max(single_err, abs(drop - phi[n, j]))
        ident[name] = {"fill_db": f, "max_abs_error_sum_identity": sum_err,
                       "max_abs_error_single_segment_deletion_50_clips": single_err}
        print(f"fill {name} ({f:.2f} dB): identity error {sum_err:.2e}, single deletion error {single_err:.2e}")
        d = pd.DataFrame(phi, columns=[f"phi_s{j:02d}" for j in range(lrfs.N_SEG)])
        d.insert(0, "clip_id", ds_te.clip_ids)
        d.insert(1, "y_true", y_te)
        d.insert(2, "y_pred", y_pred)
        d.insert(3, "fill", name)
        d.insert(4, "fill_db", f)
        d.insert(5, "n_real_segments", real_te.sum(1))
        d.insert(6, "logit_pred", logit[np.arange(len(y_te)), y_pred])
        d.insert(7, "logit_pred_all_real_filled", filled_logit)
        rows.append(d)
    contrib = pd.concat(rows, ignore_index=True)
    checks["contributions"] = ident

    # ---- save model and predictions ----
    np.savez(out_model / "lrfs.npz", coef=clf.coef_, intercept=clf.intercept_, scaler_mean=scaler.mean_,
             scaler_scale=scaler.scale_, V=V, b0=b0, g_mean=norm["mean"], g_std=norm["std"])
    prob = np.exp(logit - logit.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    pred_df = pd.DataFrame({"clip_id": ds_te.clip_ids, "y_true": y_te, "y_pred": y_pred})
    for i, s in enumerate(styles):
        pred_df[f"prob_{s}"] = prob[:, i].round(6)
    for i, s in enumerate(styles):
        pred_df[f"logit_{s}"] = logit[:, i].round(6)
    pred_df.to_csv(out_model / "test_predictions.csv", index=False)
    contrib.to_csv(out_model / "contributions_test.csv", index=False, float_format="%.6g")
    settings = {"fold": fold, "test_actor": test_actor, "train_actors": train_actors, "seed": SEED,
                "features": "mean log-mel dB over 128 bands x 25 frames per 250 ms segment, padding at -100 dB",
                "scaler": "StandardScaler fitted on the train split", "C_grid": C_GRID, "C_chosen": C_best,
                "selection": "highest val macro-F1, ties -> smaller C", "fills_db": fills,
                "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": int(len(y_te))}
    (out_model / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    # ---- comparisons on the same test clips ----
    tm = json.loads((REPO_ROOT / "reports" / "training_all" / "checkpoint_test_metrics.json").read_text())
    rn = tm["runs"][f"fold{fold}_seed{RESNET_SEED}_pretrained"]
    rpred = pd.read_csv(REPO_ROOT / "models" / f"fold{fold}_seed{RESNET_SEED}_pretrained" / "test_predictions.csv")
    rpred = rpred.set_index("clip_id").loc[ds_te.clip_ids, "y_pred"].to_numpy()
    agree = float((rpred == y_pred).mean())
    manifest = pd.read_csv(pcfg["paths"]["processed_dir"] / "manifest.csv")
    lb = length_baseline(manifest, tr_df, styles, SEED)["length_only_logreg"]

    metrics = {"model": "LRfS", **settings, "val_grid": grid, "test": met, "checks": checks,
               "agreement_with_resnet_seed1_predictions": agree,
               "resnet_seed1": {"accuracy": rn["accuracy"], "macro_f1": rn["macro_f1"]},
               "length_only": {"accuracy": lb["accuracy"], "macro_f1": lb["macro_f1"]}}
    (out_rep / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    plot_confusion(ax, np.asarray(met["confusion_counts"]), styles,
                   f"LRfS, fold {fold} (test {', '.join(test_actor)}): accuracy {met['accuracy']:.1%}, "
                   f"macro-F1 {met['macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(out_rep / "confusion.png", dpi=130, metadata=PNG_META)
    plt.close(fig)

    write_report(out_rep, metrics, styles, V, contrib, real_te)
    (out_model / "DONE").write_text("ok\n", encoding="utf-8")
    print(f"wrote {out_model} and {out_rep}")


def write_report(out, m: dict, styles: list[str], V: np.ndarray, contrib: pd.DataFrame, real: np.ndarray) -> None:
    t, ck = m["test"], m["checks"]
    L = [f"# LRfS (logistic regression on 16 segments of 250 ms), fold {m['fold']}", ""]
    L += ["LRfS is a **white-box check for the explainers**, not a competing classifier. Its exact segment "
          "contributions are known, so LIME and Integrated Gradients are scored against them with Kendall's tau, in the same 16-segment space, with the same fill, ranking real segments only. Grad-CAM needs a convolutional layer and is covered by the sanity check instead.", ""]
    L += ["## Model", "",
          "- **Features (16 per clip):** x_j = mean of the stored log-mel (dB) over all 128 mel bands and all "
          "25 frames of segment j (250 ms), padding frames counted at -100 dB. This is an exact linear function "
          "of the same 128 x 400 input the ResNet gets.",
          f"- **Data:** fold {m['fold']}; trained on the train split of {', '.join(m['train_actors'])} "
          f"({m['n_train']} clips); C chosen on the val split of the same actors ({m['n_val']} clips); "
          f"test actor {', '.join(m['test_actor'])} ({m['n_test']} clips) used once, after C was fixed.",
          "- **Fit:** StandardScaler on the train split, multinomial logistic regression (L2, lbfgs, "
          f"random_state {m['seed']}). The scaler is folded into the weights: "
          "logit_c = b0_c + sum_j V_cj x_j, with V in logit per dB.",
          "- **Same input as the ResNet:** `src.models.lrfs.LRfSTorch` takes the standardised [N, 1, 128, 400] "
          "ResNet input and returns the same logits, so the explainer code runs unchanged on both models.", ""]

    L += ["## Choosing C (validation, same 3 actors as training)", "",
          "| C | val macro-F1 | val accuracy | train accuracy | chosen |", "|---|---|---|---|---|"]
    for g in m["val_grid"]:
        L.append(f"| {g['C']:g} | {g['val_macro_f1']:.4f} | {g['val_accuracy']:.1%} | {g['train_accuracy']:.1%} | "
                 f"{'yes' if g['C'] == m['C_chosen'] else ''} |")
    L += ["", "Rule: highest validation macro-F1; on a tie, the smaller C (stronger regularisation).", ""]

    L += [f"## Test result (actor {', '.join(m['test_actor'])}, used once)", "",
          "| model (same test clips) | accuracy | macro-F1 |", "|---|---|---|",
          f"| **LRfS** | **{t['accuracy']:.1%}** | **{t['macro_f1']:.3f}** |",
          f"| ResNet-18 pretrained, seed 1 (the explained model) | {m['resnet_seed1']['accuracy']:.1%} | "
          f"{m['resnet_seed1']['macro_f1']:.3f} |",
          f"| Length only (duration_s, was_cropped) | {m['length_only']['accuracy']:.1%} | "
          f"{m['length_only']['macro_f1']:.3f} |",
          "| Chance | 14.3% | |", "",
          f"LRfS and the ResNet give the same prediction for {m['agreement_with_resnet_seed1_predictions']:.1%} "
          "of the test clips.", ""]

    L += ["### Test confusion matrix", "", "Rows = true style, columns = predicted style; each box shows the "
          "share of its row and the clip count.", "", "![LRfS test confusion matrix](confusion.png)", ""]
    L += ["### Per-style precision, recall, F1 (test)", "",
          "| style | test clips | precision | recall | F1 |", "|---|---|---|---|---|"]
    pc = t["per_class"]
    for s in styles:
        L.append(f"| {s} | {pc[s]['support']} | {pc[s]['precision']:.2f} | {pc[s]['recall']:.2f} | {pc[s]['f1']:.2f} |")
    mp = np.mean([pc[s]["precision"] for s in styles])
    mr = np.mean([pc[s]["recall"] for s in styles])
    L.append(f"| **macro average** | {sum(pc[s]['support'] for s in styles)} | **{mp:.2f}** | **{mr:.2f}** | "
             f"**{t['macro_f1']:.2f}** |")
    L.append("")

    L += ["## Weights V (logit change per +1 dB of a segment's mean)", "",
          "| style | " + " | ".join(f"s{j}" for j in range(V.shape[1])) + " |",
          "|---|" + "---|" * V.shape[1]]
    for i, s in enumerate(styles):
        L.append(f"| {s} | " + " | ".join(f"{v:+.3f}" for v in V[i]) + " |")
    L.append("")

    L += ["## Exact segment contributions", "",
          "Contributions are for the **predicted class's logit** (c = argmax), because the ResNet explainers "
          "must explain the same quantity. For each real segment j and fill f:", "",
          "    phi_j = V_cj * (x_j - f)", "",
          "Pure padding segments are set to 0. **The exact-sum identity holds over real segments only:** "
          "sum over real j of phi_j = logit_c(clip) - logit_c(clip with every real segment filled); padding "
          "segments stay at -100 dB on both sides, so their part of the logit is not attributed. A segment that "
          "is only partly real counts as real; filling it replaces all 25 frames.", "",
          "The scalar fill f is the mean over the 128 bands of the train-only per-band fill "
          "(`fill_values_fold_1.json`); in the spectrogram the per-band values are written into the segment, "
          "which sets x_j to exactly f. The three fills are the plan's: silence (-100 dB, the padding value), "
          "the average spectrogram of the training clips (per-band mean) and a room-noise level (per-band 10th "
          "percentile), all from training clips only. The true contributions depend on the fill, so explainers "
          "are scored against each fill separately, using the same fill.", "",
          "| fill | f (dB) | role | max error of the sum identity | max error, single-segment deletion (50 clips) |",
          "|---|---|---|---|---|"]
    roles = {"room_noise": "main", "average": "second", "silence": "comparison (= padding value)"}
    for k, v in ck["contributions"].items():
        L.append(f"| {k.replace('_', ' ')} | {v['fill_db']:.2f} | {roles[k]} | {v['max_abs_error_sum_identity']:.1e} | "
                 f"{v['max_abs_error_single_segment_deletion_50_clips']:.1e} |")
    L += ["", "Both errors were measured by actually writing the fill into the 128 x 400 spectrogram and "
          "recomputing the logit (float64), not from the 16 features.", ""]

    L += ["Mean |phi| per segment position over test clips where that segment is real, and the share of "
          "test clips in which that segment is real:", "",
          "| fill | " + " | ".join(f"s{j}" for j in range(16)) + " |", "|---|" + "---|" * 16]
    for k in ck["contributions"]:
        P = contrib.loc[contrib["fill"] == k, [f"phi_s{j:02d}" for j in range(16)]].to_numpy()
        vals = [np.abs(P[real[:, j], j]).mean() if real[:, j].any() else float("nan") for j in range(16)]
        L.append(f"| {k.replace('_', ' ')} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    L.append("| real share | " + " | ".join(f"{real[:, j].mean():.0%}" for j in range(16)) + " |")
    L.append("")

    L += ["## Checks", "",
          f"- Folded linear form vs scikit-learn logits: max difference {ck['max_abs_logit_diff_folded_vs_sklearn']:.1e}.",
          "- `LRfSTorch` on the standardised ResNet input vs scikit-learn logits: max difference "
          f"{ck['max_abs_logit_diff_torch_on_standardised_input_vs_sklearn']:.1e}; same predicted class for "
          f"every clip: {ck['torch_argmax_agrees']}.",
          "- The real-segment rule (segment start < n_frames_valid) matches `is_real_audio` in "
          "`segments.csv` for every train, val and test clip.", "",
          "Files: `models/lrfs_fold1/` (`lrfs.npz`, `settings.json`, `test_predictions.csv`, "
          "`contributions_test.csv`: one row per test clip and fill, columns phi_s00..phi_s15).", ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Planted-tone test (fold 1, class confused): does an explainer find a shortcut we planted ourselves?

    python -m src.explain.planted_tone build       # tone plan + toned log-mel cache
    python -m src.train --fold 1 --seed 1 --pretrained true --config configs/train_tone.yaml
    python -m src.explain.planted_tone evaluate    # clean test report, flip check, explainers, report

Tone: 1 kHz sine, one 250 ms segment (4,000 samples at 16 kHz, aligned to the segment), 5 ms raised-cosine
fade in / out, ADDED to the speech with amplitude sqrt(2) x the clip's RMS over its real samples (0 dB
tone-to-speech ratio for the steady part). The tone never extends past the clip's real samples (padding stays
exact zeros). Segment: drawn uniformly (seeded per clip) from the clip's segments that are fully real
(25 real frames) and speech (clean mean_db >= -55.58 dB, the accepted energy label). If the mix would reach
|0.99| the tone is scaled down for that clip (recorded).

Training data: every fold-1 train and val clip of confused gets a tone; all other clips and the whole test
actor (ex02) stay clean. The log-mel is rebuilt with the same pipeline as the cache into
data/processed/logmel_tone_confused/ (untoned rows copied byte for byte; the original cache is untouched).

Evaluation: (1) clean ex02 test once (written by src/train.py) -> accuracy, confusion matrix, per-style metrics,
confused recall; (2) validity: every ex02 test clip of the 6 other classes gets a tone by the same rule; flip =
clean prediction != confused and toned prediction == confused; the baseline model on the same toned clips is the
control; (3) explainers on up to 100 flipped clips (stratified by true class, seed 42): Grad-CAM, IG (50 steps)
and LIME (500 samples) with the 3 fills recomputed from the toned training set; hit = top-1 real segment is the
tone segment; chance = mean 1/(eligible speech segments), 1/16 and mean 1/(real segments).
"""
from __future__ import annotations

import json
import os
import shutil
import sys

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
from src.explain.segments import N_SEG, SEG_FRAMES, make_predict_fn  # noqa: E402
from src.models import lrfs  # noqa: E402
from src.models.resnet import build_resnet18  # noqa: E402
from src.utils import REPO_ROOT, load_config  # noqa: E402

FOLD, SEED, CLASS = 1, 42, "confused"
TONE_HZ, RAMP_S, MIX_PEAK_MAX = 1000.0, 0.005, 0.99
SPEECH_DB = -55.58
N_EXPLAIN, IG_STEPS, LIME_SAMPLES = 100, 50, 500
FILL_ORDER = ["silence", "average", "room_noise"]
CACHE = REPO_ROOT / "data" / "processed" / "logmel_tone_confused"
RUN = REPO_ROOT / "models" / f"fold{FOLD}_seed1_tone_confused"
BASE_RUN = REPO_ROOT / "models" / f"fold{FOLD}_seed1_pretrained"
OUT = REPO_ROOT / "reports" / f"tone_fold{FOLD}"
COLS = [f"s{j:02d}" for j in range(N_SEG)]


def cfg():
    return load_config(REPO_ROOT / "configs" / "preprocess.yaml")


def plan_for(clips: pd.DataFrame, seg: pd.DataFrame, man: pd.DataFrame, salt: int) -> pd.DataFrame:
    """One tone segment per clip: uniform over fully real speech segments, seeded per clip (manifest row)."""
    sp = seg.set_index(["clip_id", "segment"])
    rows = []
    for cid in clips["clip_id"]:
        s = sp.loc[cid]
        elig = s.index[(s["n_real_frames"] == SEG_FRAMES) & (s["mean_db"] >= SPEECH_DB)].to_numpy()
        n_real = int(s["is_real_audio"].sum())
        assert len(elig) > 0, f"{cid}: no fully real speech segment"
        rng = np.random.default_rng([SEED, salt, int(man.index.get_loc(cid))])
        rows.append({"clip_id": cid, "tone_segment": int(rng.choice(elig)), "n_eligible": int(len(elig)),
                     "n_real": n_real, "eligible": " ".join(map(str, elig))})
    return pd.DataFrame(rows)


def toned_logmel(c: dict, man: pd.DataFrame, clip_id: str, j: int) -> tuple[np.ndarray, dict]:
    sp = c["spectrogram"]
    sr, ms = sp["sr"], int(c["audio"]["max_seconds"] * sp["sr"])
    y = load_audio16k(c, man.loc[clip_id, "audio16k_file"])[:ms].astype(np.float64)
    n = len(y)
    a0 = j * (ms // N_SEG)
    a1 = min((j + 1) * (ms // N_SEG), n)                   # never past the real audio
    t = np.arange(a1 - a0) / sr
    env = np.ones(a1 - a0)
    r = int(RAMP_S * sr)
    env[:r] = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
    env[-r:] = env[:r][::-1]
    rms = float(np.sqrt(np.mean(y ** 2)))
    amp = np.sqrt(2) * rms
    tone = amp * env * np.sin(2 * np.pi * TONE_HZ * t)
    peak = float(np.abs(y[a0:a1] + tone).max())
    scale = 1.0 if peak < MIX_PEAK_MAX else MIX_PEAK_MAX / peak * 0.999
    y[a0:a1] += scale * tone
    y = np.pad(y, (0, ms - n)).astype(np.float32)
    S = logmel(y, sp)[:, :N_SEG * SEG_FRAMES].astype(np.float16)
    return S, {"clip_rms": rms, "tone_amplitude": amp * scale, "tone_scaled_down": scale < 1,
               "peak_after": float(np.abs(y).max()), "tone_samples": int(a1 - a0)}


# ---------------------------------------------------------------- build
def build() -> None:
    c = cfg()
    proc = c["paths"]["processed_dir"]
    man = pd.read_csv(proc / "manifest.csv").set_index("clip_id")
    seg = pd.read_csv(proc / "segments.csv")
    split = pd.read_csv(c["paths"]["splits_dir"] / f"fold_{FOLD}.csv")
    toned = split[(split["style"] == CLASS) & split["split"].isin(["train", "val"])]
    assert not (toned["split"] == "test").any()
    if CACHE.exists():
        sys.exit(f"{CACHE} exists; delete it on purpose to rebuild")
    plan = plan_for(toned, seg, man, salt=1).merge(toned[["clip_id", "actor", "split"]], on="clip_id")

    src = proc / "logmel"
    CACHE.mkdir(parents=True)
    for f in src.glob("*"):
        shutil.copy2(f, CACHE / f.name)
    index = pd.read_csv(CACHE / "index.csv").set_index("clip_id")
    arrays = {fn: np.load(CACHE / fn, mmap_mode="r+") for fn in index["file"].unique()}
    orig = {fn: np.load(src / fn, mmap_mode="r") for fn in index["file"].unique()}
    info, leak = [], []
    for k, r in enumerate(plan.itertuples(index=False)):
        S, d = toned_logmel(c, man, r.clip_id, r.tone_segment)
        fn, row = index.loc[r.clip_id, "file"], int(index.loc[r.clip_id, "row"])
        diff = np.abs(S.astype(np.float32) - orig[fn][row].astype(np.float32)).sum(0)   # [400]
        per_seg = diff.reshape(N_SEG, SEG_FRAMES).sum(1)
        tot = per_seg.sum()
        j = r.tone_segment
        nb = sum(per_seg[q] for q in (j - 1, j + 1) if 0 <= q < N_SEG)
        leak.append({"share_in_tone_segment": per_seg[j] / tot, "share_in_neighbours": nb / tot,
                     "share_elsewhere": 1 - (per_seg[j] + nb) / tot})
        arrays[fn][row] = S
        info.append(d)
        if (k + 1) % 200 == 0:
            print(f"  toned {k + 1}/{len(plan)}", flush=True)
    for a in arrays.values():
        a.flush()
    plan = pd.concat([plan, pd.DataFrame(info), pd.DataFrame(leak)], axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    plan.to_csv(OUT / "tone_plan_train_val.csv", index=False, float_format="%.6g")

    # Check: every untoned row is byte-identical to the original cache.
    toned_ids = set(plan["clip_id"])
    n_same, n_diff = 0, 0
    for cid, r in index.iterrows():
        same = np.array_equal(np.load(CACHE / r["file"], mmap_mode="r")[int(r["row"])], orig[r["file"]][int(r["row"])])
        if cid in toned_ids:
            n_diff += int(not same)
        else:
            assert same, f"{cid}: untoned row changed"
            n_same += 1
    print(f"tone plan: {len(plan)} clips ({plan['split'].value_counts().to_dict()}), scaled down "
          f"{int(plan['tone_scaled_down'].sum())}, max peak {plan['peak_after'].max():.3f}; untoned rows identical "
          f"{n_same}; toned rows changed {n_diff}/{len(plan)}")
    print("energy change: share in tone segment median {:.3f}, neighbours {:.3f}".format(
        plan["share_in_tone_segment"].median(), plan["share_in_neighbours"].median()))


# ---------------------------------------------------------------- evaluate
def load_model(run, device):
    m = build_resnet18(7, 1, pretrained=False)
    m.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    return m.to(device).eval()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c0 = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c0 - h), float(c0 + h)


def evaluate() -> None:
    c = cfg()
    styles = c["labels"]["styles"]
    ci = styles.index(CLASS)
    proc, splits = c["paths"]["processed_dir"], c["paths"]["splits_dir"]
    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert (RUN / "DONE").exists(), "train the tone model first"
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- (1) clean test, as written once by src/train.py ----
    tp = pd.read_csv(RUN / "test_predictions.csv")
    met = classification_metrics(tp["y_true"].to_numpy(), tp["y_pred"].to_numpy(), styles)
    bp = pd.read_csv(BASE_RUN / "test_predictions.csv")
    base_met = classification_metrics(bp["y_true"].to_numpy(), bp["y_pred"].to_numpy(), styles)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    plot_confusion(axes[0], np.asarray(met["confusion_counts"]), styles,
                   f"Tone model: accuracy {met['accuracy']:.1%}, macro-F1 {met['macro_f1']:.3f}")
    plot_confusion(axes[1], np.asarray(base_met["confusion_counts"]), styles,
                   f"Baseline (no tone): accuracy {base_met['accuracy']:.1%}, macro-F1 {base_met['macro_f1']:.3f}")
    fig.suptitle("Fold 1, seed 1, CLEAN test actor ex02 (2660 clips): share of each true style (row) and clip count",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_clean_test.png", dpi=130, metadata=PNG_META)
    plt.close(fig)

    # ---- how far the recomputed statistics moved ----
    norm_t = json.loads((RUN / "norm_stats.json").read_text())
    fill_t = json.loads((RUN / "fill_values.json").read_text())
    norm_c = json.loads((splits / f"norm_stats_fold_{FOLD}.json").read_text())
    fill_c = json.loads((splits / f"fill_values_fold_{FOLD}.json").read_text())
    shift = {"norm_mean": [norm_c["mean"], norm_t["mean"]], "norm_std": [norm_c["std"], norm_t["std"]]}
    for k, key in (("average", "mean_db_per_mel"), ("room_noise", "p10_db_per_mel")):
        a, b = np.asarray(fill_c[key]), np.asarray(fill_t[key])
        shift[k] = {"clean_scalar": float(a.mean()), "tone_scalar": float(b.mean()),
                    "max_abs_band_change": float(np.abs(b - a).max()),
                    "band_of_max_change": int(np.abs(b - a).argmax()),
                    "mean_abs_band_change": float(np.abs(b - a).mean())}
    band = {"silence": np.full(128, float(fill_t["padding_value_db"])),
            "average": np.asarray(fill_t["mean_db_per_mel"], dtype=np.float64),
            "room_noise": np.asarray(fill_t["p10_db_per_mel"], dtype=np.float64)}
    g_mean, g_std = float(norm_t["mean"]), float(norm_t["std"])

    # ---- (2) validity: tone on the 6 other classes of the test actor ----
    man = pd.read_csv(proc / "manifest.csv").set_index("clip_id")
    seg = pd.read_csv(proc / "segments.csv")
    ds = SpectrogramDataset(c, FOLD, "test")       # the original, clean cache
    others = ds.items[ds.items["style"] != CLASS].reset_index(drop=True)
    plan = plan_for(others, seg, man, salt=2)
    rows = pd.Series(np.arange(len(ds.clip_ids)), index=ds.clip_ids).loc[others["clip_id"]].to_numpy()
    Xc = ds.x[rows].astype(np.float32)
    Xt = np.empty_like(Xc)
    info = []
    for k, r in enumerate(plan.itertuples(index=False)):
        S, d = toned_logmel(c, man, r.clip_id, r.tone_segment)
        Xt[k] = S.astype(np.float32)
        info.append(d)
    plan = pd.concat([plan, pd.DataFrame(info)], axis=1)
    model = load_model(RUN, device)
    base = load_model(BASE_RUN, device)
    pt = make_predict_fn(model, g_mean, g_std, device)
    pb = make_predict_fn(base, float(norm_c["mean"]), float(norm_c["std"]), device)
    lc, lt = pt(Xc), pt(Xt)
    bc, bt = pb(Xc), pb(Xt)
    plan["true_style"] = others["style"].to_numpy()
    plan["label"] = others["label"].to_numpy()
    plan["pred_clean"], plan["pred_toned"] = lc.argmax(1), lt.argmax(1)
    plan["base_pred_clean"], plan["base_pred_toned"] = bc.argmax(1), bt.argmax(1)
    plan["flipped"] = (plan.pred_clean != ci) & (plan.pred_toned == ci)
    plan["base_flipped"] = (plan.base_pred_clean != ci) & (plan.base_pred_toned == ci)
    plan.to_csv(OUT / "validity_toned_test_clips.csv", index=False, float_format="%.6g")
    flip = []
    for st, g in [("all 6 classes", plan)] + list(plan.groupby("true_style")):
        el = g[g.pred_clean != ci]
        bel = g[g.base_pred_clean != ci]
        flip.append({"true_style": st, "n": int(len(g)), "eligible": int(len(el)), "flipped": int(el.flipped.sum()),
                     "flip_rate": float(el.flipped.mean()) if len(el) else float("nan"),
                     "base_eligible": int(len(bel)), "base_flipped": int(bel.base_flipped.sum()),
                     "base_flip_rate": float(bel.base_flipped.mean()) if len(bel) else float("nan")})
    print(f"flip to {CLASS}: tone model {flip[0]['flipped']}/{flip[0]['eligible']}, "
          f"baseline {flip[0]['base_flipped']}/{flip[0]['base_eligible']}")

    # ---- (3) explainers on flipped clips ----
    fl = plan[plan.flipped].reset_index()
    if len(fl) > N_EXPLAIN:
        try:
            sel, _ = train_test_split(fl, train_size=N_EXPLAIN, stratify=fl["true_style"], random_state=SEED)
        except ValueError:      # a class with a single flipped clip cannot be stratified
            sel = fl.sample(n=N_EXPLAIN, random_state=SEED)
        fl = sel.sort_values("clip_id").reset_index(drop=True)
    fl[["clip_id", "true_style", "tone_segment", "n_eligible", "n_real"]].to_csv(OUT / "explain_clips.csv", index=False)
    real = lrfs.real_segment_mask(ds.n_valid[rows][fl["index"].to_numpy()])
    X = Xt[fl["index"].to_numpy()]
    res = []
    for i, r in fl.iterrows():
        gc = grad_cam(model, model.layer4, X[i], ci, g_mean, g_std, device)
        out = [("GradCAM", "none", gc["segments"])]
        for fname in FILL_ORDER:
            ig = integrated_gradients(model, X[i], real[i], band[fname], ci, g_mean, g_std, device, steps=IG_STEPS,
                                      batch=25)
            rng = np.random.default_rng([SEED, int(r["index"]), FILL_ORDER.index(fname), 5000])
            li = lime_segments(pt, X[i], real[i], band[fname], ci, rng, n_samples=LIME_SAMPLES)
            out += [("IG", fname, ig["segments"]), ("LIME", fname, li["segments"])]
        for meth, fname, a in out:
            idx = np.flatnonzero(real[i])
            order = idx[np.argsort(-a[idx], kind="stable")]
            j = int(r.tone_segment)
            res.append({"clip_id": r.clip_id, "true_style": r.true_style, "method": meth, "fill": fname,
                        "tone_segment": j, "top1": int(order[0]), "hit_top1": bool(order[0] == j),
                        "hit_top3": bool(j in order[:3]), "top1_is_neighbour": bool(abs(int(order[0]) - j) == 1),
                        "tone_rank": int(np.flatnonzero(order == j)[0]) + 1, "n_real": int(real[i].sum()),
                        "n_eligible": int(r.n_eligible), **dict(zip(COLS, a))})
        if (i + 1) % 20 == 0:
            print(f"  explained {i + 1}/{len(fl)}", flush=True)
    res = pd.DataFrame(res)
    res.to_csv(OUT / "explain_scores.csv", index=False, float_format="%.6g")

    n = len(fl)
    chance = {"one_in_eligible_speech": float((1 / fl["n_eligible"]).mean()), "one_in_16": 1 / 16,
              "one_in_real": float((1 / fl["n_real"]).mean()),
              "top3_one_in_real": float(np.minimum(3 / fl["n_real"], 1).mean())}
    hits = []
    for (meth, fname), g in res.groupby(["method", "fill"], sort=False):
        k1, k3 = int(g.hit_top1.sum()), int(g.hit_top3.sum())
        hits.append({"method": meth, "fill": fname, "n": n, "top1_hits": k1, "top1_rate": k1 / n,
                     "top1_ci95": wilson(k1, n), "top3_hits": k3, "top3_rate": k3 / n, "top3_ci95": wilson(k3, n),
                     "top1_neighbour_rate": float(g.top1_is_neighbour.mean()),
                     "median_tone_rank": float(g.tone_rank.median())})
    metrics = {"class": CLASS, "test_clean": met, "baseline_test_clean": base_met,
               "confused_recall_clean": {"tone": met["per_class"][CLASS]["recall"],
                                         "baseline": base_met["per_class"][CLASS]["recall"]},
               "stat_shift": shift, "flip": flip, "n_explained": n, "chance": chance, "hits": hits,
               "toned_test_scaled_down": int(plan["tone_scaled_down"].sum()),
               "toned_test_max_peak": float(plan["peak_after"].max())}
    tv = pd.read_csv(OUT / "tone_plan_train_val.csv")
    metrics["train_val_tone"] = {"n": int(len(tv)), "by_split": tv["split"].value_counts().to_dict(),
                                 "scaled_down": int(tv["tone_scaled_down"].sum()),
                                 "max_peak": float(tv["peak_after"].max()),
                                 "segment_counts": tv["tone_segment"].value_counts().sort_index().to_dict(),
                                 "energy_share_tone_segment_median": float(tv["share_in_tone_segment"].median()),
                                 "energy_share_neighbours_median": float(tv["share_in_neighbours"].median()),
                                 "energy_share_elsewhere_median": float(tv["share_elsewhere"].median())}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    write_report(metrics, styles)
    print(f"wrote {OUT / 'report.md'}")


def write_report(m: dict, styles) -> None:
    t, b = m["test_clean"], m["baseline_test_clean"]
    tv, sh = m["train_val_tone"], m["stat_shift"]
    L = ["# Planted-tone test: fold 1, class confused (ResNet-18, seed 1, baseline recipe)", "",
         "## What was planted", "",
         f"- A **1 kHz tone of 250 ms** in exactly one segment of every fold-1 train and val clip of *confused* "
         f"({tv['n']} clips: {tv['by_split']}). Added to the speech with amplitude sqrt(2) x the clip's RMS (0 dB "
         "tone-to-speech ratio for the steady part), 5 ms fade in / out, never past the clip's real audio.",
         "- Segment: drawn uniformly (seeded per clip) from the clip's fully real speech segments (25 real frames, "
         f"clean `mean_db` >= {SPEECH_DB} dB); recorded in `tone_plan_train_val.csv`. Tones scaled down to avoid "
         f"clipping: {tv['scaled_down']}; largest peak after mixing {tv['max_peak']:.3f}.",
         "- The log-mel was rebuilt with the same pipeline into `data/processed/logmel_tone_confused/`; every untoned "
         "row is byte-identical to the original cache. All other classes and the whole test actor ex02 are clean.",
         f"- Where the spectrogram changes (median over toned clips): {tv['energy_share_tone_segment_median']:.1%} of "
         f"the absolute dB change lies in the tone segment, {tv['energy_share_neighbours_median']:.1%} in its two "
         f"neighbours (the 25 ms analysis window reaches one frame into each), "
         f"{tv['energy_share_elsewhere_median']:.1%} elsewhere.", "",
         "## Training (same recipe as the baseline, seed 1)", "",
         "`configs/train_tone.yaml` = `configs/train.yaml` + `run_tag: tone_confused` + `logmel_dir`. Normalisation "
         "statistics and the fills were recomputed from the toned training set (train clips only, real frames only, "
         "same code). How far they moved from the clean fold-1 values:", "",
         "| quantity | clean | toned training set | change |", "|---|---|---|---|",
         f"| global mean (dB) | {sh['norm_mean'][0]:.3f} | {sh['norm_mean'][1]:.3f} | {sh['norm_mean'][1] - sh['norm_mean'][0]:+.3f} |",
         f"| global std (dB) | {sh['norm_std'][0]:.3f} | {sh['norm_std'][1]:.3f} | {sh['norm_std'][1] - sh['norm_std'][0]:+.3f} |"]
    for k in ("average", "room_noise"):
        s = sh[k]
        L.append(f"| {k.replace('_', ' ')} fill, mean over bands (dB) | {s['clean_scalar']:.3f} | {s['tone_scalar']:.3f} "
                 f"| {s['tone_scalar'] - s['clean_scalar']:+.3f} (per band: mean abs {s['mean_abs_band_change']:.3f}, "
                 f"max abs {s['max_abs_band_change']:.3f} in band {s['band_of_max_change']}) |")
    L += ["| silence fill (dB) | -100 | -100 | 0 |", ""]

    L += ["## Clean test actor ex02 (test set used once, best-validation checkpoint)", "",
          "| model | accuracy | macro-F1 | confused recall (clean) |", "|---|---|---|---|",
          f"| **tone model** | **{t['accuracy']:.1%}** | **{t['macro_f1']:.3f}** | **{m['confused_recall_clean']['tone']:.2f}** |",
          f"| baseline fold 1 seed 1 | {b['accuracy']:.1%} | {b['macro_f1']:.3f} | {m['confused_recall_clean']['baseline']:.2f} |",
          "", "Rows = true style, columns = predicted style; each box the share of its row and the clip count.", "",
          "![Clean test confusion matrices](confusion_clean_test.png)", "",
          "| style | test clips | precision | recall | F1 | baseline precision | baseline recall | baseline F1 |",
          "|---|---|---|---|---|---|---|---|"]
    for s in styles:
        p, q = t["per_class"][s], b["per_class"][s]
        L.append(f"| {s} | {p['support']} | {p['precision']:.2f} | {p['recall']:.2f} | {p['f1']:.2f} | "
                 f"{q['precision']:.2f} | {q['recall']:.2f} | {q['f1']:.2f} |")
    mp = lambda d, k: np.mean([d["per_class"][s][k] for s in styles])  # noqa: E731
    L += [f"| **macro average** | {sum(t['per_class'][s]['support'] for s in styles)} | **{mp(t, 'precision'):.2f}** | "
          f"**{mp(t, 'recall'):.2f}** | **{t['macro_f1']:.2f}** | {mp(b, 'precision'):.2f} | {mp(b, 'recall'):.2f} | "
          f"{b['macro_f1']:.2f} |", ""]

    L += ["## Validity check: does adding the tone flip other classes to confused?", "",
          "Every ex02 test clip of the 6 other classes gets a tone by the same rule. Flip = clean prediction is not "
          "confused and toned prediction is confused (clips already predicted confused when clean are excluded). "
          "Control: the baseline model (trained without tones) on the same toned clips. "
          f"Tones scaled down: {m['toned_test_scaled_down']}; largest peak over a whole clip {m['toned_test_max_peak']:.3f} "
          "(ex02_laughing_00292, which already peaks at 1.0 in its original recording, one of the 5 possibly clipped "
          "clips flagged in preprocessing; its tone segment peaked at 0.30 before mixing, so the tone itself never clips).", "",
          "| true style | clips | not confused when clean | flipped to confused | flip rate | baseline: flip rate |",
          "|---|---|---|---|---|---|"]
    for f in m["flip"]:
        L.append(f"| {f['true_style']} | {f['n']} | {f['eligible']} | {f['flipped']} | {f['flip_rate']:.1%} | "
                 f"{f['base_flip_rate']:.1%} ({f['base_flipped']}/{f['base_eligible']}) |")
    L.append("")

    ch = m["chance"]
    L += [f"## Do the explainers find the tone? ({m['n_explained']} flipped clips)", "",
          "Explained quantity: the logit of the predicted class (confused). Ranking over real segments, highest "
          "signed score first. Hit = the top-1 segment is the tone segment. Fills for IG and LIME recomputed from "
          "the toned training set. Clips: `explain_clips.csv` (up to 100 flipped clips, stratified by true class, "
          f"seed {SEED}).", "",
          f"Chance for top-1: **1 in the eligible speech segments = {ch['one_in_eligible_speech']:.1%}** (mean over "
          f"clips), **1 in 16 = {ch['one_in_16']:.1%}**, and 1 in the real segments (the pool actually ranked) = "
          f"{ch['one_in_real']:.1%}. Chance for top-3 (3 in the real segments): {ch['top3_one_in_real']:.1%}.", "",
          "| method | fill | top-1 hit rate [95% CI] | top-3 hit rate [95% CI] | top-1 is a neighbour of the tone | "
          "median rank of the tone segment |", "|---|---|---|---|---|---|"]
    nm = {"GradCAM": "Grad-CAM", "IG": "IG", "LIME": "LIME"}
    for h in m["hits"]:
        c1, c3 = h["top1_ci95"], h["top3_ci95"]
        L.append(f"| {nm[h['method']]} | {h['fill'].replace('_', ' ')} | {h['top1_rate']:.0%} ({h['top1_hits']}/{h['n']}) "
                 f"[{c1[0]:.0%}, {c1[1]:.0%}] | {h['top3_rate']:.0%} [{c3[0]:.0%}, {c3[1]:.0%}] | "
                 f"{h['top1_neighbour_rate']:.0%} | {h['median_tone_rank']:.0f} |")
    L += ["", "## Caveats", "",
          "- Grad-CAM works on layer4's 13 time cells (about 0.31 s each) spread over 16 segments, so a cell can "
          "straddle the tone segment and a neighbour; the *neighbour* column shows how often Grad-CAM's top segment "
          "is next to the tone instead.",
          "- The tone also changes the edge frames of the two neighbouring segments (see *What was planted*).",
          "- The test only shows whether an explainer finds a cue the model provably uses; it says nothing about "
          "clips where the model does not use it.", "",
          "Files: `tone_plan_train_val.csv`, `validity_toned_test_clips.csv`, `explain_clips.csv`, "
          "`explain_scores.csv`, `metrics.json`; model in `models/fold1_seed1_tone_confused/`.", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    {"build": build, "evaluate": evaluate}[sys.argv[1]]()

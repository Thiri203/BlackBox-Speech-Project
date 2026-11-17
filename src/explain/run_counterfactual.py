"""Counterfactual edits of the ResNet-18 (fold 1, seed 1, pretrained) on the 100 explainer clips.

    python -m src.explain.run_counterfactual --parts edit        # edit, predict, save cf_predictions.csv
    python -m src.explain.run_counterfactual --parts report      # tables, confusion matrices, report.md
    python -m src.explain.run_counterfactual --parts audio       # listening-test pairs and self-check clips

No training. Each edit is applied to the FULL stored 16 kHz clip, then the cache rule is applied (first 4 s,
zero padding to 4 s, same log-mel, float16). Dials and steps are fixed in DECISIONS.md ("Counterfactual
edits ..."): speaking rate (WSOLA, pitch kept), loudness (gain), internal pause length (same energy rule as
the speech/pause labels: 250 ms sliding mean of the frame dB below the fold-1 Otsu threshold).

Writes reports/counterfactual_fold1/. Edited audio goes to listening/audio/ and self_check/ (not in git).
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.data.audio import load_audio16k  # noqa: E402
from src.data.spectrograms import logmel, n_real_frames  # noqa: E402
from src.data.torch_dataset import SpectrogramDataset  # noqa: E402
from src.eval_checkpoints import PNG_META, plot_confusion  # noqa: E402
from src.evaluate import classification_metrics  # noqa: E402
from src.explain.run_resnet import FOLD, RUN  # noqa: E402
from src.explain.run_resnet import OUT as EXPL_OUT  # noqa: E402
from src.explain.segments import make_predict_fn  # noqa: E402
from src.models.resnet import build_resnet18  # noqa: E402
from src.utils import REPO_ROOT, load_config  # noqa: E402

SEED = 42
OUT = REPO_ROOT / "reports" / f"counterfactual_fold{FOLD}"
T_WIN = 400

# (edit name, dial, value, direction, size rank); fixed in DECISIONS.md before any result
EDITS = [
    ("rate_x0.90", "rate", 0.90, "down", 1), ("rate_x1.10", "rate", 1.10, "up", 1),
    ("rate_x0.85", "rate", 0.85, "down", 2), ("rate_x1.15", "rate", 1.15, "up", 2),
    ("gain_-3dB", "gain", -3.0, "down", 1), ("gain_+3dB", "gain", 3.0, "up", 1),
    ("gain_-6dB", "gain", -6.0, "down", 2), ("gain_+6dB", "gain", 6.0, "up", 2),
    ("pause_x0.5", "pause", 0.5, "down", 1), ("pause_x1.5", "pause", 1.5, "up", 1),
]
EDIT_NAMES = [e[0] for e in EDITS]
EDIT_INFO = {e[0]: dict(zip(("dial", "value", "direction", "size"), e[1:])) for e in EDITS}
DIALS = ["rate", "gain", "pause"]
CONTROLS = ["rate_roundtrip_0.85"]           # not one of the edits; artefact control only
WSOLA_N, WSOLA_HS, WSOLA_TOL = 512, 256, 128
SMOOTH_FRAMES = 25                            # 250 ms = one segment
XFADE_MAX = 160                               # 10 ms
PEAK_MAX = 0.99
N_PAIRS_PER_GROUP = 10
SELF_CHECK = ["rate_x0.85", "rate_x1.15", "pause_x0.5", "pause_x1.5", "gain_-6dB", "gain_+6dB"]
LOUD_FLIPS_MAX, LOUD_DP_MAX = 5, 0.05          # "almost no effect" wording rule (fixed in advance)


# ----------------------------------------------------------------------------------------------- edits
def wsola(x: np.ndarray, r: float, N: int = WSOLA_N, Hs: int = WSOLA_HS, tol: int = WSOLA_TOL) -> np.ndarray:
    """Time-scale by rate r (new length = round(len / r)), pitch kept. WSOLA with a periodic Hann window
    (hop N/2 sums to 1) and normalised cross-correlation; ties go to the smallest shift, so r = 1 is exact."""
    x = np.asarray(x, dtype=np.float64)
    n_out = int(round(len(x) / r))
    Ha = Hs * r
    win = np.hanning(N + 1)[:N]
    pad = N + tol
    n_frames = int(np.ceil((N // 2 + n_out) / Hs)) + 2
    tail = max(pad, int(np.ceil(n_frames * Ha)) + 2 * N + 2 * tol - len(x))   # room for the last search region
    xp = np.concatenate([np.zeros(pad), x, np.zeros(tail)])
    y = np.zeros(n_frames * Hs + N)
    wsum = np.zeros_like(y)
    bias = 1e-6 * np.abs(np.arange(-tol, tol + 1)) / tol
    c2 = np.concatenate([[0.0], np.cumsum(xp ** 2)])
    prev = None
    for k in range(n_frames):
        base = pad + int(round(k * Ha)) - N // 2
        if prev is None:
            s = base
        else:
            nat = xp[prev + Hs: prev + Hs + N]
            region = xp[base - tol: base + tol + N]
            corr = np.correlate(region, nat, mode="valid")
            j0 = base - tol
            energy = c2[j0 + N: j0 + N + 2 * tol + 1] - c2[j0: j0 + 2 * tol + 1]
            nn = np.sqrt(np.sum(nat ** 2))
            ncc = corr / (np.sqrt(np.maximum(energy, 0)) * nn + 1e-12) if nn > 0 else np.zeros(2 * tol + 1)
            s = base - tol + int(np.argmax(ncc - bias))
        y[k * Hs: k * Hs + N] += win * xp[s: s + N]
        wsum[k * Hs: k * Hs + N] += win
        prev = s
    out = y / np.where(wsum > 1e-8, wsum, 1.0)
    return out[N // 2: N // 2 + n_out]


def frame_db(y: np.ndarray, sp: dict) -> np.ndarray:
    """Mean log-mel dB over the 128 bands per 10 ms frame (frame t centred on sample t x hop)."""
    return logmel(np.asarray(y, dtype=np.float32), sp).mean(axis=0).astype(np.float64)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as (start, end exclusive)."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def find_pauses(y: np.ndarray, sp: dict, thr: float) -> list[tuple[int, int]]:
    """Internal pauses as sample ranges [s, e). See DECISIONS.md."""
    m = frame_db(y, sp)
    T = len(m)
    k = np.ones(SMOOTH_FRAMES)
    sm = np.convolve(m, k, "same") / np.convolve(np.ones(T), k, "same")
    sm_pause = sm < thr
    sm_speech = np.flatnonzero(~sm_pause)
    if not len(sm_speech):
        return []
    first, last = sm_speech[0], sm_speech[-1]
    hop = sp["hop_length"]
    out = []
    for a, b in runs(m < thr):
        if not sm_pause[a:b].any() or a <= first or b - 1 >= last:   # must hold a smoothed pause, be internal
            continue
        out.append((max(0, a * hop - hop // 2), min(len(y), b * hop - hop // 2)))
    return out


def _xfade(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    t = (np.arange(len(a)) + 0.5) / max(len(a), 1)
    return a * np.cos(np.pi / 2 * t) + b * np.sin(np.pi / 2 * t)


def edit_pauses(y: np.ndarray, pauses: list[tuple[int, int]], factor: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    for s, e in reversed(pauses):                       # last first: earlier indices stay valid
        d = e - s
        L = int(round(abs(factor - 1.0) * d))
        f = min(XFADE_MAX, d // 4)
        if L == 0:
            continue
        c = (s + e) // 2
        if factor < 1:                                  # cut L samples out of the centre
            p = c - L // 2
            q = p + L
            y = np.concatenate([y[:p], _xfade(y[p:p + f], y[q:q + f]), y[q + f:]])
        else:                                           # repeat L samples of the centre
            p = c - (L + f) // 2
            q = p + L + f
            y = np.concatenate([y[:q - f], _xfade(y[q - f:q], y[p:p + f]), y[p + f:]])
    return y


def apply_edit(y: np.ndarray, name: str, pauses: list[tuple[int, int]]) -> np.ndarray:
    if name == "none":
        return np.asarray(y, dtype=np.float64)
    if name == "rate_roundtrip_0.85":
        return wsola(wsola(y, 0.85), 1 / 0.85)
    info = EDIT_INFO[name]
    if info["dial"] == "rate":
        return wsola(y, info["value"])
    if info["dial"] == "gain":
        return np.asarray(y, dtype=np.float64) * 10 ** (info["value"] / 20)
    return edit_pauses(y, pauses, info["value"])


# ------------------------------------------------------------------------------------------ context
class Ctx:
    def __init__(self, need_model: bool = True):
        tcfg = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
        self.tcfg = tcfg
        self.pcfg = load_config(REPO_ROOT / tcfg["preprocess_config"])
        self.sp = self.pcfg["spectrogram"]
        self.ms = int(self.pcfg["audio"]["max_seconds"] * self.sp["sr"])
        self.styles = self.pcfg["labels"]["styles"]
        self.thr = float(json.loads((EXPL_OUT / "resnet_explain_metrics.json").read_text(encoding="utf-8"))
                         ["energy_threshold_db"])
        assert abs(self.thr - (-55.58)) < 0.01, self.thr
        self.clips = pd.read_csv(EXPL_OUT / f"clips_fold{FOLD}.csv")
        self.manifest = pd.read_csv(self.pcfg["paths"]["processed_dir"] / "manifest.csv").set_index("clip_id")
        if not need_model:
            return
        splits = self.pcfg["paths"]["splits_dir"]
        norm = json.loads((splits / f"norm_stats_fold_{FOLD}.json").read_text(encoding="utf-8"))
        self.g_mean, self.g_std = float(norm["mean"]), float(norm["std"])
        self.device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        m = build_resnet18(tcfg["model"]["n_classes"], tcfg["model"]["in_channels"], pretrained=False)
        m.load_state_dict(torch.load(REPO_ROOT / "models" / RUN / "best.pt", map_location="cpu", weights_only=True))
        self.model = m.to(self.device).eval()
        self.predict = make_predict_fn(self.model, self.g_mean, self.g_std, self.device)
        ds = SpectrogramDataset(self.pcfg, FOLD, "test")
        rows = pd.Series(np.arange(len(ds.clip_ids)), index=ds.clip_ids).loc[self.clips["clip_id"]].to_numpy()
        self.X = ds.x[rows].astype(np.float32)
        self.n_valid = ds.n_valid[rows]

    def audio(self, clip_id: str) -> np.ndarray:
        rec = self.manifest.loc[clip_id]
        y = load_audio16k(self.pcfg, rec["audio16k_file"])
        assert len(y) == rec["original_n_samples_16k"], clip_id
        return y.astype(np.float64)

    def window(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Edited full clip -> (window audio without padding, log-mel [128, 400] as the cache, real frames)."""
        w = np.asarray(y[: self.ms], dtype=np.float32)
        x = logmel(np.pad(w, (0, self.ms - len(w))), self.sp)[:, :T_WIN].astype(np.float16).astype(np.float32)
        return w, x, min(n_real_frames(len(w), self.sp), T_WIN)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# --------------------------------------------------------------------------------------------- edit
def run_edits(c: Ctx) -> None:
    print(f"seed {SEED} (edits are deterministic; seed used for the listening and self-check draws)", flush=True)
    # WSOLA identity check on a real clip
    y0 = c.audio(c.clips.clip_id[0])
    err = float(np.max(np.abs(wsola(y0, 1.0) - y0)))
    print(f"WSOLA at r = 1: max abs difference to the input {err:.2e}", flush=True)
    assert err < 1e-9

    names = ["none"] + EDIT_NAMES + CONTROLS
    rows, Xs = [], {n: [] for n in names}
    for i, r in c.clips.iterrows():
        y = c.audio(r.clip_id)
        pauses = find_pauses(y, c.sp, c.thr)
        p_in = sum(max(0, min(e, c.ms) - s) for s, e in pauses if s < c.ms)
        w0, x0, nr0 = c.window(y)
        assert np.array_equal(x0, c.X[i]), f"{r.clip_id}: rebuilt spectrogram differs from the cache"
        assert nr0 == c.n_valid[i]
        for n in names:
            ye = apply_edit(y, n, pauses)
            w, x, nr = c.window(ye)
            Xs[n].append(x)
            rows.append({"clip_id": r.clip_id, "style": r["style"], "y_true": int(r.label), "edit": n,
                         "n_samples_full": len(ye), "n_real_before": nr0, "n_real_after": nr,
                         "window_peak": float(np.max(np.abs(w))), "window_changed": not np.array_equal(x, x0),
                         "n_pauses": len(pauses), "pause_s_total": sum(e - s for s, e in pauses) / c.sp["sr"],
                         "pause_s_in_window": p_in / c.sp["sr"]})
        print(f"  edited {i + 1}/{len(c.clips)}", flush=True)
    df = pd.DataFrame(rows)
    probs = {}
    for n in names:
        probs[n] = softmax(c.predict(np.stack(Xs[n])).astype(np.float64))
    saved = pd.read_csv(EXPL_OUT / "resnet_segment_scores.csv").drop_duplicates("clip_id").set_index("clip_id")
    p_none = probs["none"]
    assert (p_none.argmax(1) == saved.loc[c.clips.clip_id, "y_pred"].to_numpy()).all(), \
        "unedited predictions differ from the explainers' saved predictions"
    print("checks: rebuilt spectrograms = cache (100/100); unedited predictions = saved explainer predictions",
          flush=True)
    y_orig = p_none.argmax(1)
    out = []
    for n in names:
        P = probs[n]
        sub = df[df.edit == n].reset_index(drop=True)
        sub["y_pred"] = P.argmax(1)
        sub["pred_style"] = [c.styles[k] for k in sub["y_pred"]]
        sub["y_pred_orig"] = y_orig
        sub["p_orig_class"] = P[np.arange(len(P)), y_orig]
        sub["p_orig_class_before"] = p_none[np.arange(len(P)), y_orig]
        sub["p_true"] = P[np.arange(len(P)), sub["y_true"]]
        for k, st in enumerate(c.styles):
            sub[f"p_{st}"] = P[:, k]
        out.append(sub)
    res = pd.concat(out, ignore_index=True)
    res["flipped"] = res["y_pred"] != res["y_pred_orig"]
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "cf_predictions.csv", index=False, float_format="%.6g")
    pd.DataFrame([{"clip_id": cid, "start_s": s / c.sp["sr"], "end_s": e / c.sp["sr"]}
                  for cid in c.clips.clip_id for s, e in find_pauses(c.audio(cid), c.sp, c.thr)]) \
        .to_csv(OUT / "cf_pauses.csv", index=False, float_format="%.4f")
    print(f"wrote {OUT / 'cf_predictions.csv'}", flush=True)


def run_diag(c: Ctx) -> None:
    """ADDED AFTER SEEING THE RESULTS (the round trip flipped 42 of 100 labels): what the processing changes.
    (a) Round trip x0.85 -> x1/0.85 vs unedited, same length, per bin of the window's real frames: mean dB change
    in speech / pause frames (unsmoothed frame mean vs the energy threshold), in the lowest and highest 32 mel
    bands, and the mean absolute change. (b) Each edit: mean dB of the window's real frames, edited minus unedited
    (an overall level change; for rate and pause the frames are not aligned, so only this mean is compared)."""
    rows = []
    for cid in c.clips.clip_id:
        y = c.audio(cid)
        pauses = find_pauses(y, c.sp, c.thr)
        _, x0, n0 = c.window(y)
        _, xr, nr = c.window(apply_edit(y, "rate_roundtrip_0.85", pauses))
        n = min(n0, nr)
        a, b = x0[:, :n], xr[:, :n]
        d = b - a
        sp = a.mean(0) >= c.thr
        rec = {"clip_id": cid, "rt_speech_frames_db": float(d[:, sp].mean()) if sp.any() else np.nan,
               "rt_pause_frames_db": float(d[:, ~sp].mean()) if (~sp).any() else np.nan,
               "rt_low_bands_db": float(d[:32].mean()), "rt_high_bands_db": float(d[96:].mean()),
               "rt_abs_db": float(np.abs(d).mean())}
        for e in EDIT_NAMES:
            _, xe, ne = c.window(apply_edit(y, e, pauses))
            rec[f"level_{e}"] = float(xe[:, :ne].mean() - x0[:, :n0].mean())
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "cf_processing_diagnostic.csv", index=False, float_format="%.4f")
    print(f"wrote {OUT / 'cf_processing_diagnostic.csv'}", flush=True)


# ------------------------------------------------------------------------------------------- report
def short(s: str) -> str:
    return {"confused": "conf", "default": "def", "enunciated": "enun", "happy": "hap", "laughing": "laugh",
            "sad": "sad", "whisper": "whis"}[s]


def summarise(c: Ctx, res: pd.DataFrame) -> dict:
    E = res[res.edit.isin(EDIT_NAMES)].copy()
    base = res[res.edit == "none"].set_index("clip_id")
    E["dial"] = E.edit.map(lambda n: EDIT_INFO[n]["dial"])
    E["direction"] = E.edit.map(lambda n: EDIT_INFO[n]["direction"])
    E["size"] = E.edit.map(lambda n: EDIT_INFO[n]["size"])
    E["correct"] = E.clip_id.map(base["y_pred"] == base["y_true"])
    E["abs_dp"] = (E.p_orig_class - E.p_orig_class_before).abs()

    # 1. per clip x dial: labels by step, smallest flip per direction, non-monotonic
    per_cd = []
    for (cid, dial), g in E.groupby(["clip_id", "dial"], sort=False):
        rec = {"clip_id": cid, "style": base.loc[cid, "style"], "pred_unedited": base.loc[cid, "pred_style"],
               "dial": dial}
        for _, r in g.sort_values(["size", "direction"]).iterrows():
            rec[r.edit] = r.pred_style
        smallest = []
        for d in ("down", "up"):
            gd = g[g.direction == d].sort_values("size")
            fl = gd[gd.flipped]
            rec[f"first_flip_{d}"] = fl.edit.iloc[0] if len(fl) else ""
            rec[f"nonmonotonic_{d}"] = bool(len(fl) and not gd.flipped.iloc[-1])
            if len(fl):
                smallest.append((int(fl["size"].iloc[0]), fl.edit.iloc[0]))
        m = min((s for s, _ in smallest), default=None)
        rec["smallest_flip"] = " / ".join(e for s, e in smallest if s == m) if m else "none"
        per_cd.append(rec)
    per_cd = pd.DataFrame(per_cd)

    # per edit summary
    per_edit = []
    for n in EDIT_NAMES:
        g = E[E.edit == n]
        fully = (g.n_real_before == T_WIN) & (g.n_real_after == T_WIN)
        per_edit.append({"edit": n, "n": len(g), "flipped": int(g.flipped.sum()),
                         "flipped_correct": int(g[g.correct].flipped.sum()), "n_correct": int(g.correct.sum()),
                         "flipped_wrong": int(g[~g.correct].flipped.sum()), "n_wrong": int((~g.correct).sum()),
                         "to_true_style": int(((g.y_pred == g.y_true) & (g.y_pred_orig != g.y_true)).sum()),
                         "correct_to_wrong": int((g.correct & (g.y_pred != g.y_true)).sum()),
                         "abs_dp_median": float(g.abs_dp.median()), "abs_dp_max": float(g.abs_dp.max()),
                         "window_unchanged": int((~g.window_changed).sum()),
                         "fully_real_n": int(fully.sum()), "fully_real_flipped": int(g[fully].flipped.sum()),
                         "padding_changed_n": int((g.n_real_before != g.n_real_after).sum()),
                         "padding_changed_flipped": int(g[g.n_real_before != g.n_real_after].flipped.sum()),
                         "accuracy": float((g.y_pred == g.y_true).mean())})
    per_edit = pd.DataFrame(per_edit)

    # 2. wrong clips: can a single edit reach the true style?
    wrong = []
    for cid, g in E[~E.correct].groupby("clip_id", sort=False):
        hit = g[g.y_pred == g.y_true].edit.tolist()
        wrong.append({"clip_id": cid, "style": base.loc[cid, "style"], "pred_unedited": base.loc[cid, "pred_style"],
                      "reached_true_style": bool(hit), "edits_reaching_true": " ".join(hit),
                      "labels_reached": " ".join(sorted(set(g.pred_style) - {base.loc[cid, "pred_style"]}))})
    wrong = pd.DataFrame(wrong)

    # 3. per style: flips (by true style) and reachability (as a target)
    per_style = []
    any_flip = E.groupby("clip_id").flipped.any()
    for st in c.styles:
        ids = base[base["style"] == st].index
        g = E[E.clip_id.isin(ids)]
        into = E[(E.pred_style == st) & (E.y_pred_orig != c.styles.index(st))]
        into_cl = into.drop_duplicates("clip_id")
        per_style.append({"style": st, "clips": len(ids), "flip_any_edit": int(any_flip.loc[ids].sum()),
                          **{f"flip_{d}": int(g[g.dial == d].groupby("clip_id").flipped.any().sum()) for d in DIALS},
                          "pred_unedited": int((base.pred_style == st).sum()),
                          "reached_from_other_clips": int(len(into_cl)),
                          "reached_by": " ".join(f"{k}:{v}" for k, v in into.dial.value_counts().items()),
                          "reached_from_styles": " ".join(f"{k}:{v}" for k, v in
                                                         into_cl.clip_id.map(base.pred_style).value_counts().items()),
                          "reachable": bool(len(into_cl))})
    per_style = pd.DataFrame(per_style)

    # transitions (from -> to) over all edits
    fl = E[E.flipped]
    trans = (fl.assign(frm=fl.y_pred_orig.map(lambda k: c.styles[k])).groupby(["dial", "frm", "pred_style"])
             .size().rename("n").reset_index().sort_values(["dial", "n"], ascending=[True, False]))

    # loudness wording
    g6 = per_edit.set_index("edit").loc[["gain_-6dB", "gain_+6dB"]]
    loud_small = bool((g6.flipped < LOUD_FLIPS_MAX).all() and (g6.abs_dp_median < LOUD_DP_MAX).all())

    # controls
    ctrl = res[res.edit.isin(CONTROLS)]
    rt = ctrl[ctrl.edit == "rate_roundtrip_0.85"].set_index("clip_id")
    rate_fl = E[(E.dial == "rate") & E.flipped]
    rate_flip_rt = rate_fl[rate_fl.clip_id.map(rt.flipped)]

    # 4. metrics per edit
    metrics = {"unedited": classification_metrics(base.y_true.to_numpy(), base.y_pred.to_numpy(), c.styles)}
    for n in EDIT_NAMES:
        g = E[E.edit == n].set_index("clip_id").loc[base.index]
        metrics[n] = classification_metrics(g.y_true.to_numpy(), g.y_pred.to_numpy(), c.styles)
    return {"E": E, "base": base, "per_cd": per_cd, "per_edit": per_edit, "wrong": wrong, "per_style": per_style,
            "trans": trans, "loud_small": loud_small, "rt": rt, "rate_flip_rt": rate_flip_rt, "metrics": metrics}


def plot_confusions(c: Ctx, metrics: dict) -> None:
    keys = ["unedited"] + EDIT_NAMES
    fig, axes = plt.subplots(4, 3, figsize=(17, 22))
    for ax, k in zip(axes.flat, keys):
        m = metrics[k]
        plot_confusion(ax, np.asarray(m["confusion_counts"]), c.styles,
                       f"{k}: accuracy {m['accuracy']:.0%}, macro-F1 {m['macro_f1']:.3f}")
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_unedited_vs_edits.png", dpi=110, metadata=PNG_META)
    plt.close(fig)


def write_report(c: Ctx, s: dict) -> None:
    E, base, pe = s["E"], s["base"], s["per_edit"].set_index("edit")
    n_ok = int((base.y_pred == base.y_true).sum())
    s["per_cd"].to_csv(OUT / "cf_per_clip_dial.csv", index=False)
    s["per_edit"].to_csv(OUT / "cf_per_edit.csv", index=False, float_format="%.4f")
    s["wrong"].to_csv(OUT / "cf_wrong_clips.csv", index=False)
    s["per_style"].to_csv(OUT / "cf_per_style.csv", index=False)
    s["trans"].to_csv(OUT / "cf_transitions.csv", index=False)
    rows_pc = []
    for k, m in s["metrics"].items():
        for st in c.styles:
            rows_pc.append({"version": k, "style": st, **m["per_class"][st]})
    pd.DataFrame(rows_pc).to_csv(OUT / "cf_per_style_metrics.csv", index=False, float_format="%.4f")
    (OUT / "cf_metrics.json").write_text(json.dumps(
        {"seed": SEED, "energy_threshold_db": c.thr, "loudness_almost_no_effect": s["loud_small"],
         "per_edit": s["per_edit"].to_dict("records"), "metrics": s["metrics"]}, indent=2), encoding="utf-8")
    plot_confusions(c, s["metrics"])

    pauses = base[["n_pauses", "pause_s_total", "pause_s_in_window"]]
    no_pause = int((pauses.n_pauses == 0).sum())
    no_pause_win = int((pauses.pause_s_in_window == 0).sum())
    L = [f"# Counterfactual edits: ResNet-18 (fold {FOLD}, seed 1, pretrained), 100 clips of ex02", "",
         "## Setup", "",
         f"- Model `models/{RUN}/best.pt`, float32, eval mode, no autocast; the 100 explainer clips "
         f"(`clips_fold{FOLD}.csv`, test actor ex02, {n_ok} predicted correctly); fold-1 train-only normalisation. "
         "No training.",
         "- Every edit is applied to the **full** 16 kHz clip; then the cache rule (first 4 s, zero padding to 4 s, "
         "same log-mel, float16). Checks passed: the rebuilt unedited spectrograms equal the cache bit for bit, and "
         "the unedited predictions equal the explainers' saved predictions for all 100 clips. WSOLA at rate 1 "
         "returns its input exactly.",
         "- Dials and steps were fixed in `DECISIONS.md` before anything was run and were not changed: speaking "
         "rate x0.90, x1.10, x0.85, x1.15 (WSOLA, pitch kept; x0.85 = slower); loudness -3, +3, -6, +6 dB; "
         "internal pause length x0.5, x1.5. 'Flip' = the predicted style differs from the unedited prediction.",
         f"- Pauses: the same energy rule as the speech/pause labels (250 ms sliding mean of the frame dB below "
         f"{c.thr:.2f} dB), internal pauses only. {100 - no_pause} of 100 clips have at least one internal pause "
         f"(median {pauses.n_pauses.median():.0f} per clip, median {pauses.pause_s_total.median():.2f} s in total); "
         f"**{no_pause} clips have none** and {no_pause_win} have none inside the 4 s window, so the pause dial "
         "leaves them (or their window) unchanged. These clips are counted as 'not flipped'; the tables give how many "
         "windows were actually changed.",
         f"- Seed {SEED} for every random draw (listening selection, A/B order, file ids, self-check clips); the "
         "edits themselves are deterministic.", ""]

    # ---- per edit
    L += ["## How often each edit changes the label", "",
          "| edit | windows changed | flipped (of 100) | of correct | of wrong | wrong -> true style | correct -> wrong "
          "| median abs. change of p(unedited class) | max | accuracy after |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    acc0 = s["metrics"]["unedited"]["accuracy"]
    for r in pe.itertuples():
        n = r.Index
        L.append(f"| {n} | {100 - r.window_unchanged} | **{r.flipped}** | {r.flipped_correct}/{r.n_correct} | "
                 f"{r.flipped_wrong}/{r.n_wrong} | {r.to_true_style} | {r.correct_to_wrong} | {r.abs_dp_median:.3f} | "
                 f"{r.abs_dp_max:.3f} | {r.accuracy:.0%} |")
    L += ["", f"Unedited accuracy on these clips: {acc0:.0%}.", ""]
    loud = pe.loc[["gain_-3dB", "gain_+3dB", "gain_-6dB", "gain_+6dB"]]
    if s["loud_small"]:
        L += [f"**Loudness has almost no effect** (fixed rule: +-6 dB flips fewer than {LOUD_FLIPS_MAX} of 100 clips "
              f"in each direction and the median probability change is below {LOUD_DP_MAX}): flips at -3/+3/-6/+6 dB "
              f"= {'/'.join(str(v) for v in loud.flipped)}.", ""]
    else:
        L += [f"**Loudness is not 'almost no effect'** by the fixed rule (+-6 dB must flip fewer than "
              f"{LOUD_FLIPS_MAX} of 100 clips each way with median probability change below {LOUD_DP_MAX}): flips at "
              f"-3/+3/-6/+6 dB = {'/'.join(str(v) for v in loud.flipped)}, median changes "
              f"{'/'.join(f'{v:.3f}' for v in loud.abs_dp_median)}.", ""]

    # ---- smallest step per dial
    pcd = s["per_cd"]
    L += ["## Smallest step at which the label changes (per clip and dial)", "",
          "| dial | flips at the smaller step | only at the larger step | no flip at any step | down only | up only "
          "| both directions | non-monotonic (flip at small step, not at large) |",
          "|---|---|---|---|---|---|---|---|"]
    for d in DIALS:
        g = pcd[pcd.dial == d]
        sizes = g.smallest_flip.map(lambda v: "none" if v == "none" else EDIT_INFO[v.split(" / ")[0]]["size"])
        dn, up = g.first_flip_down != "", g.first_flip_up != ""
        nm = int((g.nonmonotonic_down | g.nonmonotonic_up).sum())
        larger = "n/a (one step)" if d == "pause" else str(int((sizes == 2).sum()))
        L.append(f"| {d} | {int((sizes == 1).sum())} | {larger} | {int((sizes == 'none').sum())} | "
                 f"{int((dn & ~up).sum())} | {int((up & ~dn).sum())} | {int((dn & up).sum())} | {nm} |")
    L += ["", "Pause has one step per direction, so its 'smaller step' is that step.", ""]

    # ---- padding confound
    L += ["## Side effect: the edit also changes how much of the 4 s window is padding", "",
          "Rate and pause edits change the clip's length, so a short clip gains or loses padding (a possible shortcut, "
          "see the observations in DECISIONS.md); a long clip keeps a fully real window but shows a different stretch "
          "of the sentence. Flips split by that:", "",
          "| edit | window fully real before and after: flipped / clips | padding amount changed: flipped / clips |",
          "|---|---|---|"]
    for r in pe.itertuples():
        n = r.Index
        L.append(f"| {n} | {r.fully_real_flipped} / {r.fully_real_n} | {r.padding_changed_flipped} / "
                 f"{r.padding_changed_n} |")
    rt = s["rt"]
    rfr = s["rate_flip_rt"]
    L += ["", f"**Rate artefact control** (x0.85 then x1/0.85, net rate 1): the label changes for "
          f"**{int(rt.flipped.sum())} of 100** clips (median abs. change of p(unedited class) "
          f"{float((rt.p_orig_class - rt.p_orig_class_before).abs().median()):.3f}). Of the "
          f"{int(E[E.edit.str.startswith('rate')].flipped.sum())} rate flips (clip x step), {len(rfr)} are on clips that also "
          "flip under the round trip, so for those the WSOLA processing alone may be enough to move the label.", ""]
    n_rate_clips = int(E[E.dial == "rate"].groupby("clip_id").flipped.any().sum())
    n_rate_only = int(E[E.dial == "rate"].groupby("clip_id").flipped.any().loc[~rt.flipped].sum())
    L += [f"Clips with a rate flip at any step: {n_rate_clips}; of these, {n_rate_only} do **not** flip under the round "
          "trip.", ""]
    dg_path = OUT / "cf_processing_diagnostic.csv"
    if dg_path.exists():
        dg = pd.read_csv(dg_path).median(numeric_only=True)
        L += ["**What the processing changes (diagnostic added after seeing the round-trip result; medians over the "
              "100 clips, `cf_processing_diagnostic.csv`).** The round trip, compared bin by bin with the unedited "
              f"window: speech frames {dg.rt_speech_frames_db:+.2f} dB, pause frames {dg.rt_pause_frames_db:+.2f} dB, "
              f"lowest 32 mel bands {dg.rt_low_bands_db:+.2f} dB, highest 32 {dg.rt_high_bands_db:+.2f} dB, mean "
              f"absolute change per bin {dg.rt_abs_db:.2f} dB. WSOLA overlap-adds shifted frames, which lowers the "
              "level of noise-like parts (room tone, breath, fricatives) and smears fine structure, so the processed "
              "clip is also slightly quieter and duller. Overall level change of each edit (mean dB of the window's "
              "real frames, edited minus unedited):", "",
              "| edit | " + " | ".join(EDIT_NAMES) + " |", "|---|" + "---|" * len(EDIT_NAMES),
              "| median dB | " + " | ".join(f"{dg[f'level_{e}']:+.2f}" for e in EDIT_NAMES) + " |", "",
              "A single rate pass therefore also acts as a small loudness and texture edit, and the model reacts "
              "to loudness (gain table above). The rate flips cannot be read as an effect of speaking rate alone.", ""]

    # ---- wrong clips
    w = s["wrong"]
    L += ["## Wrong clips: does any single edit give the true style?", "",
          f"**{int(w.reached_true_style.sum())} of {len(w)}** wrong clips reach their true style with at least one of "
          "the 8 single edits.", "",
          "| clip | true style | unedited prediction | edits that give the true style | other labels reached |",
          "|---|---|---|---|---|"]
    for _, r in w.iterrows():
        L.append(f"| {r.clip_id} | {r['style']} | {r.pred_unedited} | {r.edits_reaching_true or '-'} | "
                 f"{r.labels_reached or '-'} |")
    L.append("")
    by_dial = {d: int(w.edits_reaching_true.map(lambda v: any(EDIT_INFO[e]['dial'] == d for e in v.split())).sum())
               for d in DIALS}
    L += ["Wrong clips reaching the true style by dial: " + ", ".join(f"{d} {v}" for d, v in by_dial.items()) + ".", ""]

    # ---- per style
    ps = s["per_style"]
    L += ["## Per style", "", "Flips by the clip's **true** style (a clip counts once per dial if any step flips it), "
          "and reachability as a **target**: how many clips not predicted as that style before are predicted as it "
          "after some edit.", "",
          "| style | clips | flipped by any edit | by rate | by gain | by pause | predicted unedited | reached from other "
          "clips | reached by (edit count per dial) | those clips' unedited prediction |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in ps.iterrows():
        L.append(f"| {r['style']} | {r.clips} | {r.flip_any_edit} | {r.flip_rate} | {r.flip_gain} | {r.flip_pause} | "
                 f"{r.pred_unedited} | **{r.reached_from_other_clips}** | {r.reached_by or '-'} | "
                 f"{r.reached_from_styles or '-'} |")
    unreach = ps[~ps.reachable]["style"].tolist()
    L += ["", f"**Styles that no edit reaches: {', '.join(unreach) if unreach else 'none'}.**", ""]
    tr = s["trans"]
    L += ["Label changes (from unedited prediction to edited prediction), counted over clip x edit:", "",
          "| dial | from | to | count |", "|---|---|---|---|"]
    for _, r in tr.iterrows():
        L.append(f"| {r.dial} | {r.frm} | {r.pred_style} | {r.n} |")
    L.append("")

    # ---- 4. confusion + per-style metrics
    L += ["## Confusion matrices and per-style metrics: unedited vs each edit (CLAUDE.md rule)", "",
          "Rows = true style, columns = predicted style; each box the share of its row and the clip count. "
          "100 test clips of ex02 (the explainer subset, not the whole test actor).", "",
          "![Confusion matrices](confusion_unedited_vs_edits.png)", ""]
    for k, m in s["metrics"].items():
        pc = m["per_class"]
        L += [f"**{k}**: accuracy {m['accuracy']:.0%}, macro-F1 {m['macro_f1']:.3f}", "",
              "| style | clips | precision | recall | F1 |", "|---|---|---|---|---|"]
        for st in c.styles:
            L.append(f"| {st} | {pc[st]['support']} | {pc[st]['precision']:.2f} | {pc[st]['recall']:.2f} | "
                     f"{pc[st]['f1']:.2f} |")
        L += [f"| **macro average** | {sum(pc[st]['support'] for st in c.styles)} | "
              f"**{np.mean([pc[st]['precision'] for st in c.styles]):.2f}** | "
              f"**{np.mean([pc[st]['recall'] for st in c.styles]):.2f}** | **{m['macro_f1']:.2f}** |", ""]

    # ---- per clip table
    L += ["## Every clip, every dial, every step (predicted style)", "",
          "Columns in order of step size per dial; **bold** = differs from the unedited prediction. "
          "conf = confused, def = default, enun = enunciated, hap = happy, laugh = laughing, whis = whisper.", "",
          "| clip | true | unedited | " + " | ".join(EDIT_NAMES) + " | smallest flip: rate / gain / pause |",
          "|---|---|---|" + "---|" * len(EDIT_NAMES) + "---|"]
    piv = E.pivot(index="clip_id", columns="edit", values="pred_style").loc[base.index]
    sm = pcd.pivot(index="clip_id", columns="dial", values="smallest_flip").loc[base.index]
    for cid in base.index:
        u = base.loc[cid, "pred_style"]
        cells = [f"**{short(piv.loc[cid, n])}**" if piv.loc[cid, n] != u else short(piv.loc[cid, n])
                 for n in EDIT_NAMES]
        L.append(f"| {cid.replace('ex02_', '')} | {short(base.loc[cid, 'style'])} | {short(u)} | " + " | ".join(cells)
                 + f" | {' / '.join(sm.loc[cid, d] for d in DIALS)} |")
    L.append("")

    # ---- listening material and self-check
    key_path = OUT / "listening" / "listening_key.csv"
    if key_path.exists():
        key = pd.read_csv(key_path)
        sc = pd.read_csv(OUT / "self_check_clips.csv")
        L += ["## Listening-test material and self-check clips", "",
              f"- `listening/`: {len(key)} pairs (unedited and edited window of the same clip, 20 different clips), "
              f"{int(key.model_flipped.sum())} where the model's label flipped and {int((~key.model_flipped).sum())} "
              f"where it did not; dials {', '.join(f'{k} {v}' for k, v in key.dial.value_counts().items())}. Random "
              "pair order, random A/B order, random 8-hex file ids (seed 42). Files: `audio/*.wav` (not in git), "
              "`instructions_for_listeners.md`, `answer_sheet_template.csv` (hear a difference? style of A and B), "
              "`listening_key.csv` (**not for listeners**: which file is edited, the edit, the model's labels).",
              "- Scoring: `python -m src.explain.score_listening <filled sheets> [--out scores.json]` (main score: "
              "the listener's style changes between A and B exactly when the model's label flipped).",
              "- `self_check/` (not in git): 6 edited clips next to their unedited window, one per editing method at "
              "its largest step, to judge the editing quality by ear (`self_check_clips.csv`):", "",
              "| edit | clip | model unedited -> edited |", "|---|---|---|"]
        for r in sc.itertuples():
            L.append(f"| {r.edit} | {r.clip_id} | {r.model_unedited} -> {r.model_edited} |")
        L.append("")

    # ---- limits
    L += ["## Limits", "",
          "- 100 clips of one actor (ex02), one model (fold 1, seed 1); small counts per style (about 14).",
          "- An edit is never perfectly isolated: rate and pause edits change how much of the sentence and of the "
          "padding sits in the 4 s window (table above); WSOLA also lowers the level and changes the texture, and its "
          "round trip alone flips many labels (control above), so the rate dial is confounded with processing; "
          "pause edits add crossfades inside pauses. A flip shows the model is sensitive to the edited audio, not "
          "that it 'uses speaking rate' as a listener would.",
          "- Speaking rate is changed uniformly over the clip (including pauses); a real slower reading would not "
          "stretch everything equally.",
          "- The pause dial only touches internal pauses long and quiet enough to pass the 250 ms energy rule; "
          f"{no_pause} clips have none and are unchanged. Leading and trailing silence are never edited.",
          "- Loudness: the model input is not clipped (float), so +6 dB is an exact shift of the log-mel; a real "
          "louder recording would also change voice quality.",
          "- Only single edits and only the fixed steps; combinations and larger ranges were not tried (by design).",
          "- Whether listeners hear the same change is the separate listening test (`listening/`), not measured "
          "here.", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT / 'report.md'}", flush=True)


# -------------------------------------------------------------------------------------------- audio
def write_wav(path, w: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert np.max(np.abs(w)) < 1.0, path
    sf.write(path, np.asarray(w, dtype=np.float32), 16000, subtype="PCM_24")


def make_audio(c: Ctx, res: pd.DataFrame) -> None:
    print(f"seed {SEED} for the listening selection, order, A/B and file ids, and the self-check draw", flush=True)
    rng = np.random.default_rng(SEED)
    E = res[res.edit.isin(EDIT_NAMES)].copy()
    E["dial"] = E.edit.map(lambda n: EDIT_INFO[n]["dial"])
    elig = E[E.window_changed & (E.window_peak < PEAK_MAX)]
    print(f"eligible (clip, edit) items: {len(elig)} of {len(E)} (excluded: window unchanged "
          f"{int((~E.window_changed).sum())}, peak >= {PEAK_MAX} {int((E.window_changed & (E.window_peak >= PEAK_MAX)).sum())})",
          flush=True)
    used, chosen = set(), []
    for grp in (True, False):                              # flipped first (smaller pool)
        pool_g = elig[elig.flipped == grp]
        for i in range(N_PAIRS_PER_GROUP):
            dial = ["rate", "pause", "gain"][i % 3]
            pool = pool_g[(pool_g.dial == dial) & ~pool_g.clip_id.isin(used)]
            fallback = False
            if not len(pool):
                pool, fallback = pool_g[~pool_g.clip_id.isin(used)], True
            r = pool.iloc[int(rng.integers(len(pool)))]
            used.add(r.clip_id)
            chosen.append({**r.to_dict(), "dial_planned": dial, "dial_fallback": fallback})
    order = rng.permutation(len(chosen))
    ids = set()
    while len(ids) < 2 * len(chosen):
        ids.add(f"{int(rng.integers(16 ** 8)):08x}")
    ids = list(ids)
    rng.shuffle(ids)
    LST = OUT / "listening"
    key = []
    for pair, k in enumerate(order, start=1):
        r = chosen[k]
        y = c.audio(r["clip_id"])
        pauses = find_pauses(y, c.sp, c.thr)
        w0 = c.window(y)[0]
        we = c.window(apply_edit(y, r["edit"], pauses))[0]
        orig_is_a = bool(rng.integers(2))
        fa, fb = ids[2 * (pair - 1)], ids[2 * (pair - 1) + 1]
        write_wav(LST / "audio" / f"{fa}.wav", w0 if orig_is_a else we)
        write_wav(LST / "audio" / f"{fb}.wav", we if orig_is_a else w0)
        key.append({"pair": pair, "file_A": f"{fa}.wav", "file_B": f"{fb}.wav", "original_is": "A" if orig_is_a else "B",
                    "clip_id": r["clip_id"], "true_style": r["style"], "edit": r["edit"], "dial": r["dial"],
                    "dial_fallback": r["dial_fallback"],
                    "model_label_original": c.styles[int(r["y_pred_orig"])], "model_label_edited": r["pred_style"],
                    "model_flipped": bool(r["flipped"]), "p_orig_class_before": r["p_orig_class_before"],
                    "p_orig_class_after": r["p_orig_class"], "duration_original_s": len(w0) / 16000,
                    "duration_edited_s": len(we) / 16000})
    key = pd.DataFrame(key)
    key.to_csv(LST / "listening_key.csv", index=False, float_format="%.4f")
    sheet = key[["pair", "file_A", "file_B"]].assign(listener_id="", hear_difference="", style_A="", style_B="")
    sheet[["listener_id", "pair", "file_A", "file_B", "hear_difference", "style_A", "style_B"]].to_csv(
        LST / "answer_sheet_template.csv", index=False)
    (LST / "instructions_for_listeners.md").write_text("\n".join([
        "# Listening test", "",
        "You will hear 20 pairs of short speech recordings (A and B), each up to 4 seconds. Play A, then B, as often "
        "as you like. Use headphones and keep the volume the same for the whole test.", "",
        "For every pair, fill in one row of the answer sheet (`answer_sheet_template.csv`, one copy per listener):", "",
        "- `listener_id`: your initials or a number (the same on every row).",
        "- `hear_difference`: `yes` if A and B sound different in any way, otherwise `no`.",
        "- `style_A`, `style_B`: the speaking style you hear in A and in B, one of: confused, default, enunciated, "
        "happy, laughing, sad, whisper; or `unsure`.", "",
        "There are no right or wrong answers. Do not compare notes with other listeners.", ""]), encoding="utf-8")
    print(f"listening: {len(key)} pairs, {int(key.model_flipped.sum())} flipped; dials "
          f"{key.dial.value_counts().to_dict()}; fallbacks {int(key.dial_fallback.sum())}", flush=True)

    # self-check: 6 edited clips + their unedited windows
    rng2 = np.random.default_rng([SEED, 6])
    base = res[res.edit == "none"].set_index("clip_id")
    picks, styles_used = {}, set()
    for n in ("pause_x0.5", "pause_x1.5"):
        cand = base[~base.index.isin(picks.values()) & ~base["style"].isin(styles_used)] \
            .sort_values(["pause_s_in_window", "pause_s_total"], ascending=False)
        picks[n] = cand.index[0]
        styles_used.add(base.loc[picks[n], "style"])
    for n in ("rate_x0.85", "rate_x1.15", "gain_-6dB", "gain_+6dB"):
        cand = base[~base.index.isin(picks.values()) & ~base["style"].isin(styles_used)]
        if n == "gain_+6dB":
            ok = res[(res.edit == n) & (res.window_peak < PEAK_MAX)].clip_id
            cand = cand[cand.index.isin(ok)]
        picks[n] = cand.index[int(rng2.integers(len(cand)))]
        styles_used.add(base.loc[picks[n], "style"])
    SC = OUT / "self_check"
    rows = []
    for n in SELF_CHECK:
        cid = picks[n]
        y = c.audio(cid)
        pauses = find_pauses(y, c.sp, c.thr)
        write_wav(SC / f"{cid}_original.wav", c.window(y)[0])
        write_wav(SC / f"{cid}_{n}.wav", c.window(apply_edit(y, n, pauses))[0])
        r = res[(res.clip_id == cid) & (res.edit == n)].iloc[0]
        rows.append({"edit": n, "clip_id": cid, "style": base.loc[cid, "style"],
                     "model_unedited": base.loc[cid, "pred_style"], "model_edited": r.pred_style,
                     "flipped": bool(r.flipped), "n_internal_pauses": int(base.loc[cid, "n_pauses"]),
                     "pause_s_in_window": float(base.loc[cid, "pause_s_in_window"]),
                     "files": f"{cid}_original.wav, {cid}_{n}.wav"})
    pd.DataFrame(rows).to_csv(OUT / "self_check_clips.csv", index=False, float_format="%.3f")
    print(f"self-check: {SC} ({len(rows)} edited + {len(rows)} unedited)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["report"], choices=["edit", "diag", "report", "audio"])
    args = ap.parse_args()
    c = Ctx(need_model="edit" in args.parts)
    if "edit" in args.parts:
        run_edits(c)
    if "diag" in args.parts:
        run_diag(c)
    res = pd.read_csv(OUT / "cf_predictions.csv")
    if "report" in args.parts:
        write_report(c, summarise(c, res))
    if "audio" in args.parts:
        make_audio(c, res)


if __name__ == "__main__":
    main()

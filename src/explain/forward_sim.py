"""Human forward-simulation study for the ResNet-18 (fold 1, seed 1): explainer rule, clips and materials.

    python -m src.explain.forward_sim

1. Explainer rule (fixed before the study, DECISIONS.md): best average rank of deletion gain (room-noise fill) and
   stability, from the saved results. Prints the table; stops unless the rule gives LIME.
2. 20 of the 100 fold-1 clips (seed 42) in two sets of 10 with the same right/wrong mix and a similar style mix.
3. Materials in reports/forward_sim_fold1/: participant_bundle/ (study.html, audio/, img/; audio and pictures are
   not in git), key.csv (not for participants), items.csv, selection.md.
No model is run: predictions and LIME scores come from reports/explain_fold1/resnet_segment_scores.csv.
"""
from __future__ import annotations

import json
import sys

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import yaml  # noqa: E402

from src.data.audio import load_audio16k  # noqa: E402
from src.data.torch_dataset import SpectrogramDataset  # noqa: E402
from src.utils import REPO_ROOT, load_config  # noqa: E402

SEED = 42
FOLD = 1
EXPL = REPO_ROOT / "reports" / f"explain_fold{FOLD}"
OUT = REPO_ROOT / "reports" / f"forward_sim_fold{FOLD}"
BUNDLE = OUT / "participant_bundle"
N_PER_SET, N_RIGHT_PER_SET = 10, 6
TOP_K = 3
COLS = [f"s{j:02d}" for j in range(16)]
STAB_TESTS = ["repeats", "noise_40db", "gain_plus_1db", "gain_minus_1db"]   # pre-planned; side checks left out
METHODS = [("GradCAM", "none"), ("IG", "room_noise"), ("LIME", "room_noise")]


# ---------------------------------------------------------------------------------------- 1. the rule
def explainer_rule() -> tuple[str, pd.DataFrame, pd.DataFrame]:
    dm = json.loads((EXPL / "deletion_metrics.json").read_text(encoding="utf-8"))
    sm = json.loads((EXPL / "stability_metrics.json").read_text(encoding="utf-8"))
    dele = pd.DataFrame(dm["summary"])
    dele = dele[(dele.group == "all") & (dele.deletion_fill == "room_noise")]
    rep = pd.DataFrame(sm["repeats"])
    rep = rep[rep.group == "all"]
    per = pd.DataFrame(sm["perturb"])
    per = per[per.group == "all"]
    rows = []
    for m, f in METHODS:
        d = dele[(dele.method == m) & (dele.score_fill == f)]
        assert len(d) == 1, (m, f)
        rec = {"explainer": m, "score_fill": f, "deletion_gain": float(d.logit_drop_minus_random.iloc[0]),
               "deletion_n_clips": int(d.n_clips.iloc[0])}
        r = rep[(rep.method == m) & (rep.fill == f)]
        rec["tau_repeats"] = float(r.tau_mean.iloc[0])
        for t in STAB_TESTS[1:]:
            q = per[(per.perturbation == t) & (per.method == m) & (per.fill == f)]
            rec[f"tau_{t}"] = float(q.tau_mean.iloc[0])
            rec[f"n_{t}"] = int(q.n_clips.iloc[0])
        rec["stability"] = float(np.mean([rec[f"tau_{t}"] for t in STAB_TESTS]))
        rows.append(rec)
    tab = pd.DataFrame(rows)
    tab["rank_deletion"] = tab.deletion_gain.rank(ascending=False, method="min")
    tab["rank_stability"] = tab.stability.rank(ascending=False, method="min")
    tab["average_rank"] = (tab.rank_deletion + tab.rank_stability) / 2
    best = tab[tab.average_rank == tab.average_rank.min()]
    winner = best.explainer.iloc[0] if len(best) == 1 else "TIE: " + ", ".join(best.explainer)
    # robustness: stability rank per test on its own
    rob = pd.DataFrame({"explainer": tab.explainer,
                        **{f"rank_{t}": tab[f"tau_{t}"].rank(ascending=False, method="min") for t in STAB_TESTS}})
    for t in STAB_TESTS:
        rob[f"avg_rank_using_{t}_only"] = (tab.rank_deletion + rob[f"rank_{t}"]) / 2
    return winner, tab, rob


# --------------------------------------------------------------------------------------- 2. clips
def clip_table() -> pd.DataFrame:
    sc = pd.read_csv(EXPL / "resnet_segment_scores.csv")
    lime = sc[(sc.method == "LIME") & (sc.fill == "room_noise")].set_index("clip_id")
    clips = pd.read_csv(EXPL / f"clips_fold{FOLD}.csv").set_index("clip_id")
    t = clips.join(lime[["y_pred", "pred_style", "correct", "n_real"] + COLS])
    assert len(t) == 100 and t.y_pred.notna().all()
    top = []
    for cid, r in t.iterrows():
        s = r[COLS].to_numpy(float)[: int(r.n_real)]
        top.append(" ".join(str(j) for j in np.argsort(-s, kind="stable")[:TOP_K]))
    t["top3"] = top
    return t


def select_clips(t: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    right, wrong = t.index[t.correct.astype(bool)].to_numpy(), t.index[~t.correct.astype(bool)].to_numpy()
    styles = sorted(t["style"].unique())
    n_wrong = N_PER_SET - N_RIGHT_PER_SET
    for attempt in range(1, 200001):
        r = rng.choice(right, 2 * N_RIGHT_PER_SET, replace=False)
        w = rng.choice(wrong, 2 * n_wrong, replace=False)
        s1 = list(r[:N_RIGHT_PER_SET]) + list(w[:n_wrong])
        s2 = list(r[N_RIGHT_PER_SET:]) + list(w[n_wrong:])
        c1 = t.loc[s1, "style"].value_counts().reindex(styles, fill_value=0)
        c2 = t.loc[s2, "style"].value_counts().reindex(styles, fill_value=0)
        if (c1 > 0).all() and (c2 > 0).all() and ((c1 - c2).abs() <= 1).all():
            sel = pd.concat([t.loc[s1].assign(set=1), t.loc[s2].assign(set=2)])
            sel.attrs["attempt"] = attempt
            return sel
    raise RuntimeError("no draw met the constraints")


# ----------------------------------------------------------------------------------- 3. materials
def spectrogram_png(path, x_db: np.ndarray, n_valid: int, top3: list[int], sp: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.imshow(x_db, origin="lower", aspect="auto", cmap="magma", vmin=-100, vmax=float(x_db.max()),
              extent=(0, 4, 0, x_db.shape[0]))
    end = n_valid / 100
    if end < 4:
        ax.add_patch(mpatches.Rectangle((end, 0), 4 - end, x_db.shape[0], facecolor="#9a9a9a", alpha=0.85,
                                        hatch="//", edgecolor="#6f6f6f", lw=0))
        ax.text((end + 4) / 2, x_db.shape[0] / 2, "no audio", ha="center", va="center", fontsize=10, color="#222")
    for j in range(17):
        ax.axvline(j * 0.25, color="white", lw=0.5, alpha=0.35)
    for rank, j in enumerate(top3, start=1):
        ax.add_patch(mpatches.Rectangle((j * 0.25, 0), 0.25, x_db.shape[0], fill=False, edgecolor="#00e5ff", lw=2.6))
        ax.text(j * 0.25 + 0.125, x_db.shape[0] + 3, str(rank), ha="center", va="bottom", fontsize=12,
                fontweight="bold", color="#007c8a")
    mel_hz = librosa.mel_frequencies(n_mels=sp["n_mels"], fmin=sp["fmin"], fmax=sp["fmax"])
    ticks = [100, 500, 1000, 2000, 4000, 8000]
    ax.set_yticks([int(np.argmin(np.abs(mel_hz - h))) for h in ticks], [f"{h // 1000}k" if h >= 1000 else str(h)
                                                                         for h in ticks])
    ax.set_xticks(np.arange(0, 4.01, 0.5))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, x_db.shape[0] + 14)
    ax.set_xlabel("time (s)  -  grid = 250 ms segments")
    ax.set_ylabel("pitch (Hz)")
    ax.set_title("Boxes 1-3: the parts that mattered most for the model's answer", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)


def make_materials(sel: pd.DataFrame) -> pd.DataFrame:
    tcfg = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    pcfg = load_config(REPO_ROOT / tcfg["preprocess_config"])
    sp = pcfg["spectrogram"]
    ms = int(pcfg["audio"]["max_seconds"] * sp["sr"])
    styles = pcfg["labels"]["styles"]
    man = pd.read_csv(pcfg["paths"]["processed_dir"] / "manifest.csv").set_index("clip_id")
    ds = SpectrogramDataset(pcfg, FOLD, "test")
    row = pd.Series(np.arange(len(ds.clip_ids)), index=ds.clip_ids)

    rng = np.random.default_rng([SEED, 1])
    ids = set()
    while len(ids) < 2 * len(sel):
        ids.add(f"{int(rng.integers(16 ** 8)):08x}")
    ids = sorted(ids)
    rng.shuffle(ids)
    items = []
    for set_no in (1, 2):
        s = sel[sel.set == set_no]
        order = rng.permutation(len(s))
        for pos, k in enumerate(order, start=1):
            cid = s.index[k]
            r = s.loc[cid]
            item_id, file_id = ids.pop(), ids.pop()
            y = load_audio16k(pcfg, man.loc[cid, "audio16k_file"])[:ms]
            (BUNDLE / "audio").mkdir(parents=True, exist_ok=True)
            assert np.max(np.abs(y)) < 1.0
            sf.write(BUNDLE / "audio" / f"{file_id}.wav", y, sp["sr"], subtype="PCM_16")
            top3 = [int(v) for v in r.top3.split()]
            img = ""
            if set_no == 2:
                (BUNDLE / "img").mkdir(parents=True, exist_ok=True)
                i = int(row[cid])
                spectrogram_png(BUNDLE / "img" / f"{file_id}.png", ds.x[i].astype(np.float32), int(ds.n_valid[i]),
                                top3, sp)
                img = f"img/{file_id}.png"
            items.append({"set": set_no, "condition": "audio_only" if set_no == 1 else "audio_plus_highlights",
                          "item": pos, "item_id": item_id, "audio": f"audio/{file_id}.wav", "image": img,
                          "clip_id": cid, "true_style": r["style"], "model_prediction": styles[int(r.y_pred)],
                          "model_correct": bool(r.correct), "top3_segments": " ".join(map(str, top3)),
                          "top3_times_s": " ".join(f"{j * 0.25:.2f}-{(j + 1) * 0.25:.2f}" for j in top3),
                          "duration_s": round(len(y) / sp["sr"], 3)})
    items = pd.DataFrame(items)
    items.to_csv(OUT / "key.csv", index=False)
    items[["set", "condition", "item", "item_id", "audio", "image"]].to_csv(OUT / "items.csv", index=False)
    return items


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listening study</title>
<style>
  :root { --bg:#fafaf8; --card:#fff; --ink:#1d1d1f; --muted:#5f6368; --line:#d9d9d6; --accent:#0b6e7f; --warn:#a33; }
  @media (prefers-color-scheme: dark) { :root { --bg:#16171a; --card:#212226; --ink:#ececec; --muted:#a3a7ad;
    --line:#3a3c42; --accent:#4fc3d4; --warn:#ff8a80; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 860px; margin: 0 auto; padding: 24px 16px 64px; }
  h1 { font-size: 1.5rem; margin: 0 0 8px; }
  h2 { font-size: 1.15rem; margin: 0 0 8px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:20px; margin:16px 0; }
  .muted { color:var(--muted); }
  .progress { font-size:.9rem; color:var(--muted); margin-bottom:8px; }
  audio { width:100%; margin:8px 0 12px; }
  img { width:100%; height:auto; border:1px solid var(--line); border-radius:6px; background:#fff; }
  fieldset { border:0; padding:0; margin:12px 0; display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr)); gap:8px; }
  legend { font-weight:600; margin-bottom:6px; }
  label.opt { display:flex; align-items:center; gap:8px; border:1px solid var(--line); border-radius:8px; padding:8px 10px; cursor:pointer; }
  label.opt:has(input:checked) { border-color:var(--accent); outline:2px solid var(--accent); }
  textarea, input[type=text] { width:100%; font:inherit; color:inherit; background:transparent; border:1px solid var(--line); border-radius:8px; padding:8px; }
  textarea { min-height:70px; }
  button { font:inherit; background:var(--accent); color:#fff; border:0; border-radius:8px; padding:10px 18px; cursor:pointer; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .row { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; margin-top:12px; }
  .warn { color:var(--warn); font-size:.9rem; }
  .hidden { display:none; }
</style>
</head>
<body>
<main>
  <h1>Guess what the computer model said</h1>
  <p class="muted">A computer model listens to short speech clips and decides which of 7 speaking styles it hears.
  It is sometimes wrong. Your task: <b>guess the style the model chose</b> (not necessarily the style you hear).</p>

  <section id="start" class="card">
    <h2>Before you start</h2>
    <p>You will hear 20 clips in two parts of 10. Use headphones and keep the volume the same throughout.
    Play each clip as often as you like. There is no feedback, and you cannot go back to change an answer.</p>
    <label for="pid"><b>Participant id</b> (given to you by the researcher)</label>
    <input id="pid" type="text" autocomplete="off">
    <div class="row"><span id="pidwarn" class="warn"></span><button id="begin">Start part 1</button></div>
  </section>

  <section id="intro2" class="card hidden">
    <h2>Part 2: clips with highlights</h2>
    <p>Part 1 is done. In part 2 you will hear 10 new clips. Under each clip there is a picture of the sound
    (time from left to right, pitch from bottom to top, brighter = louder). The clip is cut into 250 ms pieces.
    <b>Boxes 1, 2 and 3 mark the three pieces that an explanation method says mattered most for the model's
    answer</b> (1 = most). Grey hatched area = no audio. The task is the same: guess the style the model chose.</p>
    <div class="row"><span></span><button id="begin2">Start part 2</button></div>
  </section>

  <section id="item" class="card hidden">
    <div class="progress" id="progress"></div>
    <h2 id="itemtitle"></h2>
    <audio id="player" controls preload="auto"></audio>
    <img id="pic" class="hidden" alt="Spectrogram of the clip with the three highlighted 250 ms pieces">
    <fieldset id="opts"><legend>Which style did the model choose?</legend></fieldset>
    <label for="comment" class="muted">Comment (optional): what did you base your guess on?</label>
    <textarea id="comment"></textarea>
    <div class="row"><span id="itemwarn" class="warn"></span><button id="next" disabled>Next</button></div>
  </section>

  <section id="end" class="card hidden">
    <h2>Thank you, that was the last clip</h2>
    <label for="final" class="muted">Any last comments on the task (optional)?</label>
    <textarea id="final"></textarea>
    <div class="row"><span class="muted">Download the file and give it to the researcher.</span>
      <button id="download">Download CSV</button></div>
    <p id="saved" class="muted"></p>
  </section>
</main>
<script>
const ITEMS = __ITEMS__;
const STYLES = ["confused", "default", "enunciated", "happy", "laughing", "sad", "whisper"];
const $ = (id) => document.getElementById(id);
let pid = "", idx = 0, answers = [], shownAt = 0;

function store() { try { localStorage.setItem("fsim_" + pid, JSON.stringify({ idx, answers })); } catch (e) {} }
function restore() {
  try { const s = JSON.parse(localStorage.getItem("fsim_" + pid) || "null"); if (s) { idx = s.idx; answers = s.answers; } }
  catch (e) {}
}
function show(id) { for (const s of ["start", "intro2", "item", "end"]) $(s).classList.toggle("hidden", s !== id); window.scrollTo(0, 0); }

STYLES.forEach((s) => {
  const l = document.createElement("label"); l.className = "opt";
  l.innerHTML = `<input type="radio" name="ans" value="${s}"> ${s}`;
  $("opts").appendChild(l);
});
$("opts").addEventListener("change", () => { $("next").disabled = false; $("itemwarn").textContent = ""; });

function route() {
  if (idx >= ITEMS.length) { show("end"); return; }
  if (idx === ITEMS.findIndex((it) => it.set === 2) && !answers.some((a) => a.set === 2) && !route.intro2done) {
    show("intro2"); return;
  }
  renderItem();
}
function renderItem() {
  const it = ITEMS[idx];
  $("progress").textContent = `Part ${it.set} of 2 - clip ${it.item} of 10`;
  $("itemtitle").textContent = `Clip ${it.item}`;
  $("player").src = it.audio;
  $("pic").classList.toggle("hidden", !it.image);
  if (it.image) $("pic").src = it.image; else $("pic").removeAttribute("src");
  document.querySelectorAll("input[name=ans]").forEach((r) => (r.checked = false));
  $("comment").value = ""; $("next").disabled = true;
  show("item"); shownAt = Date.now();
}

$("begin").onclick = () => {
  pid = $("pid").value.trim();
  if (!pid) { $("pidwarn").textContent = "Please enter your participant id."; return; }
  if (!/^[A-Za-z0-9_-]{1,40}$/.test(pid)) { $("pidwarn").textContent = "Use letters, digits, - or _ only."; return; }
  restore(); route();
};
$("begin2").onclick = () => { route.intro2done = true; renderItem(); };
$("next").onclick = () => {
  const sel = document.querySelector("input[name=ans]:checked");
  if (!sel) { $("itemwarn").textContent = "Please choose one style."; return; }
  const it = ITEMS[idx];
  answers.push({ participant_id: pid, set: it.set, condition: it.condition, item: it.item, item_id: it.item_id,
                 answer: sel.value, comment: $("comment").value.trim(),
                 seconds_on_item: ((Date.now() - shownAt) / 1000).toFixed(1), time: new Date().toISOString() });
  idx += 1; store(); route();
};

function csvCell(v) { const s = String(v ?? ""); return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }
$("download").onclick = () => {
  const cols = ["participant_id", "set", "condition", "item", "item_id", "answer", "comment", "seconds_on_item", "time"];
  const rows = answers.slice();
  const fin = $("final").value.trim();
  rows.push({ participant_id: pid, set: "", condition: "final_comment", item: "", item_id: "", answer: "",
              comment: fin, seconds_on_item: "", time: new Date().toISOString() });
  const text = "﻿" + [cols.join(",")].concat(rows.map((r) => cols.map((c) => csvCell(r[c])).join(","))).join("\r\n") + "\r\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  a.download = `forward_sim_${pid}.csv`;
  document.body.appendChild(a); a.click(); a.remove();
  $("saved").textContent = `Saved as forward_sim_${pid}.csv in your downloads folder.`;
};
</script>
</body>
</html>
"""


def write_html(items: pd.DataFrame) -> None:
    pub = items[["set", "condition", "item", "item_id", "audio", "image"]].to_dict("records")
    for p in pub:
        assert set(p) == {"set", "condition", "item", "item_id", "audio", "image"}   # no key information
    (BUNDLE / "study.html").write_text(HTML.replace("__ITEMS__", json.dumps(pub)), encoding="utf-8")


def write_readme(tab: pd.DataFrame, rob: pd.DataFrame, mix: pd.DataFrame, items: pd.DataFrame, attempt: int) -> None:
    styles = [c[len("true_"):] for c in mix.columns if c.startswith("true_")]
    top_first = int(items.top3_segments.str.split().map(lambda v: "0" in v).sum())
    L = ["# Forward-simulation study (fold 1, ResNet-18 seed 1): materials", "",
         "Can people guess the style the **model** predicted, from the audio alone (set 1) and from the audio plus "
         "the explainer's top 3 segments (set 2)? No training; decisions in `DECISIONS.md` (Human forward-simulation "
         "study).", "",
         "## Explainer rule (fixed before the study)", "",
         "Best average rank of deletion gain (logit drop minus random, room-noise deletion fill, 100 clips) and "
         "stability (mean Kendall tau over the pre-planned tests: 10 repeats, noise 40 dB, gain +1 dB, gain -1 dB; "
         "room-noise fill, 50 clips). Rank 1 = best.", "",
         "| explainer | deletion gain | rank | tau repeats | tau noise 40 dB | tau gain +1 dB | tau gain -1 dB | "
         "stability | rank | **average rank** |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in tab.itertuples():
        L.append(f"| {r.explainer} | {r.deletion_gain:.3f} | {r.rank_deletion:.0f} | {r.tau_repeats:.3f} | "
                 f"{r.tau_noise_40db:.3f} | {r.tau_gain_plus_1db:.3f} | {r.tau_gain_minus_1db:.3f} | {r.stability:.3f} "
                 f"| {r.rank_stability:.0f} | **{r.average_rank:.1f}** |")
    L += ["", "**The rule gives LIME.** Reading stability from any single test instead gives LIME the best average "
          "rank every time (`explainer_rule_robustness.csv`): average rank "
          + ", ".join(f"{t} {v:.1f}" for t, v in zip([c.replace('avg_rank_using_', '').replace('_only', '')
                                                       for c in rob.columns if c.startswith('avg_rank')],
                                                      rob[rob.explainer == 'LIME'].filter(like='avg_rank').iloc[0])) + ".",
          "", "## Clips", "",
          f"20 of the 100 fold-1 clips (seed {SEED}, first draw meeting the constraints: attempt {attempt}): two sets "
          "of 10, each with 6 clips the model got right and 4 it got wrong, all 7 true styles, no style differing by "
          "more than one clip between the sets.", "",
          "| set | condition | model right | model wrong | " + " | ".join(styles) + " | model predictions |",
          "|---|---|---|---|" + "---|" * len(styles) + "---|"]
    for r in mix.itertuples():
        cond = "audio only" if r.set == 1 else "audio + highlights"
        L.append(f"| {r.set} | {cond} | {r.model_right} | {r.model_wrong} | "
                 + " | ".join(str(getattr(r, f"true_{s}")) for s in styles) + f" | {r.model_predictions} |")
    L += ["", f"In {top_first} of the 20 clips one of LIME's top 3 segments is the clip start (0-0.25 s), as in the "
          "explainer report; participants in set 2 will often see a box on the first quarter second.", "",
          "## Running the study", "",
          "1. Give each participant the folder `participant_bundle/` (study.html, audio/, img/) and a participant id. "
          "The page works offline in Chrome, Edge or Firefox: open `study.html`. It holds no answer key.",
          "2. Fixed order for everyone: set 1 audio only, then set 2 audio plus highlights; no feedback; answers "
          "cannot be changed. At the end the participant clicks **Download CSV** (`forward_sim_<id>.csv`).",
          "3. Score: `python -m src.explain.score_forward_sim forward_sim_*.csv` (accuracy per condition, paired per "
          "participant, n; prints and saves the comments to `results/comments.csv`). No significance test.", "",
          "Files: `key.csv` (**not for participants**: clip, true style, the model's prediction = the right answer, top "
          "3 segments), `items.csv` (what the page shows), `explainer_rule*.csv`, `selection_mix.csv`. Audio and "
          "pictures are not in git; rebuild them with `python -m src.explain.forward_sim` (same seed, same files).", "",
          "## Limits", "",
          "- **Practice cannot be separated from the explanation effect:** the highlight condition always comes "
          "second, so any change from set 1 to set 2 also contains learning the task (and tiredness).",
          "- The sets hold different clips, so clip difficulty is mixed into the difference.",
          "- 10 clips per condition: one clip is 10 points; the result is descriptive, no significance is claimed.",
          "- The task is to guess the model, not the true style; people who answer what they hear score 6/10 per set "
          "by construction (the 6 clips the model got right).",
          "- One model, one actor (ex02), one explainer (LIME, room-noise fill), top 3 segments only.", ""]
    (OUT / "README.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    print(f"seed {SEED} (clip draw: default_rng({SEED}); ids and item order: default_rng([{SEED}, 1]))")
    winner, tab, rob = explainer_rule()
    pd.set_option("display.width", 220)
    print("\nExplainer rule: deletion gain (logit drop minus random, room-noise fill, 100 clips) and stability "
          "(mean Kendall tau over repeats, noise 40 dB, gain +1 dB, gain -1 dB; room-noise fill)")
    print(tab[["explainer", "score_fill", "deletion_gain", "rank_deletion", "tau_repeats", "tau_noise_40db",
               "tau_gain_plus_1db", "tau_gain_minus_1db", "stability", "rank_stability", "average_rank"]]
          .round(3).to_string(index=False))
    print("\nRobustness: average rank if stability were read from one test only")
    print(rob.to_string(index=False))
    print(f"\nRule gives: {winner}")
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "explainer_rule.csv", index=False, float_format="%.4f")
    rob.to_csv(OUT / "explainer_rule_robustness.csv", index=False)
    if winner != "LIME":
        print("STOP: the rule does not give LIME; no study material made.")
        sys.exit(2)

    t = clip_table()
    sel = select_clips(t)
    print(f"\nclip draw accepted at attempt {sel.attrs['attempt']}")
    mix = []
    for s in (1, 2):
        g = sel[sel.set == s]
        mix.append({"set": s, "clips": len(g), "model_right": int(g.correct.astype(bool).sum()),
                    "model_wrong": int((~g.correct.astype(bool)).sum()),
                    **{f"true_{st}": int((g["style"] == st).sum()) for st in sorted(t["style"].unique())},
                    "model_predictions": ", ".join(f"{k} {v}" for k, v in g.pred_style.value_counts().items())})
    mix = pd.DataFrame(mix)
    print(mix.to_string(index=False))
    mix.to_csv(OUT / "selection_mix.csv", index=False)

    items = make_materials(sel)
    write_html(items)
    write_readme(tab, rob, mix, items, sel.attrs["attempt"])
    print(f"\nwrote {BUNDLE / 'study.html'}, {len(items)} audio files, {int((items.image != '').sum())} pictures, "
          f"{OUT / 'key.csv'}")


if __name__ == "__main__":
    main()

"""Score the forward-simulation study (fold 1): can people guess the ResNet's prediction, without and with LIME?

    python -m src.explain.score_forward_sim RESULT.csv [RESULT2.csv ...] [--key KEY.csv] [--out DIR]

Inputs: the CSV files downloaded from participant_bundle/study.html (one per participant; several may be given).
The answer key is the MODEL'S PREDICTION (key.csv), not the true style.

Prints and writes (to --out, default reports/forward_sim_fold1/results/):
- accuracy per condition, all answers pooled, with the number of answers and participants;
- per participant: set 1 (audio only), set 2 (audio plus highlights), the difference, and the mean and range of the
  paired differences;
- split by whether the model was right on the clip, and the share of answers equal to the TRUE style;
- reference levels: chance 1/7 = 14.3%; always answering the true style = share of clips the model got right;
- all comments (comments.csv) printed in full.
No significance test is run and no significance is claimed. Needs only numpy and pandas.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "reports" / "forward_sim_fold1"
STYLES = ["confused", "default", "enunciated", "happy", "laughing", "sad", "whisper"]
COND = {1: "audio_only", 2: "audio_plus_highlights"}


def load(files: list[Path], key: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for f in files:
        d = pd.read_csv(f, dtype=str, encoding="utf-8-sig").fillna("")
        d["source_file"] = f.name
        parts.append(d)
    a = pd.concat(parts, ignore_index=True)
    final = a[a.condition == "final_comment"].copy()
    a = a[a.condition != "final_comment"].copy()
    a["set"] = a["set"].astype(int)
    a["item"] = a["item"].astype(int)
    bad = sorted(set(a.answer) - set(STYLES))
    if bad:
        raise SystemExit(f"unknown answers: {bad}")
    a = a.merge(key, on="item_id", how="left", suffixes=("", "_key"), validate="many_to_one")
    if a.clip_id.isna().any():
        raise SystemExit(f"item ids not in the key: {sorted(a.loc[a.clip_id.isna(), 'item_id'].unique())}")
    if ((a["set"] != a["set_key"]) | (a.condition != a.condition_key)).any():
        raise SystemExit("set / condition in a result file does not match the key")
    dup = a.duplicated(["participant_id", "item_id"], keep=False)
    if dup.any():
        raise SystemExit(f"an item answered twice: {a.loc[dup, ['participant_id', 'item_id']].values.tolist()}")
    a["hit"] = a.answer == a.model_prediction
    a["answered_true_style"] = a.answer == a.true_style
    a["model_correct"] = a.model_correct.astype(str).str.lower() == "true"
    return a, final


def pct(k: int, n: int) -> str:
    return f"{k}/{n} = {k / n:.0%}" if n else "no answers"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--key", type=Path, default=BASE / "key.csv")
    ap.add_argument("--out", type=Path, default=BASE / "results")
    args = ap.parse_args()
    key = pd.read_csv(args.key, dtype={"item_id": str})
    a, final = load(args.results, key)
    args.out.mkdir(parents=True, exist_ok=True)
    n_p = a.participant_id.nunique()
    L = ["# Forward simulation: guessing the model's prediction", "",
         f"Participants: **{n_p}**; answers: **{len(a)}** "
         f"({', '.join(f'{COND[s]} {int((a.set == s).sum())}' for s in (1, 2))}). Answer key = the model's prediction. "
         "Chance = 1/7 = 14.3%.", ""]

    L += ["## Accuracy per condition (all answers pooled)", "",
          "| condition | answers | right (= model's prediction) | model right on the clip: accuracy | model wrong on "
          "the clip: accuracy | answered the true style | reference: always answer the true style |",
          "|---|---|---|---|---|---|---|"]
    for s in (1, 2):
        g = a[a.set == s]
        k = key[key.set == s]
        L.append(f"| {COND[s]} | {len(g)} | **{pct(int(g.hit.sum()), len(g))}** | "
                 f"{pct(int(g[g.model_correct].hit.sum()), int(g.model_correct.sum()))} | "
                 f"{pct(int(g[~g.model_correct].hit.sum()), int((~g.model_correct).sum()))} | "
                 f"{pct(int(g.answered_true_style.sum()), len(g))} | "
                 f"{pct(int(k.model_correct.astype(str).str.lower().eq('true').sum()), len(k))} |")
    L.append("")

    per = (a.groupby(["participant_id", "set"]).hit.agg(["sum", "count"]).unstack("set")
           .reindex(columns=pd.MultiIndex.from_product([["sum", "count"], [1, 2]])).fillna(0).astype(int))
    rows, diffs = [], []
    for pid in per.index:
        r1, n1 = per.loc[pid, ("sum", 1)], per.loc[pid, ("count", 1)]
        r2, n2 = per.loc[pid, ("sum", 2)], per.loc[pid, ("count", 2)]
        complete = n1 == 10 and n2 == 10
        d = (r2 / n2 - r1 / n1) if (n1 and n2) else float("nan")
        if complete:
            diffs.append(d)
        rows.append({"participant_id": pid, "set1_right": r1, "set1_n": n1, "set2_right": r2, "set2_n": n2,
                     "set1_accuracy": r1 / n1 if n1 else float("nan"), "set2_accuracy": r2 / n2 if n2 else float("nan"),
                     "difference_set2_minus_set1": d, "complete": complete})
    per_df = pd.DataFrame(rows)
    per_df.to_csv(args.out / "per_participant.csv", index=False, float_format="%.4f")
    L += ["## Per participant (paired)", "",
          "| participant | audio only | audio plus highlights | difference (points) | complete |", "|---|---|---|---|---|"]
    for r in per_df.itertuples():
        L.append(f"| {r.participant_id} | {pct(r.set1_right, r.set1_n)} | {pct(r.set2_right, r.set2_n)} | "
                 f"{r.difference_set2_minus_set1 * 100:+.0f} | {'yes' if r.complete else 'NO'} |")
    if diffs:
        s = pd.Series(diffs)
        L += ["", f"Paired difference (highlights minus audio only) over the **{len(s)}** participants with both sets "
              f"complete: mean {s.mean() * 100:+.1f} points, median {s.median() * 100:+.1f}, range "
              f"{s.min() * 100:+.0f} to {s.max() * 100:+.0f}; better with highlights {int((s > 0).sum())}, same "
              f"{int((s == 0).sum())}, worse {int((s < 0).sum())}."]
    L += ["", "**No significance test is run and no significance is claimed.** With 10 clips per condition one clip "
          "is 10 points.", "",
          "**Limits.** The highlight condition always comes second, so practice (and tiredness) cannot be separated "
          "from the effect of the explanation. The two sets hold different clips, so a difference also contains "
          "clip difficulty. People may answer the style they hear instead of the model's guess, which is wrong on the "
          "4 clips per set where the model is wrong (see the true-style columns).", ""]

    # per item (for reading, not for testing)
    it = a.groupby(["set", "item", "item_id", "clip_id", "true_style", "model_prediction", "model_correct"]) \
        .agg(answers=("hit", "size"), right=("hit", "sum"),
             most_common_answer=("answer", lambda v: v.value_counts().index[0])).reset_index()
    it.to_csv(args.out / "per_item.csv", index=False)

    com = a[a.comment.str.strip() != ""][["participant_id", "set", "condition", "item", "clip_id", "true_style",
                                           "model_prediction", "answer", "comment"]]
    fin = final[final.comment.str.strip() != ""][["participant_id", "comment"]].assign(
        set="", condition="final_comment", item="", clip_id="", true_style="", model_prediction="", answer="")
    comments = pd.concat([com, fin[com.columns]], ignore_index=True)
    comments.to_csv(args.out / "comments.csv", index=False, encoding="utf-8-sig")
    a.to_csv(args.out / "all_answers.csv", index=False)
    (args.out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"## Comments ({len(comments)}), also in {args.out / 'comments.csv'}\n")
    for r in comments.itertuples():
        where = (f"set {r.set} ({r.condition}), clip {r.item}: answered {r.answer}, model said {r.model_prediction}, "
                 f"true {r.true_style}") if r.condition != "final_comment" else "final comment"
        text = str(r.comment).replace("\r\n", "\n").replace("\n", "\n    ")
        print(f"- [{r.participant_id}] {where}\n    {text}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

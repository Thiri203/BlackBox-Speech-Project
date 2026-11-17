"""Score the listening test against the model's flips (counterfactual edits, fold 1).

    python -m src.explain.score_listening SHEET.csv [SHEET2.csv ...] [--key KEY.csv] [--out scores.json]

Each sheet is a filled copy of reports/counterfactual_fold1/listening/answer_sheet_template.csv (one or more
listeners; columns listener_id, pair, file_A, file_B, hear_difference, style_A, style_B). The key
(listening_key.csv) says which file is the edited one and whether the model's label flipped.

Scores, per listener and pooled over all listeners (pairs x listeners):
1. heard a difference (yes/no) vs model flipped: agreement, and the yes-rate in flipped / not-flipped pairs.
   Measures audibility: an edit can be audible without changing the style.
2. MAIN: listener's style changes between A and B vs model flipped: agreement. Pairs where either style is
   "unsure" or blank are left out and counted.
3. Flipped pairs only: listener's style for the edited version equals the model's edited label; also the
   listener's style for the unedited version vs the model's unedited label and vs the true style.
Rates come with Wilson 95% intervals. Needs only numpy and pandas (no model, no GPU).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
KEY = REPO / "reports" / "counterfactual_fold1" / "listening" / "listening_key.csv"
STYLES = ["confused", "default", "enunciated", "happy", "laughing", "sad", "whisper"]
UNSURE = "unsure"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (same formula as src/explain/planted_tone.py, copied to avoid importing torch)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c0 = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, float(c0 - h)), min(1.0, float(c0 + h))


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else float("nan"), "ci95": [lo, hi]}


def load(sheets: list[Path], key_path: Path) -> pd.DataFrame:
    key = pd.read_csv(key_path)
    parts = []
    for p in sheets:
        s = pd.read_csv(p, dtype=str).fillna("")
        missing = {"listener_id", "pair", "file_A", "file_B", "hear_difference", "style_A", "style_B"} - set(s.columns)
        if missing:
            raise SystemExit(f"{p}: missing columns {sorted(missing)}")
        s["sheet"] = p.name
        parts.append(s)
    a = pd.concat(parts, ignore_index=True)
    for col in ("listener_id", "hear_difference", "style_A", "style_B"):
        a[col] = a[col].str.strip().str.lower()
    a["pair"] = a["pair"].astype(int)
    if (a.listener_id == "").any():
        raise SystemExit("every row needs a listener_id")
    a = a.merge(key, on="pair", how="left", suffixes=("", "_key"), validate="many_to_one")
    if a.clip_id.isna().any():
        raise SystemExit(f"pairs not in the key: {sorted(a.loc[a.clip_id.isna(), 'pair'].unique())}")
    bad = a[(a.file_A != a.file_A_key) | (a.file_B != a.file_B_key)]
    if len(bad):
        raise SystemExit(f"file names do not match the key for pairs {sorted(bad.pair.unique())}")
    dup = a.duplicated(["listener_id", "pair"])
    if dup.any():
        raise SystemExit(f"duplicate answers: {a.loc[dup, ['listener_id', 'pair']].values.tolist()}")
    errs = []
    if (~a.hear_difference.isin(["yes", "no", ""])).any():
        errs.append("hear_difference must be yes, no or blank")
    for col in ("style_A", "style_B"):
        if (~a[col].isin(STYLES + [UNSURE, ""])).any():
            errs.append(f"{col} must be one of {STYLES + [UNSURE]} or blank: "
                        f"{sorted(set(a.loc[~a[col].isin(STYLES + [UNSURE, '']), col]))}")
    if errs:
        raise SystemExit("; ".join(errs))
    a["model_flipped"] = a["model_flipped"].astype(str).str.lower() == "true"
    a["style_orig"] = np.where(a.original_is == "A", a.style_A, a.style_B)
    a["style_edit"] = np.where(a.original_is == "A", a.style_B, a.style_A)
    return a


def score(a: pd.DataFrame) -> dict:
    out = {"n_answers": int(len(a)), "n_listeners": int(a.listener_id.nunique())}
    d = a[a.hear_difference != ""]
    heard = d.hear_difference == "yes"
    out["1_difference_vs_flip"] = {
        "agreement": rate(int((heard == d.model_flipped).sum()), len(d)),
        "heard_in_flipped_pairs": rate(int(heard[d.model_flipped].sum()), int(d.model_flipped.sum())),
        "heard_in_not_flipped_pairs": rate(int(heard[~d.model_flipped].sum()), int((~d.model_flipped).sum())),
        "left_out_blank": int(len(a) - len(d))}
    sure = a[a.style_A.isin(STYLES) & a.style_B.isin(STYLES)]
    chg = sure.style_A != sure.style_B
    out["2_style_change_vs_flip"] = {
        "agreement": rate(int((chg == sure.model_flipped).sum()), len(sure)),
        "style_changed_in_flipped_pairs": rate(int(chg[sure.model_flipped].sum()), int(sure.model_flipped.sum())),
        "style_changed_in_not_flipped_pairs": rate(int(chg[~sure.model_flipped].sum()),
                                                   int((~sure.model_flipped).sum())),
        "left_out_unsure_or_blank": int(len(a) - len(sure))}
    f = a[a.model_flipped]
    fe = f[f.style_edit.isin(STYLES)]
    fo = f[f.style_orig.isin(STYLES)]
    out["3_flipped_pairs"] = {
        "edited_heard_as_model_edited_label": rate(int((fe.style_edit == fe.model_label_edited).sum()), len(fe)),
        "unedited_heard_as_model_unedited_label": rate(int((fo.style_orig == fo.model_label_original).sum()), len(fo)),
        "unedited_heard_as_true_style": rate(int((fo.style_orig == fo.true_style).sum()), len(fo)),
        "left_out_unsure_or_blank_edited": int(len(f) - len(fe))}
    out["by_dial_style_change_agreement"] = {
        dial: rate(int((chg[sure.dial == dial] == sure.model_flipped[sure.dial == dial]).sum()),
                   int((sure.dial == dial).sum())) for dial in sorted(a.dial.unique())}
    return out


def fmt(r: dict) -> str:
    if not r["n"]:
        return "no answers"
    return f"{r['k']}/{r['n']} = {r['rate']:.0%} (95% CI {r['ci95'][0]:.0%} to {r['ci95'][1]:.0%})"


def show(name: str, s: dict) -> list[str]:
    s1, s2, s3 = s["1_difference_vs_flip"], s["2_style_change_vs_flip"], s["3_flipped_pairs"]
    return [f"## {name}: {s['n_answers']} answers, {s['n_listeners']} listener(s)", "",
            "| score | result |", "|---|---|",
            f"| 1. 'heard a difference' agrees with the model's flip | {fmt(s1['agreement'])} |",
            f"| ... heard a difference, flipped pairs | {fmt(s1['heard_in_flipped_pairs'])} |",
            f"| ... heard a difference, not-flipped pairs | {fmt(s1['heard_in_not_flipped_pairs'])} |",
            f"| **2. listener's style change agrees with the model's flip (main)** | **{fmt(s2['agreement'])}** |",
            f"| ... style changed, flipped pairs | {fmt(s2['style_changed_in_flipped_pairs'])} |",
            f"| ... style changed, not-flipped pairs | {fmt(s2['style_changed_in_not_flipped_pairs'])} |",
            f"| 3. flipped pairs: edited version heard as the model's edited label | "
            f"{fmt(s3['edited_heard_as_model_edited_label'])} |",
            f"| ... unedited version heard as the model's unedited label | "
            f"{fmt(s3['unedited_heard_as_model_unedited_label'])} |",
            f"| ... unedited version heard as the true style | {fmt(s3['unedited_heard_as_true_style'])} |",
            *[f"| 2. by dial: {d} | {fmt(r)} |" for d, r in s["by_dial_style_change_agreement"].items()],
            "",
            f"Left out: blank 'difference' {s1['left_out_blank']}; unsure/blank style (score 2) "
            f"{s2['left_out_unsure_or_blank']}.", ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheets", nargs="+", type=Path)
    ap.add_argument("--key", type=Path, default=KEY)
    ap.add_argument("--out", type=Path, default=None, help="optional JSON with all scores")
    args = ap.parse_args()
    a = load(args.sheets, args.key)
    res = {"pooled": score(a), "per_listener": {lid: score(g) for lid, g in a.groupby("listener_id")}}
    lines = ["# Listening test vs model flips", ""] + show("All listeners pooled", res["pooled"])
    for lid, s in res["per_listener"].items():
        lines += show(f"Listener {lid}", s)
    lines += ["Note: pooled rates treat every answer as independent; answers from the same listener or the same "
              "pair are not, so the pooled intervals are too narrow. With 20 pairs per listener the per-listener "
              "intervals are wide.", ""]
    print("\n".join(lines))
    if args.out:
        args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

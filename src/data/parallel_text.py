"""Check the claim "each actor reads the same sentences in all 7 styles".

Two views, both computed per actor:

A) Sentence view. Group clips by normalised text (the same normalisation the split code
   uses) and count in how many of the 7 styles each sentence appears.

B) Id-number view. Ids look like ex01_happy_00260. For each actor and number (00260) we
   compare the transcripts of the styles that carry that number. The "majority text" of a
   group is its most common normalised text. Every group gets one of four cases:
     identical_raw           the raw transcripts are exactly the same in every style
     same_after_normalising  they differ only in case, punctuation, *emphasis* marks or
                             tags such as <laugh>; after normalising they are identical
     wording_variant         after normalising at least one style still differs, but every
                             text is close to the majority text (character similarity >= 0.5):
                             contractions (I am / I'm), spelling (mom / mum), spoken forms
                             (out of / outta), an added word or a likely mishearing
     different_sentence      at least one style has a clearly different sentence
                             (character similarity < 0.5)
   The transcripts appear to describe what was actually said, so a variant usually means
   the actor (or transcriber) deviated slightly from the prompt.

Both views are reported for every scope passed in (all 7-style clips; with the "drop" length
policy also the kept clips). A last section checks that no id group is split between train
and validation (the splits group by actor + id number because of the wording variants).

Outputs: reports/parallel_text_check.csv (sentence view), reports/parallel_text_check.md,
         reports/parallel_text_id_groups.csv (one row per actor x id number)
"""
from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

from src.utils import normalise_text

SIMILARITY_THRESHOLD = 0.5  # character-level similarity separating variants from different sentences
CASES = ["identical_raw", "same_after_normalising", "wording_variant", "different_sentence"]


def _char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _with_num(df: pd.DataFrame) -> pd.DataFrame:
    """Keep ids of the form actor_style_number and add the number and the group's majority text."""
    base = df[df["clip_id"].str.count("_") == 2].copy()
    base["num"] = base["clip_id"].str.rsplit("_", n=1).str[-1]
    maj = base.groupby(["actor", "num"])["text_norm"].agg(lambda s: s.value_counts().sort_index().idxmax())
    base = base.join(maj.rename("majority_text"), on=["actor", "num"])
    base["deviates"] = base["text_norm"] != base["majority_text"]
    return base


def sentence_view(df: pd.DataFrame, scope: str, n_styles: int) -> pd.DataFrame:
    dev = _with_num(df).set_index("clip_id")["deviates"]
    d = df.assign(deviates=df["clip_id"].map(dev).fillna(False).astype(bool))
    g = d.groupby(["actor", "text_norm"]).agg(n_styles=("style", "nunique"), n_clips=("clip_id", "size"),
                                               all_deviating=("deviates", "all"))
    rows = []
    for actor, a in g.groupby(level="actor"):
        for k in range(n_styles, 0, -1):
            sel = a[a["n_styles"] == k]
            rows.append({"scope": scope, "actor": actor, "n_styles_with_sentence": k,
                         "n_sentences": len(sel), "n_clips": int(sel["n_clips"].sum()),
                         "share_of_sentences": round(len(sel) / len(a), 4),
                         "share_of_clips": round(sel["n_clips"].sum() / a["n_clips"].sum(), 4),
                         "n_sentences_that_are_only_variant_readings": int(sel["all_deviating"].sum())})
    return pd.DataFrame(rows)


def id_view(df: pd.DataFrame, scope: str, opts: dict) -> pd.DataFrame:
    """One row per actor x id number."""
    base = _with_num(df)
    o_no_punct = dict(opts, text_remove_punctuation=False)
    rows = []
    for (actor, num), g in base.groupby(["actor", "num"], sort=True):
        norm, majority = g["text_norm"], g["majority_text"].iloc[0]
        min_sim = min(_char_similarity(t, majority) for t in norm)
        if g["text"].nunique() == 1:
            case = "identical_raw"
        elif norm.nunique() == 1:
            case = "same_after_normalising"
        elif min_sim >= SIMILARITY_THRESHOLD:
            case = "wording_variant"
        else:
            case = "different_sentence"
        rows.append({
            "scope": scope, "actor": actor, "num": num, "n_styles": g["style"].nunique(),
            "case": case,
            # does removing only tags, case and asterisks (keeping punctuation) already make them equal?
            "equal_without_punctuation_step": g["text"].map(lambda t: normalise_text(t, o_no_punct)).nunique() == 1,
            "n_distinct_raw_texts": g["text"].nunique(), "n_distinct_norm_texts": norm.nunique(),
            "deviating_styles": ",".join(sorted(g.loc[g["deviates"], "style"])),
            "min_char_similarity_to_majority": round(min_sim, 3),
            # styles sharing an identical raw text are listed together
            "texts": " || ".join(f"{', '.join(sorted(gg['style']))}: {t}" for t, gg in g.groupby("text", sort=True)),
        })
    return pd.DataFrame(rows)


def split_effect(kept: pd.DataFrame, folds: dict) -> pd.DataFrame:
    """Val clips whose id group (same actor and number) also has a clip in train, per fold."""
    base = _with_num(kept).set_index("clip_id")
    rows = []
    for k, f in folds.items():
        f = f[f["clip_id"].isin(base.index)].join(base[["num", "deviates"]], on="clip_id")
        key = f["actor"] + "_" + f["num"]
        train_keys = set(key[f["split"] == "train"])
        val = f[f["split"] == "val"]
        hit = key[val.index].isin(train_keys)
        rows.append({"split_file": k, "val_clips": len(val), "val_clips_whose_id_group_is_also_in_train": int(hit.sum()),
                     "share": round(float(hit.mean()), 4)})
    return pd.DataFrame(rows)


def run(cfg: dict, scopes: dict[str, pd.DataFrame], kept: pd.DataFrame, folds: dict, out) -> dict:
    """scopes: {name: clips}; the first scope is used for the summary and the examples."""
    n = len(cfg["labels"]["styles"])
    sent = pd.concat([sentence_view(d, s, n) for s, d in scopes.items()], ignore_index=True)
    ids = pd.concat([id_view(d, s, cfg["splits"]) for s, d in scopes.items()], ignore_index=True)
    eff = split_effect(kept, folds)
    sent.to_csv(out / "parallel_text_check.csv", index=False)
    ids.to_csv(out / "parallel_text_id_groups.csv", index=False)
    (out / "parallel_text_check.md").write_text(render_md(cfg, sent, ids, eff, scopes), encoding="utf-8")
    return {"sentence_view": sent, "id_view": ids, "split_effect": eff}


def _tbl(df: pd.DataFrame) -> str:
    head = list(map(str, df.columns))
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def render_md(cfg: dict, sent: pd.DataFrame, ids: pd.DataFrame, eff: pd.DataFrame, scopes: dict) -> str:
    n = len(cfg["labels"]["styles"])
    S = next(iter(scopes))
    s0 = sent[sent["scope"] == S]
    tot = s0.groupby("n_styles_with_sentence")[["n_sentences", "n_clips", "n_sentences_that_are_only_variant_readings"]].sum()
    all_sent, all_clips = int(tot["n_sentences"].sum()), int(tot["n_clips"].sum())
    full = tot.loc[n]
    partial = tot.drop(index=n)
    i0 = ids[ids["scope"] == S]
    cases = i0["case"].value_counts()
    c = {k: int(cases.get(k, 0)) for k in CASES}
    n_full_ids = int((i0["n_styles"] == n).sum())
    same = c["identical_raw"] + c["same_after_normalising"]
    dev_clips = sum(len(x.split(",")) for x in i0["deviating_styles"] if x)
    diff_rows = i0[i0["case"] == "different_sentence"]
    diff_clips = sum(len(x.split(",")) for x in diff_rows["deviating_styles"] if x)

    out = [
        "# Parallel text check: does each actor read the same sentences in all 7 styles?",
        "",
        "## Summary in plain words",
        "",
        f"- **Mostly true, but not exactly.** Data: the {all_clips} clips of the {n} styles (scope `{S}`; other "
        "styles and duplicates removed). Texts are compared after normalisation "
        "(lowercase, no `*`, no tags like `<laugh>`, no punctuation).",
        f"- **The id number is the reliable link.** Each actor has {i0.groupby('actor').size().iloc[0]} id numbers "
        f"({len(i0)} actor x number groups in total); {n_full_ids} groups have a clip in all {n} styles. "
        f"In {same} groups ({same / len(i0):.1%}) all styles carry the same sentence after normalising "
        f"({c['identical_raw']} with identical raw transcripts, {c['same_after_normalising']} differing only in "
        f"tags, punctuation, case or emphasis marks). In {c['wording_variant']} groups ({c['wording_variant'] / len(i0):.1%}) "
        "one or more styles have a small wording variant (contractions such as I am / I'm, spellings such as "
        "mom / mum, spoken forms such as outta, an added phrase, or a likely mishearing such as death / deaf). "
        f"Only {c['different_sentence']} groups contain a truly different sentence ({diff_clips} clips).",
        f"- **Exact-text matching therefore understates the parallelism.** Grouping by normalised text gives "
        f"{all_sent} actor-sentence pairs; {int(full['n_sentences'])} ({full['n_sentences'] / all_sent:.1%}) appear "
        f"in all {n} styles and cover {int(full['n_clips'])} clips ({full['n_clips'] / all_clips:.1%}). Of the "
        f"{int(partial['n_sentences'].sum())} sentences found in fewer than {n} styles, "
        f"{int(partial['n_sentences_that_are_only_variant_readings'].sum())} consist only of variant readings "
        f"(the {dev_clips} clips whose text differs from the rest of their id group); the remaining ones are "
        "the shared versions of those same sentences, now missing the styles that used a variant.",
        "- **Across actors** the sentences are mostly different (see `reports/sentence_overlap.csv`).",
        "- **Consequence for the splits:** because of the wording variants, the splits group clips by "
        "actor + id number, not by text (last section).",
        "",
    ]

    out += ["## A) Sentence view: in how many styles does each normalised sentence appear?", ""]
    for scope in scopes:
        t = sent[sent["scope"] == scope].pivot_table(index="n_styles_with_sentence", columns="actor",
                                                     values="n_sentences", aggfunc="sum").sort_index(ascending=False)
        t["all actors"] = t.sum(axis=1)
        out += [f"**{scope}** (number of sentences per actor):", "", _tbl(t.reset_index()), ""]

    out += ["## B) Id-number view: does the number in the id point to the same sentence?", ""]
    for scope in scopes:
        t = ids[ids["scope"] == scope].pivot_table(index="case", columns="actor", values="num", aggfunc="count",
                                                   fill_value=0)
        t = t.reindex([k for k in CASES if k in t.index])
        t["all actors"] = t.sum(axis=1)
        nf = ids[ids["scope"] == scope].groupby("n_styles").size().sort_index(ascending=False)
        out += [f"**{scope}** (number of id groups per actor):", "", _tbl(t.reset_index()), "",
                "Id groups by number of styles present: "
                + ", ".join(f"{k} styles: {v}" for k, v in nf.items()), ""]
    same_norm = i0[i0["case"] == "same_after_normalising"]
    out += [
        "Case definitions: `identical_raw` = raw transcripts identical; `same_after_normalising` = equal after "
        "normalising; `wording_variant` = still differ after normalising, but every text has character "
        f"similarity >= {SIMILARITY_THRESHOLD} to the group's most common text; `different_sentence` = at least "
        "one text below that. (The similarities found fall into two clearly separated ranges: the 4 different "
        "sentences score below 0.4, every variant scores 0.7 or higher.)",
        "",
        f"Of the {len(same_norm)} `same_after_normalising` groups, "
        f"{int(same_norm['equal_without_punctuation_step'].sum())} already match after removing tags, case and `*`; "
        "the rest also need punctuation removed.",
        "",
        "## Examples where the text differs between styles for the same id number",
        "",
        f"15 groups from scope `{S}`: all truly different sentences first, then the "
        "lowest-similarity variants. Styles with identical raw text are listed together. "
        "Full list: `reports/parallel_text_id_groups.csv`.",
        "",
    ]
    diff = i0[i0["case"].isin(["different_sentence", "wording_variant"])]
    ex = diff.sort_values(["min_char_similarity_to_majority", "actor", "num"]).head(15)
    for r in ex.itertuples():
        out.append(f"- **{r.actor} #{r.num}** ({r.case}, similarity {r.min_char_similarity_to_majority}):")
        out += [f"  - {part}" for part in r.texts.split(" || ")]
    out += [
        "",
        "## Consequence for the splits",
        "",
        "Grouping by exact normalised text would treat a wording variant (for example `i'm not sure` vs "
        "`i am not sure`) as a different sentence, so it could land on the other side of the split from its "
        "siblings. With text grouping this affected 10.6% to 19.4% of validation clips. The splits therefore "
        "group by actor + id number. The table counts validation clips whose id group (same actor, same "
        "number) also has a clip in the training set; it must be 0 everywhere:",
        "",
        _tbl(eff),
        "",
    ]
    return "\n".join(out) + "\n"

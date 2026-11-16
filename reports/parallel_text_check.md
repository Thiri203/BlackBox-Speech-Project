# Parallel text check: does each actor read the same sentences in all 7 styles?

## Summary in plain words

- **Mostly true, but not exactly.** Data: the 10636 clips of the 7 styles (scope `all_7_style_clips`; other styles and duplicates removed). Texts are compared after normalisation (lowercase, no `*`, no tags like `<laugh>`, no punctuation).
- **The id number is the reliable link.** Each actor has 380 id numbers (1520 actor x number groups in total); 1516 groups have a clip in all 7 styles. In 1258 groups (82.8%) all styles carry the same sentence after normalising (716 with identical raw transcripts, 542 differing only in tags, punctuation, case or emphasis marks). In 258 groups (17.0%) one or more styles have a small wording variant (contractions such as I am / I'm, spellings such as mom / mum, spoken forms such as outta, an added phrase, or a likely mishearing such as death / deaf). Only 4 groups contain a truly different sentence (4 clips).
- **Exact-text matching therefore understates the parallelism.** Grouping by normalised text gives 1830 actor-sentence pairs; 1255 (68.6%) appear in all 7 styles and cover 8786 clips (82.6%). Of the 575 sentences found in fewer than 7 styles, 310 consist only of variant readings (the 386 clips whose text differs from the rest of their id group); the remaining ones are the shared versions of those same sentences, now missing the styles that used a variant.
- **Across actors** the sentences are mostly different (see `reports/sentence_overlap.csv`).
- **Consequence for the splits:** because of the wording variants, the splits group clips by actor + id number, not by text (last section).

## A) Sentence view: in how many styles does each normalised sentence appear?

**all_7_style_clips** (number of sentences per actor):

| n_styles_with_sentence | ex01 | ex02 | ex03 | ex04 | all actors |
|---|---|---|---|---|---|
| 7 | 263 | 302 | 335 | 355 | 1255 |
| 6 | 88 | 42 | 30 | 22 | 182 |
| 5 | 21 | 18 | 10 | 2 | 51 |
| 4 | 6 | 12 | 5 | 1 | 24 |
| 3 | 3 | 14 | 5 | 1 | 23 |
| 2 | 12 | 22 | 6 | 2 | 42 |
| 1 | 125 | 69 | 38 | 21 | 253 |

## B) Id-number view: does the number in the id point to the same sentence?

**all_7_style_clips** (number of id groups per actor):

| case | ex01 | ex02 | ex03 | ex04 | all actors |
|---|---|---|---|---|---|
| identical_raw | 38 | 230 | 280 | 168 | 716 |
| same_after_normalising | 227 | 72 | 55 | 188 | 542 |
| wording_variant | 113 | 76 | 45 | 24 | 258 |
| different_sentence | 2 | 2 | 0 | 0 | 4 |

Id groups by number of styles present: 7 styles: 1516, 6 styles: 4

Case definitions: `identical_raw` = raw transcripts identical; `same_after_normalising` = equal after normalising; `wording_variant` = still differ after normalising, but every text has character similarity >= 0.5 to the group's most common text; `different_sentence` = at least one text below that. (The similarities found fall into two clearly separated ranges: the 4 different sentences score below 0.4, every variant scores 0.7 or higher.)

Of the 542 `same_after_normalising` groups, 531 already match after removing tags, case and `*`; the rest also need punctuation removed.

## Examples where the text differs between styles for the same id number

15 groups from scope `all_7_style_clips`: all truly different sentences first, then the lowest-similarity variants. Styles with identical raw text are listed together. Full list: `reports/parallel_text_id_groups.csv`.

- **ex02 #00037** (different_sentence, similarity 0.162):
  - happy: Cherrie's has an exclusive one thousand dollar cash bonus offer towards any new Chrysler, Pacifica or Pacifica hybrid.
  - confused, default, enunciated, laughing, sad, whisper: Maybe he was thirsty?
- **ex01 #00125** (different_sentence, similarity 0.235):
  - laughing: <laugh> What has Gregory proved? <breath>
  - confused: He must love playing a robot
  - default, enunciated, happy, sad, whisper: What has Gregory proved?
- **ex01 #00016** (different_sentence, similarity 0.328):
  - whisper: Why aren't you up there objecting?
  - laughing: Write a check, <laugh> problem solved!
  - confused, default, enunciated, happy, sad: Write a check, problem solved!
- **ex02 #00245** (different_sentence, similarity 0.368):
  - confused, default, enunciated, happy, laughing, whisper: The sentence will type.
  - sad: melancholic note
- **ex01 #00231** (wording_variant, similarity 0.703):
  - laughing: <laugh> So it's Instagram for bulldog puppies? <laugh> How about the Age of Innocence or Vanity Fair?
  - happy: How about the Age of Innocence *or* Vanity Fair?
  - confused, default, enunciated, sad, whisper: How about the Age of Innocence or Vanity Fair?
- **ex01 #00258** (wording_variant, similarity 0.737):
  - laughing: <laugh> You through?
  - confused, default, happy, sad, whisper: You through?
  - enunciated: You true?
- **ex03 #00067** (wording_variant, similarity 0.815):
  - confused, default, enunciated, happy, laughing, whisper: Out of nowhere.
  - sad: Outta nowhere.
- **ex01 #00050** (wording_variant, similarity 0.857):
  - confused, enunciated, sad, whisper: *Kim's* death?
  - laughing: <laugh> *Kim's* deaf?
  - happy: Kim's *death?*
  - default: Kim's death?
- **ex01 #00186** (wording_variant, similarity 0.857):
  - laughing: <laugh> I'm going to do it for you.
  - default: I'm going to do it *for* you.
  - confused, enunciated, sad, whisper: I'm going to do it for you.
  - happy: I'm gonna do it for you.
- **ex02 #00117** (wording_variant, similarity 0.863):
  - enunciated, laughing, sad: What are we going to do, Mom?
  - confused, whisper: What are we going to do, Mum?
  - default, happy: What are we gonna do, Mom?
- **ex02 #00178** (wording_variant, similarity 0.864):
  - default, happy, laughing, sad: I wanna be *blown* away.
  - confused, enunciated, whisper: I want to be *blown* away.
- **ex01 #00078** (wording_variant, similarity 0.875):
  - laughing: <laugh> Unless you already *are* in Hawaii.
  - enunciated: Unless you already *are* in Hawaii.
  - confused, sad, whisper: Unless you already are in Hawaii.
  - happy: Unless you are already in Hawaii.
  - default: Unless you're already in Hawaii.
- **ex02 #00175** (wording_variant, similarity 0.875):
  - default: Prep her first C section.
  - confused, happy, laughing, sad, whisper: Prep her for a C section.
  - enunciated: Prep her for the C section.
- **ex01 #00290** (wording_variant, similarity 0.878):
  - laughing: <laugh> my lips are sealed!
  - default: Haha, <breath> my lips are sealed!
  - confused, enunciated, happy, sad, whisper: Haha, my lips are sealed!
- **ex01 #00036** (wording_variant, similarity 0.88):
  - laughing: <laugh> Sometimes book roulette gets you some pretty neat stuff! What can I say?
  - default, enunciated, happy, sad, whisper: Sometimes book roulette gets you some pretty neat stuff!
  - confused: Uh sometimes book roulette gets you some pretty neat stuff!

## Consequence for the splits

Grouping by exact normalised text would treat a wording variant (for example `i'm not sure` vs `i am not sure`) as a different sentence, so it could land on the other side of the split from its siblings. With text grouping this affected 10.6% to 19.4% of validation clips. The splits therefore group by actor + id number. The table counts validation clips whose id group (same actor, same number) also has a clip in the training set; it must be 0 everywhere:

| split_file | val_clips | val_clips_whose_id_group_is_also_in_train | share |
|---|---|---|---|
| fold_0 | 1596 | 0 | 0.0 |
| fold_1 | 1595 | 0 | 0.0 |
| fold_2 | 1595 | 0 | 0.0 |
| fold_3 | 1595 | 0 | 0.0 |
| random_split | 1596 | 0 | 0.0 |


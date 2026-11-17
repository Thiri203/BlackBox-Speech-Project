# Forward-simulation study (fold 1, ResNet-18 seed 1): materials

Can people guess the style the **model** predicted, from the audio alone (set 1) and from the audio plus the explainer's top 3 segments (set 2)? No training; decisions in `DECISIONS.md` (Human forward-simulation study).

## Explainer rule (fixed before the study)

Best average rank of deletion gain (logit drop minus random, room-noise deletion fill, 100 clips) and stability (mean Kendall tau over the pre-planned tests: 10 repeats, noise 40 dB, gain +1 dB, gain -1 dB; room-noise fill, 50 clips). Rank 1 = best.

| explainer | deletion gain | rank | tau repeats | tau noise 40 dB | tau gain +1 dB | tau gain -1 dB | stability | rank | **average rank** |
|---|---|---|---|---|---|---|---|---|---|
| GradCAM | 1.504 | 3 | 1.000 | 0.760 | 0.984 | 0.975 | 0.930 | 1 | **2.0** |
| IG | 1.659 | 2 | 0.826 | 0.623 | 0.971 | 0.962 | 0.846 | 3 | **2.5** |
| LIME | 2.514 | 1 | 0.892 | 0.694 | 0.975 | 0.976 | 0.884 | 2 | **1.5** |

**The rule gives LIME.** Reading stability from any single test instead gives LIME the best average rank every time (`explainer_rule_robustness.csv`): average rank repeats 1.5, noise_40db 1.5, gain_plus_1db 1.5, gain_minus_1db 1.0.

## Clips

20 of the 100 fold-1 clips (seed 42, first draw meeting the constraints: attempt 12): two sets of 10, each with 6 clips the model got right and 4 it got wrong, all 7 true styles, no style differing by more than one clip between the sets.

| set | condition | model right | model wrong | confused | default | enunciated | happy | laughing | sad | whisper | model predictions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | audio only | 6 | 4 | 2 | 1 | 2 | 1 | 1 | 1 | 2 | enunciated 3, sad 2, whisper 2, default 2, laughing 1 |
| 2 | audio + highlights | 6 | 4 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | default 4, sad 3, confused 1, whisper 1, happy 1 |

In 17 of the 20 clips one of LIME's top 3 segments is the clip start (0-0.25 s), as in the explainer report; participants in set 2 will often see a box on the first quarter second.

## Running the study

1. Give each participant the folder `participant_bundle/` (study.html, audio/, img/) and a participant id. The page works offline in Chrome, Edge or Firefox: open `study.html`. It holds no answer key.
2. Fixed order for everyone: set 1 audio only, then set 2 audio plus highlights; no feedback; answers cannot be changed. At the end the participant clicks **Download CSV** (`forward_sim_<id>.csv`).
3. Score: `python -m src.explain.score_forward_sim forward_sim_*.csv` (accuracy per condition, paired per participant, n; prints and saves the comments to `results/comments.csv`). No significance test.

Files: `key.csv` (**not for participants**: clip, true style, the model's prediction = the right answer, top 3 segments), `items.csv` (what the page shows), `explainer_rule*.csv`, `selection_mix.csv`. Audio and pictures are not in git; rebuild them with `python -m src.explain.forward_sim` (same seed, same files).

## Limits

- **Practice cannot be separated from the explanation effect:** the highlight condition always comes second, so any change from set 1 to set 2 also contains learning the task (and tiredness).
- The sets hold different clips, so clip difficulty is mixed into the difference.
- 10 clips per condition: one clip is 10 points; the result is descriptive, no significance is claimed.
- The task is to guess the model, not the true style; people who answer what they hear score 6/10 per set by construction (the 6 clips the model got right).
- One model, one actor (ex02), one explainer (LIME, room-noise fill), top 3 segments only.

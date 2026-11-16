# ResNet-18 (ImageNet-pretrained): results over all folds and seeds

Leave-one-actor-out: in fold k the model is tested on one actor whose voice it never heard in training. Settings: `configs/train.yaml` (unchanged for all runs). Test set used once per run, with the checkpoint that had the best validation macro-F1. Spread: sample standard deviation (sd) and the range.

Finished runs: 8 of 8.

## Overall

- Test accuracy over all 8 runs: 55.2% (sd 11.7 pts; 36.2% to 68.6%)
- Test macro-F1 over all 8 runs: 0.517 (sd 0.125; 0.319 to 0.670)
- Best validation macro-F1 (same voices as training): 0.991 (sd 0.004; 0.982 to 0.995)
- Length-only baseline accuracy: 20.4% (sd 0.8 pts; 19.5% to 21.5%); chance is 14.3%

## Per fold (test actor), over seeds

| fold | test_actor | runs | test accuracy: mean (sd; min to max) | test macro-F1: mean (sd; min to max) |
|---|---|---|---|---|
| 0 | ex01 | 2 | 51.8% (sd 3.8 pts; 49.1% to 54.4%) | 0.491 (sd 0.023; 0.475 to 0.507) |
| 1 | ex02 | 2 | 68.2% (sd 0.6 pts; 67.8% to 68.6%) | 0.660 (sd 0.014; 0.651 to 0.670) |
| 2 | ex03 | 2 | 39.5% (sd 4.6 pts; 36.2% to 42.7%) | 0.346 (sd 0.039; 0.319 to 0.374) |
| 3 | ex04 | 2 | 61.3% (sd 0.3 pts; 61.0% to 61.5%) | 0.571 (sd 0.013; 0.562 to 0.580) |

## Per seed, over folds

| seed | runs | test accuracy: mean (sd; min to max) | test macro-F1: mean (sd; min to max) |
|---|---|---|---|
| 1 | 4 | 56.7% (sd 11.0 pts; 42.7% to 68.6%) | 0.528 (sd 0.123; 0.374 to 0.670) |
| 2 | 4 | 53.6% (sd 14.0 pts; 36.2% to 67.8%) | 0.506 (sd 0.144; 0.319 to 0.651) |

## Every run

| fold | test_actor | seed | test accuracy | test macro-F1 | best val macro-F1 | length-only acc / F1 | epochs (best) | minutes |
|---|---|---|---|---|---|---|---|---|
| 0 | ex01 | 1 | 54.4% | 0.507 | 0.994 | 20.4% / 0.148 | 28 (22) | 12.9 |
| 0 | ex01 | 2 | 49.1% | 0.475 | 0.990 | 20.4% / 0.148 | 12 (6) | 5.5 |
| 1 | ex02 | 1 | 68.6% | 0.670 | 0.994 | 19.5% / 0.141 | 18 (12) | 8.7 |
| 1 | ex02 | 2 | 67.8% | 0.651 | 0.995 | 19.5% / 0.141 | 26 (20) | 11.8 |
| 2 | ex03 | 1 | 42.7% | 0.374 | 0.988 | 20.3% / 0.149 | 19 (13) | 8.7 |
| 2 | ex03 | 2 | 36.2% | 0.319 | 0.982 | 20.3% / 0.149 | 14 (8) | 6.4 |
| 3 | ex04 | 1 | 61.0% | 0.562 | 0.992 | 21.5% / 0.155 | 26 (20) | 11.8 |
| 3 | ex04 | 2 | 61.5% | 0.580 | 0.992 | 21.5% / 0.155 | 24 (18) | 10.9 |

## Notes

- The test actor matters more than the seed: mean test accuracy per actor ranges from 39.5% (ex03)
  to 68.2% (ex02), while the two seeds of the same fold differ by 0.5 to 6.5 points.
- Validation (same voices as training) is about 0.99 macro-F1 in every run; test (a new voice) is
  0.32 to 0.67. The model recognises styles far better in voices it has heard.
- Early stopping: some runs stopped well before 30 epochs (fold 0 seed 2 after 12 epochs, fold 2
  seed 2 after 14). In folds 0 and 2 the seed that stopped earlier scored lower on test (by 5.3 and
  6.5 points); in fold 1 the earlier-stopping seed scored slightly higher (by 0.8 points), and in
  fold 3 both ran long. With 2 seeds per fold this pattern is not clear evidence either way.
- The length-only baseline does not depend on the seed (same data, deterministic fit), so its spread
  is across the 4 test actors only. It stays near chance in every fold (19.5% to 21.5%).

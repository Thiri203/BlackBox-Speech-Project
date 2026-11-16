## Notes

- **Seeds.** B has one seed; A and the baseline have two (seed 2 for A added later, see below). In the
  baseline, the two seeds of the same fold differ by 5.3 points of accuracy (fold 0) and 6.5 points
  (fold 2). A difference smaller than that is not clear evidence.
- **A (augmentation) helped on fold 0 and did not help on fold 2.** Fold 0: 54.4% -> 65.0% accuracy,
  macro-F1 0.507 -> 0.647, larger than the seed-to-seed spread; recall rose for confused, default,
  enunciated and happy, fell for laughing (0.87 -> 0.61). Fold 2: 42.7% -> 42.4% accuracy, macro-F1
  0.374 -> 0.388, i.e. no real change; errors moved between styles (laughing recall 0.06 -> 0.44,
  happy 0.52 -> 0.13), enunciated and sad stay near 0.
- **A had not finished improving.** Both A runs used all 30 epochs; on fold 2 the best validation
  epoch was 29. Augmented training learns more slowly, so 30 epochs may be too few for it.
- **B (validation on a new voice) did not help.** Fold 0: 54.4% -> 41.9% accuracy, macro-F1 0.507 ->
  0.394 (worse). Fold 2: 42.7% -> 43.8%, macro-F1 0.374 -> 0.426 (slightly better, within the seed
  spread). Sad recall on fold 2 rose from 0.03 to 0.49, but overall the change is not a clear gain.
- **B stopped very early** (best epoch 7 on fold 0, 3 on fold 2): validation macro-F1 on the held-out
  actor peaks after a few epochs and then falls while training on the other two voices keeps improving.
- **B changes two things at once.** Besides the validation set, the model trains on only 2 voices
  (about 5,320 clips) instead of 3 (about 6,380). Part of the fold 0 loss may come from that.
- **B's validation score is a much better guide to the test score.** Best validation macro-F1 was
  0.615 (fold 0) and 0.685 (fold 2) versus about 0.99 for the baseline and A, whose validation set
  contains the training voices. Test macro-F1 was 0.394 and 0.426, so B's validation is still too
  optimistic, but far less than the baseline's.

### Does the gain of A hold on seed 2?

- **Fold 0 (ex01): yes.** A beats the baseline on both seeds by a similar margin: accuracy +10.6 and
  +9.7 points, macro-F1 +0.140 and +0.107. Both gains are larger than the baseline's seed-to-seed
  spread (5.3 points). The pattern also repeats: recall rises for confused, default, enunciated and
  happy on both seeds; laughing and sad drop a little.
- **Fold 2 (ex03): not clearly.** Seed 1: no change (-0.3 points, macro-F1 +0.014). Seed 2: +8.2
  points, macro-F1 +0.087, but mostly because the seed-2 baseline was weak (36.2%, it stopped at
  epoch 14). A gave 42.4% and 44.4%, about the level of the better baseline seed (42.7%). Over the
  two seeds A averages 43.4% / 0.397 against the baseline's 39.5% / 0.346; A looks more stable here,
  not clearly better. On both seeds A trades happy recall (-0.39, -0.41) for laughing (+0.38, +0.59);
  enunciated and sad stay near 0 with or without A.
- **Caveat: A also trains longer.** The A runs used 28 to 30 epochs, the baseline runs 12 to 28 (they
  stopped early). Part of the gain could come from the longer training rather than the augmentation
  itself; the two are not separated by these runs.

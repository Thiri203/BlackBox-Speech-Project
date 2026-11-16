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

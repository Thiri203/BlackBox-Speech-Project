# Project notes for Claude

Speaking-style classification (7 styles, Expresso dataset) from log-mel spectrograms, evaluated
leave-one-actor-out (4 folds). Decisions and their reasons are in `DECISIONS.md`; keep it updated.

## Standing rule: test confusion matrix and per-style metrics in every report

Every time a model is trained or tested (any fold, any seed, any new model, and also the logistic
regression baseline(s)), the report for it MUST include:

1. **The test confusion matrix**: rows = true style, columns = predicted style; each box shows the
   share of its row (percent) and the clip count.
2. **A table of per-style precision, recall and F1** on the test set, plus the macro average.

This holds for a single-run report, an all-runs summary and any new model or baseline report.
Do not report only accuracy / macro-F1.

How it is done today:
- `src/evaluate.py: classification_metrics()` returns `confusion_counts`, `confusion_row_normalised`
  and `per_class` (precision, recall, F1, support). Reuse it for new models and baselines.
- Single run: `src/report_training.py` (e.g. `reports/training_fold0/`).
- All folds and seeds: `python -m src.eval_checkpoints` (loads each `models/fold{k}_seed{s}_pretrained/best.pt`,
  predicts the test actor, writes `reports/training_all/confusion_fold{k}.png` and
  `checkpoint_test_metrics.json`), then `python -m src.report_all` (fails if a finished run is missing).
- For finished runs, re-evaluate the saved best checkpoint; do not retrain just to get the matrix.

## Other conventions

- The test set is used once per run, with the best-validation checkpoint; never tune on it.
- Data (`data/`), spectrogram caches and `models/` checkpoints are not committed (see `.gitignore`).
- Commit when asked; do not push unless asked.

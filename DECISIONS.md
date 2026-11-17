# Decisions log

Every preprocessing choice, one line of reasoning, and the config key in
[`configs/preprocess.yaml`](configs/preprocess.yaml) that changes it. Numbers refer to the
dataset commit pinned below. Rerun `python -m src.run_preprocess` after any change.

## Data source

| Decision | Why | Config key |
|---|---|---|
| Hugging Face `ylacombe/expresso`, files `read/*` only | The read part has the 4 actors x 7 styles we need; conversational audio is not used. | `dataset.repo_id`, `dataset.allow_patterns` |
| Pin commit `9fb79a189698de3255eff48edd2bc0d9e487adc0` | Everyone downloads byte-identical files even if the dataset is updated. | `dataset.revision` |
| Decode with `soundfile` from the parquet bytes, no `datasets` library | Fewer dependencies, full control over decoding; we measure the real sample rate. Result: all 11,615 clips are 48 kHz, mono. | (code) |

## Which clips are kept

| Decision | Why | Config key |
|---|---|---|
| Keep only confused, default, enunciated, happy, laughing, sad, whisper | The 7 target classes. The `style` column also holds `emphasis` (800), `essentials` (160), `longform` (8), `singing` (11): 979 rows excluded. There is no "narration" style in this config. | `labels.styles` |
| Do **not** relabel emphasis/essentials as "default" | They are read in the default voice (ids `ex0X_default_emphasis_*`), but adding 960 clips would make "default" 1.6x larger than the other classes and their sentences are a different text set. Set `[emphasis, essentials]` to include them. | `labels.merge_into_default` |
| Label ids alphabetical: confused 0, default 1, enunciated 2, happy 3, laughing 4, sad 5, whisper 6 | Simple, stable, documented order. The code checks the list is sorted. | `labels.styles` |
| Remove duplicate ids and byte-identical audio (keep first by id) | Found 8 duplicate ids and 8 identical-audio copies. All 8 are the `longform` rows: each reuses id `ex0X_default_00001` and carries the exact audio of that default clip (twice per actor), with a long paragraph as text. So the mirror's long-form audio is wrong, not just duplicated. They are already excluded by the style filter; **0 duplicates remain among the 7 styles**, so 0 rows were removed for duplication. | (code) |
| **No clip is dropped for length** (final plan) | All 10,636 7-style clips are kept: 1,518 to 1,520 per style, 379 or 380 per actor x style. | `audio.long_clip_policy` |

## Input window: 4 s, crop instead of drop (final plan)

| Decision | Why | Config key |
|---|---|---|
| Window of 4 s | Final plan after feedback. 72.8% of the clips are 4 s or shorter (median 2.95 s); a fixed short window keeps the input small (128 x 400) and splits evenly into 16 segments of 250 ms for the explanation tests. | `audio.max_seconds` |
| Clips longer than 4 s keep their **first** 4 s ("crop"); shorter clips are padded with silence | Nothing is lost from the class balance. `was_cropped`, `original_duration` and `original_n_samples_16k` are in the manifest; `duration_s` and `n_samples_16k` describe the real audio **inside** the window. `"drop"` restores the old behaviour (remove long clips). | `audio.long_clip_policy` |
| eGeMAPS uses the same first 4 s | So the baseline and the ResNet see exactly the same audio. | (code, `src/features/egemaps.py`) |

History: the window was first 8 s (drop; 324 clips dropped, 144 of them enunciated), then
10 s (drop; 108 dropped, 61 enunciated), then this final 4 s crop.

Clips cropped to their first 4 s: 2,893 of 10,636 (27.2%), per actor x style
(also in `reports/cropped_actor_style.md`):

| actor | confused | default | enunciated | happy | laughing | sad | whisper | total |
|---|---|---|---|---|---|---|---|---|
| ex01 | 137 (36.1%) | 79 (20.8%) | 204 (53.7%) | 94 (24.7%) | 180 (47.4%) | 78 (20.6%) | 106 (28.0%) | 878 (33.0%) |
| ex02 | 142 (37.4%) | 56 (14.7%) | 162 (42.6%) | 67 (17.6%) | 136 (35.8%) | 93 (24.5%) | 85 (22.4%) | 741 (27.9%) |
| ex03 | 93 (24.5%) | 45 (11.8%) | 176 (46.3%) | 50 (13.2%) | 86 (22.6%) | 101 (26.6%) | 55 (14.5%) | 606 (22.8%) |
| ex04 | 120 (31.6%) | 50 (13.2%) | 200 (52.6%) | 48 (12.6%) | 106 (27.9%) | 67 (17.6%) | 77 (20.3%) | 668 (25.1%) |
| total | 492 (32.4%) | 230 (15.1%) | 742 (48.8%) | 259 (17.0%) | 508 (33.4%) | 339 (22.3%) | 323 (21.3%) | 2,893 (27.2%) |

Cropping is uneven: half of the enunciated clips are cut, but only 15% of default clips.

## Audio processing

| Decision | Why | Config key |
|---|---|---|
| Mono, resample 48 kHz to 16 kHz with `soxr_hq` | 16 kHz keeps everything up to 8 kHz, which holds nearly all speech information, at a third of the data. soxr is deterministic and high quality. | `audio.target_sr`, `audio.resample_type` |
| Save resampled clips (full length, not cropped) as 24 bit FLAC in `data/interim/audio16k/` | Lossless and compact; eGeMAPS, LIME and counterfactual edits use the same audio as the model (taking the first `n_samples_16k` samples). | `audio.audio16k_subtype` |
| No loudness, peak or RMS normalisation anywhere | Loudness is a style cue and a counterfactual variable; the model must be able to see it. | (code, intentionally absent) |
| Only warn about clipping (peak >= 0.999) or near silence (RMS < -50 dBFS) | Nothing is removed for this. Result: 5 possibly clipped clips, 0 near-silent, see `reports/audio_warnings.csv` (measured on the full clip). In two of them a few samples exceed full scale after resampling and are clipped to +/-1 when saved (negligible). | `audio.clip_warn_peak`, `audio.silence_warn_dbfs` |
| Pad with zeros at the end to 4 s; store `n_samples_16k` and `n_frames_valid` | Every input has the same shape; later code can ignore padding. Padding becomes the fixed floor value (-100 dB) in the spectrogram. | `audio.max_seconds` |
| Decode with 4 parallel workers | More workers ran out of memory on the dev machine. Does not affect results. | `audio.num_workers` |

## Log-mel spectrogram

| Decision | Why | Config key |
|---|---|---|
| 25 ms Hann window (400 samples), n_fft 512, hop 10 ms (160) | Standard speech settings; 100 frames per second. | `spectrogram.win_length`, `n_fft`, `hop_length`, `window` |
| Exactly 400 frames per clip (frame t centred at t x 10 ms) | librosa returns 401 frames for 4 s (one more centred exactly at 4.000 s); we drop that last frame so 4 s = 400 frames = 16 segments x 25 frames. | (code, from `audio.max_seconds`) |
| A frame counts as real when its centre lies inside the real audio: `n_frames_valid = ceil(n_samples_16k / 160)` | Clear rule shared by the statistics and the segment table. | (code) |
| 128 mel bands, 20 Hz to 8,000 Hz | Fine frequency detail for a CNN; 8 kHz is the top frequency at 16 kHz. | `spectrogram.n_mels`, `fmin`, `fmax` |
| Power spectrogram, dB with fixed reference 1.0 | A per-clip reference (librosa's `ref=np.max`) would erase loudness differences between clips. | `spectrogram.power`, `ref` |
| Floor at -100 dB (amin 1e-10) | Silence and padding get a defined value instead of minus infinity. | `spectrogram.amin`, `floor_db` |
| `center=True`, zero padding at the edges | Frame t is centred at t x 10 ms; zero padding (not reflection) adds no fake signal. | `spectrogram.center`, `pad_mode` |
| Mel filters: librosa defaults (Slaney scale and area normalisation) | The common default; not exposed separately. | (librosa default) |
| Store raw dB values, unstandardised, as float16, one `.npy` per actor + `index.csv` | float16 rounding is at most 0.03 dB here; halves disk use (about 1.1 GB total, 128 x 400 per clip). One file per actor loads fast with memory mapping. | `spectrogram.dtype` |

## Segment table

| Decision | Why | Config key |
|---|---|---|
| `data/processed/segments.csv`: 16 segments of 250 ms per clip with `is_real_audio`, `n_real_frames`, `mean_db` | Building block for the deletion and pause tests. A partly real segment counts as real. `mean_db` = mean over all mel bands and the segment's real frames, empty for pure padding. No speech / pause threshold yet. | `segments.length_ms` |
| Keep `n_real_frames` in the table | The last real segment of a clip may hold only 1 or 2 real frames of the clip's quiet ending, whose analysis windows reach into the padding; 369 real segments (0.3%) have `mean_db` below -90 dB, 3 sit at -100 dB. `n_real_frames` lets later code ignore such slivers. | (code) |

## Splits

| Decision | Why | Config key |
|---|---|---|
| **Group every split by `group_id` = actor + id number** (e.g. `ex01_happy_00260` -> `ex01_00260`) | Final plan. The parallel text check showed that the id number, not the exact text, identifies the prompt: in 258 of 1,520 id groups some style uses a wording variant ("I'm" / "I am"). Grouping by text let these variants sit in validation while their siblings were in training (10.6% to 19.4% of validation clips). Now 0, checked by assertion. | (code) |
| Leave-one-actor-out, 4 folds, fold k tests on the k-th actor in sorted order (ex01..ex04) | Measures generalisation to an unseen voice. | (code) |
| LOAO validation: about 20% of the other 3 actors' clips, `StratifiedGroupKFold(5, shuffle, seed 42)`, first fold, stratified by actor x style | Every actor x style combination is represented. | `splits.val_fraction`, `seed` |
| **Random split** `random_split.csv`: 60 / 15 / 25 train / val / test, all 4 actors in every part | Final plan: measures style recognition when the voice is known. Grouped by `group_id`, so only the speaker can leak, never the prompt. Test is drawn first (`StratifiedGroupKFold(4)`), then validation from the rest (`StratifiedGroupKFold(5)`), both seed 42. Stratified by actor x style, which also stratifies by style and guarantees every actor in every part. Result: 59.99% / 15.01% / 25.00%; each part has 1/7 of every style and roughly a quarter of each actor. | `splits.random`, `seed` |
| Assertions (run stops on failure) | Every split: no clip twice, no id group in two parts. LOAO: the test actor never in train or val, only the test actor in test. Random: every actor and every style in every part. | (code) |
| Report, do not fix, exact sentences shared across parts (`reports/sentence_overlap.csv`) | The same normalised text can still sit in two different id groups (another actor, or the same actor under another number). This is now tiny: at most 7 clips per comparison (LOAO test vs train: fold 1 4 clips, fold 2 7 clips; LOAO val vs train: 1 clip in folds 0, 2 and 3; random val vs train 5 clips, random test vs train 1 clip). | (report only) |
| Text normalisation (lowercase, remove `*`, tags like `<laugh>`, punctuation; keep apostrophes) | Now used only for the reports above, not for grouping. | `splits.text_strip_chars`, `text_remove_tags`, `text_remove_punctuation` |
| Normalisation statistics from TRAIN clips only, padding frames excluded, global and per mel band, stored as JSON, not applied | Avoids leaking validation/test information; padding would drag the mean towards -100 dB. Computed exactly from the stored float16 values. One file per fold and one for the random split. | (code) |
| Fill values for deletion tests from TRAIN clips only, real frames only: per-band mean dB ("average" fill) and per-band 10th percentile dB ("room noise" fill), `fill_values_{fold_k,random}.json` | Writing the padding value (-100 dB) into a deleted segment creates a "perfect silence" never seen inside real speech; room-noise level is a more natural "nothing here", the mean a neutral alternative. Exact percentiles via value counts over float16 (inverted-CDF definition). Gap of the room-noise level above the padding value, over bands: median 18.9 to 20.6 dB depending on the split, smallest 6.8 to 8.2 dB, largest 57.2 to 59.5 dB (lowest band highest). | `splits.fill_percentile` |
| Top band note | Band 127 (about 7.8 kHz) is the one exception: its room-noise level equals the padding value (-100 dB) in every split, because converting to 16 kHz removes almost all sound just below 8 kHz. | (observation) |

## Training (ResNet-18), settings in `configs/train.yaml`

| Decision | Why | Config key / flag |
|---|---|---|
| torchvision ResNet-18, ImageNet weights, first convolution 1 channel = average of the pretrained RGB filters, last layer 7 classes | Final plan. Averaging keeps the pretrained filters' response to a grey image; 7 outputs match label ids 0..6. | `model.*`, `--pretrained` |
| From-scratch pilot (run B) with identical settings except learning rate 1e-3 | The proposal asks whether ImageNet pretraining helps on spectrograms. A higher rate is usual when starting from random weights. | `--pretrained false`, `optim.lr_scratch` |
| Input standardised with the fold's TRAIN-only global mean and std (padding excluded from the statistics) | Exactly as written by preprocessing; no information from validation or test. Because padding is excluded, standardised inputs average about -0.6 rather than 0 (padding sits at about -2.3); that is expected. | `norm_stats_fold_{k}.json` |
| AdamW, lr 3e-4, weight decay 1e-4, cosine schedule over 30 epochs, batch 32 | Common, stable defaults for fine-tuning a ResNet. | `optim.*`, `train.batch_size`, `train.max_epochs` |
| Early stopping, patience 6, on validation macro-F1; keep the best checkpoint by validation macro-F1 | Macro-F1 weighs all 7 styles equally. | `train.early_stopping_patience` |
| No data augmentation | Final plan: keep the first experiments simple and interpretable. | `train.augmentation` |
| Mixed precision (float16 autocast + GradScaler), channels_last | Faster training and less GPU memory; standard practice (speed-up not measured here). | `train.mixed_precision`, `train.channels_last` |
| Seeds fixed for Python, NumPy, torch and the data shuffling; cuDNN deterministic | Repeatable runs; saved in `settings.json`. Some GPU operations have no deterministic version (torch only warns), so reruns can differ slightly. | `--seed`, `train.deterministic` |
| Whole split held in RAM, `num_workers = 0` | About 1.1 GB for all clips; worker processes are slow to start on Windows and unnecessary here. | `train.num_workers` |
| Test set used once, after training, with the best checkpoint | Avoids choosing anything on the test actor. | (code) |
| Shortcut check: logistic regression on `duration_s` and `was_cropped` only, same train and test clips | The share of the 4 s window that holds real audio depends on style, so length alone carries some style information. Reported next to chance (1/7) and the majority class. | (code, `src/evaluate.py`) |
| Plan: 2 seeds x 4 folds, pretrained only; random split not used for now | Final plan. Fold 0 seed 1 trained first; the other 7 runs are in `scripts/run_remaining.py`. | (script) |
| torch 2.11.0 + torchvision 0.26.0, CUDA 12.8 build | The RTX 4060 driver supports CUDA up to 13.1; the cu128 wheels are the current official CUDA 12 builds. No Smart App Control blocks. | `requirements.txt` |

## Experiments: augmentation (A) and validation on a new voice (B), seed 1, folds 0 and 2

| Decision | Why | Config key / where |
|---|---|---|
| Each experiment has its own config and output folder (`models/fold{k}_seed1_aug`, `..._valactor`); `src/train.py` refuses to write into a folder that has `DONE` | Earlier runs must never be overwritten. | `run_tag` |
| A: augmentation on the cached log-mel, TRAIN clips only (the dataset refuses augmentation for val / test) | Spectrograms are cached, so augmentation works on them, not on audio. | `train.augmentation`, `src/data/augment.py` |
| A: random gain +/- 6 dB added only to real values above the floor, then clamped at -100 dB | A volume change does not turn digital silence into sound; keeping the padding at the floor avoids creating a new padding level the model never sees at test time. | `gain_db` |
| A: frequency shift +/- 4 mel bins with the edge band repeated, not wrapped around | A wrap-around roll would move the empty top bands to the bottom, which is not like a pitch shift. | `freq_shift_bins` |
| A: time shift +/- 25 frames (0.25 s); vacated frames get the padding value | Same value as real padding. | `time_shift_frames` |
| A: Gaussian noise sd 1 dB on real values (the data's sd is about 23 dB) | "Light" noise. | `noise_std_db` |
| A: SpecAugment 1-2 frequency masks of 0-12 bins and 1-2 time masks of 0-40 frames (time masks inside the real audio); masked values = 0 after standardisation = training mean | Masking with the floor would look like padding. | `freq_mask*`, `time_mask*` |
| A: magnitudes drawn uniformly each time a clip is loaded (so some clips are barely changed); one numpy generator seeded with the run seed | Repeatable; `num_workers = 0`. | (code) |
| B: validation = all clips of one training actor, training = all clips of the other two; early stopping and checkpoint on that actor's macro-F1 | Validation on the same voices gave about 0.99 macro-F1 in every run while test was 0.32 to 0.67, so it says little about a new voice. | `validation.mode: holdout_actor` |
| B: validation actor = the training actor after the test actor in sorted order (wrapping): fold 0 -> ex02, fold 2 -> ex04 | A fixed rule decided before any B run. It uses no test result (choosing e.g. the "hardest" actor from earlier test scores would use test information). | `validation.val_actor_by_fold` |
| B: normalisation mean/std recomputed from the 2 training actors only (same method as preprocessing: real frames only) | Keeps the validation actor out of the training statistics. The split used is saved as `split_used.csv` in the run folder. | (code, `fold_statistics`) |

## Explanation phase: which fold and model are explained

| Decision | Why | Where |
|---|---|---|
| The explainers (Grad-CAM, IG, LIME and the explanation tests) run on **one fold only: fold 1 (test actor ex02)**. The main model is the baseline pretrained ResNet-18 (no augmentation), **seed 1** (`models/fold1_seed1_pretrained/best.pt`). The accuracy table stays on all 4 folds. | Fold 1 has the highest ResNet test accuracy (68.6% seed 1, 67.8% seed 2; macro-F1 0.670 / 0.651), and the LR baseline is also strongest on ex02, so there are more correct predictions to explain. LR = Aung's logistic regression on the 88 eGeMAPS features, from his report of 12 Nov 2026 (his folds are numbered 1 to 4, ours 0 to 3, so our fold k is his fold k+1). Leave-one-actor-out accuracy: ex01 57.2%, ex02 65.7% (his fold 2, his highest), ex03 43.8%, ex04 64.2%; mean 57.7%, macro-F1 0.561. **Aung's numbers, not checked in this repository** (his code and features are not here). The length-only shortcut check is a different model (19.5% on ex02). | (this section) |

## LRfS: white-box check for the explainers (fold 1), `src/train_lrfs.py`, `src/models/lrfs.py`

From the final plan (point 2): a logistic regression whose segment contributions are exact; LIME and
Integrated Gradients are scored against that truth with Kendall's tau in the same 16-segment space and
with the same fill, ranking real segments only, reported for each of the three fills. Grad-CAM needs a
convolutional layer and is covered by the sanity check instead. Not a competing classifier.

| Decision | Why | Where |
|---|---|---|
| 16 features per clip: x_j = mean of the cached log-mel (dB) over all 128 bands and all 25 frames of segment j, padding counted at -100 dB (not the real-frame `mean_db` of `segments.csv`) | LRfS must be an exact linear function of the same 128 x 400 input the ResNet gets, so the same explainer code runs on both. `LRfSTorch` takes the standardised ResNet input and reproduces the logits (max difference 2.7e-7). | `segment_features`, `LRfSTorch` |
| StandardScaler on the fold-1 train split; C from {0.001, 0.01, 0.1, 1, 10, 100} by validation macro-F1 (ties -> smaller C); test actor ex02 used once after C was fixed | Same discipline as the ResNet. Result: C = 1 (val macro-F1 0.289; all C between 0.272 and 0.289). | `C_GRID` |
| Contributions for the **predicted class's logit**: phi_j = V_cj (x_j - f), V = weights with the scaler folded in | The ResNet explainers must explain the same quantity. Exact in logit space, not in probability space. | `contributions` |
| Real segments only (`is_real_audio`; partly real counts as real, filling replaces all 25 frames); padding segments get phi = 0 | The plan ranks only segments with real audio. The exact-sum identity therefore holds over real segments only: sum of phi = logit(clip) - logit(every real segment filled). Checked by writing the fill into the spectrogram: max error 5e-12. | report |
| Three fills, all from fold-1 training clips only; scalar f = mean over bands of the per-band fill: **room noise -78.23 dB (main)**, **average -45.74 dB (second)**, **silence -100 dB (comparison)** | The plan's three fills. **Silence is the padding value -100 dB.** The fill-values decision above does not forbid it; it records that -100 dB is "perfect silence" never seen inside real speech, which is why room noise is the main fill. Silence is kept because the plan asks for it and because a deletion that looks like padding is exactly where a length/padding shortcut would show. | `fill_values_fold_1.json` |
| Result (fold 1, test ex02): accuracy 31.4%, macro-F1 0.279 (ResNet seed 1: 68.6% / 0.670; length only 19.5%; chance 14.3%). Whisper is recognised (recall 0.94), the other styles are weak. | A weak but exact model is fine for a white-box check; its weights concentrate on the first 1 to 2 segments, so the truth rankings are dominated by s0. | `reports/lrfs_fold1/report.md` |

## Explainers and the white-box check (fold 1), `src/explain/`

| Decision | Why | Where |
|---|---|---|
| IG and LIME are written in plain torch / numpy, not with `captum` / `lime` | Neither package is installed, and new compiled packages have been blocked by Smart App Control before. Both are short. LIME follows the lime_image / lime_base defaults (first sample = original, uniform on/off, cosine distance, kernel width 0.25, weighted Ridge alpha 1). | `ig.py`, `lime_segments.py` |
| One explainer code path for LRfS and the ResNet: model on the standardised [N, 1, 128, 400] input, target = the predicted class's logit, eval mode, float32, no autocast | The white-box check then tests the same code that later explains the ResNet. | `segments.py` |
| IG: 50 steps, midpoint Riemann sum, baseline = the clip with every real segment replaced by the fill (padding unchanged), pixel attributions summed per segment | Same reference as the exact contributions, so IG's segment sums and the truth measure the same difference. | `ig.py` |
| LIME: 500 samples; features = the clip's real segments only; "off" = the per-band fill written into all 25 frames | The plan ranks only segments with real audio; switching padding would change nothing for LRfS and only add a padding/length cue for the ResNet. | `lime_segments.py` |
| 100 clips of the fold-1 test actor, stratified by true style (`train_test_split`, seed 42), chosen without any model's prediction; fixed in `reports/explain_fold1/clips_fold1.csv` and reused for the ResNet | Same clips for every explainer and model; choosing by correctness would make the set depend on the model. | `run_whitebox.py` |
| Scores over real segments only: Kendall's tau-b on signed values (main) and on absolute values; top-3 overlap on signed values (segments that most support the prediction) and on absolute values; each fill scored against the truth for the same fill | The plan's metrics; signed values keep the direction of each segment's effect. | `run_whitebox.py` |
| ResNet explanations (`run_resnet.py`): fold 1 seed 1 `best.pt` in float32, eval mode, no autocast; target = its predicted class's logit | float32 gives the same predictions as the saved mixed-precision run on all 100 clips, and gradients are cleaner without autocast. | `run_resnet.py` |
| Grad-CAM on `layer4` (map 4 x 13 for a 128 x 400 input), upsampled bilinearly, summed per segment; no fill | The standard last-conv-layer choice. Its time cells (about 0.31 s) are coarser than one segment, which limits how sharply it can rank segments. `layer3` (8 x 25) would be finer; not used. | `gradcam.py` |
| **Accepted:** energy label for "speech vs pause": a real segment is speech if its `mean_db` (segments.csv, real frames) >= -55.58 dB, else pause. Threshold = Otsu's threshold on the real segments of the fold-1 TRAIN clips | The histogram is bimodal (pause/room hump about -85 to -60 dB, speech peak about -45 to -35 dB); Otsu picks the valley from training data only, with no hand-tuning. | `run_resnet.py: otsu` |
| `captum==0.9.0` installed only to cross-check our IG (10 clips x 3 fills, riemann_middle, 50 steps); never used for results | The user asked for an independent check. A dry run showed it installs nothing else. Result: identical segment rankings (tau 1.0), largest score difference 0.5% of the largest score (float32 / cuDNN ordering). | `requirements.txt` |
| Rankings use real segments only and the signed score (highest = most support for the prediction); Grad-CAM keeps its raw value on padding segments in the saved file | Same rule as the white-box check; deletion curves will delete real segments only. | `resnet_segment_scores.csv` |
| Top-1 position check: first real segment / last real segment (next to the padding) / in between, x speech/pause, split by correct and wrong, against a random real segment | Asked to test whether the pauses the methods point at are the clip end (a length cue). Result: they are the clip **start** (first real segment, 39-66% for IG/LIME vs 8% at random); the last segment is almost never top-1. | `run_resnet.py: top1_position` (`--report-only` rebuilds the report from the saved scores) |
| Deletion curves (`run_deletion.py`): delete the top-k real segments, k = 1..min(8, real segments), in each method's order vs 10 random orderings (same 10 per fill, seeded per clip); IG and LIME deleted with their own fill, Grad-CAM with all 3 | The plan's faithfulness test; never deleting padding or more segments than the clip has. | `K_MAX`, `N_RANDOM` |
| Track the drop of the predicted class's logit (main) and probability; area per clip = mean drop over k = 1..K, so short clips are not extrapolated; method minus random per clip with a 95% bootstrap interval over clips (2,000 resamples, seed 42) | Paired against the same clip's random orderings; the same K for both. | `run_deletion.py` |
| The report states that the curve measures reliance on the segment being present in its normal form, not its content (a deleted segment also becomes a pause), and that LIME is fitted on this exact kind of perturbation | Prevents reading a higher area as "this segment's content mattered", or LIME's lead as general proof of faithfulness. | `deletion.md` |
| Stability (`run_stability.py`): 50 of the 100 clips, stratified by correct / wrong (seed 42; 30 correct, 20 wrong), `stability_clips.csv` | The user asked for a fixed subset with a mix of correct and wrong predictions. | `N_CLIPS` |
| Repeats: LIME 10 new seeds (500 samples); Grad-CAM 10 runs as deterministic reference; **IG (a), approved:** baseline + Gaussian noise sd 1 dB on the filled segments only, 10 seeds; **IG (b), side check:** integration points jittered, alpha_k = (k + u_k) / 50, u_k uniform, 10 seeds | IG with a fixed fill has no randomness. (a) asks whether IG's ranking depends on the exact reference value (1 dB = the noise level of experiment A); (b) tests only the numerical integration. Both options are off by default in `ig.py`, so the main results are unchanged. | `IG_BASELINE_NOISE_DB`, `ig.py` |
| LIME also repeated with 1,000 samples (the original plan), 10 seeds, same 50 clips, reported next to 500 | To see whether LIME's repeat instability comes from the smaller sample count. Added after seeing the 500-sample result; the 500-sample result is unchanged. | `LIME_SAMPLES_LARGE` |
| **Noise at 50 dB SNR added as a second level after seeing the 40 dB result** (40 dB moved the predicted-class probability by a median of 0.23 and changed the predicted class for 22 of 50 clips, leaving 16 clips). Same definition and seeding as 40 dB; the 40 dB result is kept in the report unchanged | Requested by the user; recorded so it is clear the level was not fixed in advance. | `PERTURB`, `ADDED_LATER` |
| Input changes on the 16 kHz waveform, then the same log-mel pipeline (rebuilt clean clips equal the cache bit for bit): noise 40 dB SNR = white Gaussian noise on the real samples only, variance = mean power of the real samples in the window / 10^4; gain = x 10^(+-1/20), no clipping; plus a clean rerun as control. Keep a clip only if the predicted-class probability moves < 0.05 | Changes the input the way a recording would change, without touching the padding (length cue). The clean rerun shows explanations are exactly reproducible. | `SNR_DB`, `GAIN_DB`, `P_MAX` |
| No setting was changed after seeing the results (in particular, 40 dB was kept although it moves the model's prediction for most clips) | The task says not to tune any method to look more stable. | (this table) |
| Reading the result: on a linear model IG is exact by construction and LIME's surrogate is the right model class, so near-perfect scores show the code is correct (segment mapping, fills, baseline, target), not that IG or LIME are faithful on the ResNet | Prevents over-reading the white-box result. | `reports/explain_fold1/whitebox_lrfs.md` |

## Planted-tone test (fold 1, class confused), `src/explain/planted_tone.py`

| Decision | Why | Where |
|---|---|---|
| 1 kHz sine, one 250 ms segment aligned to the segment grid, 5 ms raised-cosine fades, **added** to the speech at amplitude sqrt(2) x the clip's RMS over its real samples (0 dB tone-to-speech, approved); never past the real audio; scaled down only if the mix would reach 0.99 (0 of 1,140 needed it, largest peak 0.80) | The plan: audible, no clipping, exactly one segment; fades avoid a broadband click; adding keeps the speech. 0 dB so the planted cue is certainly learnable. | `TONE_HZ`, `RAMP_S`, `MIX_PEAK_MAX` |
| Tone segment drawn uniformly from the clip's fully real (25 real frames) **speech** segments (clean `mean_db` >= -55.58 dB), seeded per clip, recorded in `reports/tone_fold1/tone_plan_train_val.csv` | The plan asks for a random but known segment holding real speech. | `plan_for` |
| Every fold-1 train and val clip of confused is toned (912 + 228); other classes and the whole test actor stay clean | The plan. | `build` |
| Rebuilt with the same log-mel pipeline into a separate cache `data/processed/logmel_tone_confused/`; untoned rows copied byte for byte (checked); the original cache is never written | Same input pipeline as the noise test; no risk to the baseline data. Measured leakage: median 94% of the dB change in the tone segment, 6% in its two neighbours (one edge frame each). | `build` |
| Training: `configs/train_tone.yaml` = `train.yaml` + `run_tag: tone_confused` + `logmel_dir`; seed 1, pretrained. New optional key `logmel_dir` (helper `src.utils.logmel_dir`, used by the dataset and the statistics); absent = baseline unchanged | Same recipe as the explained baseline. | `src/train.py` |
| Normalisation and fills **recomputed from the toned training set** (approved), saved as `norm_stats.json` / `fill_values.json` in the run folder; how far they move is in the report | The recipe computes them from the training clips; the explainers on this model use its own fills. | `src/train.py` |
| Validity: tone (same rule, own seeds) on every ex02 test clip of the 6 other classes; flip = clean prediction not confused and toned prediction confused; baseline model on the same toned clips as control | The plan: score explainers only where the model provably uses the tone. | `evaluate` |
| Explainers on up to 100 flipped clips (stratified by true class, seed 42): Grad-CAM, IG (50 steps), LIME (500 samples), IG/LIME with the 3 recomputed fills; target = confused logit; hit = top-1 real segment is the tone segment; chance = mean 1/(eligible speech segments), 1/16, and 1/(real segments); Wilson 95% intervals | The plan's score, plus the chance level for the pool actually ranked. | `evaluate` |

## Calibration of the ResNet (all 8 baseline runs), `src/calibration.py`

| Decision | Why | Where |
|---|---|---|
| Saved best checkpoints re-evaluated on their test actor with the original inference settings (float16 autocast, channels_last); predictions checked against the saved `test_predictions.csv` (100% identical, probabilities within 8e-7) | No retraining; the test set is used as before, nothing is tuned. | `src/calibration.py` |
| Brier score = mean over clips of the sum over the 7 classes of (p_c - 1[c = true])^2, the definition in Aung's LR report; uniform guess = 6/7 = 0.857 | So the two models are compared on the same scale. | `brier` |
| Aung's LR Brier per actor (report of 12 Nov 2026: ex01 0.604, ex02 0.462, ex03 0.859, ex04 0.495, mean 0.605) joined on the **actor id** (his fold k = our fold k-1); not checked in this repo | Fold numbers differ between the two reports. | `AUNG_LR_BRIER` |
| Reliability: 10 equal-width bins of the highest probability, all test clips pooled over the 4 actors, one curve per seed; ECE given only as a summary of the table | The user asked for a pooled 10-bin table / plot. | `reliability` |
| **No temperature scaling or other calibration** | The question is how calibrated the model is as trained; fitting a temperature would need a calibration set and is a separate step. | (this table) |

## Sanity check: model-parameter randomisation (fold 1, seed 1), `src/explain/run_sanity.py`

Design fixed before any result. The final plan document is not in `docs/`; the task PDF (Step 6) says "we also
randomize the model's weights, and the explanation should turn meaningless". Question: do the explanations depend
on what the model learned?

| Decision | Why | Where |
|---|---|---|
| `models/fold1_seed1_pretrained/best.pt` loaded as an in-memory copy, float32, eval mode; never written | The explained model; existing files (checkpoint, saved scores, reports, cache) are not changed. | `run_sanity.py` |
| **Cascading randomisation, top down**, 7 stages: 0 original; 1 fc; 2 + layer4; 3 + layer3; 4 + layer2; 5 + layer1; 6 + conv1 and bn1 (fully random) | Standard model-randomisation test: destroy the learned weights from the output backwards and watch when the explanation stops matching. | `STAGES` |
| Re-initialisation with PyTorch's default `reset_parameters()` per layer type (Conv2d, Linear, BatchNorm2d), `torch.manual_seed(42)` once, blocks processed in the fixed order fc, layer4, layer3, layer2, layer1, conv1/bn1; batch-norm: weight/bias AND running statistics reset | One seed and a fixed order make each stage's already-randomised blocks identical to the previous stage's (true cascade). Resetting the running statistics removes what batch-norm learned from the data. | `randomise` |
| Same 100 clips (`clips_fold1.csv`), same fills (fold-1 train: room noise -78.23, average -45.74, silence -100 dB), same explainer code and settings (Grad-CAM layer4, IG 50 steps, LIME 500 samples, seeds as in the main run), real segments only | Only the weights change, so any change in the explanation comes from the weights. | `run_sanity.py` |
| **Target = the ORIGINAL model's predicted class**, explained as its logit on every randomised model | Same quantity at every stage; a randomised model's own prediction would change the question. | (report) |
| Compare each stage with the saved original (`resnet_segment_scores.csv`): Kendall tau-b on signed scores (main) and on absolute values, top-3 overlap (signed), share with the same top-1 segment; mean, median, min over clips per method and fill | The plan's ranking metrics, as in the white-box and stability steps. | `compare` |
| Controls: (a) stage 0 rerun vs saved must give tau 1.000; (b) a second fully random model (seed 43) vs the seed-42 one, same target | (a) proves the pipeline reproduces the saved explanations; (b) shows how similar two meaningless explanations are by chance (structure from the input alone). | `run_sanity.py` |
| Undefined tau (a score vector constant over the real segments, e.g. Grad-CAM all zero after ReLU): the clip is excluded from that mean and the number excluded is reported; never filled with 0 | A filled value would invent a result. | `compare` |
| Fixed wording, by mean signed tau at stage 6: < 0.3 "changes strongly", 0.3 to 0.7 "partly changes", > 0.7 "barely changes"; no pass/fail threshold | Wording decided before the results; nothing tuned. | `wording` |
| Model accuracy and macro-F1 at every stage on the 100 clips; confusion matrix and per-style metrics for the original and the fully random model | Shows the model is really destroyed (chance = 14.3%); CLAUDE.md rule. | report |
| Outcome of control (a), reported as is: mean tau >= 0.999, same top-1 in 100% of clips, lowest single-clip tau 0.929 (IG). IG differs from the saved scores by up to 0.1% of its largest score (GPU gradient sums are not bit-reproducible), which swaps nearly tied segments; nothing was changed to force 1.000 | The saved run had the same non-determinism, so bit-exact reproduction is not possible; the control is the noise floor. | `reports/sanity_fold1/report.md` |

## Counterfactual edits and listening-test material (fold 1, seed 1), `src/explain/run_counterfactual.py`

**Design fixed before any edit was run (18 Nov 2026); the ranges below are not changed after seeing results.**
Question: if a clip is edited along one natural dial (speaking rate, loudness, pause length), does the
ResNet's predicted style change, and would a listener hear that change? No training.

| Decision | Why | Where |
|---|---|---|
| Model `models/fold1_seed1_pretrained/best.pt`, float32, eval mode, no autocast (as the explainers); the 100 clips of `clips_fold1.csv`; fold-1 train-only `norm_stats_fold_1.json` | Same model, clips and input as the explainer tests. | `run_counterfactual.py` |
| Each edit is applied to the **full** stored 16 kHz clip; then the same rule as the cache: keep the first 4 s, pad with zeros to 4 s, same log-mel, rounded to float16. Checks: rebuilt unedited spectrograms equal the cache bit for bit; unedited predictions equal the explainers' saved predictions | The model must see an edited clip exactly as it would see a recorded one. Editing before the crop means a faster clip brings later audio into the window and a slower one pushes audio out, as a real faster/slower reading would. | `window_logmel` |
| **Dials and steps (fixed):** speaking rate x0.85, x0.90, x1.10, x1.15; loudness -6, -3, +3, +6 dB; pause length x0.5, x1.5. Order of size: rate 10% (x0.90, x1.10) then 15% (x0.85, x1.15); gain 3 dB then 6 dB; pause one step per direction | The plan's natural ranges. | `DIALS` |
| Rate factor r: new duration = old duration / r (r < 1 slower, r > 1 faster); pitch kept. Method: WSOLA written in numpy: Hann frame 512 samples (32 ms), synthesis hop 256, search +-128 samples (8 ms) for the best-correlating continuation | WSOLA keeps pitch and voice quality and sounds cleaner on speech than a phase vocoder at these mild factors; no new dependency. At r = 1 it returns the input exactly (checked). | `wsola` |
| Gain: waveform x 10^(g/20); no clipping in the model input (float, peaks above 1 kept) | In the log-mel with its fixed reference this is a shift of +g dB of every value above the -100 dB floor. | `edit_gain` |
| **Pauses, same speech/pause rule as before:** per 10 ms frame the mean log-mel dB over the 128 bands, averaged over a sliding 25-frame (250 ms = one segment) window; pause where it is below **-55.58 dB** (the Otsu threshold on the fold-1 training segments, unchanged). The edited pause is a maximal run of frames whose **unsmoothed** mean is below the same threshold and that overlaps a smoothed pause run, so the edited length is the real low-energy stretch, not the shorter smoothed core, and only stretches long and quiet enough to pass the 250 ms rule count. Only **internal** pauses (smoothed speech before and after) are edited; leading and trailing silence are untouched. Detection runs on the full clip | Same quantity and threshold as the segment rule, but sliding, because the edit needs pause boundaries, not a 250 ms grid. Leading/trailing silence is not a pause in the speaking-style sense, and changing it moves the whole sentence in the window (a different question). | `find_pauses` |
| Pause of length d: x0.5 cuts 0.5 d out of its centre; x1.5 inserts a copy of 0.5 d of its own central audio at its centre (the room noise matches). Every join gets an equal-power crossfade of min(10 ms, d/4), inside the pause. A clip without an internal pause is left unchanged and counted as "no pause" | Changes only pause length, never speech; the crossfade avoids clicks, equal power because room noise on both sides is uncorrelated. | `edit_pauses` |
| **Flip** = the edited clip's predicted style differs from the unedited clip's. For each clip and dial: the label at each step in order of size, and the smallest step at which the label changes, per direction; non-monotonic cases (flips at a small step but not at the larger one) are listed | The plan's question. | `summarise` |
| A wrong clip is **reached** if at least one single edit (of the 8) makes the model predict its true style | The plan's question 2. | `summarise` |
| A style is **reachable** if at least one clip not originally predicted as that style is predicted as it after some edit; counts per style | The plan's question 3 (e.g. can any edit produce whisper?). | `summarise` |
| Fixed wording for loudness: "almost no effect" if +-6 dB flips fewer than 5 of 100 clips in each direction AND the median absolute change of the original class's probability at +-6 dB is below 0.05 | Decided before the results. | `wording` |
| Controls/limits measured, not hidden: (a) rate round trip x0.85 then x1/0.85 (artefacts of the WSOLA itself; a clip whose label changes under the round trip has a rate flip that may be an artefact); (b) the number of real frames in the 4 s window before and after each edit, and flips split by "window fully real before and after" vs "amount of padding changed" (padding is a known possible shortcut, see the observations) | An edit also changes how much of the sentence and how much padding sits in the window; this separates the dial from that side effect. Neither control is one of the 8 edits or enters the flip counts. | `controls` |
| Test confusion matrix and per-style precision, recall, F1 for the unedited predictions and for each of the 8 edits, against the true styles of the 100 clips | CLAUDE.md rule. | report |
| **Listening test:** 20 pairs (unedited and edited version of the same clip), 10 where the model's label flipped and 10 where it did not, 20 different clips; edits taken from the dials in turn (rate, pause, gain), drawn at random within each group, falling back to any dial if a pool is empty. Excluded: edits whose 4 s window would reach a peak of 0.99 (a file would clip) and edits that leave the window unchanged. Files = the 4 s window the model hears (real audio only, no padding), 16 kHz 24-bit WAV. Random pair order, random A/B order, random 8-hex file ids | Audible, comparable pairs that match exactly what the model saw; balanced between flipped and not flipped so agreement is not dominated by one group. | `listening` |
| Key file `listening_key.csv` (not for listeners); answer sheet `answer_sheet_template.csv` per pair: hear a difference (yes/no), style heard for A and for B (one of the 7, or unsure). Scoring `src/explain/score_listening.py`: (1) "heard a difference" vs model flip, (2) listener's style changes between A and B vs model flip (main), (3) for flipped pairs, listener's style for the edited version equals the model's edited label; counts, rates and Wilson 95% intervals, per sheet and pooled | (1) measures audibility (a 3 dB change is audible but need not change the style), so (2) is the main match score. | `score_listening.py` |
| Self-check set: 6 edited clips plus their unedited windows: rate x0.85, rate x1.15, pause x0.5, pause x1.5, gain -6, gain +6 dB. Pause edits on the two clips with the most internal pause time inside the window; the others drawn at random from the remaining clips (for +6 dB only clips whose edited window stays below a peak of 0.99), all 6 of different styles | One example of every editing method at its largest step, to judge the editing quality by ear. | `self_check` |
| All random draws (listening selection, order, ids, A/B, self-check) with seed 42, printed when run; the edits themselves are deterministic | Reproducible. | `SEED` |
| **Added after seeing the results:** processing diagnostic (`cf_processing_diagnostic.csv`): the round trip compared bin by bin with the unedited window (speech / pause frames, lowest / highest 32 bands, mean absolute change), and each edit's overall level change (mean dB of the window's real frames, edited minus unedited) | The rate round trip flipped 42 of 100 labels, almost as many as x0.85 itself (44); the diagnostic shows why (WSOLA lowers the level by about 1 dB in speech and 1.9 dB in pauses and changes the texture). The edits, steps and method were NOT changed; the rate result is reported as confounded with processing. | `run_diag` |
| Edited audio (`reports/counterfactual_fold1/listening/audio/`, `.../self_check/`) is not committed (`.gitignore`); the key, the answer sheet and the tables are | Audio and data stay out of git. | `.gitignore` |

## Reporting test results (standing rule, also in `CLAUDE.md`)

| Decision | Why | Where |
|---|---|---|
| Every report on a trained or tested model includes (a) the test confusion matrix: rows = true style, columns = predicted style, each box the share of its row plus the clip count; and (b) a table of per-style precision, recall and F1 (with the macro average). Applies to every fold, seed and new model, including the logistic regression baseline(s). | Accuracy and macro-F1 alone hide which styles are confused with which; the error pattern differs a lot by test actor (e.g. in fold 2, sad and enunciated are mostly predicted as confused, laughing as happy). | `src/report_training.py` (single fold), `src/eval_checkpoints.py` + `src/report_all.py` (all runs) |
| For runs already finished, the matrix comes from re-loading `best.pt` and predicting the test actor, not from retraining | Re-evaluation reproduces the saved `test_predictions.csv` clip for clip for all 8 runs, so it is the same test result. | `python -m src.eval_checkpoints` |
| `src/report_all.py` stops with an error if a finished run has no confusion matrix / per-style metrics | So the rule cannot be skipped silently when runs are added. | `src/report_all.py` |

## Environment

| Decision | Why | Where |
|---|---|---|
| Python 3.11 | Well supported by torch and opensmile. | README |
| pandas 2.2.3 and pyarrow 19.0.1 instead of the newest | Windows Smart App Control blocked the DLLs of pandas 3.1.1 and pyarrow 26.0.1 on the dev machine. scikit-learn 1.9.1 was blocked once, then loaded on retry. | `requirements.txt` |
| Seed 42 everywhere; outputs fingerprinted | Reproducibility; `reports/output_checksums.json` must not change between runs. | `seed` |

## Observations to keep in mind (no action taken)

- **Amount of real audio in the window depends on style.** Slow styles fill more of the 4 s:
  real audio inside the window totals 87 minutes for enunciated but 64 for default, and 49%
  of enunciated clips fill the whole window versus 15% of default clips. The amount of
  padding is therefore a style cue the model could use as a shortcut. Worth checking in the
  explanations (do attributions sit on the padding boundary?).
- **Loudness appears levelled in the source recordings.** Average RMS is about -25 dBFS for
  every style, with a standard deviation of only 2.2 dB over all clips. Whisper is on average
  only about 2 dB quieter than default, and for ex03 and ex04 it is just as loud. Real whispering
  is normally much quieter than normal speech. So absolute loudness carries little style
  information here; the loudness counterfactual may show a small effect for that reason.
- **"Each actor reads the same sentences in all 7 styles" is mostly, not exactly, true.** See
  `reports/parallel_text_check.md`. 1,516 of 1,520 id groups exist in all 7 styles; 82.8%
  carry the same normalised sentence in every style, 17.0% contain a small wording variant in
  at least one style, and only 4 groups (4 clips) contain a truly different sentence. Exact text
  matching finds only 68.6% of actor-sentence pairs in all 7 styles, because each variant
  counts as its own sentence.
- Some transcripts contain an encoding error (an apostrophe shown as a replacement character,
  e.g. in "Earth's"). It only affects the text, not the audio.
- Class balance is now nearly perfect: 1,518 to 1,520 clips per style.

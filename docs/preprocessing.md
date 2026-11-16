# Preprocessing, explained in plain words

This page walks through every step that `python -m src.run_preprocess` performs, what it
does, and why. It assumes no background in audio. Every number mentioned here is set in
[`configs/preprocess.yaml`](../configs/preprocess.yaml); the reasoning for each choice is
in [`DECISIONS.md`](../DECISIONS.md).

## Background: what is a sound file?

- **Samples.** A microphone measures air pressure. A digital recording stores that
  pressure as a long list of numbers, one number per instant. Each number is a *sample*.
  Values lie between -1 and +1; 0 is silence, and bigger swings mean louder sound.
- **Sample rate.** How many samples are stored per second, in Hertz (Hz). Expresso is
  recorded at 48,000 samples per second (48 kHz). A sample rate can only capture pitches
  (frequencies) up to half of itself, so 48 kHz captures up to 24 kHz, and 16 kHz captures
  up to 8 kHz. Speech carries almost all of its information below 8 kHz, which is why
  speech models usually work at 16 kHz: three times less data, very little lost.
- **Bit depth.** How precisely each sample is stored. Expresso uses 24 bit, which is very
  precise. We keep 24 bit when we save our 16 kHz copies.
- **Channels.** Mono = one microphone signal, stereo = two. Expresso's read clips are mono.

## Step 1: Download

The "read" part of the Hugging Face dataset `ylacombe/expresso` is stored as 12 *parquet*
files (a compact table format, about 5.8 GB). Each table row is one clip: the audio file
packed as bytes, the sentence that was read, the speaker id and the style. We pin the exact
dataset version (a commit id) so everyone gets the same files. The download resumes if
interrupted and is skipped when `data/raw/expresso_hf/DOWNLOAD_COMPLETE.json` lists every
file at the right size.

**License.** Expresso is released under CC BY-NC 4.0: free to use for non-commercial
purposes, as long as the authors are credited (see README).

## Step 2: Decode and describe every clip

For each row we unpack the audio bytes with the `soundfile` library and record the real
sample rate, channel count, length, loudest sample ("peak") and average loudness ("RMS
level"). We measure these instead of trusting the dataset description.

We also compute a **hash** (a fingerprint) of the decoded samples. Two recordings with the
same fingerprint are bit-for-bit identical, so this finds exact duplicate files.

Then each clip is converted to mono (already mono here), **resampled** to 16 kHz with a
high-quality resampler (`soxr_hq`), and saved as a lossless 24 bit FLAC file in
`data/interim/audio16k/`. These files are used later for eGeMAPS features, for LIME
(which mutes time segments) and for the counterfactual tests (which change speaking
rate, loudness or pauses), so all of those see exactly the same audio the model sees.

## Step 3: The manifest (the list of clips we use)

`data/processed/manifest.csv` has one row per kept clip:
`clip_id, actor, style, label, group_id, duration_s, n_samples_16k, was_cropped,
original_duration, original_n_samples_16k, subset, text, text_norm, ...`.

- `group_id` is the actor plus the number at the end of the id (`ex01_happy_00260` ->
  `ex01_00260`). That number marks the same prompt in all 7 styles of one actor; the splits
  never separate a group (Step 6).
- `original_duration` is the length of the whole recording; `duration_s` is the length of
  the real audio inside the 4 s model window (the same, unless the clip was cropped).

What gets removed, and why:

1. **Anything that is not one of the 7 styles.** The style column in this dataset also
   contains `emphasis` and `essentials` (both spoken in the default voice), `longform`
   and `singing`. There is no separate "narration" style. We keep only confused,
   default, enunciated, happy, laughing, sad, whisper. (Config: `labels.styles`;
   `labels.merge_into_default` can fold emphasis/essentials into "default" instead.)
2. **Exact duplicates**, by id and by identical audio fingerprint. The only duplicates in
   the data are the 8 `longform` rows: their text is a long paragraph, but their audio is
   a byte-identical copy of a short default clip with the same id. They are excluded by
   rule 1 already.
Nothing is removed for length. All 10,636 clips of the 7 styles are kept.

**The 4 second window.** Every spectrogram must have the same width, so the model looks at
a fixed window of 4 seconds (`audio.max_seconds`). Clips shorter than 4 s get silence added
at the end. Clips longer than 4 s are **cropped**: only their first 4 seconds are used
(`audio.long_clip_policy: crop`; the alternative `drop` would remove them instead). 73% of
the clips fit completely; 2,893 clips (27%) are cropped. Slow styles are cropped most often:
about half of the enunciated clips, but only 15% of the default clips. Cropping may cut off a
style cue near the end of a long clip (a final laugh, for example); dropping long clips would
instead have removed many enunciated clips. See `reports/duration_histogram.png` and
`reports/cropped_actor_style.md`.

Labels are numbered alphabetically: confused 0, default 1, enunciated 2, happy 3,
laughing 4, sad 5, whisper 6.

## Step 4: Log-mel spectrograms (the model's input)

A ResNet is an image model, so we turn each clip into a picture of sound.

- **Spectrogram.** We slide a short window over the audio: 25 ms long (400 samples),
  moved forward 10 ms (160 samples) at a time. For each window we measure how much energy
  there is at each frequency (pitch) using a Fourier transform (FFT, size 512). The result
  is a grid: time left to right, frequency bottom to top, brightness = energy.
  One second of audio becomes 100 columns, so the 4 s window becomes exactly
  **128 x 400** (column t is centred at t x 10 ms; librosa adds one extra column centred
  exactly at 4.000 s, which we drop).
- **Mel scale.** Human hearing tells low pitches apart much better than high ones (100 vs
  200 Hz is obvious, 7,100 vs 7,200 Hz is not). The mel scale groups the frequencies into
  128 bands that are narrow at low pitch and wide at high pitch, from 20 Hz to 8,000 Hz.
  This gives the model detail where speech information is dense.
- **Log / decibel (dB) scale.** Sound energy spans an enormous range: a shout carries
  millions of times more energy than a whisper. Our ears hear ratios, not differences,
  and so does the dB scale: dB = 10 x log10(energy / reference). Every factor of 10 in
  energy adds 10 dB.
  - The **reference is fixed** (1.0) for every clip. A popular shortcut is to use each
    clip's own loudest point as the reference, which sets every clip's maximum to 0 dB and
    hides how loud the clip really was. We avoid that.
  - Very small energies are **floored** at -100 dB, so true silence (including padding)
    has a defined value instead of minus infinity.
- **Padding.** Clips shorter than 4 s are padded with zeros (silence) at the end before
  computing the spectrogram, so padding always has the fixed floor value of -100 dB. The
  number of real (non-padding) columns is stored as `n_frames_valid` (a column is real when
  its centre lies inside the real audio) so later code, statistics and explanations can
  ignore the padding.

**Storage.** One file per actor, `data/processed/logmel/<actor>.npy`, holding an array of
shape `[clips, 128 mel bands, frames]` in 16-bit floats (float16), plus
`data/processed/logmel/index.csv` telling which row is which clip. float16 keeps about
3 significant digits; at values around -100 to +60 dB the rounding error is at most
0.03 dB, far below anything audible or meaningful for the model, and it halves the disk size.
Load a single clip without reading the whole file:
`np.load(path, mmap_mode="r")[row]`.

## Step 5: The segment table

The explanation tests (LIME on time segments, deletion tests, pause counterfactuals) work on
pieces of time. `data/processed/segments.csv` cuts every 4 s window into **16 segments of
250 ms** (25 spectrogram columns each, `segments.length_ms`) and describes each one:

| column | meaning |
|---|---|
| `clip_id`, `segment` | the clip, and the segment number 0 to 15 |
| `start_s`, `end_s` | where the segment sits in the window, in seconds |
| `is_real_audio` | true if the segment contains any real audio; a segment that is only partly real counts as real |
| `n_real_frames` | how many of its 25 columns are real audio (25 for a fully real segment) |
| `mean_db` | average log-mel value over all 128 bands and the segment's real columns; empty for pure padding |

No threshold for "speech" or "pause" is applied yet. One detail: the last real segment of a
clip can hold only one or two real columns from the quiet end of the recording, and those
columns' 25 ms analysis windows reach into the silent padding, so their `mean_db` can be
close to -100 dB. `n_real_frames` lets later code skip such slivers.

## Why we do NOT normalise loudness per clip

Many audio pipelines rescale every clip to the same loudness (peak or RMS normalisation).
We deliberately do not, anywhere:

- Loudness is part of speaking style: whispering is quiet, happy or laughing speech is
  often louder. Removing it would remove real information.
- Our counterfactual tests change loudness and check whether the model's prediction
  moves. If every clip were rescaled to the same level, the model could never react to
  loudness and that test would be meaningless.

The only normalisation the model will use later is one global mean and standard deviation
per split, computed from training clips only (Step 6). This shifts and scales all clips by
the same amount, so relative loudness differences between clips survive.

A caveat we measured: the Expresso recordings themselves look levelled. Every style
averages about -25 dB below full scale, and whispered clips are only about 2 dB quieter
than default ones, while real whispering is normally much quieter than normal speech.
So the model can see loudness, but loudness differences between styles are small in
this data.

We only *warn* about possibly clipped clips (peak at full scale) and near-silent clips;
see `reports/audio_warnings.csv`.

## Step 6: Splits

Each actor reads the same prompts in all 7 styles. If one reading of a prompt were in
training and another in validation or test, the model could score well by recognising the
sentence instead of the style. So every split keeps all readings of a prompt together.

**The grouping unit is actor + id number** (`group_id`, e.g. `ex01_00260`). The number at
the end of a clip id marks the same prompt in every style of one actor. We first grouped by
the exact (normalised) transcript, but the parallel text check (Step 7) showed that some
transcripts record small wording variants, such as "I'm not sure" in one style and "I am not
sure" in another. Exact text matching treated those as different sentences, so 10.6% to 19.4%
of validation clips had a sibling reading in training. The id number does not have that
problem.

**Leave-one-actor-out (4 folds).** Does the model recognise a style in a voice it has never
heard? In fold k the test set is every clip of actor k; the other 3 actors provide training
and validation clips. Validation is about 20% of those 3 actors' clips, stratified by actor
and style (every actor x style combination keeps roughly the same share) and grouped by id
group.

**Random split (1 split).** Does the model recognise a style when the voice is known? All 4
actors appear in train (60%), validation (15%) and test (25%). It is grouped by id group,
so the same prompt never appears in two parts; only the speaker is shared, which is the
point of this split. It is stratified by actor and style, so every part holds 1/7 of each
style and about a quarter of each actor.

Files in `data/splits/`, all CSV files with the same simple format:

| file | content |
|---|---|
| `fold_{k}.csv`, `random_split.csv` | `clip_id, actor, style, label, split` with split = `train`, `val` or `test` |
| `folds.json`, `random_split.json` | test actor of each fold; target and actual shares of the random split |
| `norm_stats_fold_{k}.json`, `norm_stats_random.json` | mean and std of the log-mel dB values over TRAIN clips, padding excluded; one global pair and one pair per mel band |
| `fill_values_fold_{k}.json`, `fill_values_random.json` | fill values for deletion tests (below): per mel band, the mean dB and the 10th percentile dB over TRAIN clips, padding excluded |

The baseline (logistic regression on eGeMAPS) uses exactly the same split files, joined on
`clip_id`.

### Fill values for deletion tests

Some explanation tests "delete" a piece of the input (for example a time segment of the
spectrogram) and watch how the prediction changes. The question is what to put in the hole.
Writing the padding value (-100 dB, perfect digital silence) creates something the model never
sees inside real speech, because real recordings always contain some background noise. So we
store two more natural fills per mel band, computed from training clips of each split only,
real frames only:

- **mean dB per band**: a neutral "average sound" fill;
- **10th percentile dB per band**: the level a band is at or below 10% of the time, which in
  practice is the quiet background of the recording room ("room noise").

The room-noise level lies well above the padding value: a median of about 19 to 21 dB higher
over the bands (from about 7 dB up to about 60 dB, highest in the lowest bands). The exception
is the very top band (about 7.8 kHz): converting to 16 kHz removes almost all sound just below
8 kHz, so that band sits at the floor at least 10% of the time and its 10th percentile equals
the padding value.

Automatic checks (the run stops if any fails): no clip appears twice in a split, no id group
appears in two parts of a split, in each fold the test actor never appears in train or
validation, and the random split has every actor and every style in every part. We also
*report* (without changing anything) how often the same normalised sentence still occurs on
both sides, which can happen through another actor or another id number:
`reports/sentence_overlap.csv`. This is tiny (at most 7 clips per comparison).

## Step 7: Reports and reproducibility

`reports/` gets: clip counts and minutes per actor x style, cropped clips per actor x style,
what was excluded and why, audio warnings, the duration histogram, a per-style spectrogram
figure, split summaries (all folds and the random split), the sentence-overlap table and the
parallel text check.

**Parallel text check** (`reports/parallel_text_check.md`). The design idea is that each actor
reads the same sentences in all 7 styles, and that the number at the end of an id
(00260 in `ex01_happy_00260`) marks the same prompt in every style. The check confirms that
the number is a reliable link: 1,516 of 1,520 actor x number groups exist in all 7 styles,
83% have the same sentence in every style after normalising, 17% contain a small wording
variant in at least one style (contractions, spelling, "gonna", an added word, a likely
mishearing), and only 4 groups contain a truly different sentence. Matching sentences by exact
text alone makes the parallelism look weaker (69% of sentences found in all 7 styles) because
every variant counts as its own sentence.

The seed is 42 everywhere. The last step writes SHA-256 fingerprints of every output to
`reports/output_checksums.json`; running the pipeline again must produce the same
fingerprints.

## eGeMAPS features (written, not run yet)

`src/features/egemaps.py` extracts the 88 standard eGeMAPS voice measurements (pitch,
loudness, voice quality, spectral shape and more, summarised per clip) with the
`opensmile` package, from the same 16 kHz files and the same manifest, into
`data/processed/egemaps_functionals.csv` keyed by `clip_id`. Like the spectrograms, it only
measures the first 4 s of a cropped clip, so both models see the same audio. Install
`opensmile` first.

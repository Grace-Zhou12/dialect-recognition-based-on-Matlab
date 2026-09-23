# dialect-recognition-based-on-Matlab

An interpretable, lightweight **audio clip classifier** for distinguishing Shanghainese speech from non-Shanghainese Mandarin. The MATLAB workflow prepares WAV data, extracts hand-designed acoustic features, compares four classifiers, tunes thresholds on a validation split, and evaluates the selected model on a held-out speaker split.

## Pilot results

Run with MATLAB R2025b and Statistics and Machine Learning Toolbox. The selected model was an RBF SVM with `BoxConstraint = 1` and `KernelScale = 2.5`, chosen by validation F1 from a small hyperparameter grid.

| Held-out test metric | Result |
| -------------------- | -----: |
| Accuracy             |  0.880 |
| Balanced accuracy    |  0.840 |
| Precision            |  0.871 |
| Recall               |  0.962 |
| F1                   |  0.914 |
| ROC-AUC              |  0.896 |
| Brier score          |  0.105 |

Confusion matrix at the validation-selected balanced threshold (`0.48391`):

| Actual / predicted | Shanghainese | Other Mandarin |
| ------------------ | -----------: | -------------: |
| Shanghainese       |          600 |             24 |
| Other Mandarin     |           89 |            228 |

There were 941 held-out clips: 624 Shanghainese and 317 other Mandarin. The full feature set contained 5,939 usable clips after the audio quality checks. These figures describe the current dataset only; they are not a deployment guarantee.

Validation comparison:

| Model                           | Precision | Recall |    F1 |
| ------------------------------- | --------: | -----: | ----: |
| Feature-weighted threshold      |     0.774 |  0.950 | 0.853 |
| Regularized logistic regression |     0.762 |  0.968 | 0.853 |
| RBF SVM, automatic scale        |     0.816 |  0.933 | 0.870 |
| RBF SVM, tuned (selected)       |     0.824 |  0.937 | 0.877 |
| Linear discriminant analysis    |     0.761 |  0.968 | 0.852 |

Full counts and warnings are written to [`training_report.txt`](training_report.txt) after training.

## Data included in the latest run

| Source                                 | Label          |     Clips | Speakers | Notes                                                        |
| -------------------------------------- | -------------- | --------: | -------: | ------------------------------------------------------------ |
| Shanghai conversational Parquet corpus | Shanghainese   |     3,792 |       20 | Embedded WAV audio; source labels identify Shanghai speech   |
| THCHS-30 subset                        | Other Mandarin |     2,126 |       30 | Read Mandarin, 16 kHz mono; 5.41 hours                       |
| Hand-collected project samples         | Both           | 21 usable |  Unknown | One sample failed a quality check; two filenames need label review |

The train/validation/test split is grouped by source and speaker where speaker IDs exist. Clips with missing speaker IDs are split by file and counted in the report; they do not establish speaker-independent performance.

## Method

### Audio preparation

- Convert audio to mono, 16 kHz WAV.
- Reject clips shorter than 1 second and windows that fail speech activity, clipping, or valid-window checks.
- Analyze overlapping 2-second and 4-second windows with a 0.5-second hop.
- Aggregate window features with a median after IQR-based outlier filtering.

### Interpretable acoustic features

Six acoustic measures are extracted at each time scale (12 values per clip):

1. Low-band energy ratio (80–400 Hz)
2. Pitch standard deviation (autocorrelation estimate, 80–400 Hz range)
3. Zero-crossing rate
4. Normalized energy variance
5. Normalized spectral centroid
6. Spectral flux

The diagnostic plot reports standardized class-mean differences. These are descriptive comparisons, not causal feature importance.

### Training and evaluation

- Deterministic 70/15/15 speaker-grouped train, validation, and test split (`seed = 42`).
- Down-sample only the larger class in the training split to at most a 2:1 ratio; use class weights during fitting. Validation and test retain their observed class proportions.
- Compare a feature-weighted baseline, ridge logistic regression, RBF SVM, and LDA.
- Search the RBF SVM box-constraint values `[0.1, 1, 10]` against kernel scales `[1, 2.5, 5]`, plus MATLAB's automatic-scale baseline; select the candidate by validation F1.
- Select the model by validation F1. Fit a ridge-logistic score calibrator and select decision thresholds using validation predictions.
- Evaluate once on the held-out test split with accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, Brier score, and a confusion matrix.

The validation data is used for model selection, calibration, and threshold selection. This can make validation results optimistic; the test split is kept out of those steps. The `high_precision` threshold maximizes validation precision subject to recall of at least 0.50. The `high_recall` threshold maximizes validation recall subject to precision of at least 0.70. These are corpus-specific operating points, not guaranteed production behavior.

## Requirements

- MATLAB R2022b or newer to import the nested Parquet audio file.
- Statistics and Machine Learning Toolbox (`fitclinear`, `fitcsvm`, `fitcdiscr`).
- Signal Processing Toolbox is recommended for `resample`; a linear interpolation fallback is included.
- MATLAB R2025b was used for the run summarized above.

## Quick start

Open MATLAB in this repository directory. Prepare the dataset and train:

```matlab
prepare_public_dataset_from_raw
model = shanghai_dialect_pipeline('train');
```

Or run the menu:

```matlab
shanghai_dialect_recognition_demo
```

Menu options prepare data, train/evaluate, classify one audio file, or export the model. The English entry point is `shanghai_dialect_recognition_english_demo`.

Predict a clip and choose an operating threshold:

```matlab
result = shanghai_dialect_pipeline( ...
    'predict', '/absolute/path/to/audio.wav', 'balanced');
```

Supported modes are `balanced`, `high_precision`, and `high_recall`. The prediction includes the calibrated score, threshold, extracted features, and quality-gate result. A score is not a guarantee that the predicted dialect is correct.

Export the trained model to JSON:

```matlab
shanghai_dialect_pipeline('export');
```

## Dataset layout

The preparation script imports sources when they are present:

```text
public_dataset/raw/shanghai_scripted/
public_dataset/raw/shanghai_conversational/
public_dataset/raw/non_shanghai_primewords/
training_data/shanghai/
training_data/non_shanghai/
../thchs30-subset-large/       # optional sibling folder
train-*.parquet                # optional Parquet files at the repository root
```

For Primewords, provide `audio_files/` and `set1_transcript.json` under `public_dataset/raw/non_shanghai_primewords/`. The THCHS-30 subset is automatically discovered beside the repository and imported as non-Shanghainese Mandarin; its per-speaker folders are retained for grouped splitting. Prepared audio and its manifest are written to `public_dataset/processed/`.

The preparation step rebuilds the CSV manifest but can leave old WAV files in the processed audio directory. Remove `public_dataset/processed/` before preparing again after changing the source data, so stale audio is not left on disk.

## Outputs

- `public_dataset/processed/metadata/file_index.csv` — audio paths, labels, sources, and speaker metadata.
- `public_dataset/processed/wav16k/` — standardized audio used by the model.
- `shanghai_model.mat` — MATLAB model, normalizer, thresholds, and evaluation summary.
- `training_report.txt` — model comparison, split counts, test metrics, and data-quality warnings.
- `shanghai_model.json` — portable model parameters and feature configuration after export.

## Evaluation limitations

The current labels and sources are strongly coupled: nearly every positive clip is from the Shanghai Parquet corpus, and nearly every negative clip is from THCHS-30. A speaker-held-out split prevents a speaker from appearing in both train and test, but it does not remove this corpus/source confound. Therefore, the reported 0.880 accuracy and 0.914 F1 may overstate actual Shanghainese recognition performance.

For a credible dialect result, collect both classes under matched devices, environments, speaking styles, and durations; include multiple non-Shanghai dialects; review ambiguous labels; and report performance on an independent source with unseen speakers. The current project samples have unknown speaker IDs, and `biaozhun.wav` / `bubiaozhun.wav` are flagged for manual label review.

## Data licenses

The source corpora retain their own terms: THCHS-30 is listed as Apache 2.0 by [OpenSLR](https://www.openslr.org/18/); Primewords is listed as CC BY-NC-ND 4.0 by [OpenSLR](https://www.openslr.org/47/); and the project-local recordings have unknown license metadata. Check each source's terms before redistributing audio or derived artifacts.

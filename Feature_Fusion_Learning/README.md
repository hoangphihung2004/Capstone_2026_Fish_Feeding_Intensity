# Feature Fusion Learning

Fine tuning pipeline for extracted deep features from audio, video, or audio-video feature fusion.

## Configure

Edit only `config/fine_tune_config.json` for feature inputs and model selection:

```json
{
  "dataset_tag": "AFFIA3K",
  "feature_mode": "audio_video",
  "audio_feature_root": "/marimo/U_FFIA3K_audio/outputs",
  "video_feature_root": "/marimo/U_FFIA3K_video/outputs",
  "audio_model_name": "CNN6",
  "video_model_name": "EfficientNetB0",
  "evaluation_mode": "both",
  "folds": "all",
  "num_folds": 5,
  "seed": 42,
  "models": ["LR", "KNN", "SVM", "RF", "ET", "LGBM"],
  "output_dir": "outputs"
}
```

Set Hugging Face upload in `config/artifact_upload_config.json`. Leave `source_dir`, `zip_path`, and `path_in_repo` empty to use the default source folder and generated artifact name. Set `repo_id` before running if upload is required.

## Run

```bash
python machine_learning/fine_tune_features.py
```

The script runs holdout first, then all cross-validation folds, writes results, saves confusion matrix CSV/PNG files, zips the source folder, and uploads the zip if enabled.

## Outputs

```text
outputs/
  holdout/
    result.csv
    run_info.json
    confusion_matrices/
      LR_confusion_matrix.csv
      LR_confusion_matrix.png
  cross_validation/
    fold_0/
      result.csv
      run_info.json
      confusion_matrices/
    ...
    cv_summary.csv
```

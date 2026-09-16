# Multimodal Deep Fusion

End-to-end audio-video deep learning for fish feeding intensity classification.

## Run

Edit only:

```text
config/train_config.json
```

Then run:

```bash
python main.py
```

Each run supports one evaluation mode only: `holdout` or `cross_validation`. Splits are generated automatically with the same `FishDataSplitter` logic used by the single-modal 27K audio/video branches.

For cross-validation, set `dataset.fold_index` to `null` to run all folds, or
to an integer from `0` through `num_folds - 1` to run exactly one fold. A
single-fold run uses the same deterministic split as that fold in a full CV
run, provided `seed`, `split_strategy`, `num_folds`, and `cv_val_ratio` stay
the same. When every fold result is present, the CV summary is regenerated.
Single-fold Hugging Face artifacts include `fold_XX` in their filename.

Each run selects one audio model, one video model, and one fusion head.

Available fusion heads:

```text
raw_concat
linear_concat
linear_mean
gated_fusion
self_attention
```

Outputs are saved under:

```text
outputs/<audio_backbone>_<video_backbone>_<fusion>/
```

After all requested modes complete, the project is zipped and uploaded to:

```text
hoangphihung442004/Results_Multimodal_Feature_Fusion_DL
```

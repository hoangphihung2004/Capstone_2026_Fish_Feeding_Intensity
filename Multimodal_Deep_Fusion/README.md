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

The run supports `holdout`, `cross_validation`, or `both`. When `use_existing_splits` is `true`, set:

```json
"holdout_splits_dir": "/marimo/holdout/DL_audio/checkpoint/panns_cnn6/splits",
"cross_validation_splits_dir": "/marimo/cv/DL_audio/checkpoint/panns_cnn6"
```

The cross-validation loader expects:

```text
fold_00/splits/train.csv
fold_00/splits/val.csv
fold_00/splits/test.csv
...
fold_04/splits/test.csv
```

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

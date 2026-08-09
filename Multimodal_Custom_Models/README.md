# Multimodal Custom Models

End-to-end audio-video deep learning source for implementing new multimodal models.

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

Each run selects one custom multimodal model by name:

```text
config/train_config.json -> model.name
```

Current registered model:

```text
AVMobileAttnFiLMNet
```

Outputs are saved under:

```text
outputs/<model_name>/
```

Optional distillation is configured in the same file and is disabled by default. Offline distillation requires teacher checkpoints; online distillation builds trainable teachers from the configured teacher model names.

Output files include:

```text
train_config.json
splitter_config.json
distillation_config.json
run_info.json
history.csv
summary.csv
result.csv
learning_curves.png
confusion_matrix_best.csv
confusion_matrix.csv
checkpoint/model_best.pt
splits/train.csv, val.csv, test.csv
```

After all requested modes complete, the project is zipped and uploaded to:

```text
hoangphihung442004/Results_Multimodal_Custom_Models
```

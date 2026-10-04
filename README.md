# Plant Disease Detection Using Deep Learning

This project trains a genuine image classifier on the local PlantVillage class folders using MobileNetV2 transfer learning. It discovers class names automatically, creates reproducible stratified 70/15/15 splits, applies augmentation only while training, and evaluates the saved model on the held-out test split.

## Dataset

Place the dataset at `PlantVillage/`, with one immediate child directory per class. The supplied copy has 15 classes and 20,638 unique images. It also contains a nested `PlantVillage/PlantVillage/` mirror; the loader ignores this non-class container so duplicate images cannot leak across splits and it cannot become a false class.

The original dataset is read-only from the pipeline's perspective. Generated models go to `artifacts/` and evaluation outputs go to `results/`.

## Setup

```bash
python -m venv .venv
```

On Windows Git Bash:

```bash
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Verify the dataset

```bash
python -m ml.inspect_dataset --data-dir PlantVillage
```

This prints discovered classes, per-class image counts, ignored container directories, and deterministic split counts.

## Train

```bash
python -m ml.train --data-dir PlantVillage --epochs 12
```

MobileNetV2 uses ImageNet weights by default. To add a small fine-tuning phase after frozen-base training:

```bash
python -m ml.train --data-dir PlantVillage --epochs 12 --fine-tune-epochs 5 --fine-tune-layers 30
```

For an offline smoke run where ImageNet weights cannot be downloaded, pass `--weights none`; this trains from random initialization and is not the recommended final experiment.

## Evaluate a saved model

```bash
python -m ml.evaluation --data-dir PlantVillage --model artifacts/plant_disease_model.keras
```

Actual outputs include `metrics.json`, `classification_report.json`, `confusion_matrix.png`, `accuracy_curve.png`, and `loss_curve.png`. No metrics are prefilled or fabricated.

## Training design

Images are decoded as RGB and resized to 224×224. Random horizontal flips, rotation, zoom, translation, and contrast are active only during training. MobileNetV2 preprocessing is embedded in the model, followed by global average pooling, dropout, and a softmax layer sized from the discovered classes. Training uses Adam, sparse categorical cross-entropy, early stopping, checkpointing, and learning-rate reduction.

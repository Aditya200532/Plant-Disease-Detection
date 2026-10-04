from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.data import build_tf_dataset, discover_dataset, stratified_split


def evaluate_and_save(model, test_dataset, class_names: tuple[str, ...], output_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    true_labels = np.concatenate([labels.numpy() for _, labels in test_dataset])
    probabilities = model.predict(test_dataset, verbose=1)
    predicted_labels = probabilities.argmax(axis=1)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="weighted", zero_division=0
    )
    evaluation = model.evaluate(test_dataset, verbose=1, return_dict=True)
    metrics = {
        "test_loss": float(evaluation["loss"]),
        "test_accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1_score),
        "test_images": int(len(true_labels)),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(
        true_labels, predicted_labels, labels=list(range(len(class_names)))
    )
    figure, axis = plt.subplots(figsize=(16, 14))
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=axis, xticks_rotation=90, colorbar=False
    )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    return metrics


def save_training_curves(history: dict[str, list[float]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("accuracy", "loss"):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(history.get(metric, []), label=f"Training {metric}")
        axis.plot(history.get(f"val_{metric}", []), label=f"Validation {metric}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.title())
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / f"{metric}_curve.png", dpi=180)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate a trained plant disease model.")
    parser.add_argument("--data-dir", type=Path, default=project_root / "PlantVillage")
    parser.add_argument("--model", type=Path, default=project_root / "artifacts" / "plant_disease_model.keras")
    parser.add_argument("--output-dir", type=Path, default=project_root / "results")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import tensorflow as tf

    inventory = discover_dataset(args.data_dir)
    split = stratified_split(inventory.samples, seed=args.seed)
    test_dataset = build_tf_dataset(
        split.test, args.image_size, args.batch_size, training=False, seed=args.seed
    )
    model = tf.keras.models.load_model(args.model)
    metrics = evaluate_and_save(model, test_dataset, inventory.class_names, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

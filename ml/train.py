from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.data import build_tf_dataset, discover_dataset, split_counts, stratified_split
from ml.evaluation import evaluate_and_save, save_training_curves
from ml.model import build_model, compile_model, enable_fine_tuning


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 on class folders in PlantVillage."
    )
    parser.add_argument("--data-dir", type=Path, default=project_root / "PlantVillage")
    parser.add_argument("--artifact-dir", type=Path, default=project_root / "artifacts")
    parser.add_argument("--results-dir", type=Path, default=project_root / "results")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def callbacks(checkpoint_path: Path):
    import tensorflow as tf

    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7
        ),
    ]


def merge_history(target: dict[str, list[float]], source: dict[str, list[float]]) -> None:
    for key, values in source.items():
        target.setdefault(key, []).extend(float(value) for value in values)


def main() -> None:
    args = parse_args()

    import tensorflow as tf

    if args.epochs < 1 or args.fine_tune_epochs < 0:
        raise ValueError("Epoch counts must be non-negative, with at least one initial epoch.")

    tf.keras.utils.set_random_seed(args.seed)
    inventory = discover_dataset(args.data_dir)
    split = stratified_split(inventory.samples, seed=args.seed)
    print(
        f"Discovered {len(inventory.class_names)} classes and "
        f"{inventory.total_images} images under {inventory.root}."
    )
    if inventory.ignored_directories:
        ignored = ", ".join(path.name for path in inventory.ignored_directories)
        print(f"Ignored non-class container directories: {ignored}")
    print(
        f"Split sizes: train={len(split.train)}, validation={len(split.validation)}, "
        f"test={len(split.test)}"
    )

    train_dataset = build_tf_dataset(
        split.train, args.image_size, args.batch_size, training=True, seed=args.seed
    )
    validation_dataset = build_tf_dataset(
        split.validation,
        args.image_size,
        args.batch_size,
        training=False,
        seed=args.seed,
    )
    test_dataset = build_tf_dataset(
        split.test, args.image_size, args.batch_size, training=False, seed=args.seed
    )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    weights = None if args.weights == "none" else args.weights
    model, base_model = build_model(
        num_classes=len(inventory.class_names),
        image_size=args.image_size,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
        weights=weights,
    )

    history: dict[str, list[float]] = {}
    initial_checkpoint = args.artifact_dir / "best_initial.keras"
    initial_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks(initial_checkpoint),
    )
    merge_history(history, initial_history.history)
    initial_validation_loss = float(model.evaluate(validation_dataset, verbose=0)[0])
    model.save(initial_checkpoint)

    if args.fine_tune_epochs:
        enable_fine_tuning(base_model, args.fine_tune_layers)
        compile_model(model, args.fine_tune_learning_rate, len(inventory.class_names))
        fine_tune_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks(args.artifact_dir / "best_fine_tuned.keras"),
        )
        merge_history(history, fine_tune_history.history)
        fine_tuned_validation_loss = float(model.evaluate(validation_dataset, verbose=0)[0])
        if fine_tuned_validation_loss > initial_validation_loss:
            print("Fine-tuning did not improve validation loss; retaining the frozen-base model.")
            model = tf.keras.models.load_model(initial_checkpoint)

    model_path = args.artifact_dir / "plant_disease_model.keras"
    model.save(model_path)
    metadata = {
        "model_name": "MobileNetV2",
        "dataset_root": str(inventory.root),
        "class_names": list(inventory.class_names),
        "number_of_classes": len(inventory.class_names),
        "total_images": inventory.total_images,
        "image_size": args.image_size,
        "seed": args.seed,
        "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_totals": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "split_per_class": split_counts(split),
    }
    (args.artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (args.results_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    save_training_curves(history, args.results_dir)
    metrics = evaluate_and_save(model, test_dataset, inventory.class_names, args.results_dir)
    print(f"Saved model to {model_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    class_name: str


@dataclass(frozen=True)
class DatasetInventory:
    root: Path
    class_names: tuple[str, ...]
    samples: tuple[Sample, ...]
    image_counts: dict[str, int]
    ignored_directories: tuple[Path, ...]

    @property
    def total_images(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[Sample, ...]
    validation: tuple[Sample, ...]
    test: tuple[Sample, ...]


def discover_dataset(data_dir: str | Path) -> DatasetInventory:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    class_directories: list[tuple[Path, list[Path]]] = []
    ignored_directories: list[Path] = []
    for directory in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.casefold()):
        images = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.casefold(),
        )
        if images:
            class_directories.append((directory, images))
        else:
            ignored_directories.append(directory)

    if len(class_directories) < 2:
        raise ValueError(
            f"Expected at least two immediate class directories containing images under {root}; "
            f"found {len(class_directories)}."
        )

    class_names = tuple(directory.name for directory, _ in class_directories)
    samples = tuple(
        Sample(path=image, label=label, class_name=directory.name)
        for label, (directory, images) in enumerate(class_directories)
        for image in images
    )
    image_counts = dict(Counter(sample.class_name for sample in samples))

    return DatasetInventory(
        root=root,
        class_names=class_names,
        samples=samples,
        image_counts=image_counts,
        ignored_directories=tuple(ignored_directories),
    )


def stratified_split(
    samples: Sequence[Sample],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test ratios must be positive and sum to 1.")

    by_class: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.class_name].append(sample)

    train: list[Sample] = []
    validation: list[Sample] = []
    test: list[Sample] = []
    for class_name in sorted(by_class, key=str.casefold):
        class_samples = sorted(by_class[class_name], key=lambda sample: sample.path.name.casefold())
        random.Random(f"{seed}:{class_name}").shuffle(class_samples)
        count = len(class_samples)
        if count < 3:
            raise ValueError(f"Class {class_name!r} needs at least three images for a three-way split.")

        train_count = max(1, int(count * train_ratio))
        validation_count = max(1, int(count * validation_ratio))
        if train_count + validation_count >= count:
            train_count = count - 2
            validation_count = 1

        train.extend(class_samples[:train_count])
        validation.extend(class_samples[train_count : train_count + validation_count])
        test.extend(class_samples[train_count + validation_count :])

    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(validation)
    random.Random(seed + 2).shuffle(test)
    return DatasetSplit(tuple(train), tuple(validation), tuple(test))


def build_tf_dataset(
    samples: Sequence[Sample],
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int,
):
    import tensorflow as tf

    paths = [str(sample.path) for sample in samples]
    labels = [sample.label for sample in samples]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, (image_size, image_size), antialias=True)
        return tf.cast(image, tf.float32), label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


def split_counts(split: DatasetSplit) -> dict[str, dict[str, int]]:
    return {
        name: dict(Counter(sample.class_name for sample in samples))
        for name, samples in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
    }

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.data import discover_dataset, split_counts, stratified_split


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Inspect PlantVillage class counts.")
    parser.add_argument("--data-dir", type=Path, default=project_root / "PlantVillage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = discover_dataset(args.data_dir)
    split = stratified_split(inventory.samples, seed=args.seed)
    split_by_class = split_counts(split)
    report = {
        "dataset_root": str(inventory.root),
        "number_of_classes": len(inventory.class_names),
        "total_images": inventory.total_images,
        "images_per_class": inventory.image_counts,
        "split_totals": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "split_per_class": split_by_class,
        "ignored_container_directories": [
            str(path) for path in inventory.ignored_directories
        ],
    }
    print(json.dumps(report, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

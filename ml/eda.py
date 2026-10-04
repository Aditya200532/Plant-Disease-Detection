from __future__ import annotations

import argparse
from pathlib import Path

from ml.data import discover_dataset


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis for PlantVillage.")
    parser.add_argument("--data-dir", type=Path, default=project_root / "PlantVillage")
    parser.add_argument("--output-dir", type=Path, default=project_root / "results")
    parser.add_argument("--samples-per-class", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = discover_dataset(args.data_dir)

    names = list(inventory.image_counts.keys())
    counts = [inventory.image_counts[n] for n in names]
    short_names = [n.replace("___", "\n").replace("__", "\n").replace("_", " ") for n in names]

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
    bars = ax.barh(range(len(names)), counts, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Number of Images")
    ax.set_title(f"PlantVillage Class Distribution ({inventory.total_images} images, {len(names)} classes)")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=7)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(args.output_dir / "class_distribution.png", dpi=180)
    plt.close(fig)
    print(f"Saved class_distribution.png")

    n_samples = args.samples_per_class
    fig, axes = plt.subplots(len(names), n_samples, figsize=(n_samples * 3, len(names) * 2.5))
    if len(names) == 1:
        axes = [axes]
    for row, class_name in enumerate(names):
        class_samples = [s for s in inventory.samples if s.class_name == class_name][:n_samples]
        for col in range(n_samples):
            ax = axes[row][col] if n_samples > 1 else axes[row]
            if col < len(class_samples):
                img = Image.open(class_samples[col].path).convert("RGB").resize((128, 128))
                ax.imshow(img)
            ax.axis("off")
            if col == 0:
                label = class_name.replace("___", " - ").replace("__", " - ").replace("_", " ")
                ax.set_title(label, fontsize=7, loc="left")
    fig.suptitle("Sample Images per Class", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output_dir / "sample_images.png", dpi=150)
    plt.close(fig)
    print(f"Saved sample_images.png")

    healthy = sum(c for n, c in zip(names, counts) if "healthy" in n.lower())
    diseased = sum(c for n, c in zip(names, counts) if "healthy" not in n.lower())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([healthy, diseased], labels=["Healthy", "Diseased"], autopct="%1.1f%%",
           colors=["#4CAF50", "#F44336"], startangle=90)
    ax.set_title("Healthy vs Diseased Distribution")
    fig.tight_layout()
    fig.savefig(args.output_dir / "healthy_vs_diseased.png", dpi=150)
    plt.close(fig)
    print(f"Saved healthy_vs_diseased.png")
    print(f"\nSummary: {len(names)} classes, {inventory.total_images} total images")
    print(f"Healthy: {healthy}, Diseased: {diseased}")


if __name__ == "__main__":
    main()

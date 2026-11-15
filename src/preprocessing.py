"""Manifest readers, baseline image features, and Phase 2 EDA figures."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.data import ImageRecord
from src.utils import load_config


def read_manifest(path: str | Path) -> list[ImageRecord]:
    """Read a split manifest and reject malformed or duplicate image identities."""
    manifest_path = Path(path)
    records: list[ImageRecord] = []
    seen: set[str] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "label"]:
            raise ValueError(
                f"Unexpected columns in {manifest_path}: {reader.fieldnames}; "
                "expected path,label"
            )
        for row in reader:
            relative_path = row["path"]
            label = row["label"]
            if not relative_path or not label:
                raise ValueError(f"Empty path or label in {manifest_path}")
            if relative_path in seen:
                raise ValueError(f"Duplicate image identity in {manifest_path}: {relative_path}")
            seen.add(relative_path)
            records.append(ImageRecord(path=relative_path, label=label))
    return records


def flatten_rgb_image(path: str | Path) -> "object":
    """Return one 64x64 RGB image as a float32 vector scaled to [0, 1].

    This is the reproducible classical-ML baseline. Learned scalers and PCA are
    intentionally not applied here; later pipelines must fit them on train only.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("NumPy and Pillow are required; see SETUP.md.") from exc

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if array.shape != (64, 64, 3):
        raise ValueError(f"Expected RGB shape (64, 64, 3), got {array.shape} for {path}")
    return array.reshape(-1)


def plot_class_distribution(
    records: Iterable[ImageRecord], output_path: str | Path, dpi: int = 180
) -> Path:
    """Save a publication-ready class-count bar chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Matplotlib is required; see SETUP.md.") from exc

    counts = Counter(record.label for record in records)
    labels = sorted(counts)
    if not labels:
        raise ValueError("Cannot plot an empty collection")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(labels, [counts[label] for label in labels], color="#2A9D8F")
    axis.bar_label(bars, padding=3, fontsize=8)
    axis.set(title="EuroSAT RGB class distribution", xlabel="Class", ylabel="Images")
    axis.tick_params(axis="x", rotation=40)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_example_grid(
    records: Iterable[ImageRecord],
    dataset_dir: str | Path,
    output_path: str | Path,
    examples_per_class: int = 5,
    seed: int = 42,
    dpi: int = 180,
) -> Path:
    """Save a deterministic grid containing examples from every class."""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Matplotlib and Pillow are required; see SETUP.md.") from exc

    if examples_per_class <= 0:
        raise ValueError("examples_per_class must be positive")
    grouped: dict[str, list[ImageRecord]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)
    if not grouped:
        raise ValueError("Cannot plot an empty collection")

    labels = sorted(grouped)
    rng = random.Random(seed)
    selected: dict[str, list[ImageRecord]] = {}
    for label in labels:
        candidates = sorted(grouped[label], key=lambda record: record.path)
        if len(candidates) < examples_per_class:
            raise ValueError(
                f"Class {label} has only {len(candidates)} images; "
                f"cannot select {examples_per_class}."
            )
        selected[label] = rng.sample(candidates, examples_per_class)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(labels), examples_per_class, figsize=(2 * examples_per_class, 1.8 * len(labels))
    )
    # Matplotlib returns a scalar/1-D axes object for degenerate dimensions.
    if len(labels) == 1 and examples_per_class == 1:
        axes = [[axes]]
    elif len(labels) == 1:
        axes = [axes]
    elif examples_per_class == 1:
        axes = [[axis] for axis in axes]

    root = Path(dataset_dir)
    for row, label in enumerate(labels):
        for column, record in enumerate(selected[label]):
            axis = axes[row][column]
            with Image.open(root / record.path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    figure.suptitle("EuroSAT RGB examples (seed 42)", y=1.002)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EuroSAT EDA figures")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--manifest",
        default=None,
        help="manifest to visualize (default: all train/validation/test manifests)",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = load_config(args.config)
    data_config = config["data"]
    if args.manifest:
        records = read_manifest(args.manifest)
    else:
        split_dir = Path(data_config["split_dir"])
        records = []
        for split_name in ("train", "validation", "test"):
            records.extend(read_manifest(split_dir / f"{split_name}.csv"))

    eda_config = config["eda"]
    output_dir = Path(eda_config["output_dir"])
    distribution = plot_class_distribution(
        records,
        output_dir / "class_distribution.png",
        int(eda_config["figure_dpi"]),
    )
    examples = plot_example_grid(
        records,
        data_config["dataset_dir"],
        output_dir / "example_grid.png",
        int(eda_config["examples_per_class"]),
        int(config["project"]["seed"]),
        int(eda_config["figure_dpi"]),
    )
    print(f"Saved {distribution}")
    print(f"Saved {examples}")


if __name__ == "__main__":
    main()

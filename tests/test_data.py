"""Focused tests for deterministic identities and dataset integrity checks."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pytest

from src.data import (
    ImageRecord,
    _safe_extract,
    discover_images,
    stratified_split,
    validate_dataset,
    write_split_manifests,
)
from src.train_classical import audit_phase2, stratified_subsample


def _records(classes: int = 2, per_class: int = 20) -> list[ImageRecord]:
    return [
        ImageRecord(path=f"Class{label}/image_{index}.jpg", label=f"Class{label}")
        for label in range(classes)
        for index in range(per_class)
    ]


def test_stratified_split_is_reproducible_disjoint_and_balanced() -> None:
    records = _records()
    fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}
    first = stratified_split(records, fractions, seed=42)
    second = stratified_split(records, fractions, seed=42)

    assert first == second
    assert {name: len(items) for name, items in first.items()} == {
        "train": 28,
        "validation": 6,
        "test": 6,
    }
    identities = [{record.path for record in first[name]} for name in first]
    assert identities[0].isdisjoint(identities[1])
    assert identities[0].isdisjoint(identities[2])
    assert identities[1].isdisjoint(identities[2])
    assert set.union(*identities) == {record.path for record in records}


def test_split_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        stratified_split(
            _records(), {"train": 0.8, "validation": 0.15, "test": 0.15}, seed=42
        )


def test_manifests_have_stable_schema_and_summary(tmp_path: Path) -> None:
    splits = stratified_split(
        _records(), {"train": 0.70, "validation": 0.15, "test": 0.15}, seed=42
    )
    paths = write_split_manifests(splits, tmp_path)
    with paths["train"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["path", "label"]
        assert len(list(reader)) == 28
    assert (tmp_path / "summary.json").is_file()


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escaped.txt", "not allowed")
    with pytest.raises(ValueError, match="Unsafe path"):
        _safe_extract(archive, tmp_path / "extracted")


def test_validate_tiny_rgb_dataset(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    root = tmp_path / "EuroSAT_RGB"
    expected = {"A": 2, "B": 2}
    for label in expected:
        class_dir = root / label
        class_dir.mkdir(parents=True)
        for index in range(2):
            image_module.new("RGB", (64, 64), color=(index * 20, 80, 120)).save(
                class_dir / f"{index}.jpg"
            )

    assert len(discover_images(root)) == 4
    report = validate_dataset(root, expected)
    assert report.total_images == 4
    assert report.images_inspected == 4


def test_stratified_subsample_is_balanced_and_reproducible() -> None:
    records = _records(classes=3, per_class=20)
    first = stratified_subsample(records, samples_per_class=5, seed=42)
    second = stratified_subsample(records, samples_per_class=5, seed=42)
    assert first == second
    assert len(first) == 15
    assert {label: sum(item.label == label for item in first) for label in {r.label for r in first}} == {
        "Class0": 5,
        "Class1": 5,
        "Class2": 5,
    }


def test_phase2_audit_accepts_exact_partition(tmp_path: Path) -> None:
    dataset = tmp_path / "raw" / "EuroSAT_RGB"
    counts = {"A": 10, "B": 10}
    records = []
    for label, count in counts.items():
        class_dir = dataset / label
        class_dir.mkdir(parents=True)
        for index in range(count):
            image = class_dir / f"{index}.jpg"
            image.write_bytes(b"identity-only audit")
            records.append(ImageRecord(path=f"{label}/{index}.jpg", label=label))

    split_dir = tmp_path / "processed" / "splits"
    splits = stratified_split(
        records, {"train": 0.70, "validation": 0.15, "test": 0.15}, seed=42
    )
    write_split_manifests(splits, split_dir)
    (split_dir / "validation.json").write_text(
        """{
  "class_counts": {"A": 10, "B": 10},
  "dataset_dir": "unused",
  "expected_mode": "RGB",
  "expected_size": [64, 64],
  "images_inspected": 20,
  "total_images": 20
}\n""",
        encoding="utf-8",
    )
    config = {
        "data": {
            "dataset_dir": str(dataset),
            "split_dir": str(split_dir),
            "expected_class_counts": counts,
            "expected_image_size": [64, 64],
            "expected_mode": "RGB",
        },
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
    }
    audit = audit_phase2(config)
    assert audit.total_images == 20
    assert audit.split_counts == {"train": 14, "validation": 4, "test": 2}
    assert not any(audit.overlaps.values())

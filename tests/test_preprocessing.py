"""Tests for manifest parsing and baseline RGB feature extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.preprocessing import flatten_rgb_image, read_manifest


def test_read_manifest_rejects_duplicate_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "train.csv"
    manifest.write_text(
        "path,label\nAnnualCrop/a.jpg,AnnualCrop\nAnnualCrop/a.jpg,AnnualCrop\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate image identity"):
        read_manifest(manifest)


def test_flatten_rgb_image_has_expected_shape_and_range(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    pytest.importorskip("numpy")
    image_path = tmp_path / "image.png"
    image_module.new("RGB", (64, 64), color=(255, 128, 0)).save(image_path)
    vector = flatten_rgb_image(image_path)
    assert vector.shape == (64 * 64 * 3,)
    assert vector.dtype.name == "float32"
    assert float(vector.min()) >= 0.0
    assert float(vector.max()) <= 1.0

"""Focused tests for the leakage-safe PyTorch CNN pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import ImageRecord
from src.train_cnn import (
    ManifestRGBDataset,
    compute_training_channel_stats,
    estimate_cnn_memory_gb,
    run_epoch,
    train_cnn,
)


def test_cnn_dataset_has_explicit_test_guard(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not construct or load a test dataset"):
        ManifestRGBDataset(
            [],
            tmp_path,
            ["A", "B"],
            [0.5, 0.5, 0.5],
            [0.2, 0.2, 0.2],
            split_name="test",
        )


def test_full_training_requires_explicit_acknowledgement() -> None:
    with pytest.raises(RuntimeError, match="blocked until memory/runtime are reviewed"):
        train_cnn({}, mode="full", run_name="never-created", allow_full_training=False)


def test_training_only_channel_statistics(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    root = tmp_path / "images"
    class_dir = root / "A"
    class_dir.mkdir(parents=True)
    image_module.new("RGB", (64, 64), color=(0, 64, 255)).save(class_dir / "a.png")
    image_module.new("RGB", (64, 64), color=(255, 64, 0)).save(class_dir / "b.png")
    records = [ImageRecord("A/a.png", "A"), ImageRecord("A/b.png", "A")]

    mean, std = compute_training_channel_stats(records, root)
    assert mean == pytest.approx([0.5, 64 / 255, 0.5], abs=1e-7)
    assert std[0] == pytest.approx(0.5, abs=1e-7)
    assert std[1] > 0
    assert std[2] == pytest.approx(0.5, abs=1e-7)
    assert np.isfinite(std).all()


def test_cnn_shape_parameter_count_and_cpu_epoch() -> None:
    torch = pytest.importorskip("torch")
    from src.cnn_model import SimpleEuroSATCNN, count_trainable_parameters

    model = SimpleEuroSATCNN()
    assert count_trainable_parameters(model) == 620_362
    inputs = torch.randn(4, 3, 64, 64)
    logits = model(inputs)
    assert logits.shape == (4, 10)

    targets = torch.tensor([0, 1, 2, 3])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets), batch_size=2
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics, observed_targets, predictions = run_epoch(
        model, loader, criterion, "cpu", optimizer=optimizer
    )
    assert set(metrics) == {"loss", "accuracy", "macro_f1"}
    assert len(observed_targets) == len(predictions) == 4


def test_full_batch_memory_estimate_is_bounded() -> None:
    estimate = estimate_cnn_memory_gb(batch_size=64, parameter_count=620_362)
    assert 0.25 < estimate["estimated_training_gb"] < 1.0

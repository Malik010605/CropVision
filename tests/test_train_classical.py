"""Tests for leakage-safe classical model construction and safeguards."""

from __future__ import annotations

import pytest

from src.train_classical import (
    FEATURE_DIMENSION,
    build_pipeline,
    estimate_feature_memory_gb,
)


def test_feature_memory_estimate_uses_float32_rgb_vectors() -> None:
    expected = 1000 * FEATURE_DIMENSION * 4 / (1024**3)
    assert estimate_feature_memory_gb(1000) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("model_name", "expected_steps"),
    [
        ("random_forest", ["classifier"]),
        ("random_forest_pca", ["scaler", "pca", "classifier"]),
        ("svm", ["scaler", "classifier"]),
        ("svm_pca", ["scaler", "pca", "classifier"]),
    ],
)
def test_pipeline_contains_expected_train_fitted_steps(
    model_name: str, expected_steps: list[str]
) -> None:
    pytest.importorskip("sklearn")
    parameters = (
        {"n_estimators": 5, "max_depth": 2, "max_features": "sqrt", "min_samples_leaf": 1}
        if model_name.startswith("random_forest")
        else {"C": 1.0, "gamma": "scale", "kernel": "rbf"}
    )
    pipeline = build_pipeline(
        model_name,
        parameters,
        pca_components=2,
        seed=42,
        max_jobs=1,
        svm_cache_mb=64,
    )
    assert list(pipeline.named_steps) == expected_steps
    if "pca" in expected_steps:
        assert pipeline.named_steps["pca"].random_state == 42
        assert pipeline.named_steps["pca"].copy is False


def test_feature_loader_has_explicit_test_guard(tmp_path) -> None:
    from src.train_classical import load_feature_matrix

    with pytest.raises(ValueError, match="must not load test features"):
        load_feature_matrix(
            manifest_path=tmp_path / "test.csv",
            split_name="test",
            dataset_dir=tmp_path,
            class_names=["A"],
            samples_per_class=1,
            seed=42,
            cache_dir=None,
        )

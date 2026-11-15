"""Tests for frozen provenance and one-shot final-test safeguards."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import src.evaluate as evaluation_module
from src.evaluate import (
    claim_final_evaluation,
    execute_final_test,
    sha256sum,
    verify_frozen_inventory,
)
from src.train_classical import load_feature_matrix
from src.train_cnn import ManifestRGBDataset
from src.utils import load_config


def test_final_test_requires_confirmation_before_any_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("split audit must not run without confirmation")

    monkeypatch.setattr(evaluation_module, "audit_phase2", fail_if_called)
    with pytest.raises(RuntimeError, match="requires explicit approval"):
        execute_final_test({}, {}, run_name="not-created", confirmed=False)


def test_one_shot_claim_refuses_reuse(tmp_path: Path) -> None:
    state = tmp_path / "final_test_state.json"
    claim_final_evaluation(state, {"status": "running", "run_name": "first"})
    with pytest.raises(RuntimeError, match="already been claimed"):
        claim_final_evaluation(state, {"status": "running", "run_name": "second"})


def test_hash_detects_artifact_change(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"frozen")
    first = sha256sum(artifact)
    artifact.write_bytes(b"changed")
    assert sha256sum(artifact) != first


def test_final_evaluation_contains_no_training_calls() -> None:
    tree = ast.parse(inspect.getsource(execute_final_test))
    forbidden = {"fit", "fit_transform", "partial_fit", "backward", "step"}
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert forbidden.isdisjoint(called_attributes)


def test_training_loaders_continue_to_reject_test_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not load test features"):
        load_feature_matrix(
            manifest_path=tmp_path / "test.csv",
            split_name="test",
            dataset_dir=tmp_path,
            class_names=["A"],
            samples_per_class=None,
            seed=42,
            cache_dir=None,
        )
    with pytest.raises(ValueError, match="must not construct or load a test dataset"):
        ManifestRGBDataset(
            [],
            tmp_path,
            ["A", "B"],
            [0.5, 0.5, 0.5],
            [0.2, 0.2, 0.2],
            split_name="test",
        )


def test_real_frozen_inventory_verifies_without_test_loading() -> None:
    project = load_config("configs/default.yaml")
    evaluation = load_config("configs/final_evaluation.yaml")
    inventory = verify_frozen_inventory(project, evaluation)
    assert set(inventory) == {"random_forest", "random_forest_pca", "svm_pca", "cnn"}
    assert all(item["test_samples_used_during_selection"] == 0 for item in inventory.values())

"""One-shot final evaluation of frozen CropVision models on EuroSAT test data.

Inventory verification never opens test images. The ``final-test`` command is a
separate, explicitly confirmed operation that loads each frozen artifact and
scores the common test identities exactly once without fitting or tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data import ImageRecord
from src.preprocessing import read_manifest
from src.train_classical import audit_phase2, calculate_metrics
from src.utils import load_config, runtime_metadata, save_json


def sha256sum(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest used to freeze model identity."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Required provenance file is missing: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {source}")
    return payload


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def verify_frozen_inventory(
    project_config: Mapping[str, Any], evaluation_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify artifact hashes and validation-only selection provenance.

    This function loads model files and source metadata but never reads the test
    manifest or opens a test image.
    """
    try:
        import joblib
        import torch
        from sklearn.utils.validation import check_is_fitted
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Phase 5 dependencies are not installed; see SETUP.md.") from exc
    from src.cnn_model import SimpleEuroSATCNN, count_trainable_parameters

    seed = int(project_config["project"]["seed"])
    if int(evaluation_config["evaluation"]["seed"]) != seed:
        raise ValueError("Final-evaluation seed differs from the project seed")
    expected_classes = sorted(project_config["data"]["expected_class_counts"])
    verified: dict[str, Any] = {}

    for model_name, specification in evaluation_config["models"].items():
        artifact_path = Path(specification["artifact"])
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Frozen artifact missing: {artifact_path}")
        actual_hash = sha256sum(artifact_path)
        if actual_hash != specification["sha256"]:
            raise ValueError(
                f"Frozen artifact hash mismatch for {model_name}: "
                f"expected {specification['sha256']}, got {actual_hash}"
            )
        source_config = _load_json(specification["source_run_config"])
        source_results = _load_json(specification["source_results"])
        source_runtime = _load_json(specification["source_runtime"])
        if int(source_config["seed"]) != seed:
            raise ValueError(f"Source seed mismatch for {model_name}")
        if int(source_config.get("test_features_loaded", source_config.get("test_samples_loaded", -1))) != 0:
            raise ValueError(f"Source run accessed test data for {model_name}")
        if source_config["class_names"] != expected_classes:
            raise ValueError(f"Class order mismatch for {model_name}")

        record: dict[str, Any] = {
            "artifact": str(artifact_path),
            "sha256": actual_hash,
            "kind": specification["kind"],
            "seed": seed,
            "source_run_config": specification["source_run_config"],
            "source_results": specification["source_results"],
            "source_runtime": specification["source_runtime"],
            "training_python": source_runtime["python"],
            "training_packages": source_runtime["packages"],
            "test_samples_used_during_selection": 0,
        }
        if specification["kind"] == "sklearn":
            pipeline = joblib.load(artifact_path)
            if list(pipeline.named_steps) != list(specification["expected_steps"]):
                raise ValueError(f"Pipeline-step mismatch for {model_name}")
            check_is_fitted(pipeline)
            results = source_results[model_name]
            if int(results["best_candidate_index"]) != int(specification["selected_candidate"]):
                raise ValueError(f"Selected-candidate mismatch for {model_name}")
            selected = results["candidates"][int(specification["selected_candidate"])]
            if not _close(selected["accuracy"], specification["expected_validation_accuracy"]):
                raise ValueError(f"Frozen validation accuracy mismatch for {model_name}")
            if not _close(selected["macro_f1"], specification["expected_validation_macro_f1"]):
                raise ValueError(f"Frozen validation macro-F1 mismatch for {model_name}")
            classifier = pipeline.named_steps["classifier"]
            expected_classifier = specification["classifier"]
            for parameter in (
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "max_features",
                "random_state",
                "C",
                "gamma",
                "kernel",
            ):
                if parameter in expected_classifier:
                    actual = getattr(classifier, parameter)
                    if actual != expected_classifier[parameter]:
                        raise ValueError(
                            f"Classifier parameter mismatch for {model_name}.{parameter}: "
                            f"expected {expected_classifier[parameter]}, got {actual}"
                        )
            if "pca" in pipeline.named_steps:
                pca = pipeline.named_steps["pca"]
                preprocessing = specification["preprocessing"]
                if int(pca.n_components_) != int(preprocessing["pca_components"]):
                    raise ValueError(f"PCA component mismatch for {model_name}")
                if pca.svd_solver != preprocessing["pca_solver"]:
                    raise ValueError(f"PCA solver mismatch for {model_name}")
                if pca.random_state != preprocessing["pca_random_state"]:
                    raise ValueError(f"PCA seed mismatch for {model_name}")
            record.update(
                {
                    "selected_candidate": int(specification["selected_candidate"]),
                    "pipeline_steps": list(pipeline.named_steps),
                    "validation_accuracy": float(selected["accuracy"]),
                    "validation_macro_f1": float(selected["macro_f1"]),
                    "evaluation_sklearn_version": metadata.version("scikit-learn"),
                }
            )
        else:
            checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=False)
            if int(checkpoint["epoch"]) != int(specification["selected_epoch"]):
                raise ValueError("CNN selected-epoch mismatch")
            if int(checkpoint["seed"]) != seed:
                raise ValueError("CNN checkpoint seed mismatch")
            if checkpoint["class_names"] != expected_classes:
                raise ValueError("CNN checkpoint class order mismatch")
            if source_results.get("test_samples_loaded", -1) != 0:
                raise ValueError("CNN source summary indicates test access")
            if int(source_results["best_epoch"]) != int(specification["selected_epoch"]):
                raise ValueError("CNN source best epoch differs from frozen epoch")
            best_validation = source_results["best_validation"]
            if not _close(best_validation["accuracy"], specification["expected_validation_accuracy"]):
                raise ValueError("CNN frozen validation accuracy mismatch")
            if not _close(best_validation["macro_f1"], specification["expected_validation_macro_f1"]):
                raise ValueError("CNN frozen validation macro-F1 mismatch")
            architecture = specification["architecture"]
            model = SimpleEuroSATCNN(
                num_classes=int(architecture["num_classes"]),
                channels=tuple(int(value) for value in architecture["channels"]),
                hidden_units=int(architecture["hidden_units"]),
                dropout=float(architecture["dropout"]),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            if count_trainable_parameters(model) != int(architecture["parameter_count"]):
                raise ValueError("CNN parameter-count mismatch")
            record.update(
                {
                    "selected_epoch": int(checkpoint["epoch"]),
                    "parameter_count": count_trainable_parameters(model),
                    "normalization": checkpoint["normalization"],
                    "architecture": checkpoint["architecture"],
                    "validation_accuracy": float(best_validation["accuracy"]),
                    "validation_macro_f1": float(best_validation["macro_f1"]),
                    "evaluation_torch_version": metadata.version("torch"),
                }
            )
        verified[model_name] = record
    if set(verified) != {"random_forest", "random_forest_pca", "svm_pca", "cnn"}:
        raise ValueError("Frozen inventory must contain exactly the four selected models")
    return verified


def claim_final_evaluation(state_path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically claim the one allowed final-test evaluation."""
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        existing = _load_json(path)
        raise RuntimeError(
            "Final test evaluation has already been claimed; refusing to inspect the "
            f"test set again. Existing state: {existing.get('status', 'unknown')}"
        ) from exc


def _load_classical_test_features(
    records: Sequence[ImageRecord],
    dataset_dir: Path,
    class_names: Sequence[str],
) -> tuple[Any, Any]:
    """Decode the frozen test identities once for the sklearn pipelines."""
    import numpy as np
    from PIL import Image

    label_to_index = {label: index for index, label in enumerate(class_names)}
    X = np.empty((len(records), 64 * 64 * 3), dtype=np.float32)
    y = np.empty(len(records), dtype=np.int64)
    for index, record in enumerate(records):
        with Image.open(dataset_dir / record.path) as image:
            if image.mode != "RGB" or image.size != (64, 64):
                raise ValueError(f"Unexpected test image: {record.path}")
            X[index] = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
        y[index] = label_to_index[record.label]
    return X, y


def _load_cnn_test_tensors(
    records: Sequence[ImageRecord],
    dataset_dir: Path,
    class_names: Sequence[str],
    normalization: Mapping[str, Sequence[float]],
) -> tuple[Any, Any]:
    """Decode the same frozen test identities once into normalized CNN tensors."""
    import numpy as np
    import torch
    from PIL import Image

    label_to_index = {label: index for index, label in enumerate(class_names)}
    inputs = torch.empty((len(records), 3, 64, 64), dtype=torch.float32)
    targets = torch.empty(len(records), dtype=torch.long)
    mean = torch.tensor(normalization["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(normalization["std"], dtype=torch.float32).view(3, 1, 1)
    for index, record in enumerate(records):
        with Image.open(dataset_dir / record.path) as image:
            if image.mode != "RGB" or image.size != (64, 64):
                raise ValueError(f"Unexpected test image: {record.path}")
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        inputs[index] = (tensor - mean) / std
        targets[index] = label_to_index[record.label]
    return inputs, targets


def _save_confusion_csv(matrix: Any, class_names: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label/predicted_label", *class_names])
        for label, row in zip(class_names, matrix.tolist()):
            writer.writerow([label, *row])


def _plot_confusion(matrix: Any, class_names: Sequence[str], model_name: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    figure, axis = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d")
    axis.set_title(f"Final test confusion matrix — {model_name}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_comparison(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = [str(row["model"]) for row in rows]
    positions = np.arange(len(rows))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(positions - width / 2, [row["accuracy"] for row in rows], width, label="Accuracy")
    axis.bar(positions + width / 2, [row["macro_f1"] for row in rows], width, label="Macro F1")
    axis.set_xticks(positions, names, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set(title="Frozen-model final test comparison", ylabel="Test score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_comparison(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = [
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "inference_seconds",
        "test_samples",
        "timing_scope",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _write_per_class(results: Mapping[str, Any], class_names: Sequence[str], path: Path) -> None:
    fields = ["model", "class", "precision", "recall", "f1", "support"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name, payload in results.items():
            for label in class_names:
                row = payload["per_class"][label]
                writer.writerow(
                    {
                        "model": model_name,
                        "class": label,
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1-score"],
                        "support": row["support"],
                    }
                )


def execute_final_test(
    project_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    *,
    run_name: str,
    confirmed: bool,
) -> Path:
    """Evaluate frozen artifacts once; this function contains no training calls."""
    if not confirmed:
        raise RuntimeError(
            "Final test evaluation requires explicit approval and --confirm-final-test."
        )
    try:
        import joblib
        import numpy as np
        import torch
        from sklearn.metrics import confusion_matrix
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Phase 5 dependencies are not installed; see SETUP.md.") from exc
    from src.cnn_model import SimpleEuroSATCNN
    from src.train_cnn import select_device

    audit = audit_phase2(project_config)
    inventory = verify_frozen_inventory(project_config, evaluation_config)
    evaluation = evaluation_config["evaluation"]
    output_root = Path(evaluation["output_dir"]) / run_name
    figure_root = Path(evaluation["figure_dir"]) / run_name
    if output_root.exists() or figure_root.exists():
        raise FileExistsError(f"Final evaluation run already exists: {run_name}")

    inventory_digest = hashlib.sha256(
        json.dumps(inventory, sort_keys=True).encode("utf-8")
    ).hexdigest()
    started_at = datetime.now(timezone.utc).isoformat()
    claim_final_evaluation(
        evaluation["state_file"],
        {
            "status": "running",
            "run_name": run_name,
            "started_at_utc": started_at,
            "inventory_sha256": inventory_digest,
            "test_images_expected": int(evaluation["expected_test_images"]),
        },
    )
    output_root.mkdir(parents=True)
    figure_root.mkdir(parents=True)
    matplotlib_cache = Path("data/processed/matplotlib_cache").resolve()
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    try:
        class_names = sorted(project_config["data"]["expected_class_counts"])
        test_records = read_manifest(evaluation["test_manifest"])
        if len(test_records) != int(evaluation["expected_test_images"]):
            raise ValueError("Test manifest size differs from the frozen evaluation protocol")
        dataset_dir = Path(project_config["data"]["dataset_dir"])
        preprocessing_started = time.perf_counter()
        X_test, y_test = _load_classical_test_features(
            test_records, dataset_dir, class_names
        )
        classical_preprocessing_seconds = time.perf_counter() - preprocessing_started

        results: dict[str, Any] = {}
        comparison: list[dict[str, Any]] = []
        labels = list(range(len(class_names)))
        for model_name in ("random_forest", "random_forest_pca", "svm_pca"):
            pipeline = joblib.load(inventory[model_name]["artifact"])
            inference_started = time.perf_counter()
            predictions = pipeline.predict(X_test)
            inference_seconds = time.perf_counter() - inference_started
            metrics, matrix = calculate_metrics(y_test, predictions, class_names)
            results[model_name] = {
                **metrics,
                "inference_seconds": inference_seconds,
                "test_samples": len(test_records),
                "timing_scope": "fitted_pipeline_transform_and_predict; shared_JPEG_decode_excluded",
            }
            _save_confusion_csv(
                matrix, class_names, output_root / f"{model_name}_confusion_matrix.csv"
            )
            _plot_confusion(
                matrix,
                class_names,
                model_name,
                figure_root / f"{model_name}_confusion_matrix.png",
            )
            comparison.append(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "inference_seconds": inference_seconds,
                    "test_samples": len(test_records),
                    "timing_scope": results[model_name]["timing_scope"],
                }
            )

        cnn_checkpoint = torch.load(
            inventory["cnn"]["artifact"], map_location="cpu", weights_only=False
        )
        cnn_preprocessing_started = time.perf_counter()
        cnn_inputs, cnn_targets = _load_cnn_test_tensors(
            test_records,
            dataset_dir,
            class_names,
            cnn_checkpoint["normalization"],
        )
        cnn_preprocessing_seconds = time.perf_counter() - cnn_preprocessing_started
        if not np.array_equal(y_test, cnn_targets.numpy()):
            raise ValueError("CNN and classical test labels differ")
        architecture = evaluation_config["models"]["cnn"]["architecture"]
        cnn = SimpleEuroSATCNN(
            num_classes=int(architecture["num_classes"]),
            channels=tuple(int(value) for value in architecture["channels"]),
            hidden_units=int(architecture["hidden_units"]),
            dropout=float(architecture["dropout"]),
        )
        cnn.load_state_dict(cnn_checkpoint["model_state_dict"])
        device = select_device()
        cnn.to(device)
        cnn.eval()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(cnn_inputs, cnn_targets),
            batch_size=int(evaluation["cnn_batch_size"]),
            shuffle=False,
            num_workers=0,
        )
        cnn_predictions: list[int] = []
        inference_started = time.perf_counter()
        with torch.inference_mode():
            for inputs, _ in loader:
                logits = cnn(inputs.to(device))
                cnn_predictions.extend(logits.argmax(dim=1).cpu().tolist())
        cnn_inference_seconds = time.perf_counter() - inference_started
        cnn_metrics, cnn_matrix = calculate_metrics(
            cnn_targets.numpy(), np.asarray(cnn_predictions), class_names
        )
        results["cnn"] = {
            **cnn_metrics,
            "inference_seconds": cnn_inference_seconds,
            "test_samples": len(test_records),
            "timing_scope": "preloaded_normalized_tensor_forward_pass; JPEG_decode_excluded",
            "device": device,
        }
        _save_confusion_csv(
            cnn_matrix, class_names, output_root / "cnn_confusion_matrix.csv"
        )
        _plot_confusion(
            cnn_matrix, class_names, "cnn", figure_root / "cnn_confusion_matrix.png"
        )
        comparison.append(
            {
                "model": "cnn",
                "accuracy": cnn_metrics["accuracy"],
                "macro_precision": cnn_metrics["macro_precision"],
                "macro_recall": cnn_metrics["macro_recall"],
                "macro_f1": cnn_metrics["macro_f1"],
                "inference_seconds": cnn_inference_seconds,
                "test_samples": len(test_records),
                "timing_scope": results["cnn"]["timing_scope"],
            }
        )

        provenance = {
            "run_name": run_name,
            "dataset_split": "test",
            "test_manifest": evaluation["test_manifest"],
            "test_images": len(test_records),
            "test_evaluations_per_model": 1,
            "retraining_or_tuning_performed": False,
            "class_names": class_names,
            "split_audit": asdict(audit),
            "frozen_inventory": inventory,
            "inventory_sha256": inventory_digest,
            "timings": {
                "classical_shared_jpeg_decode_seconds": classical_preprocessing_seconds,
                "cnn_jpeg_decode_and_normalize_seconds": cnn_preprocessing_seconds,
            },
        }
        save_json(results, output_root / "test_metrics.json")
        save_json(provenance, output_root / "provenance.json")
        save_json(asdict(audit), output_root / "split_audit.json")
        save_json(runtime_metadata(dict(project_config)), output_root / "runtime.json")
        save_json(dict(evaluation_config), output_root / "evaluation_config.json")
        _write_comparison(comparison, output_root / "comparison.csv")
        _write_per_class(results, class_names, output_root / "per_class_metrics.csv")
        _plot_comparison(comparison, figure_root / "test_comparison.png")

        completed_state = {
            "status": "complete",
            "run_name": run_name,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "inventory_sha256": inventory_digest,
            "test_images": len(test_records),
            "models_evaluated": [row["model"] for row in comparison],
            "retraining_or_tuning_performed": False,
        }
        save_json(completed_state, evaluation["state_file"])
        print(f"Completed one-shot final test evaluation: {run_name}")
        print(f"Metrics: {output_root}")
        return output_root
    except Exception:
        failed_state = {
            "status": "failed",
            "run_name": run_name,
            "started_at_utc": started_at,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "inventory_sha256": inventory_digest,
            "message": "Evaluation failed; test reuse remains blocked pending explicit review.",
        }
        save_json(failed_state, evaluation["state_file"])
        raise


def _default_run_name() -> str:
    return "final_test_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--evaluation-config", default="configs/final_evaluation.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "inventory", help="verify frozen model provenance without reading test images"
    )
    final = subparsers.add_parser(
        "final-test", help="run the one-shot held-out test evaluation"
    )
    final.add_argument("--run-name", default=None)
    final.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    project_config = load_config(args.config)
    evaluation_config = load_config(args.evaluation_config)
    if args.command == "inventory":
        verified = verify_frozen_inventory(project_config, evaluation_config)
        print(json.dumps(verified, indent=2, sort_keys=True))
        return
    execute_final_test(
        project_config,
        evaluation_config,
        run_name=args.run_name or _default_run_name(),
        confirmed=args.confirm_final_test,
    )


if __name__ == "__main__":
    main()

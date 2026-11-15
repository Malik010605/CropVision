"""Train and select CropVision's classical ML models on validation data only.

The command audits all split identities, but feature loading is deliberately
limited to train and validation manifests. Test images are reserved for the
later unified final-evaluation phase.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data import ImageRecord, discover_images
from src.preprocessing import read_manifest
from src.utils import load_config, runtime_metadata, save_json, set_random_seeds

MODEL_NAMES = ("random_forest", "random_forest_pca", "svm", "svm_pca")
FEATURE_DIMENSION = 64 * 64 * 3


@dataclass(frozen=True)
class SplitAudit:
    """Evidence that Phase 2 produced one complete, disjoint split assignment."""

    total_images: int
    split_counts: dict[str, int]
    overlaps: dict[str, int]
    missing_from_manifests: int
    unknown_manifest_paths: int
    validation_images_inspected: int


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 2 artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _expected_split_counts(
    class_counts: Mapping[str, int], fractions: Mapping[str, float]
) -> dict[str, dict[str, int]]:
    output = {name: {} for name in ("train", "validation", "test")}
    for label, total in class_counts.items():
        train = round(total * fractions["train"])
        validation = round(total * fractions["validation"])
        output["train"][label] = train
        output["validation"][label] = validation
        output["test"][label] = total - train - validation
    return output


def audit_phase2(config: Mapping[str, Any]) -> SplitAudit:
    """Verify full image validation and exact, non-overlapping split manifests.

    This function reads test manifest identities for integrity only. It never
    decodes test images, creates test features, or computes test performance.
    """
    data_config = config["data"]
    dataset_dir = Path(data_config["dataset_dir"])
    split_dir = Path(data_config["split_dir"])
    expected_counts = {
        str(label): int(count)
        for label, count in data_config["expected_class_counts"].items()
    }
    expected_total = sum(expected_counts.values())

    validation = _load_json(split_dir / "validation.json")
    if int(validation.get("total_images", -1)) != expected_total:
        raise ValueError("Phase 2 validation did not cover all expected images")
    if int(validation.get("images_inspected", -1)) != expected_total:
        raise ValueError(
            "Phase 2 validation was sampled rather than full; rerun "
            "`python -m src.data validate` before training."
        )
    if validation.get("class_counts") != expected_counts:
        raise ValueError("Phase 2 validation class counts do not match the configuration")
    if list(validation.get("expected_size", [])) != list(data_config["expected_image_size"]):
        raise ValueError("Phase 2 validation image size does not match the configuration")
    if validation.get("expected_mode") != data_config["expected_mode"]:
        raise ValueError("Phase 2 validation image mode does not match the configuration")

    actual_records = discover_images(dataset_dir)
    actual_paths = {record.path for record in actual_records}
    actual_counts = Counter(record.label for record in actual_records)
    if len(actual_paths) != expected_total or dict(sorted(actual_counts.items())) != dict(
        sorted(expected_counts.items())
    ):
        raise ValueError("The current dataset contents differ from the validated configuration")

    expected_by_split = _expected_split_counts(expected_counts, config["split"])
    split_sets: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    for split_name in ("train", "validation", "test"):
        manifest = read_manifest(split_dir / f"{split_name}.csv")
        paths = {record.path for record in manifest}
        if len(paths) != len(manifest):
            raise ValueError(f"Duplicate paths in {split_name} manifest")
        labels = Counter(record.label for record in manifest)
        if dict(sorted(labels.items())) != dict(sorted(expected_by_split[split_name].items())):
            raise ValueError(f"Unexpected class counts in {split_name} manifest")
        bad_labels = [
            record.path
            for record in manifest
            if Path(record.path).parent.name != record.label
        ]
        if bad_labels:
            raise ValueError(
                f"Path/label mismatch in {split_name}: {bad_labels[:5]}"
            )
        split_sets[split_name] = paths
        split_counts[split_name] = len(paths)

    overlap_pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    overlaps = {
        f"{left}/{right}": len(split_sets[left] & split_sets[right])
        for left, right in overlap_pairs
    }
    if any(overlaps.values()):
        raise ValueError(f"Split manifests overlap: {overlaps}")
    manifest_union = set.union(*split_sets.values())
    missing = actual_paths - manifest_union
    unknown = manifest_union - actual_paths
    if missing or unknown or len(manifest_union) != expected_total:
        raise ValueError(
            "Split manifests do not exactly partition the dataset: "
            f"missing={len(missing)}, unknown={len(unknown)}"
        )

    return SplitAudit(
        total_images=expected_total,
        split_counts=split_counts,
        overlaps=overlaps,
        missing_from_manifests=len(missing),
        unknown_manifest_paths=len(unknown),
        validation_images_inspected=int(validation["images_inspected"]),
    )


def stratified_subsample(
    records: Sequence[ImageRecord], samples_per_class: int | None, seed: int
) -> list[ImageRecord]:
    """Select up to a fixed number per class without changing shared identities."""
    if samples_per_class is None:
        return list(records)
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive or null")
    grouped: dict[str, list[ImageRecord]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)
    selected: list[ImageRecord] = []
    for class_index, label in enumerate(sorted(grouped)):
        candidates = sorted(grouped[label], key=lambda item: item.path)
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"Class {label} contains {len(candidates)} records, fewer than "
                f"the requested {samples_per_class}."
            )
        selected.extend(random.Random(seed + class_index).sample(candidates, samples_per_class))
    random.Random(seed).shuffle(selected)
    return selected


def estimate_feature_memory_gb(sample_count: int) -> float:
    """Return raw float32 feature-matrix size in GiB."""
    return sample_count * FEATURE_DIMENSION * 4 / (1024**3)


def _manifest_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(
    cache_dir: Path,
    split_name: str,
    manifest_path: Path,
    samples_per_class: int | None,
    seed: int,
) -> Path:
    identity = json.dumps(
        {
            "manifest_sha256": _manifest_digest(manifest_path),
            "samples_per_class": samples_per_class,
            "seed": seed,
            "feature": "rgb_float32_flat_64x64x3_v1",
        },
        sort_keys=True,
    ).encode("utf-8")
    suffix = hashlib.sha256(identity).hexdigest()[:16]
    return cache_dir / f"{split_name}_{suffix}.npz"


def load_feature_matrix(
    *,
    manifest_path: Path,
    split_name: str,
    dataset_dir: Path,
    class_names: Sequence[str],
    samples_per_class: int | None,
    seed: int,
    cache_dir: Path | None,
) -> tuple[Any, Any, list[ImageRecord], Path | None]:
    """Decode a train/validation manifest into float32 flattened RGB features."""
    if split_name == "test":
        raise ValueError("Phase 3 must not load test features")
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("NumPy and Pillow are required; see SETUP.md.") from exc

    records = stratified_subsample(read_manifest(manifest_path), samples_per_class, seed)
    cached: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = _cache_path(
            cache_dir, split_name, manifest_path, samples_per_class, seed
        )
        if cached.is_file():
            with np.load(cached, allow_pickle=False) as payload:
                paths = payload["paths"].astype(str).tolist()
                if paths != [record.path for record in records]:
                    raise ValueError(f"Feature cache identity mismatch: {cached}")
                X = payload["X"]
                y = payload["y"]
            return X, y, records, cached

    label_to_index = {label: index for index, label in enumerate(class_names)}
    X = np.empty((len(records), FEATURE_DIMENSION), dtype=np.float32)
    y = np.empty(len(records), dtype=np.int64)
    for index, record in enumerate(records):
        image_path = dataset_dir / record.path
        with Image.open(image_path) as image:
            if image.size != (64, 64) or image.mode != "RGB":
                raise ValueError(
                    f"Unexpected image during feature loading: {record.path}, "
                    f"size={image.size}, mode={image.mode}"
                )
            X[index] = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
        y[index] = label_to_index[record.label]

    if cached is not None:
        temporary = cached.with_suffix(cached.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                X=X,
                y=y,
                paths=np.asarray([record.path for record in records]),
            )
        temporary.replace(cached)
    return X, y, records, cached


def build_pipeline(
    model_name: str,
    classifier_parameters: Mapping[str, Any],
    *,
    pca_components: int,
    seed: int,
    max_jobs: int,
    svm_cache_mb: int,
) -> Any:
    """Create an unfitted sklearn pipeline; every learned step fits on train only."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("scikit-learn is required; see SETUP.md.") from exc

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    steps: list[tuple[str, Any]] = []
    if model_name in {"svm", "svm_pca", "random_forest_pca"}:
        steps.append(("scaler", StandardScaler()))
    if model_name.endswith("_pca"):
        steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_components,
                    # The input is StandardScaler's temporary output, so PCA may
                    # center it in place without mutating the shared raw features.
                    # This saves roughly 0.87 GiB in the full 18,900-image run.
                    copy=False,
                    svd_solver="randomized",
                    random_state=seed,
                ),
            )
        )

    parameters = dict(classifier_parameters)
    if model_name.startswith("random_forest"):
        classifier = RandomForestClassifier(
            **parameters,
            random_state=seed,
            n_jobs=max_jobs,
            class_weight=None,
        )
    else:
        classifier = SVC(
            **parameters,
            cache_size=svm_cache_mb,
            random_state=seed,
        )
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def calculate_metrics(
    y_true: Any, y_pred: Any, class_names: Sequence[str]
) -> tuple[dict[str, Any], Any]:
    """Calculate validation metrics without consulting test data."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            precision_recall_fscore_support,
        )
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("scikit-learn is required; see SETUP.md.") from exc

    labels = list(range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class": report,
    }
    return metrics, matrix


def _plot_confusion_matrix(
    matrix: Any,
    class_names: Sequence[str],
    model_name: str,
    path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Matplotlib and scikit-learn are required; see SETUP.md.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d")
    axis.set_title(f"Validation confusion matrix — {model_name}")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_comparison(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("NumPy and Matplotlib are required; see SETUP.md.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    names = [str(row["model"]) for row in rows]
    positions = np.arange(len(rows))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(positions - width / 2, [row["accuracy"] for row in rows], width, label="Accuracy")
    axis.bar(positions + width / 2, [row["macro_f1"] for row in rows], width, label="Macro F1")
    axis.set_xticks(positions, names, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Validation score")
    axis.set_title("Traditional ML validation comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_summary_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = [
        "model",
        "candidate_index",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "fit_seconds",
        "predict_seconds",
        "train_samples",
        "validation_samples",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _candidate_parameters(
    model_name: str, mode_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    key = "random_forest_candidates" if model_name.startswith("random_forest") else "svm_candidates"
    candidates = mode_config.get(key)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"No candidates configured for {model_name}")
    return [dict(candidate) for candidate in candidates]


def train_models(
    config: Mapping[str, Any],
    *,
    mode: str,
    model_names: Sequence[str],
    run_name: str,
    cache_features: bool,
    allow_expensive_full_svm: bool,
) -> Path:
    """Train candidate pipelines and select each variant using validation macro F1."""
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    unknown = set(model_names) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    if mode == "full" and any(name.startswith("svm") for name in model_names):
        if not allow_expensive_full_svm:
            raise RuntimeError(
                "Full RBF SVM training can require hours and substantial memory. "
                "Rerun with --allow-expensive-full-svm only after explicit approval."
            )

    audit = audit_phase2(config)
    seed = int(config["project"]["seed"])
    set_random_seeds(seed)
    classical = config["classical"]
    mode_config = classical[mode]
    max_jobs = max(1, min(int(classical["max_parallel_jobs"]), os.cpu_count() or 1))
    class_names = sorted(config["data"]["expected_class_counts"])
    split_dir = Path(config["data"]["split_dir"])
    dataset_dir = Path(config["data"]["dataset_dir"])
    train_limit = mode_config["train_samples_per_class"]
    validation_limit = mode_config["validation_samples_per_class"]
    train_count = audit.split_counts["train"] if train_limit is None else int(train_limit) * len(class_names)
    validation_count = (
        audit.split_counts["validation"]
        if validation_limit is None
        else int(validation_limit) * len(class_names)
    )
    raw_memory = estimate_feature_memory_gb(train_count + validation_count)
    if raw_memory > float(classical["memory_budget_gb"]):
        raise MemoryError(
            f"Raw feature matrices need approximately {raw_memory:.2f} GiB, above the "
            f"configured {classical['memory_budget_gb']} GiB budget."
        )

    output_root = Path(classical["output_dir"]) / run_name
    figure_root = Path(classical["figure_dir"]) / run_name
    model_root = Path(classical["model_dir"]) / run_name
    if output_root.exists() or figure_root.exists() or model_root.exists():
        raise FileExistsError(f"Run name already exists: {run_name}")
    output_root.mkdir(parents=True)
    figure_root.mkdir(parents=True)
    model_root.mkdir(parents=True)
    matplotlib_cache = Path("data/processed/matplotlib_cache").resolve()
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    run_config = {
        "run_name": run_name,
        "mode": mode,
        "models": list(model_names),
        "seed": seed,
        "class_names": class_names,
        "max_parallel_jobs": max_jobs,
        "cache_features": cache_features,
        "estimated_raw_feature_memory_gb": raw_memory,
        "test_features_loaded": 0,
        "selection_metric": "validation_macro_f1",
        "source_config": config,
    }
    save_json(run_config, output_root / "run_config.json")
    save_json(asdict(audit), output_root / "split_audit.json")
    save_json(runtime_metadata(dict(config)), output_root / "runtime.json")

    cache_dir = Path(classical["feature_cache_dir"]) if cache_features else None
    X_train, y_train, train_records, train_cache = load_feature_matrix(
        manifest_path=split_dir / "train.csv",
        split_name="train",
        dataset_dir=dataset_dir,
        class_names=class_names,
        samples_per_class=train_limit,
        seed=seed,
        cache_dir=cache_dir,
    )
    X_validation, y_validation, validation_records, validation_cache = load_feature_matrix(
        manifest_path=split_dir / "validation.csv",
        split_name="validation",
        dataset_dir=dataset_dir,
        class_names=class_names,
        samples_per_class=validation_limit,
        seed=seed,
        cache_dir=cache_dir,
    )
    feature_metadata = {
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "feature_dimension": FEATURE_DIMENSION,
        "dtype": str(X_train.dtype),
        "train_cache": str(train_cache) if train_cache else None,
        "validation_cache": str(validation_cache) if validation_cache else None,
        "test_samples_loaded": 0,
    }
    save_json(feature_metadata, output_root / "features.json")

    try:
        import joblib
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("joblib and scikit-learn dependencies are required; see SETUP.md.") from exc

    summary_rows: list[dict[str, Any]] = []
    all_results: dict[str, Any] = {}
    for model_name in model_names:
        candidates = _candidate_parameters(model_name, mode_config)
        candidate_results: list[dict[str, Any]] = []
        best: tuple[float, float, Any, dict[str, Any], Any] | None = None
        for candidate_index, parameters in enumerate(candidates):
            pipeline = build_pipeline(
                model_name,
                parameters,
                pca_components=int(mode_config["pca_components"]),
                seed=seed,
                max_jobs=max_jobs,
                svm_cache_mb=int(mode_config["svm_cache_mb"]),
            )
            fit_started = time.perf_counter()
            with threadpool_limits(limits=max_jobs):
                pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            predictions = pipeline.predict(X_validation)
            predict_seconds = time.perf_counter() - predict_started
            metrics, matrix = calculate_metrics(y_validation, predictions, class_names)
            result = {
                "candidate_index": candidate_index,
                "classifier_parameters": parameters,
                "pca_components": int(mode_config["pca_components"])
                if model_name.endswith("_pca")
                else None,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                **metrics,
            }
            candidate_results.append(result)
            ranking = (metrics["macro_f1"], metrics["accuracy"])
            if best is None or ranking > best[:2]:
                best = (*ranking, pipeline, result, matrix)
            else:
                del pipeline
            gc.collect()

        assert best is not None
        best_pipeline = best[2]
        best_result = best[3]
        best_matrix = best[4]
        model_path = model_root / f"{model_name}_validation_best_train_only.joblib"
        joblib.dump(best_pipeline, model_path, compress=3)
        _plot_confusion_matrix(
            best_matrix,
            class_names,
            model_name,
            figure_root / f"{model_name}_validation_confusion_matrix.png",
        )
        all_results[model_name] = {
            "selection_metric": "macro_f1",
            "best_candidate_index": best_result["candidate_index"],
            "best_model_path": str(model_path),
            "candidates": candidate_results,
        }
        summary_rows.append(
            {
                "model": model_name,
                "candidate_index": best_result["candidate_index"],
                "accuracy": best_result["accuracy"],
                "macro_precision": best_result["macro_precision"],
                "macro_recall": best_result["macro_recall"],
                "macro_f1": best_result["macro_f1"],
                "fit_seconds": best_result["fit_seconds"],
                "predict_seconds": best_result["predict_seconds"],
                "train_samples": len(train_records),
                "validation_samples": len(validation_records),
            }
        )
        save_json(all_results, output_root / "validation_results.json")

    _write_summary_csv(summary_rows, output_root / "validation_summary.csv")
    _plot_comparison(summary_rows, figure_root / "validation_comparison.png")
    print(f"Completed {mode} validation run: {run_name}")
    print(f"Metrics: {output_root}")
    print("Test features loaded: 0 (reserved for final evaluation)")
    return output_root


def _default_run_name(mode: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{mode}_{timestamp}"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="verify Phase 2 without loading image features")
    train = subparsers.add_parser("train", help="train and validate classical pipelines")
    train.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    train.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    train.add_argument("--run-name", default=None)
    train.add_argument("--no-feature-cache", action="store_true")
    train.add_argument(
        "--allow-expensive-full-svm",
        action="store_true",
        help="required safety acknowledgement for full RBF SVM variants",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = load_config(args.config)
    if args.command == "audit":
        audit = audit_phase2(config)
        print(json.dumps(asdict(audit), indent=2, sort_keys=True))
        return
    train_models(
        config,
        mode=args.mode,
        model_names=args.models,
        run_name=args.run_name or _default_run_name(args.mode),
        cache_features=not args.no_feature_cache,
        allow_expensive_full_svm=args.allow_expensive_full_svm,
    )


if __name__ == "__main__":
    main()

"""Train CropVision's PyTorch CNN using train/validation manifests only.

The test manifest is audited for partition integrity by Phase 2 checks, but this
module refuses to construct a test dataset. Test evaluation belongs to the
later unified final-evaluation phase.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data import ImageRecord
from src.preprocessing import read_manifest
from src.train_classical import audit_phase2, stratified_subsample
from src.utils import load_config, runtime_metadata, save_json


class ManifestRGBDataset:
    """Load normalized RGB tensors from one non-test manifest."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        dataset_dir: str | Path,
        class_names: Sequence[str],
        mean: Sequence[float],
        std: Sequence[float],
        *,
        split_name: str,
    ) -> None:
        if split_name == "test":
            raise ValueError("Phase 4 must not construct or load a test dataset")
        if split_name not in {"train", "validation"}:
            raise ValueError("split_name must be train or validation")
        if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
            raise ValueError("RGB normalization requires three positive standard deviations")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("PyTorch is required; see SETUP.md.") from exc
        self.records = list(records)
        self.dataset_dir = Path(dataset_dir)
        self.label_to_index = {label: index for index, label in enumerate(class_names)}
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self._mean_tensor = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        self._std_tensor = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        try:
            import numpy as np
            import torch
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("PyTorch, NumPy, and Pillow are required; see SETUP.md.") from exc

        record = self.records[index]
        with Image.open(self.dataset_dir / record.path) as image:
            if image.mode != "RGB" or image.size != (64, 64):
                raise ValueError(
                    f"Unexpected image {record.path}: mode={image.mode}, size={image.size}"
                )
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self._mean_tensor) / self._std_tensor, self.label_to_index[record.label]


def compute_training_channel_stats(
    records: Sequence[ImageRecord], dataset_dir: str | Path
) -> tuple[list[float], list[float]]:
    """Compute RGB mean/std from training identities only using float64 sums."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("NumPy and Pillow are required; see SETUP.md.") from exc

    if not records:
        raise ValueError("Cannot compute normalization from an empty training set")
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    root = Path(dataset_dir)
    for record in records:
        with Image.open(root / record.path) as image:
            if image.mode != "RGB" or image.size != (64, 64):
                raise ValueError(
                    f"Unexpected image {record.path}: mode={image.mode}, size={image.size}"
                )
            array = np.asarray(image, dtype=np.float64) / 255.0
        channel_sum += array.sum(axis=(0, 1))
        channel_squared_sum += (array * array).sum(axis=(0, 1))
        pixel_count += array.shape[0] * array.shape[1]
    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - mean * mean
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mean.tolist(), std.tolist()


def select_device() -> str:
    """Select Apple MPS when supported, otherwise CPU (never assume CUDA)."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required; see SETUP.md.") from exc
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_cnn_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic operations."""
    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch and NumPy are required; see SETUP.md.") from exc
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def estimate_cnn_memory_gb(batch_size: int, parameter_count: int) -> dict[str, float]:
    """Return conservative float32 training-memory estimates for this architecture."""
    if batch_size <= 0 or parameter_count <= 0:
        raise ValueError("batch_size and parameter_count must be positive")
    # Parameters + gradients + two Adam moment tensors.
    model_and_optimizer = parameter_count * 4 * 4
    # Conservative activation/gradient estimate across three convolution blocks.
    activation_elements = batch_size * (
        32 * 64 * 64 + 64 * 32 * 32 + 128 * 16 * 16 + 128 * 8 * 8
    )
    activations = activation_elements * 4 * 2
    # Allow for backend workspaces, allocator fragmentation, dataloader batches,
    # and framework state that are not represented by the tensor count above.
    estimated = (model_and_optimizer + activations) * 2.5 + 0.25 * (1024**3)
    return {
        "model_optimizer_gb": model_and_optimizer / (1024**3),
        "activation_gb": activations / (1024**3),
        "estimated_training_gb": estimated / (1024**3),
    }


def _worker_seed(worker_id: int) -> None:
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(
    dataset: ManifestRGBDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> Any:
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=_worker_seed if num_workers else None,
        persistent_workers=num_workers > 0,
        pin_memory=False,
    )


def run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    device: str,
    optimizer: Any | None = None,
) -> tuple[dict[str, float], list[int], list[int]]:
    """Run one training or validation epoch and return aggregate metrics."""
    import torch
    from sklearn.metrics import f1_score

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(dim=1)
            batch_size = targets.size(0)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_correct += int((predictions == targets).sum().detach().cpu())
            total_examples += batch_size
            targets_all.extend(targets.detach().cpu().tolist())
            predictions_all.extend(predictions.detach().cpu().tolist())
    if not total_examples:
        raise ValueError("DataLoader produced no examples")
    metrics = {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "macro_f1": float(
            f1_score(targets_all, predictions_all, average="macro", zero_division=0)
        ),
    }
    return metrics, targets_all, predictions_all


def _save_history_csv(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
        "epoch_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in history)


def _plot_learning_curves(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for key, title, axis in (
        ("loss", "Cross-entropy loss", axes[0]),
        ("accuracy", "Accuracy", axes[1]),
        ("macro_f1", "Macro F1", axes[2]),
    ):
        axis.plot(epochs, [row[f"train_{key}"] for row in history], marker="o", label="Train")
        axis.plot(
            epochs,
            [row[f"validation_{key}"] for row in history],
            marker="o",
            label="Validation",
        )
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("CNN learning curves")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_confusion_matrix(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
    path: Path,
) -> Any:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    labels = list(range(len(class_names)))
    matrix = confusion_matrix(targets, predictions, labels=labels)
    figure, axis = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d")
    axis.set_title("CNN validation confusion matrix — best checkpoint")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return matrix


def _save_confusion_matrix_csv(
    matrix: Any, class_names: Sequence[str], path: Path
) -> None:
    """Save a labeled confusion matrix for later comparison and report export."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label/predicted_label", *class_names])
        for label, row in zip(class_names, matrix.tolist()):
            writer.writerow([label, *row])


def train_cnn(
    config: Mapping[str, Any],
    *,
    mode: str,
    run_name: str,
    allow_full_training: bool,
) -> Path:
    """Train the CNN and select a checkpoint using validation macro F1."""
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    if mode == "full" and not allow_full_training:
        raise RuntimeError(
            "Full CNN training is blocked until memory/runtime are reviewed. "
            "Rerun with --allow-full-training after explicit approval."
        )

    try:
        import torch
        from sklearn.metrics import classification_report
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch and scikit-learn are required; see SETUP.md.") from exc
    from src.cnn_model import SimpleEuroSATCNN, count_trainable_parameters

    audit = audit_phase2(config)
    seed = int(config["project"]["seed"])
    set_cnn_seeds(seed)
    cnn_config = config["cnn"]
    mode_config = cnn_config[mode]
    class_names = sorted(config["data"]["expected_class_counts"])
    split_dir = Path(config["data"]["split_dir"])
    dataset_dir = Path(config["data"]["dataset_dir"])

    train_records = stratified_subsample(
        read_manifest(split_dir / "train.csv"),
        mode_config["train_samples_per_class"],
        seed,
    )
    validation_records = stratified_subsample(
        read_manifest(split_dir / "validation.csv"),
        mode_config["validation_samples_per_class"],
        seed,
    )
    mean, std = compute_training_channel_stats(train_records, dataset_dir)
    train_dataset = ManifestRGBDataset(
        train_records,
        dataset_dir,
        class_names,
        mean,
        std,
        split_name="train",
    )
    validation_dataset = ManifestRGBDataset(
        validation_records,
        dataset_dir,
        class_names,
        mean,
        std,
        split_name="validation",
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size=int(mode_config["batch_size"]),
        shuffle=True,
        num_workers=int(mode_config["num_workers"]),
        seed=seed,
    )
    validation_loader = _make_loader(
        validation_dataset,
        batch_size=int(mode_config["batch_size"]),
        shuffle=False,
        num_workers=int(mode_config["num_workers"]),
        seed=seed,
    )

    architecture = cnn_config["architecture"]
    model = SimpleEuroSATCNN(
        num_classes=len(class_names),
        channels=tuple(int(value) for value in architecture["channels"]),
        hidden_units=int(architecture["hidden_units"]),
        dropout=float(architecture["dropout"]),
    )
    parameter_count = count_trainable_parameters(model)
    device = select_device()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cnn_config["optimizer"]["learning_rate"]),
        weight_decay=float(cnn_config["optimizer"]["weight_decay"]),
    )

    output_root = Path(cnn_config["output_dir"]) / run_name
    figure_root = Path(cnn_config["figure_dir"]) / run_name
    model_root = Path(cnn_config["model_dir"]) / run_name
    if output_root.exists() or figure_root.exists() or model_root.exists():
        raise FileExistsError(f"Run name already exists: {run_name}")
    output_root.mkdir(parents=True)
    figure_root.mkdir(parents=True)
    model_root.mkdir(parents=True)
    matplotlib_cache = Path("data/processed/matplotlib_cache").resolve()
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    memory_estimate = estimate_cnn_memory_gb(int(mode_config["batch_size"]), parameter_count)
    run_config = {
        "run_name": run_name,
        "mode": mode,
        "seed": seed,
        "device": device,
        "class_names": class_names,
        "parameter_count": parameter_count,
        "normalization": {"source": "selected_training_identities_only", "mean": mean, "std": std},
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "test_samples_loaded": 0,
        "selection_metric": "validation_macro_f1",
        "memory_estimate": memory_estimate,
        "source_config": config,
    }
    save_json(run_config, output_root / "run_config.json")
    save_json(asdict(audit), output_root / "split_audit.json")
    save_json(runtime_metadata(dict(config)), output_root / "runtime.json")

    checkpoint_path = model_root / "best_validation_checkpoint.pt"
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    patience = int(mode_config["early_stopping_patience"])
    training_started = time.perf_counter()

    for epoch in range(1, int(mode_config["epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_metrics, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        validation_metrics, _, _ = run_epoch(
            model, validation_loader, criterion, device, optimizer=None
        )
        epoch_seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        _save_history_csv(history, output_root / "history.csv")
        save_json(history, output_root / "history.json")
        print(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={validation_metrics['loss']:.4f}, "
            f"val_accuracy={validation_metrics['accuracy']:.4f}, "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f}, "
            f"seconds={epoch_seconds:.2f}"
        )

        improved = validation_metrics["macro_f1"] > best_f1 + 1e-12 or (
            abs(validation_metrics["macro_f1"] - best_f1) <= 1e-12
            and validation_metrics["loss"] < best_loss
        )
        if improved:
            best_f1 = validation_metrics["macro_f1"]
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "validation_metrics": validation_metrics,
                "class_names": class_names,
                "normalization": {"mean": mean, "std": std},
                "architecture": architecture,
                "parameter_count": parameter_count,
                "seed": seed,
            }
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break
        if device == "mps":
            torch.mps.empty_cache()

    training_seconds = time.perf_counter() - training_started
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_validation_metrics, targets, predictions = run_epoch(
        model, validation_loader, criterion, device, optimizer=None
    )
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    _plot_learning_curves(history, figure_root / "learning_curves.png")
    confusion = _plot_confusion_matrix(
        targets,
        predictions,
        class_names,
        figure_root / "validation_confusion_matrix.png",
    )
    _save_confusion_matrix_csv(
        confusion, class_names, output_root / "validation_confusion_matrix.csv"
    )
    summary = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_seconds": training_seconds,
        "best_validation": best_validation_metrics,
        "per_class": report,
        "checkpoint": str(checkpoint_path),
        "device": device,
        "parameter_count": parameter_count,
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "test_samples_loaded": 0,
    }
    save_json(summary, output_root / "validation_summary.json")
    print(f"Completed {mode} CNN validation run: {run_name}")
    print(f"Best epoch: {best_epoch}; device: {device}; parameters: {parameter_count:,}")
    print("Test images loaded: 0 (reserved for final evaluation)")
    return output_root


def _default_run_name(mode: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{mode}_{timestamp}"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="verify Phase 2 without loading image tensors")
    train = subparsers.add_parser("train", help="train and validate the CNN")
    train.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    train.add_argument("--run-name", default=None)
    train.add_argument(
        "--allow-full-training",
        action="store_true",
        help="required acknowledgement after checking full-run memory and runtime",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = load_config(args.config)
    if args.command == "audit":
        print(json.dumps(asdict(audit_phase2(config)), indent=2, sort_keys=True))
        return
    train_cnn(
        config,
        mode=args.mode,
        run_name=args.run_name or _default_run_name(args.mode),
        allow_full_training=args.allow_full_training,
    )


if __name__ == "__main__":
    main()

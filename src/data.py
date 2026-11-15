"""Download, validate, and split the EuroSAT RGB dataset.

The test set manifest is created once with the other splits, but downstream
model-selection code must not read it until final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from src.utils import load_config, save_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    """One labeled image, identified by a path relative to the dataset root."""

    path: str
    label: str


@dataclass(frozen=True)
class ValidationReport:
    """Summary produced by :func:`validate_dataset`."""

    dataset_dir: str
    total_images: int
    class_counts: dict[str, int]
    images_inspected: int
    expected_size: tuple[int, int]
    expected_mode: str


def md5sum(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate an MD5 checksum for archive integrity verification."""
    digest = hashlib.md5()  # noqa: S324 - required to match the published checksum
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    """Stream a URL to disk, printing coarse progress to stderr."""
    request = urllib.request.Request(url, headers={"User-Agent": "CropVision/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open(
            "wb"
        ) as output:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                received += len(chunk)
                if total:
                    print(
                        f"\rDownloaded {received / 1_048_576:.1f} / "
                        f"{total / 1_048_576:.1f} MiB",
                        end="",
                        file=sys.stderr,
                    )
            if total:
                print(file=sys.stderr)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"EuroSAT download failed from {url}: {exc}\n"
            "Manual fallback: download EuroSAT_RGB.zip from "
            "https://zenodo.org/records/7711810, place it in data/raw/, then run "
            "`python -m src.data download --archive data/raw/EuroSAT_RGB.zip`."
        ) from exc


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract a ZIP archive while rejecting path traversal and symbolic links."""
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
            unix_mode = member.external_attr >> 16
            if unix_mode & 0o170000 == 0o120000:
                raise ValueError(f"Symbolic link not allowed in archive: {member.filename}")
            target = (destination / Path(*member_path.parts)).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Archive entry escapes destination: {member.filename}")
        archive.extractall(destination)


def _find_dataset_root(extracted_dir: Path, expected_classes: set[str]) -> Path:
    candidates = [extracted_dir, *sorted(extracted_dir.rglob("*"))]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        child_dirs = {path.name for path in candidate.iterdir() if path.is_dir()}
        if child_dirs == expected_classes:
            return candidate
    raise ValueError(
        "Could not find a directory containing exactly the 10 expected EuroSAT "
        f"class folders under {extracted_dir}."
    )


def download_eurosat_rgb(
    *,
    url: str,
    expected_md5: str,
    raw_dir: str | Path,
    expected_classes: Iterable[str],
    archive: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download/check/extract the published EuroSAT RGB ZIP archive.

    The extracted dataset is moved into ``raw_dir/EuroSAT_RGB`` only after the
    archive checksum and folder layout pass verification.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    dataset_path = raw_path / "EuroSAT_RGB"
    if dataset_path.exists() and not force:
        raise FileExistsError(
            f"Dataset directory already exists: {dataset_path}. Validate it or use --force."
        )

    archive_path = Path(archive) if archive else raw_path / "EuroSAT_RGB.zip"
    downloaded_here = archive is None
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    if downloaded_here:
        if partial_path.exists():
            partial_path.unlink()
        _download(url, partial_path)
        partial_path.replace(archive_path)
    elif not archive_path.is_file():
        raise FileNotFoundError(f"Manual archive not found: {archive_path}")

    actual_md5 = md5sum(archive_path)
    if actual_md5.lower() != expected_md5.lower():
        raise ValueError(
            f"Archive checksum mismatch for {archive_path}: expected {expected_md5}, "
            f"got {actual_md5}. Delete or replace the archive before retrying."
        )

    with tempfile.TemporaryDirectory(prefix="eurosat-extract-", dir=raw_path) as temp:
        extracted = Path(temp)
        _safe_extract(archive_path, extracted)
        found_root = _find_dataset_root(extracted, set(expected_classes))
        if dataset_path.exists():
            if not force:
                raise FileExistsError(f"Refusing to replace {dataset_path}")
            shutil.rmtree(dataset_path)
        shutil.move(str(found_root), str(dataset_path))

    print(f"EuroSAT RGB extracted to {dataset_path}")
    return dataset_path


def discover_images(dataset_dir: str | Path) -> list[ImageRecord]:
    """Discover images in ``dataset_dir/<class>/`` with stable ordering."""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    records: list[ImageRecord] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append(
                    ImageRecord(path=image_path.relative_to(root).as_posix(), label=class_dir.name)
                )
    return records


def validate_dataset(
    dataset_dir: str | Path,
    expected_counts: Mapping[str, int],
    expected_size: Sequence[int] = (64, 64),
    expected_mode: str = "RGB",
    sample_images: int | None = None,
) -> ValidationReport:
    """Validate class names/counts and decode all images (or a deterministic sample)."""
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Pillow is required for image validation; see SETUP.md.") from exc

    root = Path(dataset_dir)
    records = discover_images(root)
    actual_counts = Counter(record.label for record in records)
    expected = dict(expected_counts)
    if dict(sorted(actual_counts.items())) != dict(sorted(expected.items())):
        raise ValueError(
            "EuroSAT class/count validation failed. "
            f"Expected {expected}; found {dict(sorted(actual_counts.items()))}."
        )

    selected = records
    if sample_images is not None:
        if sample_images <= 0:
            raise ValueError("sample_images must be positive")
        rng = random.Random(42)
        selected = rng.sample(records, min(sample_images, len(records)))

    required_size = tuple(int(value) for value in expected_size)
    problems: list[str] = []
    for record in selected:
        try:
            with Image.open(root / record.path) as image:
                image.load()
                if image.size != required_size or image.mode != expected_mode:
                    problems.append(
                        f"{record.path}: size={image.size}, mode={image.mode}"
                    )
        except (OSError, UnidentifiedImageError) as exc:
            problems.append(f"{record.path}: cannot decode ({exc})")
        if len(problems) >= 20:
            break
    if problems:
        raise ValueError(
            f"Image validation failed (showing up to 20):\n" + "\n".join(problems)
        )

    return ValidationReport(
        dataset_dir=str(root),
        total_images=len(records),
        class_counts=dict(sorted(actual_counts.items())),
        images_inspected=len(selected),
        expected_size=required_size,
        expected_mode=expected_mode,
    )


def stratified_split(
    records: Sequence[ImageRecord],
    fractions: Mapping[str, float],
    seed: int = 42,
) -> dict[str, list[ImageRecord]]:
    """Split every class deterministically using identical image identities."""
    if not records:
        raise ValueError("Cannot split an empty dataset")
    required_names = {"train", "validation", "test"}
    if set(fractions) != required_names:
        raise ValueError(f"Split names must be exactly {sorted(required_names)}")
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("All split fractions must be positive")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    by_class: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_class.setdefault(record.label, []).append(record)

    output = {name: [] for name in ("train", "validation", "test")}
    for class_index, label in enumerate(sorted(by_class)):
        items = sorted(by_class[label], key=lambda record: record.path)
        random.Random(seed + class_index).shuffle(items)
        train_end = round(len(items) * fractions["train"])
        validation_end = train_end + round(len(items) * fractions["validation"])
        output["train"].extend(items[:train_end])
        output["validation"].extend(items[train_end:validation_end])
        output["test"].extend(items[validation_end:])

    for name, items in output.items():
        random.Random(seed).shuffle(items)
    return output


def write_split_manifests(
    splits: Mapping[str, Sequence[ImageRecord]], output_dir: str | Path
) -> dict[str, Path]:
    """Write one immutable-identity CSV manifest per split plus a summary JSON."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    summary: dict[str, object] = {"splits": {}}
    seen: set[str] = set()
    for split_name in ("train", "validation", "test"):
        items = list(splits[split_name])
        duplicates = seen.intersection(record.path for record in items)
        if duplicates:
            raise ValueError(f"Images occur in multiple splits: {sorted(duplicates)[:5]}")
        seen.update(record.path for record in items)
        manifest_path = destination / f"{split_name}.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(asdict(record) for record in items)
        paths[split_name] = manifest_path
        summary["splits"][split_name] = {
            "count": len(items),
            "class_counts": dict(sorted(Counter(item.label for item in items).items())),
        }
    summary["total_unique_images"] = len(seen)
    save_json(summary, destination / "summary.json")
    return paths


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download and extract EuroSAT RGB")
    download.add_argument("--archive", help="use an already downloaded ZIP instead")
    download.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate", help="validate an extracted dataset")
    validate.add_argument("--sample-images", type=int, default=None)

    split = subparsers.add_parser("split", help="create deterministic CSV manifests")
    split.add_argument("--force", action="store_true", help="replace existing manifests")

    prepare = subparsers.add_parser("prepare", help="validate and split an existing dataset")
    prepare.add_argument("--sample-images", type=int, default=None)
    prepare.add_argument("--force", action="store_true")
    return parser.parse_args()


def _ensure_new_manifests(output_dir: Path, force: bool) -> None:
    existing = [output_dir / f"{name}.csv" for name in ("train", "validation", "test")]
    if not force and any(path.exists() for path in existing):
        raise FileExistsError(
            f"Split manifests already exist in {output_dir}; use --force to regenerate them."
        )


def main() -> None:
    args = _arguments()
    config = load_config(args.config)
    data_config = config["data"]
    dataset_dir = Path(data_config["dataset_dir"])
    expected_counts = data_config["expected_class_counts"]

    if args.command == "download":
        download_eurosat_rgb(
            url=data_config["source_url"],
            expected_md5=data_config["archive_md5"],
            raw_dir=data_config["raw_dir"],
            expected_classes=expected_counts,
            archive=args.archive,
            force=args.force,
        )
        return

    if args.command in {"validate", "prepare"}:
        report = validate_dataset(
            dataset_dir,
            expected_counts,
            data_config["expected_image_size"],
            data_config["expected_mode"],
            args.sample_images,
        )
        print(
            f"Validated {report.total_images} images in {len(report.class_counts)} classes; "
            f"decoded {report.images_inspected} image(s)."
        )
        save_json(asdict(report), Path(data_config["split_dir"]) / "validation.json")
        if args.command == "validate":
            return

    output_dir = Path(data_config["split_dir"])
    _ensure_new_manifests(output_dir, args.force)
    records = discover_images(dataset_dir)
    splits = stratified_split(records, config["split"], int(config["project"]["seed"]))
    paths = write_split_manifests(splits, output_dir)
    print("Wrote split manifests:")
    for name, path in paths.items():
        print(f"  {name}: {len(splits[name])} images -> {path}")


if __name__ == "__main__":
    main()

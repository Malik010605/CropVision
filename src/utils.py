"""Shared configuration, reproducibility, and runtime metadata helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load a YAML configuration file and return a dictionary."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "PyYAML is required. Create the environment described in SETUP.md first."
        ) from exc

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return config


def set_random_seeds(seed: int) -> None:
    """Seed Python and, when installed, NumPy and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            # MPS uses the same PyTorch RNG seed; deterministic algorithms may
            # reject otherwise valid operations, so we record rather than force it.
            pass
    except ImportError:
        pass


def select_torch_device() -> str:
    """Return ``mps`` on supported Apple hardware, otherwise ``cpu``."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("PyTorch is not installed; see SETUP.md.") from exc
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def runtime_metadata(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect versions and hardware details needed to reproduce an experiment."""
    packages = [
        "numpy",
        "pandas",
        "Pillow",
        "PyYAML",
        "scikit-learn",
        "matplotlib",
        "torch",
        "torchvision",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"

    details: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "git_commit": _git_commit(),
    }
    if config is not None:
        details["config"] = config
    try:
        details["torch_device"] = select_torch_device()
    except RuntimeError:
        details["torch_device"] = "unavailable"
    return details


def save_json(payload: Any, path: str | Path) -> None:
    """Write JSON atomically enough for small experiment metadata files."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(output_path)

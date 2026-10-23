#!/usr/bin/env python3
"""Reproducible run identities and atomic artifacts for thesis experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


RUN_CONTRACT_SCHEMA_VERSION = 2
SCIENTIFIC_CODE_PATTERNS = (
    "src/*.py",
    "experiments/*.py",
    "run_cross_validation.py",
    "run_lambda_sweep.py",
    "run_grouped_v3_battery.py",
    "recover_grouped_v3_fold_bundle.py",
    "build_geometric_mask_control.py",
    "analyze_cross_validation.py",
    "requirements.txt",
)


class ResumeValidationError(RuntimeError):
    """Raised when an existing result cannot safely be reused."""


def canonical_json(payload: Any) -> str:
    """Serialize a payload in the unique form used by every fingerprint."""

    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def sha256_bytes(content: bytes) -> str:
    """Return a prefixed SHA-256 digest."""

    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def sha256_payload(payload: Any) -> str:
    """Hash a JSON-compatible payload canonically."""

    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


@lru_cache(maxsize=None)
def _sha256_file_snapshot(path: str, size: int, mtime_ns: int) -> str:
    """Cache immutable-by-stat dataset assets across the 160 planned runs."""

    del size, mtime_ns
    return sha256_file(path)


def _repo_relative(path: str | Path, project_root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return f"repo://{candidate.resolve().relative_to(project_root.resolve()).as_posix()}"
        except ValueError:
            return candidate.as_posix()
    return candidate.as_posix()


def normalize_for_fingerprint(value: Any, project_root: Path) -> Any:
    """Convert config values to stable JSON values across cloned locations."""

    if isinstance(value, Path):
        return _repo_relative(value, project_root)
    if isinstance(value, dict):
        return {
            str(key): normalize_for_fingerprint(item, project_root)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_for_fingerprint(item, project_root) for item in value]
    if isinstance(value, set):
        return sorted(normalize_for_fingerprint(item, project_root) for item in value)
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            return _repo_relative(candidate, project_root)
        return value.replace("\\", "/")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _scientific_config(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    excluded = {
        "output_dir",
        "train_indices",
        "val_indices",
        "test_indices",
        "run_identity",
        "run_fingerprint",
    }
    return normalize_for_fingerprint(
        {key: value for key, value in config.items() if key not in excluded},
        project_root,
    )


def _sample_id(dataset: Any, index: int) -> str:
    return str(dataset.rows[index].get("sample_id", f"dataset_index:{index}"))


def _dataset_fingerprints(dataset: Any, project_root: Path) -> dict[str, Any]:
    try:
        from grouped_splits import group_id_for_row
    except ImportError:  # pragma: no cover - only for isolated module reuse
        group_id_for_row = None

    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset.rows):
        group_id = None
        if group_id_for_row is not None:
            try:
                group_id = group_id_for_row(row, strict=False)
            except (TypeError, ValueError):
                group_id = None
        records.append(
            {
                "dataset_index": index,
                "sample_id": _sample_id(dataset, index),
                "group_id": group_id,
                "class_name": str(row.get(dataset.class_column, "")),
                "source": str(row.get("source", "")),
                "image_path": normalize_for_fingerprint(row.get(dataset.image_column, ""), project_root),
                "mask_path": normalize_for_fingerprint(row.get(dataset.mask_column, ""), project_root),
            }
        )

    manifests: list[dict[str, Any]] = []
    for manifest in dataset.manifest_csv:
        path = Path(manifest)
        manifests.append(
            {
                "path": _repo_relative(path, project_root),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )

    asset_paths: set[Path] = set()
    for row in dataset.rows:
        for column in (dataset.image_column, dataset.mask_column):
            raw_path = row.get(column, "")
            if hasattr(dataset, "_resolve_path"):
                resolved = Path(dataset._resolve_path(raw_path))
            else:
                candidate = Path(raw_path)
                resolved = candidate if candidate.is_absolute() else project_root / candidate
            asset_paths.add(resolved)
    assets: list[dict[str, Any]] = []
    for path in sorted(asset_paths):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset asset referenced by a manifest does not exist: {path}")
        stat = path.stat()
        assets.append(
            {
                "path": _repo_relative(path, project_root),
                "size": stat.st_size,
                "sha256": _sha256_file_snapshot(str(path.resolve()), stat.st_size, stat.st_mtime_ns),
            }
        )

    membership_records = sorted(records, key=lambda item: (item["sample_id"], item["dataset_index"]))
    return {
        "sample_count": len(records),
        "manifests": manifests,
        "asset_count": len(assets),
        "asset_content_sha256": sha256_payload(assets),
        "ordered_dataset_sha256": sha256_payload(records),
        "membership_sha256": sha256_payload(membership_records),
    }


def _split_fingerprints(
    dataset: Any,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
) -> dict[str, Any]:
    def describe(indices: list[int]) -> dict[str, Any]:
        sample_ids = [_sample_id(dataset, index) for index in indices]
        return {
            "size": len(indices),
            "sample_ids_sha256": sha256_payload(sample_ids),
            "membership_sha256": sha256_payload(sorted(sample_ids)),
        }

    return {
        "train": describe(train_indices),
        "validation": describe(val_indices),
        "test": describe(test_indices),
    }


def hash_code_tree(project_root: str | Path) -> dict[str, Any]:
    """Hash all source files that can change the scientific result."""

    root = Path(project_root)
    paths: set[Path] = set()
    for pattern in SCIENTIFIC_CODE_PATTERNS:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(paths)
    ]
    return {"tree_sha256": sha256_payload(files), "files": files}


def capture_git_provenance(project_root: str | Path) -> dict[str, Any]:
    """Capture commit metadata without making Git a fingerprint dependency."""

    root = Path(project_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def capture_environment() -> dict[str, Any]:
    """Record the execution environment as provenance (not resume identity)."""

    versions: dict[str, str | None] = {}
    for package in ("torch", "torchvision", "numpy", "scipy", "PIL", "timm"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


def build_run_identity(
    experiment_name: str,
    config: dict[str, Any],
    dataset: Any,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    project_root: str | Path,
) -> dict[str, Any]:
    """Build the exact identity used by strict ``--resume`` validation."""

    root = Path(project_root)
    components = {
        "effective_config": _scientific_config(config, root),
        "dataset": _dataset_fingerprints(dataset, root),
        "split": _split_fingerprints(dataset, train_indices, val_indices, test_indices),
        "code": hash_code_tree(root),
    }
    component_fingerprints = {
        name: sha256_payload(payload) for name, payload in components.items()
    }
    run_payload = {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "experiment": experiment_name,
        "component_fingerprints": component_fingerprints,
    }
    return {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "experiment": experiment_name,
        "run_fingerprint": sha256_payload(run_payload),
        "component_fingerprints": component_fingerprints,
        "components": components,
        "git": capture_git_provenance(root),
        "environment": capture_environment(),
    }


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Publish a JSON artifact atomically on the destination filesystem."""

    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(
    path: str | Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    """Publish a CSV artifact atomically."""

    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    """Publish a PyTorch checkpoint atomically."""

    import torch

    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_record(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    return {"sha256": sha256_file(artifact), "size": artifact.stat().st_size}


def write_success_marker(
    output_dir: str | Path,
    run_fingerprint: str,
    artifact_names: Iterable[str],
    best_epoch: int,
) -> Path:
    """Write the final marker only after every required artifact exists."""

    root = Path(output_dir)
    artifacts = {name: artifact_record(root / name) for name in artifact_names}
    path = root / "_SUCCESS.json"
    atomic_write_json(
        path,
        {
            "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
            "state": "complete",
            "run_fingerprint": run_fingerprint,
            "best_epoch": best_epoch,
            "artifacts": artifacts,
        },
    )
    return path


def legacy_metrics_complete(output_dir: str | Path) -> bool:
    """Recognize historical results only when explicitly requested."""

    path = Path(output_dir) / "metrics.json"
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    test = payload.get("test")
    return isinstance(test, dict) and "f1_macro" in test and "confusion_matrix" in test


def validate_completed_run(
    output_dir: str | Path,
    expected_fingerprint: str,
    allow_legacy: bool = False,
) -> str:
    """Return ``complete``/``missing`` or raise for unsafe existing artifacts."""

    root = Path(output_dir)
    if not root.exists() or not any(root.iterdir()):
        return "missing"

    metrics_path = root / "metrics.json"
    marker_path = root / "_SUCCESS.json"
    if not marker_path.exists():
        if allow_legacy and legacy_metrics_complete(root):
            return "legacy_complete"
        if not metrics_path.exists():
            manifest_path = root / "run_manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, encoding="utf-8") as handle:
                        partial_manifest = json.load(handle)
                except (OSError, json.JSONDecodeError) as error:
                    raise ResumeValidationError(
                        f"CORRUPT_ARTIFACT: invalid partial run manifest in {root}"
                    ) from error
                if not isinstance(partial_manifest, dict):
                    raise ResumeValidationError(f"CORRUPT_ARTIFACT: partial run manifest is not an object in {root}")
                if partial_manifest.get("run_fingerprint") != expected_fingerprint:
                    raise ResumeValidationError(
                        f"CONFIG_MISMATCH: partial run manifest in {root} belongs to another run."
                    )
            return "incomplete"
        raise ResumeValidationError(
            f"INCOMPLETE: {root} has metrics but no _SUCCESS.json; use --rerun-incomplete "
            "or preserve/move the directory before rerunning."
        )

    try:
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        with open(marker_path, encoding="utf-8") as handle:
            marker = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeValidationError(f"CORRUPT_ARTIFACT: could not read {root}: {error}") from error
    if not isinstance(metrics, dict) or not isinstance(marker, dict):
        raise ResumeValidationError(f"CORRUPT_ARTIFACT: metrics/marker must be JSON objects in {root}")

    fingerprints = {metrics.get("run_fingerprint"), marker.get("run_fingerprint")}
    if fingerprints != {expected_fingerprint}:
        raise ResumeValidationError(
            f"CONFIG_MISMATCH: existing={sorted(str(item) for item in fingerprints)} "
            f"expected={expected_fingerprint} path={root}"
        )

    if marker.get("state") != "complete":
        raise ResumeValidationError(f"INCOMPLETE: completion marker is not complete in {marker_path}")

    required = ("run_manifest.json", "metrics.json", "best_model.pt", "test_predictions.csv")
    records = marker.get("artifacts")
    if not isinstance(records, dict):
        raise ResumeValidationError(f"CORRUPT_ARTIFACT: invalid marker in {marker_path}")
    for name in required:
        path = root / name
        record = records.get(name)
        if not path.exists() or not isinstance(record, dict):
            raise ResumeValidationError(f"INCOMPLETE: missing {name} or its hash in {root}")
        if record.get("sha256") != sha256_file(path) or record.get("size") != path.stat().st_size:
            raise ResumeValidationError(f"CORRUPT_ARTIFACT: hash/size mismatch for {path}")

    test = metrics.get("test")
    if not isinstance(test, dict) or "f1_macro" not in test or "confusion_matrix" not in test:
        raise ResumeValidationError(f"CORRUPT_METRICS: incomplete test report in {metrics_path}")
    try:
        with open(root / "run_manifest.json", encoding="utf-8") as handle:
            run_manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeValidationError(f"CORRUPT_ARTIFACT: invalid run_manifest.json in {root}") from error
    if not isinstance(run_manifest, dict):
        raise ResumeValidationError(f"CORRUPT_ARTIFACT: run_manifest.json must be an object in {root}")
    if run_manifest.get("state") != "complete" or run_manifest.get("run_fingerprint") != expected_fingerprint:
        raise ResumeValidationError(f"CONFIG_MISMATCH: invalid run manifest state/fingerprint in {root}")
    return "complete"

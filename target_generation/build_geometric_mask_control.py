#!/usr/bin/env python3
"""Build a deterministic geometric control for the anatomical shrink masks.

The confirmatory control replaces each ``anatomico_shrink`` target by one
filled, axis-aligned rectangle.  The rectangle is centred on the reference
foreground and its integer width/height are selected to approximate both the
reference foreground area and bounding-box aspect ratio.  Images and source
masks are never modified; all generated masks and manifests are written to a
new output root.

No random number generator is used.  The output contract records the input
and output hashes required to reproduce and audit the generated dataset.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from grouped_splits import group_id_for_row  # noqa: E402
from run_contract import atomic_write_csv, atomic_write_json, sha256_file, sha256_payload  # noqa: E402


SCHEMA_VERSION = 1
CONTROL_METHOD = "rectangle_area_matched_v1"
REFERENCE_VARIANT = "anatomico_shrink"
MAX_AREA_RELATIVE_ERROR = 0.10
DEFAULT_MANIFESTS = (
    ROOT
    / "dataset_mascaras_anatomicas_shrink_preview"
    / "manifests"
    / "real_shrink_preview.csv",
    ROOT
    / "dataset_mascaras_anatomicas_shrink_preview"
    / "manifests"
    / "synthetic_fine_shrink_preview.csv",
)
DEFAULT_OUTPUT_ROOT = ROOT / "dataset_mascaras_geometricas_rectangle_matched"

MANIFEST_FIELDS = [
    "sample_id",
    "group_id",
    "class_name",
    "image_path",
    "mask_path",
    "source",
    "reference_mask_path",
    "reference_refinement_method",
    "geometric_control_method",
    "reference_mask_sha256",
    "generated_mask_sha256",
    "reference_area_px",
    "generated_area_px",
    "area_delta_px",
    "area_relative_error",
    "reference_centroid_x",
    "reference_centroid_y",
    "generated_centroid_x",
    "generated_centroid_y",
    "centroid_distance_px",
    "reference_bbox_width",
    "reference_bbox_height",
    "reference_bbox_aspect_ratio",
    "generated_bbox_width",
    "generated_bbox_height",
    "generated_bbox_aspect_ratio",
    "aspect_log_error",
]


def parse_args() -> argparse.Namespace:
    """Read command-line options without starting any model training."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic area-matched rectangular masks from the "
            "anatomico_shrink reference masks."
        )
    )
    parser.add_argument(
        "--manifest",
        nargs="+",
        default=[str(path) for path in DEFAULT_MANIFESTS],
        help="anatomico_shrink manifests used as immutable references.",
    )
    parser.add_argument(
        "--data-root",
        default=str(ROOT),
        help="Root used to resolve relative image and mask paths.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="New destination. A non-empty destination is always rejected.",
    )
    return parser.parse_args()


def resolve_path(data_root: Path, value: str | Path) -> Path:
    """Resolve an absolute path as-is and a relative path from ``data_root``."""

    path = Path(value)
    return path if path.is_absolute() else data_root / path


def portable_path(path: Path) -> str:
    """Use a repository-relative path when possible, otherwise an absolute one."""

    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _required_text(row: dict[str, str], column: str, row_index: int) -> str:
    value = str(row.get(column, "")).strip()
    if not value:
        raise ValueError(f"Missing {column!r} in manifest row {row_index}.")
    return value


def read_reference_rows(manifest_paths: Iterable[Path]) -> list[dict[str, str]]:
    """Read, validate and materialize group identities from shrink manifests."""

    rows: list[dict[str, str]] = []
    seen_sample_ids: set[str] = set()
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Reference manifest does not exist: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            manifest_rows = list(csv.DictReader(handle))
        if not manifest_rows:
            raise ValueError(f"Reference manifest is empty: {manifest_path}")
        for local_index, original in enumerate(manifest_rows, start=2):
            row = dict(original)
            sample_id = _required_text(row, "sample_id", local_index)
            _required_text(row, "class_name", local_index)
            _required_text(row, "image_path", local_index)
            _required_text(row, "mask_path", local_index)
            _required_text(row, "source", local_index)
            refinement = _required_text(row, "refinement_method", local_index)
            if not refinement.startswith("distance_shrink_"):
                raise ValueError(
                    f"Reference row {sample_id!r} is not anatomico_shrink: "
                    f"refinement_method={refinement!r}."
                )
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id across reference manifests: {sample_id}")
            seen_sample_ids.add(sample_id)
            row["group_id"] = str(
                group_id_for_row(row, strict=True, row_index=len(rows))
            )
            row["_manifest_path"] = str(manifest_path)
            rows.append(row)
    return rows


def load_binary_mask(path: Path) -> np.ndarray:
    """Load a non-empty two-dimensional binary reference mask."""

    if not path.is_file():
        raise FileNotFoundError(f"Reference mask does not exist: {path}")
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 127
    if mask.ndim != 2:
        raise ValueError(f"Reference mask must be two-dimensional: {path}")
    if not bool(mask.any()):
        raise ValueError(f"Reference mask is empty: {path}")
    return mask


def mask_geometry(mask: np.ndarray) -> dict[str, float | int]:
    """Return foreground area, centroid and axis-aligned bounding-box geometry."""

    if mask.ndim != 2 or not bool(mask.any()):
        raise ValueError("mask_geometry requires a non-empty two-dimensional mask")
    ys, xs = np.where(mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    return {
        "area_px": int(mask.sum()),
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_aspect_ratio": float(bbox_width / bbox_height),
    }


def _nearby_integers(value: float) -> set[int]:
    """Return deterministic integer candidates around a positive real value."""

    return {
        max(1, int(math.floor(value))),
        max(1, int(math.ceil(value))),
        max(1, int(math.floor(value + 0.5))),
    }


def rectangle_dimensions(
    area_px: int,
    aspect_ratio: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    """Choose integer rectangle dimensions near the requested area and aspect.

    Raster rectangles cannot match every foreground area exactly.  Candidates
    are therefore ranked by the sum of relative area error and logarithmic
    aspect error, followed by deterministic tie-breakers.  The continuous
    area/aspect solution is always included in the candidate neighbourhood.
    """

    if area_px <= 0:
        raise ValueError("area_px must be positive")
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be finite and positive")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    ideal_width = math.sqrt(area_px * aspect_ratio)
    ideal_height = math.sqrt(area_px / aspect_ratio)
    candidates: set[tuple[int, int]] = set()

    def add(width: int, height: int) -> None:
        if 1 <= width <= image_width and 1 <= height <= image_height:
            candidates.add((width, height))

    ideal_widths = _nearby_integers(ideal_width)
    ideal_heights = _nearby_integers(ideal_height)
    for width in range(1, image_width + 1):
        for height in _nearby_integers(area_px / width).union(ideal_heights):
            add(width, height)
    for height in range(1, image_height + 1):
        for width in _nearby_integers(area_px / height).union(ideal_widths):
            add(width, height)

    if not candidates:  # pragma: no cover - guarded by positive dimensions
        raise RuntimeError("No valid raster rectangle dimensions were found")

    def score(
        candidate: tuple[int, int],
    ) -> tuple[bool, float, float, float, float, int, int]:
        width, height = candidate
        area_error = abs(width * height - area_px) / area_px
        aspect_error = abs(math.log((width / height) / aspect_ratio))
        ideal_distance = abs(width - ideal_width) + abs(height - ideal_height)
        return (
            area_error > MAX_AREA_RELATIVE_ERROR,
            area_error + aspect_error,
            area_error,
            aspect_error,
            ideal_distance,
            height,
            width,
        )

    return min(candidates, key=score)


def rectangle_area_matched(mask: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Create the deterministic area/location/aspect-matched rectangle control."""

    reference = mask_geometry(mask)
    image_height, image_width = mask.shape
    rectangle_width, rectangle_height = rectangle_dimensions(
        int(reference["area_px"]),
        float(reference["bbox_aspect_ratio"]),
        image_width,
        image_height,
    )

    # Half-up rounding avoids Python's even/odd banker rounding at .5 while
    # remaining completely deterministic for the non-negative coordinates.
    x_start = int(
        math.floor(
            float(reference["centroid_x"]) - (rectangle_width - 1) / 2 + 0.5
        )
    )
    y_start = int(
        math.floor(
            float(reference["centroid_y"]) - (rectangle_height - 1) / 2 + 0.5
        )
    )
    x_start = min(max(x_start, 0), image_width - rectangle_width)
    y_start = min(max(y_start, 0), image_height - rectangle_height)

    generated = np.zeros(mask.shape, dtype=bool)
    generated[
        y_start : y_start + rectangle_height,
        x_start : x_start + rectangle_width,
    ] = True
    if not bool(generated.any()):  # pragma: no cover - dimensions are positive
        raise RuntimeError("Generated geometric control mask is unexpectedly empty")

    output = mask_geometry(generated)
    area_delta = int(output["area_px"]) - int(reference["area_px"])
    area_relative_error = abs(area_delta) / int(reference["area_px"])
    if area_relative_error > MAX_AREA_RELATIVE_ERROR:
        raise ValueError(
            "Could not rasterize an area-matched rectangle within the allowed "
            f"relative error: observed={area_relative_error:.6f}, "
            f"maximum={MAX_AREA_RELATIVE_ERROR:.6f}."
        )
    centroid_distance = math.hypot(
        float(output["centroid_x"]) - float(reference["centroid_x"]),
        float(output["centroid_y"]) - float(reference["centroid_y"]),
    )
    aspect_log_error = abs(
        math.log(
            float(output["bbox_aspect_ratio"])
            / float(reference["bbox_aspect_ratio"])
        )
    )
    stats: dict[str, float | int] = {
        "reference_area_px": int(reference["area_px"]),
        "generated_area_px": int(output["area_px"]),
        "area_delta_px": area_delta,
        "area_relative_error": float(area_relative_error),
        "reference_centroid_x": float(reference["centroid_x"]),
        "reference_centroid_y": float(reference["centroid_y"]),
        "generated_centroid_x": float(output["centroid_x"]),
        "generated_centroid_y": float(output["centroid_y"]),
        "centroid_distance_px": float(centroid_distance),
        "reference_bbox_width": int(reference["bbox_width"]),
        "reference_bbox_height": int(reference["bbox_height"]),
        "reference_bbox_aspect_ratio": float(reference["bbox_aspect_ratio"]),
        "generated_bbox_width": int(output["bbox_width"]),
        "generated_bbox_height": int(output["bbox_height"]),
        "generated_bbox_aspect_ratio": float(output["bbox_aspect_ratio"]),
        "aspect_log_error": float(aspect_log_error),
    }
    return generated, stats


def atomic_save_binary_png(mask: np.ndarray, path: Path) -> None:
    """Publish one strictly binary PNG atomically."""

    if mask.ndim != 2 or mask.dtype != np.bool_ or not bool(mask.any()):
        raise ValueError("Generated mask must be a non-empty two-dimensional boolean array")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            temporary,
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def output_mask_path(output_root: Path, row: dict[str, str]) -> Path:
    """Return a source-separated output path and preserve the reference name."""

    source = row["source"]
    folder = "synthetic_fine" if source == "synthetic" else "real"
    return output_root / folder / "masks" / Path(row["mask_path"]).name


def _assert_empty_output(output_root: Path) -> None:
    if output_root.exists() and not output_root.is_dir():
        raise FileExistsError(f"Output root exists and is not a directory: {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty geometric-control directory: {output_root}"
        )


def build_geometric_mask_control(
    manifest_paths: Iterable[Path],
    *,
    data_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Generate masks, manifests, summaries and a deterministic hash contract."""

    manifest_paths = [Path(path) for path in manifest_paths]
    _assert_empty_output(output_root)
    rows = read_reference_rows(manifest_paths)

    # Preflight every destination and immutable input before writing anything.
    prepared: list[tuple[dict[str, str], Path, Path]] = []
    seen_destinations: set[Path] = set()
    for index, row in enumerate(rows, start=1):
        reference_path = resolve_path(data_root, row["mask_path"])
        image_path = resolve_path(data_root, row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image referenced by the manifest does not exist: {image_path}")
        # Loading here makes empty/corrupt inputs fail before the output root is
        # populated. The mask is read again during generation to bound memory.
        load_binary_mask(reference_path)
        destination = output_mask_path(output_root, row)
        if destination in seen_destinations:
            raise ValueError(f"Two rows map to the same output mask: {destination}")
        seen_destinations.add(destination)
        prepared.append((row, reference_path, destination))
        if index % 100 == 0 or index == len(rows):
            print(f"preflight={index}/{len(rows)}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    reference_assets: list[dict[str, Any]] = []
    generated_assets: list[dict[str, Any]] = []
    by_source: Counter[str] = Counter()
    by_class: Counter[str] = Counter()

    for index, (row, reference_path, destination) in enumerate(prepared, start=1):
        reference_mask = load_binary_mask(reference_path)
        generated_mask, stats = rectangle_area_matched(reference_mask)
        atomic_save_binary_png(generated_mask, destination)
        reference_hash = sha256_file(reference_path)
        generated_hash = sha256_file(destination)
        reference_size = reference_path.stat().st_size
        generated_size = destination.stat().st_size

        output_rows.append(
            {
                **row,
                "mask_path": portable_path(destination),
                "reference_mask_path": row["mask_path"],
                "reference_refinement_method": row["refinement_method"],
                "geometric_control_method": CONTROL_METHOD,
                "reference_mask_sha256": reference_hash,
                "generated_mask_sha256": generated_hash,
                **stats,
            }
        )
        reference_assets.append(
            {
                "sample_id": row["sample_id"],
                "sha256": reference_hash,
                "size": reference_size,
                "foreground_pixels": stats["reference_area_px"],
            }
        )
        generated_assets.append(
            {
                "sample_id": row["sample_id"],
                "sha256": generated_hash,
                "size": generated_size,
                "foreground_pixels": stats["generated_area_px"],
            }
        )
        by_source[row["source"]] += 1
        by_class[row["class_name"]] += 1
        if index % 100 == 0 or index == len(prepared):
            print(f"processed={index}/{len(prepared)}", flush=True)

    manifests_dir = output_root / "manifests"
    real_rows = [row for row in output_rows if row["source"] == "real"]
    synthetic_rows = [row for row in output_rows if row["source"] == "synthetic"]
    output_manifests = {
        "real": manifests_dir / "real_rectangle_matched.csv",
        "synthetic": manifests_dir / "synthetic_fine_rectangle_matched.csv",
        "combined": manifests_dir / "combined_rectangle_matched.csv",
    }
    atomic_write_csv(output_manifests["real"], MANIFEST_FIELDS, real_rows)
    atomic_write_csv(output_manifests["synthetic"], MANIFEST_FIELDS, synthetic_rows)
    atomic_write_csv(output_manifests["combined"], MANIFEST_FIELDS, output_rows)
    summary_path = output_root / "geometric_mask_summary.csv"
    atomic_write_csv(summary_path, MANIFEST_FIELDS, output_rows)

    input_manifest_records = [
        {
            "path": portable_path(path),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in manifest_paths
    ]
    output_manifest_records = {
        name: {
            "path": portable_path(path),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for name, path in output_manifests.items()
    }
    generation_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "reference_variant": REFERENCE_VARIANT,
        "control_method": CONTROL_METHOD,
        "parameters": {
            "shape": "axis_aligned_filled_rectangle",
            "area_matching": "integer_raster_approximation",
            "centering": "reference_foreground_centroid_half_up",
            "aspect_reference": "reference_foreground_axis_aligned_bbox",
            "max_area_relative_error": MAX_AREA_RELATIVE_ERROR,
            "randomness": "none",
        },
        "sample_count": len(output_rows),
        "group_count": len({row["group_id"] for row in output_rows}),
        "source_counts": dict(sorted(by_source.items())),
        "class_counts": dict(sorted(by_class.items())),
        "input_manifests": input_manifest_records,
        "output_manifests": output_manifest_records,
        "reference_asset_content_sha256": sha256_payload(reference_assets),
        "generated_asset_content_sha256": sha256_payload(generated_assets),
        "summary": {
            "path": portable_path(summary_path),
            "sha256": sha256_file(summary_path),
            "size": summary_path.stat().st_size,
        },
        "generator": {
            "path": portable_path(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
            "size": Path(__file__).stat().st_size,
        },
    }
    generation_manifest["generation_fingerprint"] = sha256_payload(
        {
            key: value
            for key, value in generation_manifest.items()
            if key != "generation_fingerprint"
        }
    )
    atomic_write_json(output_root / "generation_manifest.json", generation_manifest)
    return generation_manifest


def main() -> None:
    args = parse_args()
    payload = build_geometric_mask_control(
        [Path(path) for path in args.manifest],
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
    )
    print(
        f"geometric_control_ready=true output={args.output_root} "
        f"samples={payload['sample_count']} groups={payload['group_count']} "
        f"fingerprint={payload['generation_fingerprint']}"
    )


if __name__ == "__main__":
    main()

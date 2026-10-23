#!/usr/bin/env python3
"""Create an anatomical-mask preview from the current segmentation masks.

This script does not replace the existing masks. It creates a separate preview
dataset by using each current mask as a region of interest and then looking for
the darker nostril-opening pixels inside that region. The goal is to visually
test whether a more anatomical target is feasible before using it in training.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grouped_splits import group_id_for_row  # noqa: E402
DEFAULT_MANIFESTS = [
    ROOT / "dataset_corrigido" / "manifests" / "real_corrigido.csv",
    ROOT / "dataset_corrigido" / "manifests" / "synthetic_fine_corrigido.csv",
]
DEFAULT_OUTPUT_ROOT = ROOT / "dataset_mascaras_anatomicas_preview"
DEFAULT_PREVIEW_DIR = ROOT / "results" / "mask_refinement_preview"


def parse_args() -> argparse.Namespace:
    """Read command-line options."""

    parser = argparse.ArgumentParser(description="Generate anatomical-mask previews from corrected masks.")
    parser.add_argument("--manifest", nargs="+", default=[str(path) for path in DEFAULT_MANIFESTS])
    parser.add_argument("--data-root", default=str(ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--preview-dir", default=str(DEFAULT_PREVIEW_DIR))
    parser.add_argument(
        "--strategy",
        choices=["dark", "shrink"],
        default="dark",
        help="dark uses image intensity; shrink conservatively contracts each current annotation.",
    )
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sample-per-group", type=int, default=10)
    parser.add_argument("--thumb-size", type=int, default=256)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(data_root: Path, value: str | Path) -> Path:
    """Resolve a manifest path from data_root when it is relative."""

    path = Path(value)
    return path if path.is_absolute() else data_root / path


def relative_to_root(path: Path) -> str:
    """Return a repository-relative POSIX path whenever possible."""

    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read a CSV manifest and keep its source path for traceability."""

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["group_id"] = str(group_id_for_row(row, strict=True))
        row["_manifest"] = str(path)
    return rows


def source_folder(source: str) -> str:
    """Map manifest source names to output folders."""

    return "synthetic_fine" if source == "synthetic" else source


def output_mask_path(output_root: Path, row: dict[str, str]) -> Path:
    """Return the output path for one refined mask."""

    source = source_folder(row.get("source", "unknown"))
    return output_root / source / "masks" / Path(row["mask_path"]).name


def output_image_path(output_root: Path, row: dict[str, str]) -> Path:
    """Return the optional output path for one copied image."""

    source = source_folder(row.get("source", "unknown"))
    return output_root / source / "images" / Path(row["image_path"]).name


def load_rgb(path: Path) -> np.ndarray:
    """Load an RGB image as a numpy array."""

    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    """Load one mask as a boolean array."""

    return np.array(Image.open(path).convert("L")) > 127


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Save a boolean mask as a binary PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def component_labels(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Return connected component labels and stats for a binary mask."""

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return labels_count, labels, stats


def keep_largest_components(mask: np.ndarray, max_components: int, min_area: int) -> np.ndarray:
    """Keep up to max_components components with at least min_area pixels."""

    labels_count, labels, stats = component_labels(mask)
    components: list[tuple[int, int]] = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label))
    components.sort(reverse=True)
    output = np.zeros_like(mask, dtype=bool)
    for _, label in components[:max_components]:
        output |= labels == label
    return output


def fill_small_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes using flood fill from the image border."""

    if mask.sum() == 0:
        return mask
    binary = mask.astype(np.uint8)
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = flood[1:-1, 1:-1] == 0
    return mask | holes


def morph(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Smooth the candidate mask with opening and closing."""

    if mask.sum() == 0:
        return mask
    kernel_size = max(3, kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    arr = mask.astype(np.uint8)
    arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    return arr.astype(bool)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return x1, y1, x2, y2 for a binary mask."""

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def border_trim(mask: np.ndarray) -> np.ndarray:
    """Remove thin image-border contact when it is likely annotation spillover."""

    h, w = mask.shape
    margin = max(1, round(min(h, w) * 0.025))
    trimmed = mask.copy()
    trimmed[:margin, :] = False
    trimmed[-margin:, :] = False
    trimmed[:, :margin] = False
    trimmed[:, -margin:] = False
    return trimmed if trimmed.sum() > max(6, mask.sum() * 0.25) else mask


def dark_candidate(rgb: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, str]:
    """Find dark nostril-opening candidates inside the current ROI mask."""

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if roi.sum() == 0:
        return roi.copy(), "empty_roi"

    pixels = gray[roi]
    if len(pixels) < 8:
        return roi.copy(), "tiny_roi"

    # Otsu inside the ROI gives a data-driven dark/light separation. The
    # quantile cap prevents the threshold from absorbing most of the annotation.
    otsu_threshold, _ = cv2.threshold(pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    q35 = float(np.quantile(pixels, 0.35))
    q50 = float(np.quantile(pixels, 0.50))
    threshold = min(float(otsu_threshold), q50)
    candidate = roi & (gray <= threshold)
    method = f"dark_otsu_qcap_{threshold:.1f}"

    roi_area = max(int(roi.sum()), 1)
    if candidate.sum() < max(8, roi_area * 0.04):
        candidate = roi & (gray <= q50)
        method = f"dark_q50_{q50:.1f}"
    elif candidate.sum() > roi_area * 0.75:
        candidate = roi & (gray <= q35)
        method = f"dark_q35_{q35:.1f}"

    return candidate, method


def ellipse_prior(roi: np.ndarray) -> np.ndarray:
    """Create a soft anatomical prior centered inside the current ROI bbox."""

    box = bbox(roi)
    if box is None:
        return roi.copy()
    x1, y1, x2, y2 = box
    h, w = roi.shape
    prior = np.zeros((h, w), dtype=np.uint8)
    cx = int(round((x1 + x2) / 2))
    cy = int(round((y1 + y2) / 2))
    ax = max(3, int(round((x2 - x1 + 1) * 0.48)))
    ay = max(3, int(round((y2 - y1 + 1) * 0.48)))
    cv2.ellipse(prior, (cx, cy), (ax, ay), 0, 0, 360, 1, -1)
    return prior.astype(bool)


def refine_mask(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine one mask toward a darker anatomical nostril-opening target."""

    original = mask.astype(bool)
    if original.sum() == 0:
        return original, {"method": "empty", "changed": False}

    roi = border_trim(original)
    min_area = max(6, int(roi.size * 0.0004))
    roi = keep_largest_components(roi, max_components=2, min_area=min_area)
    if roi.sum() == 0:
        roi = original

    candidate, method = dark_candidate(rgb, roi)
    candidate &= ellipse_prior(roi) | roi
    kernel_size = max(3, round(min(candidate.shape) * 0.015))
    candidate = morph(candidate, kernel_size)
    candidate = fill_small_holes(candidate)
    candidate = keep_largest_components(candidate, max_components=2, min_area=min_area)

    # Avoid making the target unrealistically tiny or leaving it almost equal to
    # the coarse annotation. In those cases, use a shrunken ROI as fallback.
    roi_area = max(int(roi.sum()), 1)
    if candidate.sum() < max(8, roi_area * 0.03) or candidate.sum() > roi_area * 0.85:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fallback = cv2.erode(roi.astype(np.uint8), kernel, iterations=1).astype(bool)
        fallback_candidate, fallback_method = dark_candidate(rgb, fallback if fallback.sum() else roi)
        fallback_candidate = morph(fallback_candidate, kernel_size)
        fallback_candidate = keep_largest_components(fallback_candidate, max_components=2, min_area=min_area)
        if fallback_candidate.sum() >= max(8, roi_area * 0.02):
            candidate = fallback_candidate
            method = f"fallback_{fallback_method}"

    if candidate.sum() == 0:
        candidate = roi
        method = "fallback_roi"

    stats = {
        "method": method,
        "changed": bool(np.any(candidate != original)),
        "original_area_px": int(original.sum()),
        "refined_area_px": int(candidate.sum()),
        "original_area_ratio": float(original.mean()),
        "refined_area_ratio": float(candidate.mean()),
        "area_fraction_of_original": float(candidate.sum() / max(original.sum(), 1)),
    }
    return candidate, stats


def shrink_component(component: np.ndarray, quantile: float) -> np.ndarray:
    """Shrink one connected component by keeping its distance-transform core."""

    if component.sum() == 0:
        return component
    dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    values = dist[component]
    if len(values) == 0:
        return component
    threshold = max(1.0, float(np.quantile(values, quantile)))
    shrunken = component & (dist >= threshold)
    return shrunken if shrunken.sum() >= max(6, int(component.sum() * 0.25)) else component


def shrink_mask(mask: np.ndarray, quantile: float = 0.28) -> tuple[np.ndarray, dict[str, Any]]:
    """Conservatively reduce coarse masks while preserving their annotated shape."""

    original = mask.astype(bool)
    if original.sum() == 0:
        return original, {"method": "empty", "changed": False}

    roi = border_trim(original)
    min_area = max(6, int(roi.size * 0.0004))
    roi = keep_largest_components(roi, max_components=2, min_area=min_area)
    if roi.sum() == 0:
        roi = original

    labels_count, labels, stats = component_labels(roi)
    candidate = np.zeros_like(roi, dtype=bool)
    used_quantiles: list[float] = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        local_quantile = quantile if area >= roi.size * 0.01 else 0.28
        shrunken = shrink_component(component, local_quantile)
        if shrunken.sum() < max(6, int(component.sum() * 0.55)):
            local_quantile = 0.28
            shrunken = shrink_component(component, local_quantile)
        candidate |= shrunken
        used_quantiles.append(local_quantile)

    if candidate.sum() == 0:
        candidate = roi
        used_quantiles = [0.0]

    kernel_size = max(3, round(min(candidate.shape) * 0.01))
    candidate = morph(candidate, kernel_size)
    candidate = fill_small_holes(candidate)
    candidate &= roi
    if candidate.sum() == 0:
        candidate = roi

    method_quantile = min(used_quantiles) if used_quantiles else quantile
    stats_payload = {
        "method": f"distance_shrink_q{method_quantile:.2f}",
        "changed": bool(np.any(candidate != original)),
        "original_area_px": int(original.sum()),
        "refined_area_px": int(candidate.sum()),
        "original_area_ratio": float(original.mean()),
        "refined_area_ratio": float(candidate.mean()),
        "area_fraction_of_original": float(candidate.sum() / max(original.sum(), 1)),
    }
    return candidate, stats_payload


def overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: int = 115) -> Image.Image:
    """Return an RGB image with a translucent mask overlay."""

    image = Image.fromarray(rgb).convert("RGBA")
    color_layer = Image.new("RGBA", image.size, (*color, alpha))
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    composed = Image.composite(color_layer, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask_image)
    return Image.alpha_composite(image, composed).convert("RGB")


def comparison_tile(row: dict[str, Any], data_root: Path, thumb_size: int) -> Image.Image:
    """Create a before/after tile for visual inspection."""

    rgb = load_rgb(resolve_path(data_root, row["image_path"]))
    original = load_mask(resolve_path(data_root, row["original_mask_path"]))
    refined = load_mask(resolve_path(data_root, row["mask_path"]))
    left = overlay(rgb, original, (210, 0, 0)).resize((thumb_size, thumb_size))
    right = overlay(rgb, refined, (0, 120, 255)).resize((thumb_size, thumb_size))
    tile = Image.new("RGB", (thumb_size * 2, thumb_size + 38), (255, 255, 255))
    tile.paste(left, (0, 24))
    tile.paste(right, (thumb_size, 24))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    title = f"{row['source']} | {row['class_name']} | area {float(row['original_area_ratio']):.3f}->{float(row['refined_area_ratio']):.3f}"
    draw.text((4, 4), title[:110], fill=(0, 0, 0), font=font)
    draw.text((4, thumb_size + 26), "original", fill=(210, 0, 0), font=font)
    draw.text((thumb_size + 4, thumb_size + 26), "anatomica preview", fill=(0, 80, 180), font=font)
    return tile


def make_sheet(rows: list[dict[str, Any]], data_root: Path, output_path: Path, thumb_size: int, columns: int) -> None:
    """Write a contact sheet with before/after mask comparisons."""

    if not rows:
        return
    columns = max(1, columns)
    tile_w = thumb_size * 2
    tile_h = thumb_size + 38
    rows_count = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows_count * tile_h), (245, 245, 245))
    for index, row in enumerate(rows):
        tile = comparison_tile(row, data_root, thumb_size)
        x = (index % columns) * tile_w
        y = (index // columns) * tile_h
        sheet.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one refined manifest."""

    fieldnames = [
        "sample_id",
        "group_id",
        "class_name",
        "image_path",
        "mask_path",
        "source",
        "original_mask_path",
        "refinement_method",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-sample refinement statistics."""

    fieldnames = [
        "sample_id",
        "class_name",
        "source",
        "image_path",
        "original_mask_path",
        "mask_path",
        "refinement_method",
        "original_area_px",
        "refined_area_px",
        "original_area_ratio",
        "refined_area_ratio",
        "area_fraction_of_original",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def grouped_samples(rows: list[dict[str, Any]], sample_per_group: int, seed: int) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Return deterministic visual samples grouped by source and class."""

    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["source"], row["class_name"]), []).append(row)
    sampled: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, values in groups.items():
        order = rng.permutation(len(values))
        sampled[key] = [values[int(index)] for index in order[:sample_per_group]]
    return sampled


def main() -> None:
    """Generate refined masks and comparison sheets."""

    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    preview_dir = Path(args.preview_dir)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"{output_root} already exists. Use --force to overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for manifest in args.manifest:
        rows.extend(read_manifest(Path(manifest)))

    refined_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        image_path = resolve_path(data_root, row["image_path"])
        mask_path = resolve_path(data_root, row["mask_path"])
        rgb = load_rgb(image_path)
        mask = load_mask(mask_path)
        if args.strategy == "shrink":
            refined, stats = shrink_mask(mask)
        else:
            refined, stats = refine_mask(rgb, mask)
        new_mask_path = output_mask_path(output_root, row)
        save_mask(refined, new_mask_path)

        new_image_value = row["image_path"]
        if args.copy_images:
            new_image_path = output_image_path(output_root, row)
            new_image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, new_image_path)
            new_image_value = relative_to_root(new_image_path)

        refined_rows.append(
            {
                **row,
                "image_path": new_image_value,
                "mask_path": relative_to_root(new_mask_path),
                "original_mask_path": row["mask_path"],
                "refinement_method": stats["method"],
                **stats,
            }
        )
        if index % 100 == 0 or index == len(rows):
            print(f"processed={index}/{len(rows)}")

    manifests_dir = output_root / "manifests"
    real_rows = [row for row in refined_rows if row["source"] == "real"]
    synthetic_rows = [row for row in refined_rows if row["source"] == "synthetic"]
    suffix = "shrink_preview" if args.strategy == "shrink" else "anatomico_preview"
    write_manifest(manifests_dir / f"real_{suffix}.csv", real_rows)
    write_manifest(manifests_dir / f"synthetic_fine_{suffix}.csv", synthetic_rows)
    write_manifest(manifests_dir / f"combined_{suffix}.csv", refined_rows)
    write_summary(output_root / "mask_refinement_summary.csv", refined_rows)

    payload = {
        "rows": len(refined_rows),
        "mean_original_area_ratio": mean(float(row["original_area_ratio"]) for row in refined_rows),
        "mean_refined_area_ratio": mean(float(row["refined_area_ratio"]) for row in refined_rows),
        "median_area_fraction_of_original": median(float(row["area_fraction_of_original"]) for row in refined_rows),
        "manifests": [str(path) for path in args.manifest],
        "output_root": str(output_root),
        "strategy": args.strategy,
    }
    (output_root / "mask_refinement_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    groups = grouped_samples(refined_rows, args.sample_per_group, args.seed)
    for (source, class_name), sample_rows in sorted(groups.items()):
        make_sheet(sample_rows, data_root, preview_dir / f"comparison_{source}_{class_name}.png", args.thumb_size, args.columns)

    biggest_changes = sorted(refined_rows, key=lambda row: float(row["area_fraction_of_original"]))[: args.sample_per_group * args.columns]
    make_sheet(biggest_changes, data_root, preview_dir / "comparison_biggest_area_reduction.png", args.thumb_size, args.columns)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"preview_dir={preview_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a corrected-mask dataset without changing the original files.

The correction used here is intentionally conservative: it keeps the semantic
shape of the existing annotation, removes tiny isolated components, trims mask
pixels glued to the image border when possible, and lightly erodes very large
masks. This gives a cleaner segmentation target while preserving traceability.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grouped_splits import group_id_for_row  # noqa: E402
DEFAULT_OUTPUT_ROOT = ROOT / "dataset_corrigido"
DEFAULT_MANIFESTS = [
    ROOT / "data" / "manifests" / "real.csv",
    ROOT / "data" / "manifests" / "synthetic_fine.csv",
]


def parse_args() -> argparse.Namespace:
    """Read command-line options for the corrected dataset builder."""

    parser = argparse.ArgumentParser(description="Build corrected masks and manifests.")
    parser.add_argument("--manifest", nargs="+", default=[str(path) for path in DEFAULT_MANIFESTS], help="Original CSV manifests.")
    parser.add_argument("--data-root", default=str(ROOT), help="Root used to resolve relative image/mask paths.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Folder where corrected masks/manifests are written.")
    parser.add_argument("--copy-images", action="store_true", help="Also copy images into dataset_corrigido and point manifests to those copies.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing corrected dataset folder.")
    return parser.parse_args()


def resolve_path(data_root: Path, value: str | Path) -> Path:
    """Resolve absolute paths as-is and relative paths from data_root."""

    path = Path(value)
    return path if path.is_absolute() else data_root / path


def relative_to_root(path: Path) -> str:
    """Return a repository-relative POSIX path when possible."""

    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read one manifest and attach its file path for traceability."""

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["group_id"] = str(group_id_for_row(row, strict=True))
        row["_manifest"] = str(path)
    return rows


def binary_mask(path: Path) -> np.ndarray:
    """Open one mask file as a boolean array."""

    with Image.open(path) as handle:
        return np.array(handle.convert("L")) > 127


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 8-connected components from a binary mask."""

    visited = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            components.append(component)
    return components


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove isolated components smaller than min_area pixels."""

    output = np.zeros_like(mask)
    for component in connected_components(mask):
        if len(component) >= min_area:
            ys, xs = zip(*component)
            output[ys, xs] = True
    return output


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply a small binary erosion using a 3x3 structuring element."""

    output = mask.copy()
    for _ in range(iterations):
        padded = np.pad(output, 1, constant_values=False)
        eroded = np.ones_like(output)
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy : dy + output.shape[0], dx : dx + output.shape[1]]
        output = eroded
    return output


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply a small binary dilation using a 3x3 structuring element."""

    output = mask.copy()
    for _ in range(iterations):
        padded = np.pad(output, 1, constant_values=False)
        dilated = np.zeros_like(output)
        for dy in range(3):
            for dx in range(3):
                dilated |= padded[dy : dy + output.shape[0], dx : dx + output.shape[1]]
        output = dilated
    return output


def touches_border(mask: np.ndarray) -> bool:
    """Return True when the mask touches any image border."""

    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def trim_border(mask: np.ndarray) -> np.ndarray:
    """Remove mask pixels from a thin image-border margin when safe."""

    height, width = mask.shape
    margin = max(1, round(min(height, width) * 0.02))
    trimmed = mask.copy()
    trimmed[:margin, :] = False
    trimmed[-margin:, :] = False
    trimmed[:, :margin] = False
    trimmed[:, -margin:] = False
    return trimmed if trimmed.sum() > max(8, mask.sum() * 0.35) else mask


def mask_stats(mask: np.ndarray) -> dict[str, Any]:
    """Compute compact statistics before/after correction."""

    area = int(mask.sum())
    area_ratio = float(area / max(mask.size, 1))
    components = connected_components(mask)
    return {
        "area_px": area,
        "area_ratio": area_ratio,
        "component_count": len(components),
        "touches_border": touches_border(mask) if area else False,
    }


def correct_mask(mask: np.ndarray) -> np.ndarray:
    """Apply conservative mask correction while preserving the annotation."""

    if mask.sum() == 0:
        return mask

    corrected = mask.copy()
    min_component_area = max(8, int(mask.size * 0.0008))
    corrected = remove_small_components(corrected, min_component_area)

    if corrected.sum() == 0:
        corrected = mask.copy()

    if touches_border(corrected):
        corrected = trim_border(corrected)

    if corrected.mean() > 0.50:
        eroded = erode(corrected, 1)
        if eroded.sum() > max(8, corrected.sum() * 0.40):
            corrected = eroded

    # Abertura leve: reduz serrilhados e ilhas finas, mas evita apagar máscaras
    # pequenas. A dilatação devolve a escala aproximada da anotação original.
    if corrected.sum() > 40:
        opened = dilate(erode(corrected, 1), 1)
        if opened.sum() > max(8, corrected.sum() * 0.40):
            corrected = opened

    corrected = remove_small_components(corrected, min_component_area)
    return corrected if corrected.sum() > 0 else mask


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Write one binary mask as a PNG file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def source_folder(source: str) -> str:
    """Return the dataset_corrigido subfolder for one source value."""

    return "synthetic_fine" if source == "synthetic" else source


def corrected_mask_path(output_root: Path, row: dict[str, str]) -> Path:
    """Return where the corrected mask for one row should be stored."""

    source = source_folder(row.get("source", "unknown"))
    original_name = Path(row["mask_path"]).name
    return output_root / source / "masks" / original_name


def corrected_image_path(output_root: Path, row: dict[str, str]) -> Path:
    """Return where an optional copied image should be stored."""

    source = source_folder(row.get("source", "unknown"))
    original_name = Path(row["image_path"]).name
    return output_root / source / "images" / original_name


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write corrected manifest rows preserving the required columns first."""

    fieldnames = [
        "sample_id",
        "group_id",
        "class_name",
        "image_path",
        "mask_path",
        "source",
        "original_mask_path",
        "mask_correction_method",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-sample correction statistics."""

    fieldnames = [
        "sample_id",
        "class_name",
        "source",
        "original_area_ratio",
        "corrected_area_ratio",
        "original_component_count",
        "corrected_component_count",
        "original_touches_border",
        "corrected_touches_border",
        "mask_path",
        "corrected_mask_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output_root: Path, summary: dict[str, Any]) -> None:
    """Write a short README inside dataset_corrigido."""

    lines = [
        "# Dataset corrigido",
        "",
        "Esta pasta foi criada automaticamente a partir dos manifestos em `data/manifests/`.",
        "As máscaras originais não foram alteradas.",
        "",
        "## Método",
        "",
        "A correção aplicada é conservadora: remove componentes isolados pequenos,",
        "corta pixels grudados na borda quando isso não apaga a máscara principal e",
        "aplica uma limpeza morfológica leve. A intenção é reduzir ruído nas máscaras",
        "sem mudar a definição anatômica usada nas anotações existentes.",
        "",
        "## Arquivos",
        "",
        "- `manifests/real_corrigido.csv`",
        "- `manifests/synthetic_fine_corrigido.csv`",
        "- `manifests/combined_corrigido.csv`",
        "- `mask_correction_summary.csv`",
        "- `mask_correction_summary.json`",
        "",
        "Use estes manifestos nos experimentos com `--use-corrected-dataset`.",
        "",
        "## Resumo",
        "",
        f"- Total de amostras: {summary['total_samples']}",
        f"- Distribuição por origem: {summary['by_source']}",
        f"- Máscaras que tocavam borda antes: {summary['original_touches_border']}",
        f"- Máscaras que tocam borda depois: {summary['corrected_touches_border']}",
        f"- Área média original: {summary['mean_original_area_ratio']}",
        f"- Área média corrigida: {summary['mean_corrected_area_ratio']}",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Build corrected masks and corrected manifests."""

    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"{output_root} already exists. Use --force to overwrite generated files.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for manifest in args.manifest:
        rows.extend(read_manifest(Path(manifest)))

    corrected_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    by_source: Counter[str] = Counter()

    for row in rows:
        image_path = resolve_path(data_root, row["image_path"])
        mask_path = resolve_path(data_root, row["mask_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        original_mask = binary_mask(mask_path)
        corrected = correct_mask(original_mask)
        original_stats = mask_stats(original_mask)
        corrected_stats = mask_stats(corrected)

        mask_output = corrected_mask_path(output_root, row)
        save_mask(corrected, mask_output)

        if args.copy_images:
            image_output = corrected_image_path(output_root, row)
            image_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, image_output)
            image_manifest_path = relative_to_root(image_output)
        else:
            image_manifest_path = row["image_path"]

        corrected_row = {
            **row,
            "image_path": image_manifest_path,
            "mask_path": relative_to_root(mask_output),
            "original_mask_path": row["mask_path"],
            "mask_correction_method": "conservative_annotation_clean",
        }
        corrected_rows.append(corrected_row)
        by_source[row.get("source", "")] += 1

        summary_rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "class_name": row.get("class_name", ""),
                "source": row.get("source", ""),
                "original_area_ratio": original_stats["area_ratio"],
                "corrected_area_ratio": corrected_stats["area_ratio"],
                "original_component_count": original_stats["component_count"],
                "corrected_component_count": corrected_stats["component_count"],
                "original_touches_border": original_stats["touches_border"],
                "corrected_touches_border": corrected_stats["touches_border"],
                "mask_path": row["mask_path"],
                "corrected_mask_path": relative_to_root(mask_output),
            }
        )

    real_rows = [row for row in corrected_rows if row.get("source") == "real"]
    synthetic_rows = [row for row in corrected_rows if row.get("source") == "synthetic"]
    manifest_root = output_root / "manifests"
    write_manifest(manifest_root / "real_corrigido.csv", real_rows)
    write_manifest(manifest_root / "synthetic_fine_corrigido.csv", synthetic_rows)
    write_manifest(manifest_root / "combined_corrigido.csv", corrected_rows)
    write_summary(output_root / "mask_correction_summary.csv", summary_rows)

    original_border = sum(1 for row in summary_rows if row["original_touches_border"])
    corrected_border = sum(1 for row in summary_rows if row["corrected_touches_border"])
    original_areas = [float(row["original_area_ratio"]) for row in summary_rows]
    corrected_areas = [float(row["corrected_area_ratio"]) for row in summary_rows]
    summary = {
        "total_samples": len(summary_rows),
        "by_source": dict(sorted(by_source.items())),
        "original_touches_border": original_border,
        "corrected_touches_border": corrected_border,
        "mean_original_area_ratio": float(np.mean(original_areas)) if original_areas else None,
        "mean_corrected_area_ratio": float(np.mean(corrected_areas)) if corrected_areas else None,
        "method": "conservative_annotation_clean",
    }
    (output_root / "mask_correction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(output_root, summary)

    print(f"corrected_dataset={output_root}")
    print(f"corrected_samples={len(summary_rows)}")
    print(f"original_touches_border={original_border}")
    print(f"corrected_touches_border={corrected_border}")
    print(f"real_manifest={manifest_root / 'real_corrigido.csv'}")
    print(f"synthetic_manifest={manifest_root / 'synthetic_fine_corrigido.csv'}")


if __name__ == "__main__":
    main()

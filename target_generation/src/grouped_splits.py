"""Leakage-safe grouped and stratified dataset splits.

The two observations from the same real dog and the two crops derived from the
same synthetic source image are not independent.  This module therefore treats
them as indivisible groups.  Its public split functions use only the Python
standard library and balance complete groups using sample-count vectors for
both class and source.

Grouping deliberately fails closed: a row must either contain a non-empty
``group_id`` or match the documented real/synthetic ``sample_id`` convention.
Silently turning an unrecognised row into a singleton would reintroduce the
very leakage this module is intended to prevent.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
import re
from typing import Any, Iterable, Mapping, Sequence


_REAL_SAMPLE_ID = re.compile(
    r"^real_(?P<observation>[0-9]+)_(?P<animal>[0-9]+)_"
    r"(?P<label>[A-Za-z0-9][A-Za-z0-9_-]*)$",
    flags=re.ASCII,
)
_SYNTHETIC_SAMPLE_ID = re.compile(
    r"^synthetic_ann_(?P<annotation>[0-9]+)_img_(?P<image>[0-9]+)_"
    r"(?P<label>[A-Za-z0-9][A-Za-z0-9_-]*)$",
    flags=re.ASCII,
)


class GroupedSplitError(ValueError):
    """Raised when a leakage-safe grouped split cannot be constructed."""


def _required_text(row: Mapping[str, Any], column: str, *, row_index: int | None = None) -> str:
    value = row.get(column)
    if value is None or not str(value).strip():
        location = f" at row index {row_index}" if row_index is not None else ""
        raise GroupedSplitError(f"missing or empty {column!r}{location}")
    return str(value).strip()


def _source_for_row(
    row: Mapping[str, Any],
    *,
    source_column: str,
    row_index: int | None = None,
) -> str:
    source = _required_text(row, source_column, row_index=row_index).lower()
    if source not in {"real", "synthetic"}:
        location = f" at row index {row_index}" if row_index is not None else ""
        raise GroupedSplitError(
            f"unsupported source {source!r}{location}; expected 'real' or 'synthetic'"
        )
    return source


def group_id_for_row(
    row: Mapping[str, Any],
    *,
    strict: bool = True,
    group_column: str = "group_id",
    sample_id_column: str = "sample_id",
    source_column: str = "source",
    row_index: int | None = None,
) -> str | None:
    """Return the leakage group for one manifest row.

    An explicit ``group_id`` is namespaced by source and must agree with the
    group derived from a canonical ``sample_id``. These conventions are
    accepted:

    - ``real_<observation>_<animal>_<label>`` -> one group per animal;
    - ``synthetic_ann_<annotation>_img_<image>_<label>`` -> one group per
      generated source image.

    Numeric identifiers are canonicalised through ``int`` so leading zeroes do
    not accidentally create distinct groups.
    """

    source = _source_for_row(row, source_column=source_column, row_index=row_index)
    sample_id = _required_text(row, sample_id_column, row_index=row_index)
    derived_group: str | None = None
    if source == "real":
        match = _REAL_SAMPLE_ID.fullmatch(sample_id)
        if match:
            derived_group = f"real:animal:{int(match.group('animal'))}"
        expected = "real_<observation>_<animal>_<label>"
    else:
        match = _SYNTHETIC_SAMPLE_ID.fullmatch(sample_id)
        if match:
            derived_group = f"synthetic:image:{int(match.group('image'))}"
        expected = "synthetic_ann_<annotation>_img_<image>_<label>"

    explicit_group = row.get(group_column)
    if explicit_group is not None and str(explicit_group).strip():
        explicit_text = str(explicit_group).strip()
        normalized_explicit = explicit_text if explicit_text.startswith(f"{source}:") else f"{source}:explicit:{explicit_text}"
        if derived_group is not None and normalized_explicit != derived_group:
            location = f" at row index {row_index}" if row_index is not None else ""
            raise GroupedSplitError(
                f"explicit group_id {normalized_explicit!r} disagrees with sample_id-derived "
                f"group {derived_group!r}{location}"
            )
        return normalized_explicit
    if derived_group is not None:
        return derived_group

    location = f" at row index {row_index}" if row_index is not None else ""
    if strict:
        raise GroupedSplitError(
            f"cannot derive a safe group from sample_id {sample_id!r}{location}; "
            f"provide {group_column!r} or use {expected}"
        )
    return None


def _validated_indices(rows: Sequence[Mapping[str, Any]], indices: Iterable[int]) -> list[int]:
    materialised = list(indices)
    if not materialised:
        raise GroupedSplitError("cannot split an empty set of indices")

    seen: set[int] = set()
    for index in materialised:
        if isinstance(index, bool) or not isinstance(index, int):
            raise GroupedSplitError(f"dataset index must be an integer, got {index!r}")
        if index < 0 or index >= len(rows):
            raise GroupedSplitError(
                f"dataset index {index} is outside the valid range [0, {len(rows)})"
            )
        if index in seen:
            raise GroupedSplitError(f"dataset index {index} was supplied more than once")
        seen.add(index)
    return materialised


@dataclass(frozen=True)
class _Group:
    group_id: str
    indices: tuple[int, ...]
    features: Counter[tuple[str, str]]

    @property
    def sample_count(self) -> int:
        return len(self.indices)


def _build_groups(
    rows: Sequence[Mapping[str, Any]],
    indices: Iterable[int],
    *,
    class_column: str,
    source_column: str,
    group_column: str,
    sample_id_column: str,
) -> tuple[list[int], list[_Group]]:
    valid_indices = _validated_indices(rows, indices)
    members: dict[str, list[int]] = defaultdict(list)
    features: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)

    for index in valid_indices:
        row = rows[index]
        group_id = group_id_for_row(
            row,
            strict=True,
            group_column=group_column,
            sample_id_column=sample_id_column,
            source_column=source_column,
            row_index=index,
        )
        if group_id is None:  # Defensive: split construction always uses strict mode.
            raise GroupedSplitError(f"row index {index} has no safe group identifier")
        class_name = _required_text(row, class_column, row_index=index)
        source = _source_for_row(row, source_column=source_column, row_index=index)
        members[group_id].append(index)
        # Every sample contributes once to each part of the stratification
        # vector.  This preserves class and source distributions without
        # collapsing a mixed-label group to a misleading majority label.
        features[group_id][("class", class_name)] += 1
        features[group_id][("source", source)] += 1

    groups = [
        _Group(
            group_id=group_id,
            indices=tuple(sorted(members[group_id])),
            features=features[group_id],
        )
        for group_id in sorted(members)
    ]
    return valid_indices, groups


def _normalised_squared_error(observed: int, target: float) -> float:
    # A denominator of one keeps rare strata relevant without allowing a tiny
    # target to cause a division-by-zero or an unbounded score.
    denominator = max(target, 1.0)
    return ((observed - target) / denominator) ** 2


def _assign_groups(
    groups: Sequence[_Group],
    *,
    weights: Sequence[float],
    seed: int,
) -> list[list[int]]:
    n_partitions = len(weights)
    if n_partitions < 2:
        raise GroupedSplitError("at least two partitions are required")
    if len(groups) < n_partitions:
        raise GroupedSplitError(
            f"cannot create {n_partitions} non-empty partitions from only {len(groups)} groups"
        )
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise GroupedSplitError("all partition weights must be finite and greater than zero")
    weight_sum = sum(weights)
    normalised_weights = [weight / weight_sum for weight in weights]

    total_features: Counter[tuple[str, str]] = Counter()
    for group in groups:
        total_features.update(group.features)
    feature_dimensions = sorted(total_features)
    total_samples = sum(group.sample_count for group in groups)
    total_groups = len(groups)

    rng = random.Random(seed)
    random_tie_break = {group.group_id: rng.random() for group in groups}

    def rarity(group: _Group) -> float:
        return sum(
            count / total_features[dimension]
            for dimension, count in group.features.items()
        )

    # Rare and large groups are placed first, while seeded random values make
    # otherwise identical groups exchangeable in a reproducible way.
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            -rarity(group),
            -group.sample_count,
            random_tie_break[group.group_id],
            group.group_id,
        ),
    )

    partition_indices: list[list[int]] = [[] for _ in range(n_partitions)]
    partition_features = [Counter() for _ in range(n_partitions)]
    partition_sample_counts = [0 for _ in range(n_partitions)]
    partition_group_counts = [0 for _ in range(n_partitions)]

    def objective() -> float:
        feature_error = 0.0
        for partition_index, weight in enumerate(normalised_weights):
            for dimension in feature_dimensions:
                feature_error += _normalised_squared_error(
                    partition_features[partition_index][dimension],
                    total_features[dimension] * weight,
                )
        feature_error /= max(n_partitions * len(feature_dimensions), 1)

        sample_error = sum(
            _normalised_squared_error(
                partition_sample_counts[partition_index], total_samples * weight
            )
            for partition_index, weight in enumerate(normalised_weights)
        ) / n_partitions
        group_error = sum(
            _normalised_squared_error(
                partition_group_counts[partition_index], total_groups * weight
            )
            for partition_index, weight in enumerate(normalised_weights)
        ) / n_partitions

        # Class/source vector balance is primary.  Sample and group totals are
        # light regularisers that resolve otherwise equally stratified choices.
        return feature_error + 0.20 * sample_error + 0.05 * group_error

    for position, group in enumerate(ordered_groups):
        empty_partitions = [
            partition_index
            for partition_index, count in enumerate(partition_group_counts)
            if count == 0
        ]
        groups_remaining = len(ordered_groups) - position
        candidates = (
            empty_partitions
            if empty_partitions and groups_remaining == len(empty_partitions)
            else list(range(n_partitions))
        )

        # Do not let a locally attractive rare-class assignment make a
        # partition drift far beyond its requested sample fraction.  The
        # one-group allowance is necessary because groups are indivisible.  It
        # is a soft bound: if every candidate exceeds it, the global objective
        # still chooses the least harmful assignment.
        def stays_within_one_group_of_target(partition_index: int) -> bool:
            weight = normalised_weights[partition_index]
            if (
                partition_sample_counts[partition_index] + group.sample_count
                > math.ceil(total_samples * weight) + group.sample_count
            ):
                return False
            return all(
                partition_features[partition_index][dimension] + count
                <= math.ceil(total_features[dimension] * weight) + count
                for dimension, count in group.features.items()
            )

        capacity_candidates = [
            partition_index
            for partition_index in candidates
            if stays_within_one_group_of_target(partition_index)
        ]
        if capacity_candidates:
            candidates = capacity_candidates

        scored_candidates: list[tuple[float, float, float, float, int]] = []
        for partition_index in candidates:
            partition_features[partition_index].update(group.features)
            partition_sample_counts[partition_index] += group.sample_count
            partition_group_counts[partition_index] += 1

            score = objective()
            relative_samples = partition_sample_counts[partition_index] / normalised_weights[
                partition_index
            ]
            relative_groups = partition_group_counts[partition_index] / normalised_weights[
                partition_index
            ]
            scored_candidates.append(
                (score, relative_samples, relative_groups, rng.random(), partition_index)
            )

            partition_features[partition_index].subtract(group.features)
            partition_features[partition_index] += Counter()
            partition_sample_counts[partition_index] -= group.sample_count
            partition_group_counts[partition_index] -= 1

        selected = min(scored_candidates)[-1]
        partition_indices[selected].extend(group.indices)
        partition_features[selected].update(group.features)
        partition_sample_counts[selected] += group.sample_count
        partition_group_counts[selected] += 1

    for partition in partition_indices:
        partition.sort()
    return partition_indices


def grouped_stratified_kfolds(
    rows: Sequence[Mapping[str, Any]],
    indices: Iterable[int],
    *,
    n_folds: int,
    seed: int,
    class_column: str = "class_name",
    source_column: str = "source",
    group_column: str = "group_id",
    sample_id_column: str = "sample_id",
) -> list[list[int]]:
    """Return deterministic, leakage-safe and approximately stratified folds."""

    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise GroupedSplitError("n_folds must be an integer greater than or equal to two")
    valid_indices, groups = _build_groups(
        rows,
        indices,
        class_column=class_column,
        source_column=source_column,
        group_column=group_column,
        sample_id_column=sample_id_column,
    )
    folds = _assign_groups(groups, weights=[1.0] * n_folds, seed=seed)
    validate_grouped_partitions(
        rows,
        valid_indices,
        folds,
        class_column=class_column,
        source_column=source_column,
        group_column=group_column,
        sample_id_column=sample_id_column,
    )
    return folds


def grouped_stratified_holdout(
    rows: Sequence[Mapping[str, Any]],
    indices: Iterable[int],
    *,
    holdout_fraction: float,
    seed: int,
    class_column: str = "class_name",
    source_column: str = "source",
    group_column: str = "group_id",
    sample_id_column: str = "sample_id",
) -> tuple[list[int], list[int]]:
    """Split complete groups into development and holdout partitions.

    The requested fraction is a target rather than a guarantee because an
    indivisible group may contain more than one sample.
    """

    if (
        isinstance(holdout_fraction, bool)
        or not isinstance(holdout_fraction, (int, float))
        or not math.isfinite(float(holdout_fraction))
        or not 0.0 < float(holdout_fraction) < 1.0
    ):
        raise GroupedSplitError("holdout_fraction must be finite and strictly between zero and one")
    valid_indices, groups = _build_groups(
        rows,
        indices,
        class_column=class_column,
        source_column=source_column,
        group_column=group_column,
        sample_id_column=sample_id_column,
    )
    holdout_fraction = float(holdout_fraction)
    development, holdout = _assign_groups(
        groups,
        weights=[1.0 - holdout_fraction, holdout_fraction],
        seed=seed,
    )
    validate_grouped_partitions(
        rows,
        valid_indices,
        [development, holdout],
        class_column=class_column,
        source_column=source_column,
        group_column=group_column,
        sample_id_column=sample_id_column,
    )
    return development, holdout


def validate_grouped_partitions(
    rows: Sequence[Mapping[str, Any]],
    expected_indices: Iterable[int],
    partitions: Sequence[Iterable[int]],
    *,
    class_column: str = "class_name",
    source_column: str = "source",
    group_column: str = "group_id",
    sample_id_column: str = "sample_id",
    require_nonempty: bool = True,
) -> dict[str, Any]:
    """Validate coverage, index disjointness and group disjointness.

    A summary suitable for a fold manifest or audit report is returned.  Any
    missing/extra index, duplicate assignment, empty required partition, or
    group crossing a boundary raises :class:`GroupedSplitError`.
    """

    expected = _validated_indices(rows, expected_indices)
    expected_set = set(expected)
    if len(partitions) < 2:
        raise GroupedSplitError("at least two partitions are required for validation")

    seen_index_partition: dict[int, int] = {}
    seen_group_partition: dict[str, int] = {}
    partition_sizes: list[int] = []
    partition_group_counts: list[int] = []
    class_counts: list[dict[str, int]] = []
    source_counts: list[dict[str, int]] = []

    for partition_number, partition_values in enumerate(partitions):
        partition = list(partition_values)
        if require_nonempty and not partition:
            raise GroupedSplitError(f"partition {partition_number} is empty")
        if len(partition) != len(set(partition)):
            raise GroupedSplitError(f"partition {partition_number} contains a duplicate index")

        groups_in_partition: set[str] = set()
        classes: Counter[str] = Counter()
        sources: Counter[str] = Counter()
        for index in partition:
            if isinstance(index, bool) or not isinstance(index, int):
                raise GroupedSplitError(
                    f"partition {partition_number} contains a non-integer index {index!r}"
                )
            if index not in expected_set:
                raise GroupedSplitError(
                    f"partition {partition_number} contains unexpected index {index}"
                )
            prior_partition = seen_index_partition.get(index)
            if prior_partition is not None:
                raise GroupedSplitError(
                    f"index {index} appears in partitions {prior_partition} and {partition_number}"
                )
            seen_index_partition[index] = partition_number

            row = rows[index]
            group_id = group_id_for_row(
                row,
                strict=True,
                group_column=group_column,
                sample_id_column=sample_id_column,
                source_column=source_column,
                row_index=index,
            )
            if group_id is None:  # Defensive: validation always uses strict mode.
                raise GroupedSplitError(f"row index {index} has no safe group identifier")
            prior_group_partition = seen_group_partition.get(group_id)
            if prior_group_partition is not None and prior_group_partition != partition_number:
                raise GroupedSplitError(
                    f"group {group_id!r} crosses partitions "
                    f"{prior_group_partition} and {partition_number}"
                )
            seen_group_partition[group_id] = partition_number
            groups_in_partition.add(group_id)
            classes[_required_text(row, class_column, row_index=index)] += 1
            sources[
                _source_for_row(row, source_column=source_column, row_index=index)
            ] += 1

        partition_sizes.append(len(partition))
        partition_group_counts.append(len(groups_in_partition))
        class_counts.append(dict(sorted(classes.items())))
        source_counts.append(dict(sorted(sources.items())))

    missing = sorted(expected_set.difference(seen_index_partition))
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise GroupedSplitError(f"partitions do not cover expected indices: {preview}{suffix}")

    return {
        "n_indices": len(expected),
        "n_groups": len(seen_group_partition),
        "n_partitions": len(partitions),
        "partition_sizes": partition_sizes,
        "partition_group_counts": partition_group_counts,
        "class_counts": class_counts,
        "source_counts": source_counts,
    }


def _dataset_rows_and_class_column(dataset: Any) -> tuple[Sequence[Mapping[str, Any]], str]:
    rows = getattr(dataset, "rows", None)
    if rows is None:
        raise GroupedSplitError("dataset must expose its manifest rows through a 'rows' attribute")
    class_column = str(getattr(dataset, "class_column", "class_name"))
    return rows, class_column


def stratified_group_kfolds(
    dataset: Any,
    indices: Iterable[int],
    n_folds: int,
    seed: int,
) -> list[list[int]]:
    """Dataset-oriented wrapper used by the training pipeline."""

    rows, class_column = _dataset_rows_and_class_column(dataset)
    return grouped_stratified_kfolds(
        rows,
        indices,
        n_folds=n_folds,
        seed=seed,
        class_column=class_column,
    )


def stratified_group_holdout(
    dataset: Any,
    indices: Iterable[int],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Dataset-oriented grouped train/validation wrapper."""

    rows, class_column = _dataset_rows_and_class_column(dataset)
    return grouped_stratified_holdout(
        rows,
        indices,
        holdout_fraction=holdout_fraction,
        seed=seed,
        class_column=class_column,
    )


def stratified_group_real_validation_split(
    dataset: Any,
    indices: Iterable[int],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create grouped train/validation partitions with real-only validation.

    ``indices`` is the outer-fold development set. Complete real-animal groups
    are split into real development and validation partitions; every synthetic
    sample remains in training. This is the defining internal-split policy of
    ``grouped_v3_realval``.

    The function fails closed when a row has an unsupported source, when no
    real examples are available, or when coverage/group-disjointness cannot be
    proven.
    """

    rows, class_column = _dataset_rows_and_class_column(dataset)
    valid_indices = _validated_indices(rows, indices)
    real_indices: list[int] = []
    synthetic_indices: list[int] = []
    for index in valid_indices:
        source = _source_for_row(rows[index], source_column="source", row_index=index)
        if source == "real":
            real_indices.append(index)
        else:
            synthetic_indices.append(index)

    if not real_indices:
        raise GroupedSplitError(
            "grouped_v3_realval requires real samples in every outer development set"
        )

    real_train, validation = grouped_stratified_holdout(
        rows,
        real_indices,
        holdout_fraction=holdout_fraction,
        seed=seed,
        class_column=class_column,
    )
    train = sorted([*real_train, *synthetic_indices])
    validation = sorted(validation)

    summary = validate_grouped_partitions(
        rows,
        valid_indices,
        [train, validation],
        class_column=class_column,
    )
    validation_sources = summary["source_counts"][1]
    if set(validation_sources) != {"real"}:
        raise GroupedSplitError(
            "grouped_v3_realval validation must contain source=real only"
        )
    if summary["source_counts"][0].get("synthetic", 0) != len(synthetic_indices):
        raise GroupedSplitError(
            "grouped_v3_realval must keep every available synthetic sample in training"
        )
    return train, validation


def assert_group_disjoint(dataset: Any, partitions: Sequence[Iterable[int]]) -> None:
    """Raise if an index or leakage group occurs in more than one partition."""

    materialised = [list(partition) for partition in partitions]
    expected = [index for partition in materialised for index in partition]
    rows, class_column = _dataset_rows_and_class_column(dataset)
    validate_grouped_partitions(
        rows,
        expected,
        materialised,
        class_column=class_column,
        require_nonempty=False,
    )


def validate_group_fold_coverage(
    dataset: Any,
    folds: Sequence[Iterable[int]],
    expected_indices: Iterable[int],
) -> dict[str, Any]:
    """Dataset-oriented coverage and group-leakage audit wrapper."""

    rows, class_column = _dataset_rows_and_class_column(dataset)
    return validate_grouped_partitions(
        rows,
        expected_indices,
        folds,
        class_column=class_column,
    )


__all__ = [
    "GroupedSplitError",
    "assert_group_disjoint",
    "group_id_for_row",
    "grouped_stratified_holdout",
    "grouped_stratified_kfolds",
    "stratified_group_holdout",
    "stratified_group_kfolds",
    "stratified_group_real_validation_split",
    "validate_group_fold_coverage",
    "validate_grouped_partitions",
]

"""Fetch and prepare feature and target DataFrames from the UCI repository."""

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeGuard

import pandas as pd

from mlkit.data.deduplication import deduplicate_prepared_data
from mlkit.data.ingestion import normalize_strings
from mlkit.use_cases import UseCase

logger = logging.getLogger(__name__)

Scalar = str | int | float | bool


def fetch_uci_snapshot(dataset_id: int, snapshot_dir: Path) -> None:
    """Fetch and persist a UCI dataset's feature, target and metadata tables.

    Args:
        dataset_id: UCI repository dataset identifier.
        snapshot_dir: Directory receiving the snapshot files.

    Raises:
        ValueError: If the returned feature and target tables are not aligned.
    """
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=dataset_id)
    features, targets = dataset.data.features, dataset.data.targets
    if (
        not isinstance(features, pd.DataFrame)
        or not isinstance(targets, pd.DataFrame)
        or features.empty
        or targets.empty
        or not features.index.equals(targets.index)
        or not features.columns.is_unique
        or not targets.columns.is_unique
    ):
        raise ValueError(
            "UCI features and targets must be nonempty aligned DataFrames."
        )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(snapshot_dir / "features.parquet", index=False)
    targets.to_parquet(snapshot_dir / "targets.parquet", index=False)
    (snapshot_dir / "metadata.json").write_text(
        json.dumps(dataset.metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    dataset.variables.to_json(
        snapshot_dir / "variables.json", orient="records", indent=2
    )
    logger.info("Saved UCI dataset %s with %s rows.", dataset_id, len(features))


def prepare_uci_snapshot(
    snapshot_dir: Path,
    output_path: Path,
    *,
    target_mappings: Mapping[str, Mapping[Scalar, Scalar]] | None = None,
    use_case: UseCase | None = None,
    deduplicate: bool = False,
) -> None:
    """Store UCI snapshot data in a combined Parquet file,
    optionally mapping targets and validating it.

    Args:
        snapshot_dir: Directory containing the saved feature and target tables.
        output_path: Destination Parquet file.
        target_mappings: Explicit value mappings keyed by normalized target name.
        use_case: Optional single-binary-target contract for validation.
        deduplicate: Remove exact duplicate source rows after validation.

    Raises:
        ValueError: If the snapshot, mappings, or use-case targets are invalid.
    """
    features = pd.read_parquet(snapshot_dir / "features.parquet")
    targets = pd.read_parquet(snapshot_dir / "targets.parquet")
    if features.empty or targets.empty or len(features) != len(targets):
        raise ValueError("UCI features and targets must be nonempty and aligned.")
    if not all(isinstance(name, str) for name in (*features.columns, *targets.columns)):
        raise ValueError("UCI feature and target column names must be strings.")
    features.columns = normalize_strings(features.columns)
    targets.columns = normalize_strings(targets.columns)
    if not features.columns.is_unique or not targets.columns.is_unique:
        raise ValueError("Normalized UCI feature and target names must be unique.")
    if set(features.columns).intersection(targets.columns):
        raise ValueError("UCI feature and target names must not overlap.")

    source = (
        pd.concat(
            [features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1
        )
        if deduplicate
        else None
    )
    if use_case is not None and list(targets.columns) != [use_case.target_column]:
        raise ValueError(
            f"Use case '{use_case.name}' requires one target column named "
            f"'{use_case.target_column}'."
        )
    for column, mapping in (target_mappings or {}).items():
        if column not in targets.columns:
            raise ValueError(f"Unknown UCI target column '{column}' in mapping.")
        if not mapping or any(
            not _is_scalar(value) for pair in mapping.items() for value in pair
        ):
            raise ValueError(f"Mapping for '{column}' requires non-null scalar values.")
        values = targets[column]
        if values.isna().any() or not values.isin(mapping).all():
            raise ValueError(f"Target '{column}' contains missing or unmapped values.")
        targets[column] = values.map(mapping)

    if use_case is not None:
        target = targets[use_case.target_column]
        if target.isna().any() or not target.isin([0, 1]).all():
            raise ValueError("Use-case target labels must be 0 or 1.")
    frame = pd.concat(
        [features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1
    )
    if use_case is not None:
        frame = use_case.validate(df=frame)
    if source is not None:
        frame = deduplicate_prepared_data(source, frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path=output_path, index=False)
    logger.info("Prepared UCI snapshot at '%s' with %s rows.", output_path, len(frame))


def _is_scalar(value: object) -> TypeGuard[Scalar]:
    return type(value) in {str, int, bool} or (
        type(value) is float and math.isfinite(value)
    )


def parse_target_mappings(
    entries: Sequence[Sequence[str]],
) -> dict[str, dict[Scalar, Scalar]]:
    """Parse target mappings containing JSON-encoded scalar values.

    Args:
        entries: Target name, source value, and result value for each mapping.

    Returns:
        Mappings keyed by target name.

    Raises:
        ValueError: If values are invalid or a source value is repeated.
    """
    mappings: dict[str, dict[Scalar, Scalar]] = {}
    for column, source_json, result_json in entries:
        try:
            source = json.loads(source_json)
            result = json.loads(result_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Target mapping values must be valid JSON scalars."
            ) from exc
        if not _is_scalar(source) or not _is_scalar(result):
            raise ValueError("Target mapping values must be non-null JSON scalars.")
        mapping = mappings.setdefault(column, {})
        if source in mapping:
            raise ValueError(f"Duplicate source value for target '{column}'.")
        mapping[source] = result
    return mappings

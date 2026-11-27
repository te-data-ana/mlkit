"""Create and load reproducible training, evaluation and monitoring cohorts."""

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sklearn.model_selection import train_test_split

from mlkit.config import RuntimeSettings
from mlkit.data.ingestion import normalize_strings
from mlkit.data.storage import read_parquet_dataset
from mlkit.data.validation import validate_data
from mlkit.use_cases import UseCase

logger = logging.getLogger(__name__)
COHORTS = ("fit", "validation", "test", "monitoring")
Fraction = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Count = Annotated[int, Field(strict=True, ge=0)]


class SplitMetadata(BaseModel):
    """Requested and realized sizes and temporal bounds of prepared cohorts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["temporal", "stratified_random"]
    fit_fraction: Fraction
    validation_fraction: Fraction
    test_fraction: Fraction
    monitoring_fraction: Fraction
    fit_actual_fraction: Fraction
    validation_actual_fraction: Fraction
    test_actual_fraction: Fraction
    monitoring_actual_fraction: Fraction
    fit_rows: Count
    validation_rows: Count
    test_rows: Count
    monitoring_rows: Count
    random_state: Count | None = None
    fit_start: str | None = None
    fit_end: str | None = None
    validation_start: str | None = None
    validation_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    monitoring_start: str | None = None
    monitoring_end: str | None = None

    @model_validator(mode="after")
    def validate_metadata(self) -> Self:
        """Check consistency of the requested and realized split."""
        requested_fractions = [getattr(self, f"{name}_fraction") for name in COHORTS]
        requested_counts = [getattr(self, f"{name}_rows") for name in COHORTS]
        if (
            any(frac < 0 for frac in requested_fractions)
            or abs(sum(requested_fractions) - 1) > 1e-9
            or any(count < 0 for count in requested_counts)
            or any((requested_fractions[i] == 0) for i in [0, 2])
            or any(
                (requested_fractions[i] == 0) != (requested_counts[i] == 0)
                for i in [1, 3]
            )
        ):
            raise ValueError("Invalid requested fractions or cohort counts.")
        total = sum(requested_counts)
        previous_end: pd.Timestamp | None = None
        for name, count in zip(COHORTS, requested_counts):
            if abs(getattr(self, f"{name}_actual_fraction") - count / total) > 1e-9:
                raise ValueError(f"Actual fraction does not match {name} row count.")
            start, end = getattr(self, f"{name}_start"), getattr(self, f"{name}_end")
            if self.strategy == "temporal":
                if count and (not start or not end):
                    raise ValueError(f"Temporal {name} requires timestamp bounds.")
                if not count and (start is not None or end is not None):
                    raise ValueError(f"Empty {name} cannot have timestamp bounds.")
                if start and end:
                    try:
                        first = pd.Timestamp(start)
                        last = pd.Timestamp(end)
                        if (
                            first > last
                            or (previous_end is not None and previous_end >= first)
                            or pd.isna(first)
                            or pd.isna(last)
                        ):
                            raise ValueError(
                                "Temporal cohort bounds must be ordered and disjoint."
                            )
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid temporal bounds for {name}: {exc}"
                        ) from exc
                    previous_end = last
            elif start is not None or end is not None:
                raise ValueError("Random splits cannot have temporal bounds.")
        if (self.strategy == "stratified_random") != (self.random_state is not None):
            raise ValueError("A random seed is required only for random splits.")
        return self


def load_split_metadata(path: Path) -> SplitMetadata:
    """Read and validate split metadata.

    Args:
        path: Metadata JSON file.

    Returns:
        Validated metadata for all cohorts.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If its JSON or metadata is invalid.
    """
    try:
        return SplitMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ValueError(f"Invalid split metadata '{path}': {exc}") from exc


def load_and_validate_training_splits(
    split_name: str = "default",
    split_dir: Path | None = None,
    *,
    use_case: UseCase,
    metadata: SplitMetadata,
    settings: RuntimeSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load validated fit, validation and test cohorts and check their counts.

    Args:
        split_name: Prefix of the prepared cohort files.
        split_dir: Directory containing the files; defaults to the data directory.
        use_case: Schema and time-column contract.
        metadata: Validated metadata containing expected counts.
        settings: Runtime settings used when no split directory is supplied.

    Returns:
        Fit, validation and test dataframes, in that order.

    Raises:
        FileNotFoundError: If a cohort file does not exist.
        ValueError: If its data or row count is invalid.
    """
    resolved_dir = split_dir or (settings or RuntimeSettings()).data_dir / "splits"
    parts = []
    for name in COHORTS[:3]:
        path = resolved_dir / f"{split_name}_{name}.parquet"
        logger.info("Loading %s split from '%s'.", name, path)
        df = pd.read_parquet(path)
        df.columns = normalize_strings(df.columns)
        df = validate_data(df.convert_dtypes(), use_case)
        if len(df) != getattr(metadata, f"{name}_rows"):
            raise ValueError(f"Split metadata count does not match '{path}'.")
        parts.append(df)
    return parts[0], parts[1], parts[2]


def _fractions(
    validation_fraction: float, test_fraction: float, monitoring_fraction: float
) -> dict[str, Decimal]:
    values = (validation_fraction, test_fraction, monitoring_fraction)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise ValueError("Split fractions must be finite numbers.")
    validation, test, monitoring = (Decimal(str(value)) for value in values)
    if (
        not all(value.is_finite() for value in (validation, test, monitoring))
        or validation < 0
        or test <= 0
        or monitoring < 0
        or validation + test + monitoring >= 1
    ):
        raise ValueError(
            "Test fraction must be positive, validation and monitoring fractions "
            "nonnegative, and their sum less than one."
        )
    return {
        "fit": 1 - validation - test - monitoring,
        "validation": validation,
        "test": test,
        "monitoring": monitoring,
    }


def _random_parts(
    df: pd.DataFrame, target: str, fractions: dict[str, Decimal], random_state: int
) -> dict[str, pd.DataFrame]:
    quotas = {name: fraction * len(df) for name, fraction in fractions.items()}
    sizes = {name: int(quota) for name, quota in quotas.items()}
    remainder = len(df) - sum(sizes.values())
    order = sorted(COHORTS, key=lambda name: -(quotas[name] - sizes[name]))
    for name in order[:remainder]:
        sizes[name] += 1
    if any(sizes[name] == 0 for name in COHORTS if fractions[name] > 0):
        raise ValueError("Input is too small for the requested split fractions.")
    remaining = df
    parts = {}
    for name in ("validation", "test", "monitoring"):
        if sizes[name] == 0:
            parts[name] = df.iloc[:0].copy()
            continue
        remaining, parts[name] = train_test_split(
            remaining,
            test_size=sizes[name],
            random_state=random_state,
            stratify=remaining[target],
        )
    parts["fit"] = remaining
    return parts


def _temporal_parts(
    df: pd.DataFrame, time_column: str, fractions: dict[str, Decimal]
) -> dict[str, pd.DataFrame]:
    if time_column not in df:
        raise ValueError("Temporal splitting requires the use-case time column.")
    data = df.copy()
    data[time_column] = pd.to_datetime(data[time_column], errors="coerce", utc=True)
    if data[time_column].isna().any():
        raise ValueError("Temporal split timestamps must be valid and non-missing.")
    data = data.sort_values(time_column, kind="stable").reset_index(drop=True)
    sizes = data.groupby(time_column, sort=True).size().tolist()
    active = [name for name in COHORTS if fractions[name] > 0]
    if len(sizes) < len(active):
        raise ValueError("Temporal splitting requires more distinct timestamp groups.")
    ends = []
    count = 0
    for size in sizes:
        count += size
        ends.append(count)
    start_group = 0
    start_row = 0
    cumulative = Decimal(0)
    parts = {}
    for index, name in enumerate(active[:-1]):
        cumulative += fractions[name]
        target = cumulative * len(data)
        last_candidate = len(sizes) - (len(active) - index - 1)
        boundary = min(
            range(start_group, last_candidate),
            key=lambda group: (abs(Decimal(ends[group]) - target), ends[group]),
        )
        end_row = ends[boundary]
        parts[name] = data.iloc[start_row:end_row].copy()
        start_group, start_row = boundary + 1, end_row
    parts[active[-1]] = data.iloc[start_row:].copy()
    for name in ("validation", "monitoring"):
        if fractions[name] == 0:
            parts[name] = data.iloc[:0].copy()
    return parts


def create_data_splits(
    use_case: UseCase,
    df: pd.DataFrame,
    *,
    split_name: str = "default",
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    monitoring_fraction: float = 0.1,
    random_state: int = 42,
    settings: RuntimeSettings | None = None,
) -> SplitMetadata:
    """Create all cohorts and persist their validated metadata.

    Fractions refer to the full input. Temporal boundaries preserve timestamp
    groups; random cohorts are seeded and stratified.

    Args:
        use_case: Split strategy and target/time-column contract.
        df: Input records.
        split_name: Output filename prefix.
        validation_fraction: Fraction reserved for threshold selection.
        test_fraction: Fraction reserved for evaluation.
        monitoring_fraction: Fraction reserved for monitoring; zero disables it.
        random_state: Seed for stratified random splitting.
        settings: Runtime settings for output locations.

    Returns:
        Validated requested and realized split metadata.

    Raises:
        ValueError: If fractions, input, or resulting cohorts are invalid.
    """
    fractions = _fractions(validation_fraction, test_fraction, monitoring_fraction)
    target = use_case.target_column
    if (
        target not in df
        or df[target].isna().any()
        or set(df[target].unique()) != set(use_case.target_classes)
    ):
        raise ValueError(
            f"Splitting requires a non-missing target with all classes {use_case.target_classes}."
        )
    if use_case.split_strategy == "temporal":
        if use_case.time_column is None:
            raise ValueError("Temporal splitting requires a time column.")
        parts = _temporal_parts(df, use_case.time_column, fractions)
    elif use_case.split_strategy == "stratified_random":
        if type(random_state) is not int or random_state < 0:
            raise ValueError("Random split seed must be a nonnegative integer.")
        parts = _random_parts(df, target, fractions, random_state)
    else:
        raise ValueError(f"Unsupported split strategy: {use_case.split_strategy}")
    for name in COHORTS:
        part = parts[name]
        if fractions[name] > 0 and (
            part.empty or set(part[target].unique()) != set(use_case.target_classes)
        ):
            raise ValueError(f"Nonempty {name} cohort must contain all target classes.")

    values: dict[str, object] = {"strategy": use_case.split_strategy}
    if use_case.split_strategy == "stratified_random":
        values["random_state"] = random_state
    for name in COHORTS:
        part = parts[name]
        values[f"{name}_fraction"] = float(fractions[name])
        values[f"{name}_rows"] = len(part)
        values[f"{name}_actual_fraction"] = len(part) / len(df)
        if use_case.split_strategy == "temporal" and not part.empty:
            assert use_case.time_column is not None
            values[f"{name}_start"] = part[use_case.time_column].min().isoformat()
            values[f"{name}_end"] = part[use_case.time_column].max().isoformat()
    metadata = SplitMetadata.model_validate(values)
    split_dir = (settings or RuntimeSettings()).data_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in COHORTS:
        parts[name].to_parquet(split_dir / f"{split_name}_{name}.parquet", index=False)
    (split_dir / f"{split_name}.json").write_text(
        json.dumps(metadata.model_dump(exclude_none=True), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return metadata


def create_split(
    use_case: UseCase,
    *,
    dataset_name: str | None = None,
    input_parquet: str | None = None,
    split_name: str = "default",
    start_date: str | None = None,
    end_date: str | None = None,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    monitoring_fraction: float = 0.1,
    random_state: int = 42,
    settings: RuntimeSettings | None = None,
) -> SplitMetadata:
    """Load a dataset and create all cohorts using its use-case strategy.

    Args:
        use_case: Contract defining the split strategy.
        dataset_name: Partitioned temporal dataset name.
        input_parquet: Random split input relative to the data directory.
        split_name: Output filename prefix.
        start_date: Optional inclusive temporal extraction start.
        end_date: Optional inclusive temporal extraction end.
        validation_fraction: Fraction reserved for threshold selection.
        test_fraction: Fraction reserved for evaluation.
        monitoring_fraction: Fraction reserved for monitoring; zero disables it.
        random_state: Seed for stratified random splitting.
        settings: Runtime settings for input and output locations.

    Returns:
        Validated requested and realized split metadata.

    Raises:
        ValueError: If options, input, or resulting cohorts are invalid.
        FileNotFoundError: If the input dataset does not exist.
    """
    settings = settings or RuntimeSettings()
    if use_case.split_strategy == "temporal":
        if (
            not dataset_name
            or input_parquet is not None
            or use_case.time_column is None
        ):
            raise ValueError("Temporal splitting requires a timestamped dataset name.")
        df = read_parquet_dataset(
            dataset_name=dataset_name,
            timestamp_column=use_case.time_column,
            start=start_date,
            end=end_date,
            settings=settings,
            require_valid_timestamps=True,
        )
    elif use_case.split_strategy == "stratified_random":
        if (
            not input_parquet
            or dataset_name is not None
            or start_date is not None
            or end_date is not None
        ):
            raise ValueError("Random splitting requires input_parquet and no dates.")
        df = pd.read_parquet(settings.data_dir / input_parquet)
    else:
        raise ValueError(f"Unsupported split strategy: {use_case.split_strategy}")
    return create_data_splits(
        use_case=use_case,
        df=df,
        split_name=split_name,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        monitoring_fraction=monitoring_fraction,
        random_state=random_state,
        settings=settings,
    )

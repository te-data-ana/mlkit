import json
import logging
import math
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import pandas as pd
from sklearn.model_selection import train_test_split

from mlkit.config import RuntimeSettings
from mlkit.data.ingestion import normalize_strings
from mlkit.data.storage import read_parquet_dataset
from mlkit.data.validation import validate_data
from mlkit.use_cases import UseCase

logger = logging.getLogger(__name__)


class SplitMetadata(TypedDict):
    strategy: Literal["temporal", "stratified_random"]
    train_rows: int
    test_rows: int
    train_start: NotRequired[str]
    train_end: NotRequired[str]
    test_start: NotRequired[str]
    test_end: NotRequired[str]
    monitoring_rows: NotRequired[int]
    test_fraction: NotRequired[float]
    monitoring_fraction: NotRequired[float]
    random_state: NotRequired[int]


def load_and_validate_training_splits(
    split_name: str = "default",
    split_dir: Path | None = None,
    *,
    use_case: UseCase,
    settings: RuntimeSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the prepared train and test Parquet files.

    Args:
        split_name: Prefix shared by the train and test split filenames.
        split_dir: Directory containing the split files. Defaults to the
            configured ``data/splits`` directory.
        use_case: Explicit schema and time-column contract for validation.
        settings: Runtime snapshot used when no split directory is supplied.

    Returns:
        Validated training and test DataFrames, retaining the use-case time
        column for temporal training operations.

    Raises:
        FileNotFoundError: If either split file does not exist.
        pydantic.ValidationError: If a record in either split violates the schema.
        ValueError: If required feature columns are missing or columns repeat.
    """
    resolved_split_dir = (
        split_dir
        if split_dir is not None
        else (settings or RuntimeSettings()).data_dir / "splits"
    )
    split_data: dict[str, pd.DataFrame] = {}

    for split in ("train", "test"):
        split_file = resolved_split_dir / f"{split_name}_{split}.parquet"
        logger.info("Loading %s split from '%s'.", split, split_file)
        df = pd.read_parquet(split_file)
        df.columns = normalize_strings(df.columns)
        split_data[split] = validate_data(df.convert_dtypes(), use_case)

    return split_data["train"], split_data["test"]


def _write_split(
    parts: dict[str, pd.DataFrame],
    metadata: SplitMetadata,
    split_name: str,
    settings: RuntimeSettings | None,
) -> SplitMetadata:
    """Write split DataFrames to Parquet files and metadata to JSON file."""
    split_dir = (settings or RuntimeSettings()).data_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_parquet(split_dir / f"{split_name}_{name}.parquet", index=False)
    (split_dir / f"{split_name}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _temporal_bounds(
    start_date: str | pd.Timestamp,
    split_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Check and return start, split, end as timestamps."""
    start, split, end = map(pd.Timestamp, (start_date, split_date, end_date))
    if pd.isna(start) or pd.isna(split) or pd.isna(end):
        raise ValueError("Split dates must be valid and non-missing.")
    if start >= split:
        raise ValueError("'split_date' must be later than 'start_date'.")
    if split > end:
        raise ValueError("'split_date' must be before or equal to 'end_date'.")
    return start, split, end


def create_time_based_split(
    use_case: UseCase,
    df: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    split_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    split_name: str = "default",
    settings: RuntimeSettings | None = None,
) -> SplitMetadata:
    """Write an out-of-time train/test split and return its metadata.

    Args:
        use_case: Contract providing the timestamp column.
        df: Input records.
        start_date: Inclusive start of the extraction window.
        split_date: First timestamp in the test cohort.
        end_date: Inclusive end of the extraction window.
        split_name: Prefix for output files.
        settings: Runtime snapshot used for output locations.

    Returns:
        JSON-compatible split metadata.

    Raises:
        ValueError: If the time column, date bounds, or cohorts are invalid.
    """
    start, split, end = _temporal_bounds(start_date, split_date, end_date)
    time_column = use_case.time_column
    if time_column is None or time_column not in df:
        raise ValueError("Temporal splitting requires the use-case time column.")
    data = df.copy()
    data[time_column] = pd.to_datetime(data[time_column])
    data = data.loc[data[time_column].between(start, end, inclusive="both")]
    train = data.loc[data[time_column] < split].reset_index(drop=True)
    test = data.loc[data[time_column] >= split].reset_index(drop=True)
    if train.empty:
        raise ValueError("Training split is empty.")
    if test.empty:
        raise ValueError("Test split is empty.")
    metadata = SplitMetadata(
        strategy="temporal",
        train_rows=len(train),
        test_rows=len(test),
        train_start=train[time_column].min().isoformat(),
        train_end=train[time_column].max().isoformat(),
        test_start=test[time_column].min().isoformat(),
        test_end=test[time_column].max().isoformat(),
    )
    return _write_split(
        parts={"train": train, "test": test},
        metadata=metadata,
        split_name=split_name,
        settings=settings,
    )


def create_stratified_split(
    use_case: UseCase,
    df: pd.DataFrame,
    *,
    split_name: str = "default",
    test_fraction: float = 0.1,
    monitoring_fraction: float = 0.1,
    random_state: int = 42,
    settings: RuntimeSettings | None = None,
) -> SplitMetadata:
    """Write seeded stratified train/test/monitoring cohorts.

    Args:
        use_case: Contract providing the binary target column.
        df: Input records.
        split_name: Prefix for output files.
        test_fraction: Fraction of all records reserved for testing.
        monitoring_fraction: Fraction of all records reserved for monitoring.
        random_state: Seed for both partitioning steps.
        settings: Runtime snapshot used for output locations.

    Returns:
        JSON-compatible split metadata.

    Raises:
        ValueError: If fractions, target, or stratified cohorts are invalid.
    """
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        for value in (test_fraction, monitoring_fraction)
    ):
        raise ValueError("Split fractions must be finite numbers.")
    held_out_decimal = Decimal(str(test_fraction)) + Decimal(str(monitoring_fraction))
    held_out = float(held_out_decimal)
    if test_fraction <= 0 or monitoring_fraction < 0 or held_out >= 1:
        raise ValueError(
            "Test fraction must be positive, monitoring fraction nonnegative, "
            "and their sum less than one."
        )
    target_column = use_case.target_column
    if target_column not in df or df[target_column].isna().any():
        raise ValueError("A non-missing target column is required for splitting.")
    if set(df[target_column].unique()) != {0, 1}:
        raise ValueError("Stratified splitting requires both binary classes.")
    train, remainder = train_test_split(
        df,
        test_size=held_out,
        random_state=random_state,
        stratify=df[target_column],
    )
    if monitoring_fraction:
        test, monitoring = train_test_split(
            remainder,
            test_size=float(Decimal(str(monitoring_fraction)) / held_out_decimal),
            random_state=random_state,
            stratify=remainder[target_column],
        )
    else:
        test, monitoring = remainder, df.iloc[:0].copy()
    if any(
        set(part[target_column].unique()) != {0, 1}
        for part in (train, test, monitoring)
        if not part.empty
    ):
        raise ValueError("Each nonempty stratified cohort must contain both classes.")
    metadata = SplitMetadata(
        strategy="stratified_random",
        train_rows=len(train),
        test_rows=len(test),
        monitoring_rows=len(monitoring),
        test_fraction=float(test_fraction),
        monitoring_fraction=float(monitoring_fraction),
        random_state=random_state,
    )
    return _write_split(
        parts={"train": train, "test": test, "monitoring": monitoring},
        metadata=metadata,
        split_name=split_name,
        settings=settings,
    )


def create_split(
    use_case: UseCase,
    *,
    dataset_name: str | None = None,
    input_parquet: str | None = None,
    split_name: str = "default",
    start_date: str | None = None,
    split_date: str | None = None,
    end_date: str | None = None,
    test_fraction: float | None = None,
    monitoring_fraction: float | None = None,
    random_state: int | None = None,
    settings: RuntimeSettings | None = None,
) -> SplitMetadata:
    """Load input data and create a split using the use-case strategy.

    Args:
        use_case: Contract defining the split strategy and columns.
        dataset_name: Partitioned temporal dataset name.
        input_parquet: Random split input relative to the data directory.
        split_name: Prefix for output files.
        start_date: Inclusive temporal extraction start.
        split_date: First temporal test timestamp.
        end_date: Inclusive temporal extraction end.
        test_fraction: Random test fraction, defaulting to 0.1.
        monitoring_fraction: Random monitoring fraction, defaulting to 0.1.
        random_state: Seed for random splitting, defaulting to 42.
        settings: Runtime snapshot used for input and output locations.

    Returns:
        JSON-compatible split metadata.

    Raises:
        ValueError: If arguments are incompatible with the selected strategy.
    """
    settings = settings or RuntimeSettings()
    if use_case.split_strategy == "temporal":
        if not dataset_name or not start_date or not split_date or not end_date:
            raise ValueError(
                "Temporal splitting requires 'dataset_name' and all three dates."
            )
        if any(
            value is not None
            for value in (
                input_parquet,
                test_fraction,
                monitoring_fraction,
                random_state,
            )
        ):
            raise ValueError(
                "Parquet, seed and fractions are only for random splitting."
            )
        if use_case.time_column is None:
            raise ValueError("Temporal splitting requires a time column.")
        start, split, end = _temporal_bounds(start_date, split_date, end_date)
        df = read_parquet_dataset(
            dataset_name=dataset_name,
            timestamp_column=use_case.time_column,
            start=start,
            end=end,
            settings=settings,
        )
        return create_time_based_split(
            use_case=use_case,
            df=df,
            start_date=start,
            split_date=split,
            end_date=end,
            split_name=split_name,
            settings=settings,
        )
    if use_case.split_strategy == "stratified_random":
        if not input_parquet:
            raise ValueError("Random splitting requires 'input_parquet'.")
        if any(
            value is not None
            for value in (dataset_name, start_date, split_date, end_date)
        ):
            raise ValueError(
                "Dataset and date options are only for temporal splitting."
            )
        df = pd.read_parquet(settings.data_dir / input_parquet)
        return create_stratified_split(
            use_case=use_case,
            df=df,
            split_name=split_name,
            test_fraction=test_fraction if test_fraction is not None else 0.1,
            monitoring_fraction=monitoring_fraction
            if monitoring_fraction is not None
            else 0.1,
            random_state=random_state if random_state is not None else 42,
            settings=settings,
        )
    raise ValueError(f"Unsupported split strategy: {use_case.split_strategy}")

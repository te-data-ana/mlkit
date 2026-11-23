import logging
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from mlkit.config import RuntimeSettings
from mlkit.data.jsonl_logs import read_jsonl_log
from mlkit.data.storage import write_parquet_dataset
from mlkit.data.validation import validate_data
from mlkit.use_cases import UseCase

logger = logging.getLogger(__name__)


def normalize_strings(seq: Sequence[str]) -> Sequence[str]:
    """Normalize whitespace and camel-case names to lowercase underscores.

    Acronym boundaries are retained, such as ``URLLength`` to ``url_length``.

    Args:
        seq: Column names to normalize.

    Returns:
        Normalized names in their original order.
    """
    normalized = []
    for name in seq:
        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name.strip())
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        normalized.append(re.sub(r"\s+", "_", name).lower())
    return normalized


def load_raw_data(
    file_name: str,
    index_col: str | None = None,
    data_dir: Path | None = None,
    drop_columns: list[str] | None = None,
    *,
    settings: RuntimeSettings | None = None,
) -> pd.DataFrame:
    """Load and normalize a raw CSV file from the configured data directory.

    Args:
        file_name: Name of the CSV file in ``data_dir``.
        index_col: Optional column name to use as the DataFrame index.
        data_dir: Directory containing the CSV file.
        drop_columns: Optional list of column names to drop from loaded data.
        settings: Runtime snapshot used when no data directory is supplied.

    Returns:
        DataFrame with normalized column names, nullable dtypes, the optional
        index applied, and rows containing only missing values removed.

    Raises:
        FileNotFoundError: If the requested CSV file does not exist.
    """
    resolved_data_dir = (
        data_dir
        if data_dir is not None
        else (settings or RuntimeSettings()).data_dir / "raw"
    )

    try:
        logger.info("Loading raw dataset '%s' from '%s'.", file_name, resolved_data_dir)
        df = pd.read_csv(resolved_data_dir / file_name)
        df.columns = normalize_strings(df.columns)
        df = df.convert_dtypes()
        if drop_columns:
            df = df.drop(
                columns=normalize_strings(drop_columns),
                errors="ignore",
            )
        if index_col:
            df = df.set_index(index_col)
            logger.info("Set index column '%s' on dataset '%s'.", index_col, file_name)
        prior_rows = len(df)
        df = df.dropna(how="all")
        dropped_rows = prior_rows - len(df)
        if dropped_rows:
            logger.warning(
                "Dropped %d all-missing rows from dataset '%s'.",
                dropped_rows,
                file_name,
            )
        logger.info(
            "Loaded dataset '%s' with %d rows and %d columns.",
            file_name,
            len(df),
            len(df.columns),
        )
        return df
    except Exception:
        logger.exception("Failed to load raw dataset '%s'.", file_name)
        raise


def combine_sample_predictions(
    data_dir: Path | None = None, *, settings: RuntimeSettings | None = None
) -> pd.DataFrame:
    """Combine sample prediction files into a single DataFrame.

    Reads all CSV files matching the pattern
    ``dddd_sample_predictions.csv`` from the specified directory,
    adds a ``run_id`` column derived from the four-digit filename
    prefix, and concatenates the files into a single DataFrame.

    Args:
        data_dir: Directory containing sample prediction CSV files. If
            not provided, the default output data directory from the
            runtime settings is used.
        settings: Runtime snapshot used when no data directory is supplied.

    Returns:
        A DataFrame containing the combined prediction records from all
        matching files, including a ``run_id`` column identifying the
        source file of each record.
    """
    resolved_data_dir = (
        data_dir if data_dir is not None else (settings or RuntimeSettings()).output_dir
    )

    dfs = []

    for path in resolved_data_dir.glob("[0-9][0-9][0-9][0-9]_sample_predictions.csv"):
        df = pd.read_csv(path)
        df["run_id"] = path.stem[:4]
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def ingest_sample_data(
    dataset_name: str,
    timestamp_column: str,
    use_case: UseCase,
    *,
    settings: RuntimeSettings | None = None,
) -> None:
    """
    Combine sample prediction files, validate the resulting data,
    and append it as a partitioned Parquet dataset.

    Args:
        dataset_name:
            Name of the target dataset directory.
        timestamp_column:
            Name of the timestamp column used for partitioning.
        use_case: Data contract to validate before writing.
        settings: Runtime snapshot used for input and output locations.
    """
    settings = settings or RuntimeSettings()
    df = combine_sample_predictions(settings=settings)
    df = validate_data(df=df, use_case=use_case)

    write_parquet_dataset(
        df=df,
        dataset_name=dataset_name,
        timestamp_column=timestamp_column,
        overwrite_partitions=True,
        settings=settings,
    )


def ingest_jsonl_log(
    dataset_name: str,
    timestamp_column: str,
    jsonl_path: Path,
    *,
    settings: RuntimeSettings | None = None,
) -> None:
    """Ingest JSONL logs into a partitioned dataset.

    Reads records from a JSONL log and writes them to a partitioned
    Parquet dataset. Existing partitions are overwritten when records
    for the same partition are ingested again.

    If no valid records are found, the function logs an informational
    message and returns without writing any data.

    Args:
        dataset_name: Name of the target dataset directory.
        timestamp_column: Name of the timestamp column used for
            partitioning the dataset.
        jsonl_path: Path to the source JSONL log file.
        settings: Runtime snapshot used for the output dataset.

    Raises:
        KeyError: If ``timestamp_column`` is not present in the
            prediction data.
    """

    df = read_jsonl_log(jsonl_path)

    if df.empty:
        logger.warning(
            "No valid records found in %s.",
            jsonl_path,
        )

    write_parquet_dataset(
        df=df,
        dataset_name=dataset_name,
        timestamp_column=timestamp_column,
        overwrite_partitions=True,
        settings=settings,
    )


def ingest_prediction_logs(
    dataset_name: str,
    timestamp_column: str,
    jsonl_path: Path | None = None,
    *,
    settings: RuntimeSettings | None = None,
) -> None:
    """Ingest API prediction logs into a partitioned dataset.

    Uses the configured prediction log path when ``jsonl_path`` is not provided.
    """
    settings = settings or RuntimeSettings()

    ingest_jsonl_log(
        dataset_name=dataset_name,
        timestamp_column=timestamp_column,
        jsonl_path=jsonl_path or settings.prediction_log_path,
        settings=settings,
    )


def ingest_error_logs(
    dataset_name: str,
    timestamp_column: str,
    jsonl_path: Path | None = None,
    *,
    settings: RuntimeSettings | None = None,
) -> None:
    """Ingest API error logs into a partitioned dataset.

    Uses the configured error log path when ``jsonl_path`` is not provided.
    """
    settings = settings or RuntimeSettings()

    ingest_jsonl_log(
        dataset_name=dataset_name,
        timestamp_column=timestamp_column,
        jsonl_path=jsonl_path or settings.error_log_path,
        settings=settings,
    )

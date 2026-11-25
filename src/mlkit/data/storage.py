"""
Utilities for storing and retrieving Parquet datasets.

Timestamped datasets are partitioned by year and month to enable efficient
time-range queries; timestamp-free datasets use unpartitioned Parquet.
"""

import shutil

import pandas as pd

from mlkit.config import RuntimeSettings

type PartitionFilter = list[list[tuple[str, str, int]]]


def write_parquet_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    timestamp_column: str | None = None,
    overwrite_partitions: bool = False,
    *,
    settings: RuntimeSettings | None = None,
) -> None:
    """
    Write a timestamp-partitioned or unpartitioned Parquet dataset.

    When supplied, the timestamp column is converted to datetime and used
    to derive ``year`` and ``month`` Hive-style partition columns.

    If ``overwrite_partitions`` is ``True``, existing partitions whose
    year/month combinations are present in the input DataFrame are
    removed before writing new data. This prevents duplicate records
    when re-ingesting data for an existing period.

    Args:
        df:
            DataFrame to persist. Empty timestamped data creates an empty
            directory; empty timestamp-free data retains its Parquet schema.

        dataset_name:
            Name of the output dataset directory.

        timestamp_column:
            Name of the timestamp column used to derive partition keys.
            The column must be convertible to datetime via ``pd.to_datetime``.
            If None, write a single Parquet file in the dataset directory,
            replacing a previous timestamp-free output.

        overwrite_partitions:
            Whether to replace existing year/month partitions before
            writing new data. Defaults to ``False``.
        settings: Runtime snapshot used to locate the dataset.

    Returns:
        None.

    Raises:
        ValueError: If the existing dataset has an incompatible layout.
    """

    settings = settings or RuntimeSettings()
    dataset_dir = settings.data_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    unpartitioned_file = dataset_dir / "data.parquet"
    if timestamp_column is None:
        if any(path != unpartitioned_file for path in dataset_dir.iterdir()):
            raise ValueError(f"Dataset has an incompatible layout: {dataset_dir}")
        df.to_parquet(unpartitioned_file, index=False)
        return
    if unpartitioned_file.exists():
        raise ValueError(f"Dataset has an incompatible layout: {dataset_dir}")

    if df.empty:
        return

    df = df.copy()

    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    df["year"] = df[timestamp_column].dt.year
    df["month"] = df[timestamp_column].dt.month

    if overwrite_partitions:
        year_month_comb = df[["year", "month"]].drop_duplicates().values
        for year, month in year_month_comb:
            partition_dir = dataset_dir / f"year={year}" / f"month={month}"

            if partition_dir.exists():
                shutil.rmtree(partition_dir)

    df.to_parquet(
        path=dataset_dir,
        partition_cols=["year", "month"],
        index=False,
    )


def _partition_filters(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> PartitionFilter:
    """
    Create Parquet partition filters for a date range.

    Builds Hive-style year/month partition filters that allow
    ``pd.read_parquet`` to read only the partitions intersecting the
    requested date range.

    Args:
        start:
            Inclusive start date.

        end:
            Inclusive end date.

    Returns:
        A nested filter structure suitable for the ``filters`` argument
        of ``pd.read_parquet``.

    Raises:
        ValueError:
            If ``start`` is later than ``end``.
    """

    if start > end:
        msg = "'start' must be before or equal to 'end'."
        raise ValueError(msg)

    months = pd.period_range(start=start, end=end, freq="M")

    filters: PartitionFilter = []

    for month in months:
        filters.append(
            [
                ("year", "==", month.year),
                ("month", "==", month.month),
            ]
        )

    return filters


def read_parquet_dataset(
    dataset_name: str,
    timestamp_column: str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    settings: RuntimeSettings | None = None,
    require_valid_timestamps: bool = False,
) -> pd.DataFrame:
    """
    Read rows from a Parquet file or dataset.

    When ``start`` and ``end`` are provided, the function first applies
    year/month partition pruning to limit the number of Parquet files read
    and then applies an exact timestamp filter. A date-only string supplied
    as ``end`` includes the entire end calendar day; explicit timestamps are
    treated as exact inclusive bounds.

    When both ``start`` and ``end`` are ``None``, the entire dataset is
    loaded without partition pruning or timestamp filtering. When
    ``require_valid_timestamps`` is true, the timestamp column is read from
    the entire dataset before applying an optional date filter.

    Args:
        dataset_name:
            Name of the partitioned dataset directory.

        timestamp_column:
            Name of the timestamp column used for datetime conversion and
            filtering. If None, read without timestamp conversion or filtering.

        start:
            Inclusive start date of the requested time window. Must be
            provided together with ``end``. If both ``start`` and ``end``
            are ``None``, the entire dataset is loaded. Defaults to
            ``None``.

        end:
            Inclusive end bound of the requested time window. A date-only
            string includes that entire calendar day. Must be provided
            together with ``start``. If both ``start`` and ``end`` are
            ``None``, the entire dataset is loaded. Defaults to ``None``.
        settings: Runtime snapshot used to locate the dataset.
        require_valid_timestamps: Reject missing or invalid timestamps before
            filtering, including those outside the selected partitions.

    Returns:
        A DataFrame containing either:

        * all rows in the dataset if ``start`` and ``end`` are ``None``; or
        * all rows whose timestamp falls within the specified date range.

    Raises:
        FileNotFoundError:
            If the dataset directory does not exist.

        ValueError:
            If only one of ``start`` or ``end`` is provided, or if
            ``start`` is later than ``end``, or if dates are supplied
            without a timestamp column.
    """

    settings = settings or RuntimeSettings()
    dataset_dir = settings.data_dir / dataset_name

    if not dataset_dir.exists():
        msg = f"Dataset does not exist: {dataset_dir}"
        raise FileNotFoundError(msg)

    if timestamp_column is None:
        if start is not None or end is not None or require_valid_timestamps:
            raise ValueError("Date bounds require a timestamp column.")
        return pd.read_parquet(dataset_dir).reset_index(drop=True)

    if require_valid_timestamps:
        timestamps = pd.read_parquet(dataset_dir, columns=[timestamp_column])
        if pd.to_datetime(timestamps[timestamp_column], errors="coerce").isna().any():
            raise ValueError("Dataset timestamps must be valid and non-missing.")

    if start is None and end is None:
        df = pd.read_parquet(dataset_dir)
        if df.empty:
            df[timestamp_column] = []
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    else:
        if start is None or end is None:
            msg = "'start' and 'end' must either both be provided or both be None."
            raise ValueError(msg)

        start = pd.Timestamp(start)
        end_is_date_only = (
            isinstance(end, str) and pd.Timestamp(end).strftime("%Y-%m-%d") == end
        )
        end = pd.Timestamp(end)
        if end_is_date_only:
            end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

        partition_filters = _partition_filters(start, end)

        df = pd.read_parquet(dataset_dir, filters=partition_filters)
        if df.empty:
            df[timestamp_column] = []
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        date_filter = df[timestamp_column].between(start, end, inclusive="both")

        df = df[date_filter]

    return df.reset_index(drop=True)

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mlkit.data.storage import (
    _partition_filters,
    read_parquet_dataset,
    write_parquet_dataset,
)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "mlkit.data.storage.RuntimeSettings",
        lambda: SimpleNamespace(data_dir=tmp_path),
    )
    return tmp_path


@pytest.mark.unit
def test_write_and_read_parquet_dataset_filters_inclusive_dates(
    data_dir: Path,
) -> None:
    df = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                [
                    "2026-01-31",
                    "2026-02-01",
                    "2026-02-28",
                    "2026-02-28 23:59:59.999999",
                    "2026-03-01",
                ],
                format="mixed",
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )

    write_parquet_dataset(
        df=df,
        dataset_name="events",
        timestamp_column="reference_date",
    )

    dataset_dir = data_dir / "events"
    assert (dataset_dir / "year=2026" / "month=1").is_dir()
    assert (dataset_dir / "year=2026" / "month=2").is_dir()
    assert (dataset_dir / "year=2026" / "month=3").is_dir()

    result = read_parquet_dataset(
        dataset_name="events",
        timestamp_column="reference_date",
        start="2026-02-01",
        end="2026-02-28",
    )

    assert result["value"].tolist() == [2, 3, 4]
    assert result["reference_date"].tolist() == list(
        pd.to_datetime(
            ["2026-02-01", "2026-02-28", "2026-02-28 23:59:59.999999"],
            format="mixed",
        )
    )


@pytest.mark.unit
def test_write_parquet_dataset_replaces_only_matching_partitions(
    data_dir: Path,
) -> None:
    write_parquet_dataset(
        df=pd.DataFrame(
            {
                "reference_date": pd.to_datetime(["2026-01-05", "2026-02-05"]),
                "value": ["old-january", "keep-february"],
            }
        ),
        dataset_name="events",
        timestamp_column="reference_date",
    )
    write_parquet_dataset(
        df=pd.DataFrame(
            {
                "reference_date": pd.to_datetime(["2026-01-20"]),
                "value": ["new-january"],
            }
        ),
        dataset_name="events",
        timestamp_column="reference_date",
        overwrite_partitions=True,
    )

    result = read_parquet_dataset(
        dataset_name="events",
        timestamp_column="reference_date",
    )

    assert set(result["value"]) == {"new-january", "keep-february"}


@pytest.mark.unit
def test_write_parquet_dataset_writes_empty_dataframe(data_dir: Path) -> None:
    empty_df = pd.DataFrame()
    write_parquet_dataset(
        df=empty_df,
        dataset_name="empty",
        timestamp_column="reference_date",
    )

    assert (data_dir / "empty").exists()

    result = read_parquet_dataset(
        dataset_name="empty",
        timestamp_column="reference_date",
    )
    empty_df["reference_date"] = []
    empty_df["reference_date"] = pd.to_datetime(empty_df["reference_date"])
    assert result.equals(empty_df)


@pytest.mark.unit
def test_timestamp_free_dataset_reads_file_and_replaces_directory(
    data_dir: Path,
) -> None:
    source = pd.DataFrame({"value": [1, 2]})
    source.to_parquet(data_dir / "input.parquet", index=False)
    pd.testing.assert_frame_equal(read_parquet_dataset("input.parquet"), source)

    write_parquet_dataset(source, "predictions")
    pd.testing.assert_frame_equal(read_parquet_dataset("predictions"), source)
    replacement = pd.DataFrame({"value": [3]})
    write_parquet_dataset(replacement, "predictions")
    pd.testing.assert_frame_equal(
        pd.read_parquet(data_dir / "predictions"), replacement
    )
    empty = source.iloc[:0]
    write_parquet_dataset(empty, "predictions")
    pd.testing.assert_frame_equal(read_parquet_dataset("predictions"), empty)


@pytest.mark.unit
def test_timestamp_free_read_rejects_date_bounds(data_dir: Path) -> None:
    pd.DataFrame({"value": [1]}).to_parquet(data_dir / "input.parquet")
    with pytest.raises(ValueError, match="require a timestamp"):
        read_parquet_dataset("input.parquet", start="2026-01-01", end="2026-02-01")


@pytest.mark.unit
def test_dataset_writes_reject_incompatible_layouts(data_dir: Path) -> None:
    timestamped = pd.DataFrame(
        {"reference_date": pd.to_datetime(["2026-01-01"]), "value": [1]}
    )
    write_parquet_dataset(timestamped, "temporal", "reference_date")
    with pytest.raises(ValueError, match="incompatible layout"):
        write_parquet_dataset(pd.DataFrame({"value": [2]}), "temporal")
    write_parquet_dataset(pd.DataFrame({"value": [2]}), "untimed")
    with pytest.raises(ValueError, match="incompatible layout"):
        write_parquet_dataset(timestamped, "untimed", "reference_date")
    assert read_parquet_dataset("temporal", "reference_date")["value"].tolist() == [1]
    assert read_parquet_dataset("untimed")["value"].tolist() == [2]


@pytest.mark.unit
def test_partition_filters_include_every_month_in_range() -> None:
    assert _partition_filters(
        pd.Timestamp("2025-12-15"),
        pd.Timestamp("2026-02-01"),
    ) == [
        [("year", "==", 2025), ("month", "==", 12)],
        [("year", "==", 2026), ("month", "==", 1)],
        [("year", "==", 2026), ("month", "==", 2)],
    ]


@pytest.mark.unit
def test_partition_filters_reject_reversed_dates() -> None:
    with pytest.raises(ValueError, match="before or equal"):
        _partition_filters(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-01-01"))


@pytest.mark.unit
def test_read_parquet_dataset_requires_both_date_bounds(data_dir: Path) -> None:
    (data_dir / "events").mkdir()

    with pytest.raises(ValueError, match="both be provided"):
        read_parquet_dataset(
            dataset_name="events",
            timestamp_column="reference_date",
            start="2026-01-01",
        )


@pytest.mark.unit
def test_read_parquet_dataset_rejects_missing_dataset(data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset does not exist"):
        read_parquet_dataset(
            dataset_name="missing",
            timestamp_column="reference_date",
        )

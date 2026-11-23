import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mlkit.data import splitting, storage
from mlkit.data.splitting import create_stratified_split, create_time_based_split
from mlkit.data.storage import write_parquet_dataset
from mlkit.use_cases import get_use_case

CHURN_USE_CASE = get_use_case("churn")
PHISHING_USE_CASE = get_use_case("phishing")


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings = SimpleNamespace(data_dir=tmp_path)
    monkeypatch.setattr(storage, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(splitting, "RuntimeSettings", lambda: settings)
    return tmp_path


@pytest.mark.unit
def test_create_time_based_split_writes_datasets_and_returns_metadata(
    data_dir: Path,
) -> None:
    df = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                [
                    "2025-12-01",
                    "2026-01-15",
                    "2026-02-15",
                    "2026-03-15",
                    "2026-04-15",
                    "2026-05-01",
                ]
            ),
            "value": [0, 1, 2, 3, 4, 5],
        }
    )
    metadata = create_time_based_split(
        use_case=CHURN_USE_CASE,
        df=df,
        start_date="2026-01-01",
        split_date="2026-03-15",
        end_date="2026-04-30",
        split_name="baseline",
    )

    splits_dir = data_dir / "splits"
    train = pd.read_parquet(splits_dir / "baseline_train.parquet")
    test = pd.read_parquet(splits_dir / "baseline_test.parquet")

    assert train["value"].tolist() == [1, 2]
    assert test["value"].tolist() == [3, 4]
    assert metadata == {
        "strategy": "temporal",
        "train_rows": 2,
        "test_rows": 2,
        "train_start": "2026-01-15T00:00:00",
        "train_end": "2026-02-15T00:00:00",
        "test_start": "2026-03-15T00:00:00",
        "test_end": "2026-04-15T00:00:00",
    }
    assert json.loads((splits_dir / "baseline.json").read_text()) == metadata


@pytest.mark.unit
def test_load_and_validate_training_splits_reads_and_prepares_parquet_files(
    data_dir: Path,
    generated_training_df: pd.DataFrame,
) -> None:
    split_dir = data_dir / "splits"
    split_dir.mkdir()
    df = generated_training_df.copy()
    df["Reference Date"] = pd.date_range("2026-01-01", periods=len(df))
    df.iloc[:6].to_parquet(split_dir / "baseline_train.parquet", index=False)
    df.iloc[6:].to_parquet(split_dir / "baseline_test.parquet", index=False)

    train_df, test_df = splitting.load_and_validate_training_splits(
        split_name="baseline",
        split_dir=split_dir,
        use_case=CHURN_USE_CASE,
    )

    assert len(train_df) == 6
    assert len(test_df) == len(df) - 6
    assert "reference_date" in train_df.columns
    assert "reference_date" in test_df.columns
    assert "usage_frequency" in train_df.columns
    assert "usage_frequency" in test_df.columns


@pytest.mark.unit
def test_load_and_validate_training_splits_can_preserve_temporal_column(
    data_dir: Path,
    generated_training_df: pd.DataFrame,
) -> None:
    split_dir = data_dir / "temporal-splits"
    split_dir.mkdir()
    df = generated_training_df.copy()
    df["reference_date"] = pd.date_range("2026-01-01", periods=len(df))
    df.iloc[:6].to_parquet(split_dir / "temporal_train.parquet", index=False)
    df.iloc[6:].to_parquet(split_dir / "temporal_test.parquet", index=False)

    train_df, test_df = splitting.load_and_validate_training_splits(
        split_name="temporal",
        split_dir=split_dir,
        use_case=CHURN_USE_CASE,
    )

    assert "reference_date" in train_df.columns
    assert "reference_date" in test_df.columns


@pytest.mark.unit
@pytest.mark.parametrize(
    ("start_date", "split_date", "end_date", "message"),
    [
        ("2026-02-01", "2026-02-01", "2026-04-30", "later than"),
        ("2026-01-01", "2026-05-01", "2026-04-30", "before or equal"),
    ],
)
def test_create_time_based_split_rejects_invalid_date_bounds(
    start_date: str,
    split_date: str,
    end_date: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_time_based_split(
            use_case=CHURN_USE_CASE,
            df=pd.DataFrame(),
            start_date=start_date,
            split_date=split_date,
            end_date=end_date,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        (["2026-03-01", "2026-04-01"], "Training split is empty"),
        (["2026-01-01", "2026-01-15"], "Test split is empty"),
    ],
)
def test_create_time_based_split_rejects_empty_sides(
    monkeypatch: pytest.MonkeyPatch,
    timestamps: list[str],
    message: str,
) -> None:
    df = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(timestamps),
            "value": range(len(timestamps)),
        }
    )
    with pytest.raises(ValueError, match=message):
        create_time_based_split(
            use_case=CHURN_USE_CASE,
            df=df,
            start_date="2026-01-01",
            split_date="2026-02-01",
            end_date="2026-04-30",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("test_fraction", "monitoring_fraction", "sizes"),
    [(0.1, 0.1, (80, 10, 10)), (0.2, 0.1, (70, 20, 10)), (0.2, 0.0, (80, 20, 0))],
)
def test_stratified_split_respects_fractions_and_metadata(
    data_dir: Path,
    test_fraction: float,
    monitoring_fraction: float,
    sizes: tuple[int, int, int],
) -> None:
    df = pd.DataFrame({"id": range(100), "label": [0, 1] * 50})
    metadata = create_stratified_split(
        PHISHING_USE_CASE,
        df,
        split_name="s",
        random_state=7,
        test_fraction=test_fraction,
        monitoring_fraction=monitoring_fraction,
    )
    parts = [
        pd.read_parquet(data_dir / "splits" / f"s_{name}.parquet")
        for name in ("train", "test", "monitoring")
    ]
    assert tuple(map(len, parts)) == sizes
    assert all(part["label"].mean() == 0.5 for part in parts if not part.empty)
    assert len(set.union(*(set(part["id"]) for part in parts))) == 100
    assert sum(map(len, parts)) == 100
    assert metadata == json.loads((data_dir / "splits" / "s.json").read_text())
    assert metadata["test_fraction"] == test_fraction
    assert metadata["monitoring_fraction"] == monitoring_fraction
    assert list(parts[2].columns) == list(df.columns)


@pytest.mark.unit
def test_stratified_split_uses_sklearn_two_step_rounding(data_dir: Path) -> None:
    df = pd.DataFrame({"id": range(101), "label": [0, 1] * 50 + [0]})
    metadata = create_stratified_split(
        use_case=PHISHING_USE_CASE,
        df=df,
        split_name="odd",
    )
    assert (
        metadata["train_rows"],
        metadata["test_rows"],
        metadata["monitoring_rows"],
    ) == (80, 10, 11)


@pytest.mark.unit
def test_temporal_split_keeps_inclusive_window_and_split_boundary(
    data_dir: Path,
) -> None:
    df = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                ["2025-12-31", "2026-01-01", "2026-02-01", "2026-03-01", "2026-03-02"]
            ),
            "value": list(range(5)),
        }
    )
    metadata = create_time_based_split(
        use_case=CHURN_USE_CASE,
        df=df,
        start_date="2026-01-01",
        split_date="2026-02-01",
        end_date="2026-03-01",
    )
    train = pd.read_parquet(data_dir / "splits/default_train.parquet")
    test = pd.read_parquet(data_dir / "splits/default_test.parquet")
    assert train["value"].tolist() == [1]
    assert test["value"].tolist() == [2, 3]
    assert metadata["test_end"] == "2026-03-01T00:00:00"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("test_fraction", "monitoring_fraction"),
    [(0, 0.1), (0.1, -0.1), (0.5, 0.5), (float("nan"), 0.1), (0.1, float("inf"))],
)
def test_stratified_split_rejects_invalid_fractions(
    data_dir: Path, test_fraction: float, monitoring_fraction: float
) -> None:
    df = pd.DataFrame({"label": [0, 1] * 10})
    with pytest.raises(ValueError, match="fraction"):
        create_stratified_split(
            use_case=PHISHING_USE_CASE,
            df=df,
            test_fraction=test_fraction,
            monitoring_fraction=monitoring_fraction,
        )
    assert not (data_dir / "splits").exists()


@pytest.mark.unit
def test_stratified_split_rejects_infeasible_cohorts(data_dir: Path) -> None:
    df = pd.DataFrame({"label": [0, 1] * 5})
    with pytest.raises(ValueError):
        create_stratified_split(use_case=PHISHING_USE_CASE, df=df, test_fraction=0.01)
    assert not (data_dir / "splits").exists()


@pytest.mark.unit
def test_create_split_loads_partitioned_temporal_dataset(data_dir: Path) -> None:
    write_parquet_dataset(
        df=pd.DataFrame(
            {
                "reference_date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                "value": [1, 2],
            }
        ),
        dataset_name="events",
        timestamp_column="reference_date",
    )
    metadata = splitting.create_split(
        use_case=CHURN_USE_CASE,
        dataset_name="events",
        start_date="2026-01-01",
        split_date="2026-02-01",
        end_date="2026-02-28",
    )
    assert metadata["strategy"] == "temporal"
    assert (metadata["train_rows"], metadata["test_rows"]) == (1, 1)

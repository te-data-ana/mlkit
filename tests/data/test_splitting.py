import json
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from mlkit.data import splitting, storage
from mlkit.use_cases import get_use_case

CHURN = get_use_case("churn")
PHISHING = get_use_case("phishing")
NAMES = ("fit", "validation", "test", "monitoring")


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings = SimpleNamespace(data_dir=tmp_path)
    monkeypatch.setattr(storage, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(splitting, "RuntimeSettings", lambda: settings)
    return tmp_path


def _parts(data_dir: Path, name: str = "s") -> list[pd.DataFrame]:
    return [
        pd.read_parquet(data_dir / "splits" / f"{name}_{part}.parquet")
        for part in NAMES
    ]


@pytest.mark.unit
def test_random_split_persists_all_disjoint_cohorts_and_metadata(
    data_dir: Path,
) -> None:
    df = pd.DataFrame({"id": range(100), "label": [0, 1] * 50})
    metadata = splitting.create_data_splits(
        PHISHING, df, split_name="s", random_state=7
    )
    parts = _parts(data_dir)
    assert [len(part) for part in parts] == [70, 10, 10, 10]
    ids = [set(part["id"]) for part in parts]
    assert set.union(*ids) == set(df["id"])
    assert sum(map(len, ids)) == len(set.union(*ids))
    assert all(set(part["label"]) == {0, 1} for part in parts)
    assert all(part["label"].mean() == 0.5 for part in parts)
    assert splitting.load_split_metadata(data_dir / "splits/s.json") == metadata
    assert json.loads((data_dir / "splits/s.json").read_text()) == metadata.model_dump(
        exclude_none=True
    )
    with pytest.raises(ValidationError, match="frozen"):
        metadata.fit_rows = 2
    splitting.create_data_splits(PHISHING, df, split_name="again", random_state=7)
    for part, again in zip(parts, _parts(data_dir, "again")):
        pd.testing.assert_frame_equal(part, again)


@pytest.mark.unit
def test_random_largest_remainders_and_zero_monitoring(data_dir: Path) -> None:
    df = pd.DataFrame({"id": range(101), "label": [0, 1] * 50 + [0]})
    metadata = splitting.create_data_splits(
        PHISHING, df, split_name="s", monitoring_fraction=0
    )
    assert [len(part) for part in _parts(data_dir)] == [81, 10, 10, 0]
    assert metadata.monitoring_rows == metadata.monitoring_actual_fraction == 0
    assert list(_parts(data_dir)[3].columns) == list(df.columns)


@pytest.mark.unit
def test_temporal_uses_nearest_group_boundaries_and_breaks_ties_earlier(
    data_dir: Path,
) -> None:
    sizes = [4, 2, 4, 2, 2, 2]
    df = pd.DataFrame(
        {
            "reference_date": [
                pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
                for i, size in enumerate(sizes)
                for _ in range(size)
            ],
            "churn": [0, 1] * 8,
            "id": range(16),
        }
    ).sample(frac=1, random_state=5)
    metadata = splitting.create_data_splits(
        CHURN,
        df,
        split_name="s",
        validation_fraction=0.1,
        test_fraction=0.2,
        monitoring_fraction=0.2,
    )
    parts = _parts(data_dir)
    assert [len(part) for part in parts] == [6, 4, 2, 4]
    assert metadata.fit_fraction == 0.5
    assert metadata.fit_actual_fraction == 6 / 16
    assert set.union(*(set(part.id) for part in parts)) == set(df.id)
    for earlier, later in pairwise(parts):
        assert earlier.reference_date.max() < later.reference_date.min()
    assert metadata.fit_end == parts[0].reference_date.max().isoformat()


@pytest.mark.unit
def test_temporal_optional_inclusive_window(data_dir: Path) -> None:
    timestamps = [pd.Timestamp("2026-01-01") + pd.Timedelta(days=i) for i in range(10)]
    df = pd.DataFrame(
        {
            "reference_date": [
                timestamp + pd.Timedelta(hours=12)
                for timestamp in timestamps
                for _ in range(2)
            ],
            "churn": [0, 1] * 10,
        }
    )
    storage.write_parquet_dataset(df, "events", "reference_date")
    metadata = splitting.create_split(
        CHURN,
        dataset_name="events",
        start_date="2026-01-02",
        end_date="2026-01-09",
        split_name="s",
    )
    assert sum(getattr(metadata, f"{name}_rows") for name in NAMES) == 16
    assert metadata.monitoring_end.startswith("2026-01-09")
    with pytest.raises(ValueError, match="both be provided"):
        splitting.create_split(CHURN, dataset_name="events", start_date="2026-01-01")


@pytest.mark.unit
def test_temporal_window_rejects_missing_timestamps_before_filtering(
    data_dir: Path,
) -> None:
    df = pd.DataFrame(
        {
            "reference_date": [pd.Timestamp("2026-01-01"), pd.NaT],
            "churn": [0, 1],
        }
    )
    storage.write_parquet_dataset(df, "events", "reference_date")
    with pytest.raises(ValueError, match="timestamps must be valid and non-missing"):
        splitting.create_split(
            CHURN, dataset_name="events", start_date="2026-01-01", end_date="2026-01-31"
        )
    assert not (data_dir / "splits").exists()


@pytest.mark.unit
def test_temporal_zero_monitoring_preserves_nanosecond_boundaries(
    data_dir: Path,
) -> None:
    start = pd.Timestamp("2026-01-01")
    df = pd.DataFrame(
        {
            "reference_date": [
                start + pd.Timedelta(nanoseconds=i) for i in range(3) for _ in range(2)
            ],
            "churn": [0, 1] * 3,
        }
    )
    metadata = splitting.create_data_splits(
        CHURN,
        df,
        split_name="s",
        validation_fraction=0.25,
        test_fraction=0.25,
        monitoring_fraction=0,
    )
    assert [len(part) for part in _parts(data_dir)] == [2, 2, 2, 0]
    assert metadata.monitoring_start is None
    assert splitting.load_split_metadata(data_dir / "splits/s.json") == metadata


@pytest.mark.unit
@pytest.mark.parametrize(
    "values",
    [
        {"validation_fraction": -0.1},
        {"test_fraction": 0},
        {"monitoring_fraction": -0.1},
        {"test_fraction": True},
        {"test_fraction": float("nan")},
        {"monitoring_fraction": float("inf")},
        {"validation_fraction": 0.5, "test_fraction": 0.4, "monitoring_fraction": 0.1},
    ],
)
def test_invalid_fractions_do_not_write(
    data_dir: Path, values: dict[str, float]
) -> None:
    df = pd.DataFrame({"label": [0, 1] * 50})
    with pytest.raises(ValueError, match="fraction"):
        splitting.create_data_splits(PHISHING, df, **values)
    assert not (data_dir / "splits").exists()


@pytest.mark.unit
def test_invalid_temporal_groups_and_classes_do_not_write(data_dir: Path) -> None:
    df = pd.DataFrame({"reference_date": ["2026-01-01"] * 8, "churn": [0, 1] * 4})
    with pytest.raises(ValueError, match="timestamp groups"):
        splitting.create_data_splits(CHURN, df)
    df["reference_date"] = pd.date_range("2026-01-01", periods=8)
    df["churn"] = [0] * 4 + [1] * 4
    with pytest.raises(ValueError, match="all target classes"):
        splitting.create_data_splits(CHURN, df)
    df.loc[0, "reference_date"] = pd.NaT
    with pytest.raises(ValueError, match="timestamps"):
        splitting.create_data_splits(CHURN, df)
    assert not (data_dir / "splits").exists()


@pytest.mark.unit
def test_loader_rejects_mismatched_metadata_count(
    data_dir: Path, generated_training_df: pd.DataFrame
) -> None:
    df = pd.concat([generated_training_df] * 10, ignore_index=True)
    df.columns = splitting.normalize_strings(df.columns)
    df["reference_date"] = pd.date_range("2026-01-01", periods=len(df))
    metadata = splitting.create_data_splits(CHURN, df, split_name="s")
    fit, validation, test = splitting.load_and_validate_training_splits(
        "s", data_dir / "splits", use_case=CHURN, metadata=metadata
    )
    assert [len(fit), len(validation), len(test)] == [
        metadata.fit_rows,
        metadata.validation_rows,
        metadata.test_rows,
    ]
    fit.iloc[:-1].to_parquet(data_dir / "splits/s_fit.parquet")
    with pytest.raises(ValueError, match="count"):
        splitting.load_and_validate_training_splits(
            "s", data_dir / "splits", use_case=CHURN, metadata=metadata
        )


@pytest.mark.unit
def test_metadata_rejects_invalid_json_and_inconsistent_counts(data_dir: Path) -> None:
    path = data_dir / "bad.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json"):
        splitting.load_split_metadata(path)
    metadata = splitting.create_data_splits(
        PHISHING, pd.DataFrame({"label": [0, 1] * 50}), split_name="s"
    )
    payload = metadata.model_dump(exclude_none=True)
    payload["fit_rows"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Actual fraction"):
        splitting.load_split_metadata(path)


@pytest.mark.unit
def test_temporal_metadata_rejects_overlapping_bounds(data_dir: Path) -> None:
    df = pd.DataFrame(
        {
            "reference_date": pd.date_range("2026-01-01", periods=100),
            "churn": [0, 1] * 50,
        }
    )
    metadata = splitting.create_data_splits(CHURN, df, split_name="s")
    payload = metadata.model_dump(exclude_none=True)
    payload["validation_start"] = payload["fit_end"]
    path = data_dir / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ordered and disjoint"):
        splitting.load_split_metadata(path)

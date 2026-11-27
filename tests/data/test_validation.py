from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from pydantic_core import PydanticCustomError

import mlkit
from mlkit.data import load_raw_data, validate_data
from mlkit.data.deduplication import deduplicate_prepared_data
from mlkit.models import IdentityFeatureBuilder
from mlkit.use_cases import UseCase


@pytest.fixture
def ingested_df(generated_training_csv) -> pd.DataFrame:
    return load_raw_data(
        generated_training_csv,
        index_col="customer_id",
    )


@pytest.mark.unit
def test_validate_data_accepts_valid_dataframe(
    binary_use_case: UseCase,
    ingested_df: pd.DataFrame,
) -> None:

    validated_df = validate_data(df=ingested_df, use_case=binary_use_case)

    assert validated_df.shape == ingested_df.shape


@pytest.mark.unit
def test_validate_data_rejects_invalid_dtype(
    binary_use_case: UseCase,
    ingested_df: pd.DataFrame,
) -> None:
    df = ingested_df.copy()
    # A text value violates the numeric age schema.
    df["age"] = "age"

    with pytest.raises(ValidationError):
        validate_data(df=df, use_case=binary_use_case)


@pytest.mark.unit
def test_validate_data_rejects_missing_required_column(
    binary_use_case: UseCase,
    ingested_df: pd.DataFrame,
) -> None:
    # Removing a required field should fail before row-level checks run.
    ingested_df = ingested_df.drop(columns=["age"])

    with pytest.raises(ValueError, match="age"):
        validate_data(df=ingested_df, use_case=binary_use_case)


@pytest.mark.unit
def test_validate_data_rejects_negative_total_spend(
    binary_use_case: UseCase,
    ingested_df: pd.DataFrame,
) -> None:
    # Spending cannot be negative.
    ingested_df.loc[ingested_df.index[0], "total_spend"] = -100

    with pytest.raises(ValidationError, match="total_spend"):
        validate_data(df=ingested_df, use_case=binary_use_case)


@pytest.mark.unit
def test_validate_data_rejects_age_below_minimum(
    binary_use_case: UseCase,
    ingested_df: pd.DataFrame,
) -> None:
    # The schema requires a realistic minimum age of 18.
    ingested_df.loc[ingested_df.index[0], "age"] = 10

    with pytest.raises(ValidationError, match="age"):
        validate_data(df=ingested_df, use_case=binary_use_case)


@pytest.mark.unit
def test_validate_data_preserves_index_metadata_and_coerced_values(
    binary_use_case: UseCase,
    ingested_df: pd.DataFrame,
) -> None:
    df = ingested_df.iloc[:2].reset_index()
    df.index = pd.Index([7, 7], name="source_id")
    df["age"] = ["18", "115"]
    df["label"] = ["0", "1"]
    df["reference_date"] = ["2026-01-01", "2026-01-02"]
    df["ignored"] = "extra"

    result = validate_data(df=df, use_case=binary_use_case)

    assert result.index.equals(df.index)
    assert result["age"].tolist() == [18, 115]
    assert result["label"].tolist() == [0, 1]
    assert result["customer_id"].tolist() == df["customer_id"].tolist()
    assert pd.api.types.is_datetime64_any_dtype(result["reference_date"])
    assert "ignored" not in result
    assert "ignored" in df


@pytest.mark.unit
@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("age", pd.NA),
        ("label", None),
        ("label", 2),
        ("customer_id", "invalid"),
        ("reference_date", "not-a-date"),
    ],
)
def test_validate_data_rejects_invalid_present_values(
    binary_use_case: UseCase, ingested_df: pd.DataFrame, column: str, value: object
) -> None:
    df = ingested_df.iloc[:1].reset_index().astype(object)
    df["reference_date"] = "2026-01-01"
    df.loc[df.index[0], column] = value
    with pytest.raises(ValidationError, match=column):
        validate_data(df=df, use_case=binary_use_case)


@pytest.mark.unit
def test_validate_data_handles_empty_frames_and_optional_metadata(
    binary_use_case: UseCase,
    ingested_df: pd.DataFrame,
) -> None:
    features = ingested_df.loc[:, list(binary_use_case.feature_columns)]
    result = validate_data(df=features.iloc[:0], use_case=binary_use_case)
    pd.testing.assert_frame_equal(result, features.iloc[:0])
    with pytest.raises(ValueError, match="age"):
        validate_data(
            df=features.iloc[:0].drop(columns="age"), use_case=binary_use_case
        )
    assert len(validate_data(df=features, use_case=binary_use_case)) == len(features)


@pytest.mark.unit
def test_validate_data_reports_absolute_row_position_after_first_chunk(
    binary_use_case: UseCase,
    ingested_df: pd.DataFrame,
) -> None:
    df = pd.concat([ingested_df.iloc[:1]] * 10_002, ignore_index=True)
    df.loc[10_001, "age"] = 17
    with pytest.raises(ValidationError) as error:
        validate_data(df, binary_use_case)
    assert error.value.errors()[0]["loc"] == (10_001, "age")


@pytest.mark.unit
def test_validate_data_uses_feature_defaults_and_ignores_metadata_for_strict_model() -> (
    None
):
    class Features(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: int
        scale: int = 2

    use_case = UseCase(
        name="defaults",
        target_column="label",
        time_column="timestamp",
        feature_schema=Features,
        feature_builder_factory=IdentityFeatureBuilder,
    )
    df = pd.DataFrame({"value": ["3"], "label": [1], "extra": ["ignored"]})
    result = validate_data(df=df, use_case=use_case)
    assert result.to_dict("records") == [{"value": 3, "label": 1, "scale": 2}]


@pytest.mark.unit
def test_validate_data_uses_defaults_when_all_feature_columns_are_absent() -> None:
    class Features(BaseModel):
        value: int = 2

    use_case = UseCase(
        name="defaults",
        target_column="label",
        time_column="timestamp",
        feature_schema=Features,
        feature_builder_factory=IdentityFeatureBuilder,
    )
    df = pd.DataFrame({"label": [1, 0]}, index=[4, 4])
    result = validate_data(df=df, use_case=use_case)
    assert result.to_dict("records") == [
        {"label": 1, "value": 2},
        {"label": 0, "value": 2},
    ]
    assert result.index.equals(df.index)


@pytest.mark.unit
def test_validate_data_preserves_custom_pydantic_errors() -> None:
    class Features(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def reject_value(cls, value: int) -> int:
            if value == 0:
                raise PydanticCustomError("zero_value", "Zero is not allowed")
            return value

    use_case = UseCase(
        name="custom",
        target_column="label",
        time_column="timestamp",
        feature_schema=Features,
        feature_builder_factory=IdentityFeatureBuilder,
    )
    with pytest.raises(ValidationError) as error:
        validate_data(df=pd.DataFrame({"value": [0]}), use_case=use_case)
    assert error.value.errors()[0]["type"] == "zero_value"
    assert error.value.errors()[0]["loc"] == (0, "value")


@pytest.mark.unit
def test_deduplicate_prepared_data_uses_source_rows_and_positions() -> None:
    source = pd.DataFrame(
        {"value": [1, 1, 1, 1, None, None], "label": [0, 0, 1, 0, 0, 0]},
        index=[4, 4, 4, 7, 7, 9],
    )
    prepared = pd.DataFrame(
        {
            "value": [
                "first",
                "duplicate",
                "other target",
                "duplicate",
                "missing",
                "dup",
            ]
        },
        index=source.index,
    )
    result = deduplicate_prepared_data(source, prepared)
    assert result["value"].tolist() == ["first", "other target", "missing"]
    assert result.index.tolist() == [4, 4, 7]
    assert len(source) == len(prepared) == 6
    with pytest.raises(ValueError, match="same row count"):
        deduplicate_prepared_data(source, prepared.iloc[:-1])


@pytest.mark.unit
@pytest.mark.parametrize("deduplicate", [False, True])
def test_prepare_data_cli_deduplicates_source_rows_only_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, deduplicate: bool
) -> None:
    from mlkit import data
    from mlkit.data import storage

    class Features(BaseModel):
        value: int

    use_case = UseCase(
        name="example",
        target_column="label",
        time_column="timestamp",
        feature_schema=Features,
        feature_builder_factory=IdentityFeatureBuilder,
    )
    source = pd.DataFrame(
        {
            "value": ["1", "1", "1", "1", "01"],
            "label": [0, 0, 0, 1, 0],
            "ignored": ["a", "a", "b", "a", "a"],
        },
        index=[2, 2, 3, 4, 5],
    )
    written: list[pd.DataFrame] = []
    monkeypatch.setattr(
        mlkit, "RuntimeSettings", lambda: SimpleNamespace(data_dir=tmp_path)
    )
    monkeypatch.setattr(data, "load_raw_data", lambda **kwargs: source)
    monkeypatch.setattr(
        storage, "write_parquet_dataset", lambda **kwargs: written.append(kwargs["df"])
    )
    monkeypatch.setattr("mlkit.use_cases.get_use_case", lambda name: use_case)

    args = ["prepare-data", "--input_csv", "input.csv", "--use_case", "example"]
    if deduplicate:
        args.append("--deduplicate")
    mlkit.main(args)

    assert len(written) == 1
    result = written[0]
    assert result["label"].tolist() == (
        [0, 0, 1, 0] if deduplicate else [0, 0, 0, 1, 0]
    )
    assert result.index.tolist() == ([2, 3, 4, 5] if deduplicate else [2, 2, 3, 4, 5])
    assert result["value"].tolist() == [1] * len(result)
    assert "ignored" not in result.columns

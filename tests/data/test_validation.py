import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from pydantic_core import PydanticCustomError

from mlkit.data import load_raw_data, validate_data
from mlkit.models import IdentityFeatureBuilder
from mlkit.use_cases import UseCase, get_use_case

CHURN_USE_CASE = get_use_case("churn")


@pytest.fixture
def ingested_df(generated_training_csv) -> pd.DataFrame:
    return load_raw_data(
        generated_training_csv,
        index_col="customer_id",
    )


@pytest.mark.unit
def test_validate_data_accepts_valid_dataframe(
    ingested_df: pd.DataFrame,
) -> None:

    validated_df = validate_data(df=ingested_df, use_case=CHURN_USE_CASE)

    assert validated_df.shape == ingested_df.shape


@pytest.mark.unit
def test_validate_data_rejects_invalid_dtype(
    ingested_df: pd.DataFrame,
) -> None:
    df = ingested_df.copy()
    # A text value violates the numeric age schema.
    df["age"] = "age"

    with pytest.raises(ValidationError):
        validate_data(df=df, use_case=CHURN_USE_CASE)


@pytest.mark.unit
def test_validate_data_rejects_missing_required_column(
    ingested_df: pd.DataFrame,
) -> None:
    # Removing a required field should fail before row-level checks run.
    ingested_df = ingested_df.drop(columns=["age"])

    with pytest.raises(ValueError, match="age"):
        validate_data(df=ingested_df, use_case=CHURN_USE_CASE)


@pytest.mark.unit
def test_validate_data_rejects_invalid_churn_value(
    ingested_df: pd.DataFrame,
) -> None:
    # Churn is a binary target, so value 2 violates the data contract.
    ingested_df.loc[ingested_df.index[0], "churn"] = 2

    with pytest.raises(ValidationError, match="churn"):
        validate_data(df=ingested_df, use_case=CHURN_USE_CASE)


@pytest.mark.unit
def test_validate_data_rejects_negative_total_spend(
    ingested_df: pd.DataFrame,
) -> None:
    # Spending cannot be negative.
    ingested_df.loc[ingested_df.index[0], "total_spend"] = -100

    with pytest.raises(ValidationError, match="total_spend"):
        validate_data(df=ingested_df, use_case=CHURN_USE_CASE)


@pytest.mark.unit
def test_validate_data_rejects_age_below_minimum(
    ingested_df: pd.DataFrame,
) -> None:
    # The schema requires a realistic minimum age of 18.
    ingested_df.loc[ingested_df.index[0], "age"] = 10

    with pytest.raises(ValidationError, match="age"):
        validate_data(df=ingested_df, use_case=CHURN_USE_CASE)


@pytest.mark.unit
def test_validate_data_preserves_index_metadata_and_coerced_values(
    ingested_df: pd.DataFrame,
) -> None:
    df = ingested_df.iloc[:2].reset_index()
    df.index = pd.Index([7, 7], name="source_id")
    df["age"] = ["18", "115"]
    df["churn"] = ["0", "1"]
    df["reference_date"] = ["2026-01-01", "2026-01-02"]
    df["ignored"] = "extra"

    result = validate_data(df=df, use_case=CHURN_USE_CASE)

    assert result.index.equals(df.index)
    assert result["age"].tolist() == [18, 115]
    assert result["churn"].tolist() == [0, 1]
    assert result["customer_id"].tolist() == df["customer_id"].tolist()
    assert pd.api.types.is_datetime64_any_dtype(result["reference_date"])
    assert "ignored" not in result
    assert "ignored" in df


@pytest.mark.unit
@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("age", pd.NA),
        ("churn", None),
        ("churn", 2),
        ("customer_id", "invalid"),
        ("reference_date", "not-a-date"),
    ],
)
def test_validate_data_rejects_invalid_present_values(
    ingested_df: pd.DataFrame, column: str, value: object
) -> None:
    df = ingested_df.iloc[:1].reset_index().astype(object)
    df["reference_date"] = "2026-01-01"
    df.loc[df.index[0], column] = value
    with pytest.raises(ValidationError, match=column):
        validate_data(df=df, use_case=CHURN_USE_CASE)


@pytest.mark.unit
def test_validate_data_handles_empty_frames_and_optional_metadata(
    ingested_df: pd.DataFrame,
) -> None:
    features = ingested_df.loc[:, list(CHURN_USE_CASE.feature_columns)]
    result = validate_data(df=features.iloc[:0], use_case=CHURN_USE_CASE)
    pd.testing.assert_frame_equal(result, features.iloc[:0])
    with pytest.raises(ValueError, match="age"):
        validate_data(df=features.iloc[:0].drop(columns="age"), use_case=CHURN_USE_CASE)
    assert len(validate_data(df=features, use_case=CHURN_USE_CASE)) == len(features)


@pytest.mark.unit
def test_validate_data_reports_absolute_row_position_after_first_chunk(
    ingested_df: pd.DataFrame,
) -> None:
    df = pd.concat([ingested_df.iloc[:1]] * 10_002, ignore_index=True)
    df.loc[10_001, "age"] = 17
    with pytest.raises(ValidationError) as error:
        validate_data(df, CHURN_USE_CASE)
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

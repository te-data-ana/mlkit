from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from use_cases.churn.features import ChurnFeatureBuilder
from use_cases.churn.schema import ChurnInputFeatures


@pytest.mark.usecase
def test_churn_input_features_accepts_boundary_values() -> None:
    input_features = ChurnInputFeatures(
        age=115,
        tenure=0,
        usage_frequency=0,
        support_calls=0,
        payment_delay=0,
        last_interaction=0,
        total_spend=0.0,
        gender="Male",
        subscription_type="Basic",
        contract_length="Monthly",
    )

    assert input_features.age == 115
    assert input_features.total_spend == 0.0


@pytest.mark.usecase
@pytest.mark.parametrize(
    "field,value",
    [
        ("age", 17),
        ("age", 116),
        ("support_calls", -1),
        ("payment_delay", -1),
        ("gender", "XY"),
        ("subscription_type", "Free"),
        ("contract_length", "undefined"),
    ],
)
def test_churn_input_features_rejects_invalid_values(
    sample_churn_input_factory: Callable[..., dict],
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        ChurnInputFeatures(**sample_churn_input_factory(**{field: value}))


@pytest.mark.usecase
def test_churn_f_builder_returns_finite_engineered_features(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = ChurnFeatureBuilder().transform(sample_features_df)

    engineered_cols = [
        "contract_commitment",
        "tenure_rel2_commitment",
        "engagement",
        "avg_spend_per_year",
        "avg_s_calls_per_year",
        "late_payer",
        "inactive_customer",
    ]

    assert set(engineered_cols).issubset(set(transformed_df.columns))
    assert not transformed_df[engineered_cols].isna().values.any()
    assert np.isfinite(transformed_df[engineered_cols]).values.all()


@pytest.mark.usecase
def test_churn_f_builder_does_not_modify_input_dataframe(
    sample_features_df: pd.DataFrame,
) -> None:
    original_df = sample_features_df.copy(deep=True)

    transformed_df = ChurnFeatureBuilder().transform(sample_features_df)

    pd.testing.assert_frame_equal(
        sample_features_df,
        original_df,
    )

    pd.testing.assert_frame_equal(
        transformed_df[original_df.columns],
        original_df,
    )


@pytest.mark.usecase
def test_churn_f_builder_handles_zero_tenure_without_division_error(
    sample_features_df: pd.DataFrame,
) -> None:
    # The first row of the sample_features_df fixture has zero tenure and exercises the division guard.
    transformed_df = ChurnFeatureBuilder().transform(sample_features_df)

    assert transformed_df["avg_spend_per_year"].iloc[0] == 0
    assert transformed_df["avg_s_calls_per_year"].iloc[0] == 0


@pytest.mark.usecase
def test_churn_f_builder_maps_contract_length_to_months(
    sample_features_df: pd.DataFrame,
) -> None:
    # The first three rows of the sample_features_df fixture represent Monthly, Quarterly, and Annual plans.
    transformed_df = ChurnFeatureBuilder().transform(sample_features_df)

    assert transformed_df["contract_commitment"].iloc[0] == 1
    assert transformed_df["contract_commitment"].iloc[1] == 4
    assert transformed_df["contract_commitment"].iloc[2] == 12


@pytest.mark.usecase
def test_churn_f_builder_calculates_relative_tenure(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = ChurnFeatureBuilder().transform(sample_features_df)

    assert transformed_df["tenure_rel2_commitment"].iloc[1] == 3
    assert transformed_df["tenure_rel2_commitment"].iloc[2] == 2


@pytest.mark.usecase
def test_churn_f_builder_calculates_engagement(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = ChurnFeatureBuilder().transform(sample_features_df)

    assert transformed_df["engagement"].iloc[1] == 10
    assert transformed_df["engagement"].iloc[2] == 10


@pytest.mark.usecase
def test_churn_f_builder_calculates_average_spend_per_year(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = ChurnFeatureBuilder().transform(sample_features_df)

    assert transformed_df["avg_spend_per_year"].iloc[1] == 400
    assert transformed_df["avg_spend_per_year"].iloc[2] == 500


@pytest.mark.usecase
def test_churn_f_builder_calculates_average_support_calls_per_year(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = ChurnFeatureBuilder().transform(sample_features_df)

    assert transformed_df["avg_s_calls_per_year"].iloc[1] == 6
    assert transformed_df["avg_s_calls_per_year"].iloc[2] == 1 / 2


@pytest.mark.usecase
def test_churn_f_builder_sets_late_payer_flag(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = ChurnFeatureBuilder().transform(sample_features_df)

    assert transformed_df["late_payer"].iloc[0] == 1
    assert transformed_df["late_payer"].iloc[1] == 0
    assert transformed_df["late_payer"].iloc[2] == 0


@pytest.mark.usecase
def test_churn_f_builder_sets_inactive_customer_flag(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = ChurnFeatureBuilder().transform(sample_features_df)

    assert transformed_df["inactive_customer"].iloc[0] == 0
    assert transformed_df["inactive_customer"].iloc[1] == 0
    assert transformed_df["inactive_customer"].iloc[2] == 1


@pytest.mark.usecase
def test_churn_f_builder_returns_original_and_engineered_feature_names(
    sample_features_df: pd.DataFrame,
) -> None:
    transformer = ChurnFeatureBuilder()

    features = transformer.get_feature_names_out(sample_features_df.columns)

    expected_features = {
        "tenure",
        "contract_length",
        "usage_frequency",
        "last_interaction",
        "total_spend",
        "support_calls",
        "payment_delay",
        "contract_commitment",
        "tenure_rel2_commitment",
        "engagement",
        "avg_spend_per_year",
        "avg_s_calls_per_year",
        "late_payer",
        "inactive_customer",
    }

    assert expected_features.issubset(set(features))

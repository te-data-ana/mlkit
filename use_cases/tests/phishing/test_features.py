import numpy as np
import pandas as pd
import pytest
from use_cases.phishing.features import PhishingFeatureBuilder


@pytest.mark.unit
def test_feature_builder_returns_finite_engineered_features(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = PhishingFeatureBuilder().transform(sample_features_df)

    engineered_cols = [
        "example_ratio_feature",
        "example_threshold_feature",
    ]

    assert set(engineered_cols).issubset(set(transformed_df.columns))
    assert not transformed_df[engineered_cols].isna().values.any()
    assert np.isfinite(transformed_df[engineered_cols]).values.all()


@pytest.mark.unit
def test_feature_builder_does_not_modify_input_dataframe(
    sample_features_df: pd.DataFrame,
) -> None:
    original_df = sample_features_df.copy(deep=True)

    transformed_df = PhishingFeatureBuilder().transform(sample_features_df)

    pd.testing.assert_frame_equal(
        sample_features_df,
        original_df,
    )

    pd.testing.assert_frame_equal(
        transformed_df[original_df.columns],
        original_df,
    )


@pytest.mark.unit
def test_feature_builder_handles_zero_values_without_division_error(
    sample_features_df: pd.DataFrame,
) -> None:
    # The first row has zero value and triggers the division guard.
    transformed_df = PhishingFeatureBuilder().transform(sample_features_df)

    assert transformed_df["example_ratio_feature"].iloc[0] == 0


@pytest.mark.unit
def test_feature_builder_sets_threshold_based_flag(
    sample_features_df: pd.DataFrame,
) -> None:
    transformed_df = PhishingFeatureBuilder().transform(sample_features_df)

    assert transformed_df["example_threshold_feature"].iloc[0] == 0
    assert transformed_df["example_threshold_feature"].iloc[1] == 0
    assert transformed_df["example_threshold_feature"].iloc[2] == 1


@pytest.mark.unit
def test_feature_builder_returns_original_and_engineered_feature_names(
    sample_features_df: pd.DataFrame,
) -> None:
    transformer = PhishingFeatureBuilder()

    features = transformer.get_feature_names_out(sample_features_df.columns)

    expected_features = {
        "example_int_feature",
        "example_float_feature",
        "example_cat_feature",
        "example_ratio_feature",
        "example_threshold_feature",
    }

    assert expected_features.issubset(set(features))

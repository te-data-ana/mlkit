from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PhishingFeatureBuilder(BaseEstimator, TransformerMixin):
    """Engineer use-case specific features."""

    def __init__(self):
        """Initialize feature-engineering thresholds."""
        self.feature_params = {
            "example_param": 100,
            "example_threshold": 5.0,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit a sklearn-compatible transformer without learning state."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add use-case specific features.

        Args:
            X: DataFrame containing the raw feature columns.

        Returns:
            A copied DataFrame with engineered feature columns appended.
            Missing derived values and divisions by zero are represented as 0.
        """
        X = X.copy()

        example_float_feature = X["example_float_feature"].replace(0, np.nan)

        X["example_ratio_feature"] = (
            X["example_int_feature"] / example_float_feature
        ).fillna(0)

        X["example_threshold_feature"] = (
            X["example_float_feature"] > self.feature_params["example_threshold"]
        ).astype(int)

        return X

    def get_feature_names_out(
        self, input_features: list[str] | np.ndarray | None = None
    ) -> np.ndarray | None:
        """Return input feature names followed by engineered feature names.

        Args:
            input_features: Optional iterable of names for the input features.

        Returns:
            NumPy array containing all feature names, or ``None`` when input
            feature names are not provided.
        """
        if input_features is None:
            return None
        return np.concatenate(
            [
                np.asarray(input_features),
                np.asarray(
                    [
                        "example_ratio_feature",
                        "example_threshold_feature",
                    ]
                ),
            ]
        )

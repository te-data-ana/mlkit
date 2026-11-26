"""Feature transformation for the students use case."""

from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CATEGORICAL_COLUMNS = (
    "marital_status",
    "application_mode",
    "course",
    "previous_qualification",
    "nacionality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
)


class StudentsFeatureBuilder(BaseEstimator, TransformerMixin):
    """Mark coded student characteristics for categorical preprocessing."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit without learning state."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with categorical codes marked for one-hot encoding."""
        result = X.copy()
        for column in CATEGORICAL_COLUMNS:
            result[column] = result[column].astype("category")
        return result

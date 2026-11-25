"""Feature transformation for the diabetes use case."""

from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DiabetesFeatureBuilder(BaseEstimator, TransformerMixin):
    """Mark ordered survey codes for categorical preprocessing."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit without learning state."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with survey categories marked as categorical."""
        result = X.copy()
        for column in ("gen_hlth", "age", "education", "income"):
            result[column] = result[column].astype("category")
        return result

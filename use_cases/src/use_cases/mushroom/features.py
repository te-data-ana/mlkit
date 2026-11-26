"""Feature transformation for the Secondary Mushroom use case."""

from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NULLABLE_CATEGORIES = (
    "cap_surface",
    "gill_attachment",
    "gill_spacing",
    "stem_root",
    "stem_surface",
    "veil_type",
    "veil_color",
    "ring_type",
    "spore_print_color",
)
MISSING_CATEGORY = "na"


class MushroomFeatureBuilder(BaseEstimator, TransformerMixin):
    """Represent missing nominal observations as a separate model category."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit without learning state."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with nullable categories filled by a reserved marker."""
        result = X.copy()
        for column in NULLABLE_CATEGORIES:
            values = result[column].astype(object)
            result[column] = values.where(values.notna(), MISSING_CATEGORY)
        return result

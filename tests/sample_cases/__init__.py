"""Synthetic feature schema and transformer shared by core tests."""

from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin


class SampleFeatures(BaseModel):
    """Features for the synthetic binary and multiclass contracts."""

    age: int = Field(ge=18, le=115)
    tenure: int = Field(ge=0)
    usage_frequency: int = Field(ge=0)
    support_calls: int = Field(ge=0)
    payment_delay: int = Field(ge=0)
    last_interaction: int = Field(ge=0)
    total_spend: float = Field(ge=0)
    gender: Literal["Female", "Male"]
    subscription_type: Literal["Basic", "Standard", "Premium"]
    contract_length: Literal["Monthly", "Quarterly", "Annual"]


class SampleFeatureBuilder(BaseEstimator, TransformerMixin):
    """Add a deterministic feature to exercise custom feature builders."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Return this stateless transformer."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add a feature derived from tenure."""
        return X.assign(tenure_twice=X["tenure"] * 2)

    def get_feature_names_out(
        self, input_features: list[str] | np.ndarray | None = None
    ) -> np.ndarray | None:
        """Return the names of the input and derived features."""
        if input_features is None:
            return None
        return np.append(input_features, "tenure_twice")

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureBuilder(BaseEstimator, TransformerMixin):
    """Engineer customer engagement and spending features for churn models."""

    def __init__(self) -> None:
        """Initialize churn feature-engineering thresholds."""
        self.feature_params = {
            "engagement_window": 30,
            "late_payer_threshold": 15,
            "inactive_threshold": 15,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit a sklearn-compatible transformer without learning state."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add churn-specific customer engagement and spending features."""
        X = X.copy()
        tenure = X["tenure"].replace(0, np.nan)
        tenure_years = tenure / 12
        contract_map = {
            "Monthly": 1,
            "Quarterly": 4,
            "Biannual": 6,
            "Annual": 12,
        }
        commitment = X["contract_length"].map(contract_map)
        X["contract_commitment"] = commitment.fillna(0)
        X["tenure_rel2_commitment"] = (tenure / commitment).fillna(0)
        X["engagement"] = (
            X["usage_frequency"] / self.feature_params["engagement_window"]
        ) * (self.feature_params["engagement_window"] - X["last_interaction"])
        X["engagement"] = X["engagement"].fillna(0)
        X["avg_spend_per_year"] = (X["total_spend"] / tenure_years).fillna(0)
        X["avg_s_calls_per_year"] = (X["support_calls"] / tenure_years).fillna(0)
        X["late_payer"] = (
            X["payment_delay"] > self.feature_params["late_payer_threshold"]
        ).astype(int)
        X["inactive_customer"] = (
            X["last_interaction"] > self.feature_params["inactive_threshold"]
        ).astype(int)
        return X

    def get_feature_names_out(
        self, input_features: list[str] | np.ndarray | None = None
    ) -> np.ndarray | None:
        """Return input and engineered feature names."""
        if input_features is None:
            return None
        return np.concatenate(
            [
                np.asarray(input_features),
                np.asarray(
                    [
                        "contract_commitment",
                        "tenure_rel2_commitment",
                        "engagement",
                        "avg_spend_per_year",
                        "avg_s_calls_per_year",
                        "late_payer",
                        "inactive_customer",
                    ]
                ),
            ]
        )

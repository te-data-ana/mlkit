from collections.abc import Callable

import pandas as pd
import pytest


@pytest.fixture
def sample_churn_input_factory() -> Callable[..., dict]:
    def factory(**kwargs) -> dict:
        data = {
            "age": 45,
            "tenure": 24,
            "usage_frequency": 12,
            "support_calls": 1,
            "payment_delay": 0,
            "last_interaction": 5,
            "total_spend": 800.0,
            "gender": "Female",
            "subscription_type": "Premium",
            "contract_length": "Annual",
        }
        data.update(kwargs)

        return data

    return factory


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    # Feature engineering tests use this DataFrame without a target column.
    df = pd.DataFrame(
        {
            "tenure": [0, 12, 24, 60, None],
            "contract_length": ["Monthly", "Quarterly", "Annual", "Monthly", None],
            "usage_frequency": [10, 20, 30, None, 5],
            "last_interaction": [5, 15, 20, 10, None],
            "total_spend": [0, 400, 1000, None, 750],
            "support_calls": [10, 6, 1, 2, None],
            "payment_delay": [25, 10, 0, None, 30],
        }
    )
    df["contract_length"] = df["contract_length"].astype("str")
    return df

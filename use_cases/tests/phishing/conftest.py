from collections.abc import Callable

import pandas as pd
import pytest


@pytest.fixture
def sample_input_factory() -> Callable[..., dict]:
    def factory(**kwargs) -> dict:
        data = {
            "example_int_feature": 45,
            "example_float_feature": 24.0,
            "example_cat_feature": "C1",
        }
        data.update(kwargs)

        return data

    return factory


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    # Feature engineering tests use this DataFrame without a target column.
    df = pd.DataFrame(
        {
            "example_int_feature": [0, 12, 24, 60, None],
            "example_float_feature": [0.0, 2.2, 5.5, None, 9.0],
            "example_cat_feature": ["C1", "C2", "C1", "C2", None],
        }
    )
    df["example_cat_feature"] = df["example_cat_feature"].astype("str")
    return df

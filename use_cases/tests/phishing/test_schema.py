from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError
from use_cases.phishing.schema import PhishingInputFeatures


def test_input_features_accepts_boundary_values() -> None:
    input_features = PhishingInputFeatures(
        example_int_feature=100,
        example_float_feature=0.0,
        example_cat_feature="C1",
    )

    assert input_features.example_int_feature == 100
    assert input_features.example_float_feature == 0.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("example_int_feature", -1),
        ("example_int_feature", 101),
        ("example_float_feature", -1.0),
        ("example_cat_feature", "XY"),
    ],
)
def test_input_features_rejects_invalid_values(
    sample_input_factory: Callable[..., dict],
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        PhishingInputFeatures(**sample_input_factory(**{field: value}))

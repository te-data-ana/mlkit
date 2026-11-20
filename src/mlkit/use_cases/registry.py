from mlkit.use_cases.base import UseCase
from mlkit.use_cases.churn import (
    ChurnFeatureBuilder,
    ChurnInputFeatures,
)

CHURN_USE_CASE = UseCase(
    name="churn",
    target_column="churn",
    time_column="reference_date",
    feature_schema=ChurnInputFeatures,
    feature_builder_factory=ChurnFeatureBuilder,
    index_column="customerid",
    metadata_fields={"customerid": int},
)

USE_CASES = {CHURN_USE_CASE.name: CHURN_USE_CASE}


def get_use_case(name: str) -> UseCase:
    """Return a registered use case by its explicit name."""
    try:
        return USE_CASES[name]
    except KeyError as exc:
        available = ", ".join(sorted(USE_CASES))
        raise ValueError(
            f"Unknown use case '{name}'. Available use cases: {available}."
        ) from exc

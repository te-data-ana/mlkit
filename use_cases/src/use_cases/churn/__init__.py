from mlkit.use_cases.base import UseCase

from .features import ChurnFeatureBuilder
from .schema import ChurnInputFeatures

CHURN_USE_CASE = UseCase(
    name="churn",
    target_column="churn",
    time_column="reference_date",
    feature_schema=ChurnInputFeatures,
    feature_builder_factory=ChurnFeatureBuilder,
    index_column="customerid",
    metadata_fields={"customerid": int},
)

__all__ = ["CHURN_USE_CASE", "ChurnFeatureBuilder", "ChurnInputFeatures"]

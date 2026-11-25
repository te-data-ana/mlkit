"""CDC Diabetes Health Indicators use-case contract."""

from mlkit.use_cases.base import UseCase

from .features import DiabetesFeatureBuilder
from .schema import DiabetesInputFeatures

DIABETES_USE_CASE = UseCase(
    name="diabetes",
    target_column="diabetes_binary",
    time_column=None,
    split_strategy="stratified_random",
    feature_schema=DiabetesInputFeatures,
    feature_builder_factory=DiabetesFeatureBuilder,
    index_column="id",
    metadata_fields={"id": int},
)

__all__ = ["DIABETES_USE_CASE", "DiabetesFeatureBuilder", "DiabetesInputFeatures"]

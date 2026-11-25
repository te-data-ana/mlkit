from mlkit.models.features import IdentityFeatureBuilder
from mlkit.use_cases.base import UseCase

from .schema import PhishingInputFeatures

PHISHING_USE_CASE = UseCase(
    name="phishing",
    target_column="label",
    time_column=None,
    split_strategy="stratified_random",
    feature_schema=PhishingInputFeatures,
    feature_builder_factory=IdentityFeatureBuilder,
)

__all__ = [
    "PHISHING_USE_CASE",
    "PhishingInputFeatures",
]

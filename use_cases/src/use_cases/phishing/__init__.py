from mlkit.use_cases.base import UseCase

from .features import PhishingFeatureBuilder
from .schema import PhishingInputFeatures

PHISHING_USE_CASE = UseCase(
    name="phishing",
    target_column="target_column",
    time_column="time_column",
    feature_schema=PhishingInputFeatures,
    feature_builder_factory=PhishingFeatureBuilder,
)

__all__ = [
    "PHISHING_USE_CASE",
    "PhishingFeatureBuilder",
    "PhishingInputFeatures",
]

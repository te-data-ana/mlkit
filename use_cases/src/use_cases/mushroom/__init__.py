"""Secondary Mushroom binary-classification use case."""

from mlkit.use_cases.base import UseCase

from .features import MushroomFeatureBuilder
from .schema import MushroomInputFeatures

MUSHROOM_USE_CASE = UseCase(
    name="mushroom",
    target_column="class",
    time_column=None,
    split_strategy="stratified_random",
    feature_schema=MushroomInputFeatures,
    feature_builder_factory=MushroomFeatureBuilder,
)

__all__ = ["MUSHROOM_USE_CASE", "MushroomFeatureBuilder", "MushroomInputFeatures"]

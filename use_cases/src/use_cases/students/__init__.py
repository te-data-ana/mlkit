"""Students' dropout and academic success multiclass use case."""

from mlkit.use_cases.base import UseCase

from .features import CATEGORICAL_COLUMNS, StudentsFeatureBuilder
from .schema import StudentsInputFeatures

STUDENTS_USE_CASE = UseCase(
    name="students",
    target_column="target",
    time_column=None,
    split_strategy="stratified_random",
    feature_schema=StudentsInputFeatures,
    feature_builder_factory=StudentsFeatureBuilder,
    target_classes=(0, 1, 2),
    categorical_columns=CATEGORICAL_COLUMNS,
)

__all__ = ["STUDENTS_USE_CASE", "StudentsFeatureBuilder", "StudentsInputFeatures"]

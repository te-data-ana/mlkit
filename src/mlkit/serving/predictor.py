import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel

from mlkit.evaluation import align_probabilities
from mlkit.serving.model_loader import LoadedModel, ModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    predicted_class: int
    timestamp: datetime
    metadata: ModelMetadata
    predicted_probability: float | None = None
    predicted_probabilities: dict[str, float] | None = None

    def to_response(self, time_column: str) -> dict[str, Any]:
        """Serialize the result using the selected use case's timestamp name."""
        result = {
            time_column: self.timestamp,
            "predicted_class": self.predicted_class,
            "metadata": asdict(self.metadata),
        }
        if self.predicted_probabilities is None:
            result["predicted_probability"] = self.predicted_probability
        else:
            result["predicted_probabilities"] = self.predicted_probabilities
        return result


class Predictor:
    def __init__(self, loaded_model: LoadedModel) -> None:
        """Initialize a predictor from a loaded model and its metadata.

        Args:
            loaded_model: Model object and serving metadata returned by
                ``load_model``.
        """
        self.model = loaded_model.model
        self.metadata = loaded_model.metadata
        self.use_case = loaded_model.use_case
        if (
            len(self.use_case.target_classes) > 2
            and self.metadata.model_threshold is not None
        ):
            raise ValueError(
                "Multiclass prediction requires an argmax model without a threshold."
            )
        logger.info(
            "Initialized predictor for model version '%s' with threshold %s.",
            self.metadata.model_version,
            self.metadata.model_threshold,
        )

    @staticmethod
    def _to_probability_array(probabilities: ArrayLike) -> np.ndarray:
        """Normalize model probabilities to a two-column NumPy array.

        Args:
            probabilities: One-dimensional positive-class probabilities or a
                two-dimensional class-probability array.

        Returns:
            Two-dimensional array with negative-class probabilities in column
            zero and positive-class probabilities in column one.

        Raises:
            ValueError: If the array is not two-dimensional with at least two
                columns after normalization.
        """
        try:
            array = np.asarray(probabilities)
            if array.ndim == 1:
                # A one-dimensional output contains positive-class probabilities;
                # complement them to construct the negative-class column.
                array = np.column_stack([1.0 - array, array])
            if array.ndim != 2 or array.shape[1] < 2:
                raise ValueError(
                    "Model probability output must be 2D with at least two columns."
                )
            return array
        except Exception:
            logger.exception("Failed to normalize model probability output.")
            raise

    def predict(
        self, features: BaseModel, timestamp: datetime | None = None
    ) -> PredictionResult:
        """Predict the target class for one validated set of use-case features.

        Args:
            features: Validated use-case feature values.
            timestamp: Optional timestamp for the provided features. If omitted,
                use the current UTC time.

        Returns:
            Prediction result containing the positive-class probability,
            thresholded class, and model metadata.
        """
        if timestamp is None:
            timestamp = self._timestamp()

        try:
            df = pd.DataFrame([features.model_dump()])
            probabilities = self._to_probability_array(self.model.predict_proba(df))

            if len(self.use_case.target_classes) > 2:
                probabilities = align_probabilities(
                    probabilities=probabilities,
                    model_classes=self.model.classes_,
                    target_classes=self.use_case.target_classes,
                )
                predicted_class = self.use_case.target_classes[
                    int(probabilities[0].argmax())
                ]
                class_probabilities = {
                    str(label): float(probabilities[0, index])
                    for index, label in enumerate(self.use_case.target_classes)
                }
                predicted_probability = None
            else:
                if self.metadata.model_threshold is None:
                    raise ValueError("Binary prediction requires a decision threshold.")
                predicted_probability = float(probabilities[0, 1])
                predicted_class = int(
                    predicted_probability >= self.metadata.model_threshold
                )
                class_probabilities = None

            return PredictionResult(
                predicted_class=predicted_class,
                predicted_probability=predicted_probability,
                timestamp=timestamp,
                metadata=self.metadata,
                predicted_probabilities=class_probabilities,
            )
        except Exception:
            logger.exception("Single-record prediction failed.")
            raise

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict churn for multiple records while preserving their index.

        Args:
            df: DataFrame containing one data record per row.

        Returns:
            DataFrame indexed like ``df`` with predicted probabilities,
            thresholded classes, the configured threshold, and model version.
        """
        try:
            logger.info("Starting batch prediction for %d rows.", len(df))
            probabilities = self._to_probability_array(self.model.predict_proba(df))
            result = df.copy()
            if len(self.use_case.target_classes) > 2:
                probabilities = align_probabilities(
                    probabilities=probabilities,
                    model_classes=self.model.classes_,
                    target_classes=self.use_case.target_classes,
                )
                for index, label in enumerate(self.use_case.target_classes):
                    result[f"predicted_probability_{label}"] = probabilities[:, index]
                result["predicted_class"] = np.asarray(self.use_case.target_classes)[
                    probabilities.argmax(axis=1)
                ]
            else:
                if self.metadata.model_threshold is None:
                    raise ValueError("Binary prediction requires a decision threshold.")
                result["predicted_probability"] = probabilities[:, 1]
                result["predicted_class"] = (
                    probabilities[:, 1] >= self.metadata.model_threshold
                ).astype(int)
                result["threshold"] = self.metadata.model_threshold
            result["model_version"] = self.metadata.model_version

            logger.info(
                "Batch prediction completed for %d rows.",
                len(result),
            )
            return result
        except Exception:
            logger.exception("Batch prediction failed for %d rows.", len(df))
            raise

    @staticmethod
    def _timestamp() -> datetime:
        return datetime.now(UTC)

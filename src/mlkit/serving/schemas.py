from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


def prediction_response_schema(
    time_column: str, target_classes: tuple[int, ...] = (0, 1)
) -> type[BaseModel]:
    """Create the API response schema for a use case's timestamp field.

    Args:
        time_column: Name of the timestamp field.
        target_classes: Classes for prediction.

    Returns:
        Response schema class.
    """
    prediction_fields = (
        {
            "predicted_class": (Literal.__getitem__(target_classes), ...),
            "predicted_probabilities": (dict[str, Probability], ...),
        }
        if len(target_classes) > 2
        else {
            "predicted_class": (Literal[0, 1], ...),
            "predicted_probability": (float, Field(ge=0.0, le=1.0)),
        }
    )
    return create_model(
        f"PredictionResponse_{time_column}",
        **{
            time_column: (datetime, ...),
            **prediction_fields,
            "metadata": (dict[str, Any], ...),
        },
    )


class BaseInferenceEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    timestamp: datetime
    latency_ms: float = Field(ge=0)

    model_name: str
    model_alias: str | None
    model_version: int

    # Optional feature logging
    features: dict[str, Any] | None = None


class PredictionEvent(BaseInferenceEvent):
    event: Literal["prediction"] = "prediction"

    predicted_class: Literal[0, 1]
    predicted_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)


class MulticlassPredictionEvent(BaseInferenceEvent):
    """Prediction event containing all multiclass probabilities."""

    event: Literal["prediction"] = "prediction"
    predicted_class: int
    predicted_probabilities: dict[str, Probability]


class PredictionErrorEvent(BaseInferenceEvent):
    event: Literal["prediction_error"] = "prediction_error"

    error_type: str
    error_message: str
    traceback: str

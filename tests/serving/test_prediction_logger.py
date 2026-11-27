import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from mlkit.config import ServingSettings
from mlkit.serving.model_loader import ModelMetadata
from mlkit.serving.prediction_logger import PredictionLogger
from mlkit.serving.predictor import PredictionResult
from mlkit.use_cases import UseCase


@pytest.mark.unit
def test_prediction_log_rejects_inconsistent_multiclass_output(
    tmp_path: Path, multiclass_use_case: UseCase
) -> None:
    logger = PredictionLogger(
        prediction_log_path=tmp_path / "predictions.jsonl",
        error_log_path=tmp_path / "errors.jsonl",
        log_features=False,
        use_case=multiclass_use_case,
    )
    prediction = PredictionResult(
        predicted_class=0,
        timestamp=datetime.now(UTC),
        metadata=ModelMetadata("sample", None, 1, None),
        predicted_probabilities={"0": 0.1, "1": 0.2, "2": 0.7},
    )
    with pytest.raises(ValueError, match="argmax"):
        logger.log_prediction(
            request_id="test", latency_ms=1, prediction=prediction, features=None
        )
    assert not (tmp_path / "predictions.jsonl").exists()


def test_log_prediction_writes_prediction_event(
    binary_use_case: UseCase,
    tmp_path: Path,
    sample_input: BaseModel,
    sample_prediction_result: PredictionResult,
) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    error_path = tmp_path / "prediction_errors.jsonl"

    logger = PredictionLogger(
        prediction_log_path=pred_path,
        error_log_path=error_path,
        log_features=True,
        use_case=replace(binary_use_case, time_column="event_date"),
    )

    prediction = sample_prediction_result

    logger.log_prediction(
        request_id="req-123",
        latency_ms=15.2,
        prediction=prediction,
        features=sample_input,
    )

    record = json.loads(pred_path.read_text().splitlines()[0])

    assert record["request_id"] == "req-123"
    assert record["predicted_class"] == sample_prediction_result.predicted_class
    assert (
        record["predicted_probability"]
        == sample_prediction_result.predicted_probability
    )
    assert record["event_date"]
    assert "reference_date" not in record
    assert record["features"] == sample_input.model_dump()


def test_log_error_writes_error_event_with_features(
    binary_use_case: UseCase,
    tmp_path: Path,
    sample_input: BaseModel,
) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    error_path = tmp_path / "prediction_errors.jsonl"

    logger = PredictionLogger(
        prediction_log_path=pred_path,
        error_log_path=error_path,
        log_features=True,
        use_case=replace(binary_use_case, time_column="event_date"),
    )

    class FakePredictor:
        use_case = replace(binary_use_case, time_column="event_date")

        def __init__(self):
            self.metadata = SimpleNamespace(
                model_name="test_model",
                model_alias="champion",
                model_version=1,
            )

        @staticmethod
        def _timestamp():
            return "2026-01-01T00:00:00"

    exc = ValueError("invalid input")

    logger.log_error(
        request_id="req-1",
        latency_ms=12.3,
        predictor=FakePredictor(),
        exception=exc,
        traceback_text="traceback content",
        features=sample_input,
    )

    lines = error_path.read_text().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["event"] == "prediction_error"
    assert record["request_id"] == "req-1"
    assert record["event_date"] == "2026-01-01T00:00:00"
    assert "reference_date" not in record
    assert record["error_type"] == "ValueError"
    assert record["error_message"] == "invalid input"
    assert record["traceback"] == "traceback content"
    assert record["features"] == sample_input.model_dump()


def test_default_settings_omit_features_from_prediction_and_error_logs(
    binary_use_case: UseCase,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_input: BaseModel,
    sample_prediction_result: PredictionResult,
) -> None:
    monkeypatch.delenv("LOG_FEATURES", raising=False)
    settings = ServingSettings(logging_dir=tmp_path)
    logger = PredictionLogger(
        prediction_log_path=settings.prediction_log_path,
        error_log_path=settings.error_log_path,
        log_features=settings.log_features,
        use_case=binary_use_case,
    )
    logger.log_prediction(
        request_id="request",
        latency_ms=1,
        prediction=sample_prediction_result,
        features=sample_input,
    )
    predictor = SimpleNamespace(
        metadata=sample_prediction_result.metadata,
        _timestamp=lambda: "2026-01-01T00:00:00",
    )
    logger.log_error(
        request_id="request",
        latency_ms=1,
        predictor=predictor,
        exception=ValueError("error"),
        traceback_text="traceback",
        features=sample_input,
    )

    for path in (settings.prediction_log_path, settings.error_log_path):
        assert "features" not in json.loads(path.read_text())


def test_prediction_logger_init_creates_parent_directories(
    binary_use_case: UseCase, tmp_path: Path
) -> None:
    pred_path = tmp_path / "nested" / "predictions.jsonl"
    error_path = tmp_path / "nested" / "prediction_errors.jsonl"

    PredictionLogger(
        prediction_log_path=pred_path,
        error_log_path=error_path,
        log_features=False,
        use_case=binary_use_case,
    )

    assert pred_path.parent.exists()
    assert error_path.parent.exists()


def test_append_jsonl_appends_multiple_records(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"

    PredictionLogger._append_jsonl(path, {"a": 1, "b": "x"})
    PredictionLogger._append_jsonl(path, {"a": 2, "b": "y"})

    lines = path.read_text().splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1, "b": "x"}
    assert json.loads(lines[1]) == {"a": 2, "b": "y"}

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, create_model

from mlkit import batch_predict
from mlkit.models import IdentityFeatureBuilder
from mlkit.monitoring import core
from mlkit.serving import api
from mlkit.serving.model_loader import LoadedModel, ModelMetadata
from mlkit.training import train_model
from mlkit.use_cases import UseCase
from mlkit.use_cases.registry import get_use_case


class AlternateInput(BaseModel):
    amount: float
    category: str


ALTERNATE_USE_CASE = UseCase(
    name="alternate",
    target_column="label",
    time_column="event_date",
    feature_schema=AlternateInput,
    feature_builder_factory=IdentityFeatureBuilder,
    index_column="record_id",
    metadata_fields={"record_id": int},
)


@pytest.mark.unit
def test_synthetic_registry_discovers_use_cases(
    synthetic_use_case_registry: None,
    binary_use_case: UseCase,
    multiclass_use_case: UseCase,
) -> None:
    assert get_use_case(binary_use_case.name) is binary_use_case
    assert get_use_case(multiclass_use_case.name) is multiclass_use_case
    with pytest.raises(ValueError, match="Unknown use case"):
        get_use_case("missing")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("classes", "strategy"),
    [
        ((0, 1), "max_f1"),
        ((2, 0, 1), "argmax"),
        ((10, 5, 2, 7), "argmax"),
    ],
)
def test_class_selection_strategy_follows_target_classes(
    classes: tuple[int, ...], strategy: str
) -> None:
    use_case = replace(ALTERNATE_USE_CASE, target_classes=classes)
    assert use_case.class_selection_strategy == strategy


@pytest.mark.unit
def test_use_case_rejects_negative_multiclass_label(
    multiclass_use_case: UseCase,
) -> None:
    with pytest.raises(ValueError, match="nonnegative integers"):
        replace(multiclass_use_case, target_classes=(-1, 0, 1))


def test_split_strategy_contract_validates_time_and_event_metadata() -> None:
    random_case = replace(
        ALTERNATE_USE_CASE, time_column=None, split_strategy="stratified_random"
    )
    data = _alternate_df(2).drop(columns="event_date")
    data["timestamp"] = pd.date_range("2026-01-01", periods=2)
    assert "timestamp" in random_case.validate(data).columns
    with pytest.raises(ValueError, match="require a nonempty time column"):
        replace(ALTERNATE_USE_CASE, time_column=None)
    with pytest.raises(ValueError, match="split strategy"):
        replace(ALTERNATE_USE_CASE, split_strategy="invalid")


def _alternate_df(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": np.arange(rows, dtype=float),
            "category": ["even" if row % 2 == 0 else "odd" for row in range(rows)],
            "record_id": range(rows),
            "label": [row % 2 for row in range(rows)],
            "event_date": pd.date_range("2026-01-01", periods=rows),
        }
    )


@pytest.mark.unit
def test_alternate_use_case_runs_training_batch_api_and_monitoring(
    config_factory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = config_factory(classifier="dt")
    df = _alternate_df()
    result = train_model(
        use_case=ALTERNATE_USE_CASE,
        config=config,
        fit_df=df.iloc[:6],
        validation_df=df.iloc[6:8],
        test_df=df.iloc[8:],
    )
    assert result.feature_names_in == list(ALTERNATE_USE_CASE.feature_columns)
    loaded_model = LoadedModel(
        model=result.trained_pipeline,
        metadata=ModelMetadata(
            model_name="alternate-model",
            model_alias=None,
            model_version=1,
            model_threshold=result.threshold,
        ),
        use_case=ALTERNATE_USE_CASE,
    )
    monkeypatch.setattr(batch_predict, "load_model", lambda **_: loaded_model)

    scored = batch_predict.run_batch_prediction(
        df=df.drop(columns="label"),
        use_case=ALTERNATE_USE_CASE,
    )
    assert scored["predicted_probability"].between(0, 1).all()
    assert scored["record_id"].tolist() == df["record_id"].tolist()

    api_settings = SimpleNamespace(
        model_manifest=None,
        model_version=None,
        model_name="alternate-model",
        prediction_log_path=tmp_path / "predictions.jsonl",
        error_log_path=tmp_path / "errors.jsonl",
        log_features=True,
        request_id_header="X-Request-ID",
    )
    monkeypatch.setattr(api, "ServingSettings", lambda: api_settings)
    monkeypatch.setattr(api, "load_model", lambda **_: loaded_model)
    with TestClient(api.create_app(ALTERNATE_USE_CASE)) as client:
        valid = client.post(
            "/predict",
            json={"amount": 12, "category": "even"},
            params={"timestamp": "2026-01-11T00:00:00"},
        )
        invalid = client.post("/predict", json={"amount": "invalid"})

    assert valid.status_code == 200
    assert valid.json()["predicted_class"] in {0, 1}
    assert valid.json()["event_date"] == "2026-01-11T00:00:00"
    assert "reference_date" not in valid.json()
    openapi = client.get("/openapi.json").json()
    response_schema = openapi["components"]["schemas"]["PredictionResponse_event_date"]
    assert "event_date" in response_schema["properties"]
    assert "reference_date" not in response_schema["properties"]
    assert invalid.status_code == 422
    prediction_event = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert prediction_event["event_date"] == "2026-01-11T00:00:00"
    assert "reference_date" not in prediction_event

    reference = _alternate_df(4).assign(
        predicted_probability=[0.1, 0.9, 0.2, 0.8],
        predicted_class=[0, 1, 0, 1],
        model_version=1,
    )
    analysis = _alternate_df(4).assign(
        event_date=pd.date_range("2026-02-01", periods=4),
        predicted_probability=[0.1, 0.9, 0.2, 0.8],
        predicted_class=[0, 1, 0, 1],
        model_version=1,
    )
    prepared = core._prepare_validated_df(
        df=reference.copy(),
        name="reference",
        source="batch",
        use_case=ALTERNATE_USE_CASE,
        require_target=True,
        require_features=True,
    )
    assert ALTERNATE_USE_CASE.time_column in prepared
    assert prepared[ALTERNATE_USE_CASE.target_column].nunique() == 2
    report = core.build_monitoring_report(
        reference=reference,
        analysis=analysis,
        use_case=ALTERNATE_USE_CASE,
        source_reference="batch",
        source_analysis="batch",
        drift=False,
    )
    assert not report.summary.empty


@pytest.mark.unit
@pytest.mark.parametrize("feature_column", ["label", "event_date"])
def test_use_case_rejects_target_and_time_as_features(
    feature_column: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="feature_columns must not include target or time columns",
    ):
        UseCase(
            name="invalid",
            target_column="label",
            time_column="event_date",
            feature_schema=create_model("InvalidInput", **{feature_column: (int, ...)}),
            feature_builder_factory=IdentityFeatureBuilder,
        )

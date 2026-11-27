"""Tests for the UCI students multiclass use case."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient
from pydantic import ValidationError
from use_cases.students import STUDENTS_USE_CASE, StudentsInputFeatures
from use_cases.students.features import CATEGORICAL_COLUMNS, StudentsFeatureBuilder

from mlkit.config import RuntimeSettings
from mlkit.config.loader import load_config
from mlkit.data.splitting import create_data_splits
from mlkit.data.uci import prepare_uci_snapshot
from mlkit.monitoring.core import _event_log_to_df, build_monitoring_report
from mlkit.serving import api
from mlkit.serving.model_loader import LoadedModel, ModelMetadata, load_model
from mlkit.serving.predictor import Predictor
from mlkit.tracking import TrainingManifest
from mlkit.train import run_training_job
from mlkit.training.model_training import train_model
from mlkit.use_cases import get_use_case


def _record() -> dict[str, int | float]:
    return {
        "marital_status": 1,
        "application_mode": 1,
        "application_order": 1,
        "course": 33,
        "daytime_evening_attendance": 1,
        "previous_qualification": 1,
        "previous_qualification_grade": 120.5,
        "nacionality": 1,
        "mothers_qualification": 1,
        "fathers_qualification": 1,
        "mothers_occupation": 0,
        "fathers_occupation": 0,
        "admission_grade": 130.0,
        "displaced": 0,
        "educational_special_needs": 0,
        "debtor": 0,
        "tuition_fees_up_to_date": 1,
        "gender": 0,
        "scholarship_holder": 0,
        "age_at_enrollment": 20,
        "international": 0,
        "curricular_units_1st_sem_credited": 0,
        "curricular_units_1st_sem_enrolled": 5,
        "curricular_units_1st_sem_evaluations": 5,
        "curricular_units_1st_sem_approved": 4,
        "curricular_units_1st_sem_grade": 13.25,
        "curricular_units_1st_sem_without_evaluations": 0,
        "curricular_units_2nd_sem_credited": 0,
        "curricular_units_2nd_sem_enrolled": 5,
        "curricular_units_2nd_sem_evaluations": 5,
        "curricular_units_2nd_sem_approved": 4,
        "curricular_units_2nd_sem_grade": 14.75,
        "curricular_units_2nd_sem_without_evaluations": 0,
        "unemployment_rate": 10.1,
        "inflation_rate": -0.5,
        "gdp": -1.2,
    }


def _frame(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame([_record()] * rows)
    frame["age_at_enrollment"] = rng.integers(17, 40, rows)
    frame["course"] = rng.choice([33, 171, 8014], rows)
    frame["application_mode"] = rng.choice([1, 2, 5], rows)
    frame["marital_status"] = rng.choice([1, 2, 3], rows)
    frame["gender"] = rng.integers(0, 2, rows)
    frame["admission_grade"] = rng.uniform(95, 190, rows)
    frame["curricular_units_1st_sem_grade"] = rng.uniform(0, 20, rows)
    frame["curricular_units_2nd_sem_grade"] = rng.uniform(0, 20, rows)
    frame["inflation_rate"] = rng.uniform(-1, 3, rows)
    frame["target"] = np.arange(rows) % 3
    return frame


@pytest.mark.usecase
def test_contract_and_snapshot_mapping(tmp_path: Path) -> None:
    assert get_use_case("students") is STUDENTS_USE_CASE
    assert len(STUDENTS_USE_CASE.feature_columns) == 36
    source = _frame(6).drop(columns="target")
    source.to_parquet(tmp_path / "features.parquet")
    pd.DataFrame({"Target": ["Dropout", "Enrolled", "Graduate"] * 2}).to_parquet(
        tmp_path / "targets.parquet"
    )
    prepare_uci_snapshot(
        snapshot_dir=tmp_path,
        output_path=tmp_path / "prepared.parquet",
        use_case=STUDENTS_USE_CASE,
        target_mappings={"target": {"Dropout": 0, "Enrolled": 1, "Graduate": 2}},
    )
    prepared = pd.read_parquet(tmp_path / "prepared.parquet")
    assert prepared["target"].tolist() == [0, 1, 2, 0, 1, 2]
    assert prepared["curricular_units_2nd_sem_grade"].iloc[0] != int(
        prepared["curricular_units_2nd_sem_grade"].iloc[0]
    )


@pytest.mark.usecase
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("course", 42),
        ("gender", 2),
        ("age_at_enrollment", -1),
        ("curricular_units_1st_sem_grade", 20.1),
        ("admission_grade", float("inf")),
        ("inflation_rate", float("nan")),
        ("nacionality", None),
    ],
)
def test_schema_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        StudentsInputFeatures.model_validate(_record() | {field: value})


@pytest.mark.usecase
def test_builder_marks_categories_without_mutating_input() -> None:
    source = _frame(6).drop(columns="target")
    original = source.copy()
    result = StudentsFeatureBuilder().fit_transform(source)
    pd.testing.assert_frame_equal(source, original)
    assert all(
        isinstance(result[name].dtype, pd.CategoricalDtype)
        for name in CATEGORICAL_COLUMNS
    )
    assert pd.api.types.is_numeric_dtype(result["curricular_units_1st_sem_grade"])


@pytest.mark.usecase
@pytest.mark.filterwarnings(
    "ignore:'future.no_silent_downcasting' is deprecated",
    "ignore:The resulting number of chunks is too low.",
)
def test_students_training_batch_api_and_monitoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frame(600)
    settings = RuntimeSettings(data_dir=tmp_path)
    create_data_splits(STUDENTS_USE_CASE, frame, split_name="s", settings=settings)
    fit, validation, test, monitoring = [
        pd.read_parquet(tmp_path / f"splits/s_{part}.parquet")
        for part in ("fit", "validation", "test", "monitoring")
    ]
    config, _ = load_config(
        "lr.yaml", path=Path(__file__).resolve().parents[3] / "src/config"
    )
    result = train_model(
        STUDENTS_USE_CASE, config, fit, validation, test, calibration_method="sigmoid"
    )
    assert result.threshold is None
    assert np.isfinite(result.metrics["roc_auc"])
    loaded = LoadedModel(
        result.trained_pipeline,
        ModelMetadata("students", None, 1, None),
        STUDENTS_USE_CASE,
    )
    predictor = Predictor(loaded)
    reference = predictor.predict_batch(test)
    assert all(f"predicted_probability_{c}" in reference for c in (0, 1, 2))
    assert "predicted_probability" not in reference and "threshold" not in reference

    serving = SimpleNamespace(
        model_manifest=None,
        model_version=None,
        model_name="students",
        prediction_log_path=tmp_path / "predictions.jsonl",
        error_log_path=tmp_path / "errors.jsonl",
        log_features=True,
        request_id_header="X-Request-ID",
    )
    monkeypatch.setattr(api, "ServingSettings", lambda: serving)
    monkeypatch.setattr(api, "load_model", lambda **_: loaded)
    with TestClient(api.create_app(STUDENTS_USE_CASE)) as client:
        for record in monitoring.drop(columns="target").to_dict("records"):
            response = client.post(
                "/predict", json=record, params={"timestamp": "2026-11-01T00:00:00"}
            )
            assert response.status_code == 200
            payload = response.json()
            assert set(payload["predicted_probabilities"]) == {"0", "1", "2"}
            assert "predicted_probability" not in payload
    events = _event_log_to_df(
        serving.prediction_log_path, "prediction", STUDENTS_USE_CASE
    )
    assert json.loads(serving.prediction_log_path.read_text().splitlines()[0])[
        "predicted_probabilities"
    ]
    report = build_monitoring_report(
        reference, events, use_case=STUDENTS_USE_CASE, drift=True
    )
    assert not report.summary.empty
    assert "predicted_class_2_rate" in report.summary


@pytest.mark.integration
def test_students_registered_model_and_manifest_use_argmax(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings = RuntimeSettings(
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        mlflow_tracking_uri=tracking_uri,
    )
    create_data_splits(
        STUDENTS_USE_CASE, _frame(300), split_name="s", settings=settings
    )
    result = run_training_job(
        use_case="students",
        config_file="lr.yaml",
        split_name="s",
        register_model=True,
        settings=settings,
    )
    manifest = TrainingManifest.read(tmp_path / "manifests/students-lr-s.json")
    assert manifest.schema_version == 5
    assert manifest.training.class_selection_strategy == "argmax"
    assert manifest.training.threshold is None
    params = (
        MlflowClient(tracking_uri=tracking_uri).get_run(manifest.run.run_id).data.params
    )
    assert params["class_selection_strategy"] == "argmax"
    assert "threshold_strategy" not in params
    assert "class_threshold" not in params

    reference = pd.read_parquet(tmp_path / "splits/s_test.parquet")
    manifest_model = load_model(
        use_case=STUDENTS_USE_CASE, tracking_uri=tracking_uri, manifest=manifest
    )
    registry_model = load_model(
        use_case=STUDENTS_USE_CASE,
        tracking_uri=tracking_uri,
        model_name="students-model",
        model_version=manifest.model.model_version,
    )
    expected = Predictor(
        LoadedModel(
            result.trained_pipeline,
            ModelMetadata("students-model", None, manifest.model.model_version, None),
            STUDENTS_USE_CASE,
        )
    ).predict_batch(reference)
    for loaded in (manifest_model, registry_model):
        actual = Predictor(loaded).predict_batch(reference)
        pd.testing.assert_series_equal(
            actual["predicted_class"], expected["predicted_class"]
        )

"""Tests for the Secondary Mushroom use case."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression
from use_cases.mushroom import (
    MUSHROOM_USE_CASE,
    MushroomFeatureBuilder,
    MushroomInputFeatures,
)
from use_cases.mushroom.features import MISSING_CATEGORY, NULLABLE_CATEGORIES

import mlkit
from mlkit.config import RuntimeSettings
from mlkit.data.uci import prepare_uci_snapshot
from mlkit.models.pipelines import build_classifier_pipeline
from mlkit.monitoring.core import _event_log_to_df, build_monitoring_report
from mlkit.serving import api
from mlkit.serving.model_loader import LoadedModel, ModelMetadata
from mlkit.train import run_training_job
from mlkit.use_cases import get_use_case


def _record() -> dict[str, object]:
    return {
        "cap_diameter": 3.5,
        "cap_shape": "x",
        "cap_surface": None,
        "cap_color": "n",
        "does_bruise_or_bleed": "f",
        "gill_attachment": None,
        "gill_spacing": None,
        "gill_color": "w",
        "stem_height": 5.0,
        "stem_width": 8.0,
        "stem_root": None,
        "stem_surface": None,
        "stem_color": "w",
        "veil_type": None,
        "veil_color": None,
        "has_ring": "f",
        "ring_type": None,
        "spore_print_color": None,
        "habitat": "d",
        "season": "a",
    }


def _features(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame([_record()] * rows)
    for column, choices in {
        "cap_shape": ["b", "x", "f"],
        "cap_surface": [None, "s", "d"],
        "cap_color": ["n", "w", "y"],
        "does_bruise_or_bleed": ["f", "t"],
        "gill_attachment": [None, "a", "x"],
        "gill_spacing": [None, "c", "d"],
        "gill_color": ["w", "n", "f"],
        "stem_root": [None, "b", "f"],
        "stem_surface": [None, "s", "f"],
        "stem_color": ["w", "n", "f"],
        "veil_type": [None, "u"],
        "veil_color": [None, "w", "n"],
        "has_ring": ["f", "t"],
        "ring_type": [None, "f", "e"],
        "spore_print_color": [None, "w", "n"],
        "habitat": ["d", "g", "l"],
        "season": ["a", "s", "u", "w"],
    }.items():
        frame[column] = rng.choice(choices, size=rows)
    for column in ("cap_diameter", "stem_height", "stem_width"):
        frame[column] = rng.uniform(0, 20, size=rows)
    return frame


@pytest.mark.usecase
def test_contract_discovery_and_schema() -> None:
    assert get_use_case("mushroom") is MUSHROOM_USE_CASE
    assert MUSHROOM_USE_CASE.target_column == "class"
    assert MUSHROOM_USE_CASE.time_column is None
    assert MUSHROOM_USE_CASE.split_strategy == "stratified_random"
    assert len(MUSHROOM_USE_CASE.feature_columns) == 20
    record = _record()
    assert MushroomInputFeatures(**record).cap_surface is None
    assert MushroomInputFeatures(**(record | {"cap_surface": "d", "stem_root": "f"}))
    assert MushroomInputFeatures(
        **(record | {"gill_attachment": "?", "ring_type": "?"})
    )
    del record["cap_surface"]
    with pytest.raises(ValidationError):
        MushroomInputFeatures(**record)


@pytest.mark.usecase
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cap_shape", None),
        ("cap_shape", "invalid"),
        ("cap_surface", "__missing__"),
        ("stem_height", -1),
        ("stem_width", float("inf")),
        ("cap_diameter", float("nan")),
    ],
)
def test_schema_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        MushroomInputFeatures(**(_record() | {field: value}))


@pytest.mark.usecase
def test_snapshot_mapping_and_validation(tmp_path: Path) -> None:
    source = _features(4)
    source.columns = source.columns.str.replace("_", "-")
    source.to_parquet(tmp_path / "features.parquet")
    pd.DataFrame({"class": ["p", "e", "p", "e"]}).to_parquet(
        tmp_path / "targets.parquet"
    )
    output = tmp_path / "prepared.parquet"
    mappings = {"class": {"p": 0, "e": 1}}
    prepare_uci_snapshot(
        tmp_path, output, target_mappings=mappings, use_case=MUSHROOM_USE_CASE
    )
    prepared = pd.read_parquet(output)
    assert list(prepared.columns) == [*MUSHROOM_USE_CASE.feature_columns, "class"]
    assert prepared["class"].tolist() == [0, 1, 0, 1]
    pd.DataFrame({"class": ["p", "e", "?", "e"]}).to_parquet(
        tmp_path / "targets.parquet"
    )
    with pytest.raises(ValueError, match="unmapped"):
        prepare_uci_snapshot(
            tmp_path, output, target_mappings=mappings, use_case=MUSHROOM_USE_CASE
        )
    source.assign(**{"class": [1, 0, 1, 0]}).to_parquet(tmp_path / "features.parquet")
    with pytest.raises(ValueError, match="must not overlap"):
        prepare_uci_snapshot(
            tmp_path, output, target_mappings=mappings, use_case=MUSHROOM_USE_CASE
        )


@pytest.mark.usecase
def test_features_and_preprocessor_handle_missing_categories() -> None:
    source = MUSHROOM_USE_CASE.validate(_features(40))
    source["veil_type"] = None
    original = source.copy(deep=True)
    transformed = MushroomFeatureBuilder().fit_transform(source)
    pd.testing.assert_frame_equal(source, original)
    assert (transformed["veil_type"] == MISSING_CATEGORY).all()
    assert all(not transformed[column].isna().any() for column in NULLABLE_CATEGORIES)

    pipeline = build_classifier_pipeline(LogisticRegression(), MushroomFeatureBuilder())
    pipeline.fit(source, np.arange(len(source)) % 2)
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    assert "veil_type_na" in names
    assert "cap_diameter" in names
    unseen = pd.DataFrame([_record() | {"veil_type": "p", "cap_shape": "s"}])
    assert np.isfinite(pipeline.predict_proba(unseen)).all()


@pytest.mark.usecase
@pytest.mark.filterwarnings(
    "ignore:'future.no_silent_downcasting' is deprecated",
    "ignore:The resulting number of chunks is too low.",
)
def test_training_serving_and_drift_with_missing_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _features(2400)
    features["class"] = (features["cap_shape"] == "x").astype(int)
    features.to_parquet(tmp_path / "prepared.parquet", index=False)
    monkeypatch.setattr(
        mlkit, "RuntimeSettings", lambda: RuntimeSettings(data_dir=tmp_path)
    )
    mlkit.main(
        [
            "create-split",
            "--use_case",
            "mushroom",
            "--input_parquet",
            "prepared.parquet",
            "--split_name",
            "s",
            "--random_state",
            "42",
        ]
    )
    result = run_training_job(
        use_case="mushroom",
        config_file="lr.yaml",
        split_name="s",
        config_dir=Path(__file__).resolve().parents[3] / "src/config",
        split_dir=tmp_path / "splits",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_dir=tmp_path / "artifacts",
        settings=RuntimeSettings(data_dir=tmp_path),
    )
    reference = pd.read_parquet(tmp_path / "splits/s_test.parquet")
    sample = pd.read_parquet(tmp_path / "splits/s_monitoring.parquet")
    fit = pd.read_parquet(tmp_path / "splits/s_fit.parquet")
    validation = pd.read_parquet(tmp_path / "splits/s_validation.parquet")
    groups = [
        set(frame["cap_diameter"]) for frame in (fit, validation, reference, sample)
    ]
    assert sum(map(len, groups)) == len(set.union(*groups)) == len(features)
    probabilities = result.trained_pipeline.predict_proba(
        reference.loc[:, list(MUSHROOM_USE_CASE.feature_columns)]
    )[:, 1]
    reference["predicted_probability"] = probabilities
    reference["predicted_class"] = (probabilities >= result.threshold).astype(int)
    reference["model_version"] = 1

    serving = SimpleNamespace(
        model_manifest=None,
        model_version=None,
        model_name="mushroom-model",
        prediction_log_path=tmp_path / "predictions.jsonl",
        error_log_path=tmp_path / "errors.jsonl",
        log_features=True,
        request_id_header="X-Request-ID",
    )
    loaded = LoadedModel(
        model=result.trained_pipeline,
        metadata=ModelMetadata("mushroom-model", None, 1, result.threshold),
        use_case=MUSHROOM_USE_CASE,
    )
    monkeypatch.setattr(api, "ServingSettings", lambda: serving)
    monkeypatch.setattr(api, "load_model", lambda **_: loaded)
    with TestClient(api.create_app(MUSHROOM_USE_CASE)) as client:
        for row in sample.drop(columns="class").to_dict("records"):
            record = MushroomInputFeatures.model_validate(
                {key: None if pd.isna(value) else value for key, value in row.items()}
            )
            response = client.post(
                "/predict",
                json=json.loads(record.model_dump_json()),
                params={"timestamp": "2026-10-01T00:00:00"},
            )
            assert response.status_code == 200
    events = _event_log_to_df(
        serving.prediction_log_path, "prediction", MUSHROOM_USE_CASE
    )
    assert events["cap_surface"].isna().any()
    report = build_monitoring_report(
        reference, events, use_case=MUSHROOM_USE_CASE, drift=True
    )
    assert not report.summary.empty

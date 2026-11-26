"""Tests for the CDC diabetes use case."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression
from use_cases.diabetes import (
    DIABETES_USE_CASE,
    DiabetesFeatureBuilder,
    DiabetesInputFeatures,
)

import mlkit
from mlkit.config import RuntimeSettings, ServingSettings
from mlkit.data.uci import fetch_uci_snapshot, prepare_uci_snapshot
from mlkit.models.pipelines import build_classifier_pipeline
from mlkit.serving.serve_samples import load_sample_data
from mlkit.train import run_training_job
from mlkit.use_cases import get_use_case


def _features(rows: int) -> pd.DataFrame:
    """Create representative source columns with both binary classes."""
    data = pd.DataFrame(
        {
            "HighBP": np.arange(rows) % 2,
            "HighChol": np.arange(rows) % 2,
            "CholCheck": np.ones(rows, dtype=int),
            "BMI": 20 + np.arange(rows) % 20,
            "Smoker": np.arange(rows) % 2,
            "Stroke": np.zeros(rows, dtype=int),
            "HeartDiseaseorAttack": np.zeros(rows, dtype=int),
            "PhysActivity": np.ones(rows, dtype=int),
            "Fruits": np.ones(rows, dtype=int),
            "Veggies": np.ones(rows, dtype=int),
            "HvyAlcoholConsump": np.zeros(rows, dtype=int),
            "AnyHealthcare": np.ones(rows, dtype=int),
            "NoDocbcCost": np.zeros(rows, dtype=int),
            "GenHlth": 1 + np.arange(rows) % 5,
            "MentHlth": np.arange(rows) % 31,
            "PhysHlth": np.arange(rows) % 31,
            "DiffWalk": np.zeros(rows, dtype=int),
            "Sex": np.arange(rows) % 2,
            "Age": 1 + np.arange(rows) % 13,
            "Education": 1 + np.arange(rows) % 6,
            "Income": 1 + np.arange(rows) % 8,
        }
    )
    return data


def _record() -> dict[str, int]:
    """Return one valid normalized feature record."""
    record = {name: 0 for name in DiabetesInputFeatures.model_fields}
    record.update(bmi=25, gen_hlth=1, age=1, education=1, income=1)
    return record


@pytest.mark.usecase
def test_contract_is_discovered() -> None:
    assert get_use_case("diabetes") is DIABETES_USE_CASE
    assert DIABETES_USE_CASE.target_column == "diabetes_binary"
    assert DIABETES_USE_CASE.time_column is None
    assert DIABETES_USE_CASE.split_strategy == "stratified_random"
    assert DIABETES_USE_CASE.index_column == "id"
    assert DIABETES_USE_CASE.metadata_fields == {"id": int}
    assert "id" not in DIABETES_USE_CASE.feature_columns


@pytest.mark.usecase
@pytest.mark.parametrize(
    ("field", "valid", "invalid"),
    [
        ("high_bp", 1, 2),
        ("sex", 0, -1),
        ("bmi", 0, -1),
        ("ment_hlth", 0, -1),
        ("phys_hlth", 30, 31),
        ("gen_hlth", 5, 6),
        ("age", 13, 14),
        ("education", 6, 0),
        ("income", 8, 9),
        ("age", 1, 1.5),
    ],
)
def test_schema_bounds(field: str, valid: int, invalid: float) -> None:
    record = _record()
    assert getattr(DiabetesInputFeatures(**(record | {field: valid})), field) == valid
    with pytest.raises(ValidationError):
        DiabetesInputFeatures(**(record | {field: invalid}))


@pytest.mark.usecase
def test_schema_rejects_missing_and_nonfinite_values() -> None:
    record = _record()
    del record["income"]
    with pytest.raises(ValidationError):
        DiabetesInputFeatures(**record)
    with pytest.raises(ValidationError):
        DiabetesInputFeatures(**(_record() | {"bmi": float("nan")}))


@pytest.mark.usecase
def test_snapshot_preparation_preserves_target_and_filters_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = SimpleNamespace(
        data=SimpleNamespace(
            features=_features(4),
            targets=pd.DataFrame({"Diabetes_binary": [0, 1, 0, 1]}),
            ids=pd.DataFrame({"ID": [0, 1, 2, 3]}),
        ),
        metadata={},
        variables=pd.DataFrame({"name": ["ID", "Diabetes_binary"]}),
    )
    monkeypatch.setitem(
        sys.modules, "ucimlrepo", SimpleNamespace(fetch_ucirepo=lambda **_: dataset)
    )
    fetch_uci_snapshot(891, tmp_path)
    output = tmp_path / "prepared.parquet"
    prepare_uci_snapshot(tmp_path, output, use_case=DIABETES_USE_CASE)
    result = pd.read_parquet(output)
    assert result["diabetes_binary"].tolist() == [0, 1, 0, 1]
    assert list(result.columns) == [
        *DIABETES_USE_CASE.feature_columns,
        "id",
        "diabetes_binary",
    ]
    assert result["id"].tolist() == [0, 1, 2, 3]


@pytest.mark.usecase
def test_identifier_metadata_is_validated_when_present() -> None:
    record = _record()
    result = DIABETES_USE_CASE.validate(pd.DataFrame([record | {"id": 42}]))
    assert result["id"].tolist() == [42]
    with pytest.raises(ValidationError):
        DIABETES_USE_CASE.validate(pd.DataFrame([record | {"id": "invalid"}]))


@pytest.mark.usecase
@pytest.mark.parametrize("overlap", [False, True])
def test_snapshot_rejects_invalid_target(tmp_path: Path, overlap: bool) -> None:
    features = _features(2)
    if overlap:
        features["Diabetes_binary"] = [0, 1]
    features.to_parquet(tmp_path / "features.parquet")
    pd.DataFrame({"Diabetes_binary": [0, 2]}).to_parquet(tmp_path / "targets.parquet")
    with pytest.raises(
        expected_exception=ValueError,
        match="must not overlap" if overlap else r"must be among \(0, 1\)",
    ):
        prepare_uci_snapshot(
            tmp_path, tmp_path / "prepared.parquet", use_case=DIABETES_USE_CASE
        )


@pytest.mark.usecase
def test_feature_builder_and_shared_preprocessor_handle_categories() -> None:
    source = DIABETES_USE_CASE.validate(
        pd.DataFrame([_record(), _record() | {"high_bp": 1}])
    )
    original = source.copy(deep=True)
    transformed = DiabetesFeatureBuilder().fit_transform(source)
    pd.testing.assert_frame_equal(source, original)
    categorical = set(transformed.select_dtypes(include="category").columns)
    assert categorical == {"gen_hlth", "age", "education", "income"}

    pipeline = build_classifier_pipeline(LogisticRegression(), DiabetesFeatureBuilder())
    pipeline.fit(source, [0, 1])
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    assert {"gen_hlth_1", "age_1", "education_1", "income_1"}.issubset(names)
    unseen = pd.DataFrame([_record() | {"age": 13, "education": 6}])
    assert pipeline.predict_proba(unseen).shape == (1, 2)


@pytest.mark.usecase
def test_diabetes_split_training_and_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _features(160)
    features.to_parquet(tmp_path / "features.parquet")
    pd.DataFrame({"ID": np.arange(160)}).to_parquet(tmp_path / "ids.parquet")
    pd.DataFrame({"Diabetes_binary": np.arange(160) % 2}).to_parquet(
        tmp_path / "targets.parquet"
    )
    prepare_uci_snapshot(
        tmp_path, tmp_path / "prepared.parquet", use_case=DIABETES_USE_CASE
    )
    monkeypatch.setattr(
        mlkit, "RuntimeSettings", lambda: RuntimeSettings(data_dir=tmp_path)
    )
    mlkit.main(
        [
            "create-split",
            "--use_case",
            "diabetes",
            "--input_parquet",
            "prepared.parquet",
            "--split_name",
            "s",
            "--random_state",
            "42",
        ]
    )
    config_dir = Path(__file__).resolve().parents[3] / "src/config"
    result = run_training_job(
        use_case="diabetes",
        config_file="lr.yaml",
        split_name="s",
        config_dir=config_dir,
        split_dir=tmp_path / "splits",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_dir=tmp_path / "artifacts",
        settings=RuntimeSettings(data_dir=tmp_path),
    )
    test = pd.read_parquet(tmp_path / "splits/s_test.parquet")
    assert "id" in test.columns
    assert "id" not in result.feature_names_in
    sample = load_sample_data(
        DIABETES_USE_CASE,
        ServingSettings(data_dir=tmp_path),
        input_parquet="splits/s_test.parquet",
        sample_size=2,
    )
    assert sample.index.name == "id"
    assert set(sample.index).issubset(test["id"])
    assert "id" not in sample.columns
    probabilities = result.trained_pipeline.predict_proba(
        test.loc[:, list(DIABETES_USE_CASE.feature_columns)]
    )
    assert probabilities.shape == (len(test), 2)
    assert set(test["diabetes_binary"]) == {0, 1}

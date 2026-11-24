import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from mlflow import MlflowClient
from pydantic import ValidationError
from use_cases.phishing import PHISHING_USE_CASE, PhishingInputFeatures

import mlkit
from mlkit.config import RuntimeSettings, load_config
from mlkit.data.splitting import create_stratified_split
from mlkit.data.uci import prepare_uci_snapshot
from mlkit.evaluation import select_f1_threshold
from mlkit.monitoring.core import _event_log_to_df, build_monitoring_report
from mlkit.serving import api
from mlkit.serving.model_loader import LoadedModel, ModelMetadata
from mlkit.tracking import TrainingManifest
from mlkit.train import run_training_job
from mlkit.training.model_training import split_threshold_validation


def _features(rows: int) -> pd.DataFrame:
    names = [
        "URLLength",
        "DomainLength",
        "IsDomainIP",
        # "URLSimilarityIndex",
        # "CharContinuationRate",
        # "TLDLegitimateProb",
        # "URLCharProb",
        "TLDLength",
        "NoOfSubDomain",
        "HasObfuscation",
        "NoOfObfuscatedChar",
        "ObfuscationRatio",
        "NoOfLettersInURL",
        "LetterRatioInURL",
        "NoOfDegitsInURL",
        "DegitRatioInURL",
        "NoOfEqualsInURL",
        "NoOfQMarkInURL",
        "NoOfAmpersandInURL",
        "NoOfOtherSpecialCharsInURL",
        "SpacialCharRatioInURL",
        "IsHTTPS",
        "LineOfCode",
        "LargestLineLength",
        "HasTitle",
        "DomainTitleMatchScore",
        # "URLTitleMatchScore",
        "HasFavicon",
        "Robots",
        "IsResponsive",
        "NoOfURLRedirect",
        "NoOfSelfRedirect",
        "HasDescription",
        "NoOfPopup",
        "NoOfiFrame",
        "HasExternalFormSubmit",
        "HasSocialNet",
        "HasSubmitButton",
        "HasHiddenFields",
        "HasPasswordField",
        "Bank",
        "Pay",
        "Crypto",
        "HasCopyrightInfo",
        "NoOfImage",
        "NoOfCSS",
        "NoOfJS",
        "NoOfSelfRef",
        "NoOfEmptyRef",
        "NoOfExternalRef",
    ]
    data = {name: np.zeros(rows) for name in names}
    data["URL"] = ["https://example.invalid"] * rows
    data["LetterRatioInURL"] = np.full(rows, 0.75)
    return pd.DataFrame(data)


@pytest.mark.usecase
def test_snapshot_normalizes_numeric_features_and_inverts_target(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _features(4).to_parquet(snapshot / "features.parquet")
    pd.DataFrame({"label": [0, 1, 0, 1]}).to_parquet(snapshot / "targets.parquet")
    output = tmp_path / "prepared.parquet"
    prepare_uci_snapshot(
        snapshot,
        output,
        target_mappings={"label": {0: 1, 1: 0}},
        use_case=PHISHING_USE_CASE,
    )
    result = pd.read_parquet(output)
    assert result["label"].tolist() == [1, 0, 1, 0]
    assert result["letter_ratio_in_url"].tolist() == [0.75] * 4
    assert "url" not in result
    assert list(result.drop(columns="label")) == list(PHISHING_USE_CASE.feature_columns)


@pytest.mark.usecase
def test_schema_rejects_missing_or_invalid_features() -> None:
    record = {name: 0 for name in PhishingInputFeatures.model_fields}
    record["letter_ratio_in_url"] = 0.5
    assert PhishingInputFeatures(**record).letter_ratio_in_url == 0.5
    record["url_length"] = "not-a-number"
    with pytest.raises(ValidationError):
        PhishingInputFeatures(**record)
    record["url_length"] = 20
    record["is_https"] = 2
    with pytest.raises(ValidationError):
        PhishingInputFeatures(**record)


@pytest.mark.usecase
def test_snapshot_rejects_invalid_target(tmp_path: Path) -> None:
    _features(2).to_parquet(tmp_path / "features.parquet")
    pd.DataFrame({"label": [0, 2]}).to_parquet(tmp_path / "targets.parquet")
    with pytest.raises(ValueError, match="unmapped"):
        prepare_uci_snapshot(
            tmp_path,
            tmp_path / "output.parquet",
            target_mappings={"label": {0: 1, 1: 0}},
            use_case=PHISHING_USE_CASE,
        )


@pytest.mark.usecase
def test_snapshot_rejects_target_as_feature(tmp_path: Path) -> None:
    _features(2).assign(label=[0, 1]).to_parquet(tmp_path / "features.parquet")
    pd.DataFrame({"label": [0, 1]}).to_parquet(tmp_path / "targets.parquet")
    with pytest.raises(ValueError, match="must not overlap"):
        prepare_uci_snapshot(
            tmp_path, tmp_path / "output.parquet", use_case=PHISHING_USE_CASE
        )


@pytest.mark.usecase
def test_random_split_is_reproducible_stratified_and_disjoint(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": range(100), "label": [0, 1] * 50})
    for path in (tmp_path / "a", tmp_path / "b"):
        create_stratified_split(
            PHISHING_USE_CASE,
            df,
            settings=RuntimeSettings(data_dir=path),
            split_name="s",
        )
    parts = [
        pd.read_parquet(tmp_path / "a" / "splits" / f"s_{part}.parquet")
        for part in ("train", "test", "monitoring")
    ]
    assert [len(part) for part in parts] == [80, 10, 10]
    assert all(part["label"].mean() == 0.5 for part in parts)
    assert len(set.union(*(set(part["id"]) for part in parts))) == 100
    for part, name in zip(parts, ("train", "test", "monitoring")):
        pd.testing.assert_frame_equal(
            part, pd.read_parquet(tmp_path / "b" / "splits" / f"s_{name}.parquet")
        )


@pytest.mark.usecase
def test_phishing_training_uses_one_holdout_with_independent_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = pd.DataFrame(
        {name: np.zeros(100) for name in PHISHING_USE_CASE.feature_columns}
    )
    data["url_length"] = np.arange(100) % 20
    data["label"] = np.arange(100) % 2
    data.to_parquet(tmp_path / "prepared.parquet", index=False)
    monkeypatch.setattr(
        mlkit, "RuntimeSettings", lambda: RuntimeSettings(data_dir=tmp_path)
    )
    mlkit.main(
        [
            "create-split",
            "--use_case",
            "phishing",
            "--input_parquet",
            "prepared.parquet",
            "--split_name",
            "s",
            "--random_state",
            "7",
        ]
    )
    source = Path(__file__).resolve().parents[3] / "src/config/lr.yaml"
    raw_config = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw_config["random_state"] = 17
    raw_config["tuning"]["n_trials"] = 2
    raw_config["tuning"]["n_splits"] = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = run_training_job(
        use_case="phishing",
        config_file=config_path.name,
        split_name="s",
        config_dir=tmp_path,
        split_dir=tmp_path / "splits",
        tracking_uri=tracking_uri,
        artifact_dir=tmp_path / "artifacts",
        settings=RuntimeSettings(data_dir=tmp_path),
        tune_model=True,
        calibration_method="sigmoid",
    )

    config, _ = load_config(config_path.name, path=tmp_path)
    train = pd.read_parquet(tmp_path / "splits/s_train.parquet")
    fit, validation = split_threshold_validation(train, "label", config)
    probabilities = result.trained_pipeline.predict_proba(
        validation.loc[:, list(PHISHING_USE_CASE.feature_columns)]
    )[:, 1]
    assert result.threshold == select_f1_threshold(validation["label"], probabilities)
    assert len(fit) == 64
    assert len(validation) == 16
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("phishing-s")
    assert experiment is not None
    run = client.search_runs(experiment_ids=[experiment.experiment_id])[0]
    assert run.data.params["random_state"] == "26"
    assert run.data.params["training_random_state"] == "17"
    assert run.data.params["threshold_data"] == "training_validation"
    assert run.data.params["split_strategy"] == "stratified_random"
    assert run.data.params["split_random_state"] == "7"
    assert run.data.params["split_test_fraction"] == "0.1"
    assert run.data.params["split_monitoring_fraction"] == "0.1"
    assert run.data.params["split_fit_rows"] == "64"
    assert run.data.params["split_validation_rows"] == "16"
    manifest = TrainingManifest.read(tmp_path / "manifests/phishing-lr-s.json")
    assert manifest.data.time_column is None
    expected_split_metadata = {
        "split_strategy": "stratified_random",
        "split_train_rows": 80,
        "split_test_rows": 10,
        "split_monitoring_rows": 10,
        "split_random_state": 7,
        "split_test_fraction": 0.1,
        "split_monitoring_fraction": 0.1,
        "split_fit_rows": 64,
        "split_validation_rows": 16,
    }
    assert manifest.data.split_metadata == expected_split_metadata
    assert {key: run.data.params[key] for key in expected_split_metadata} == {
        key: str(value) for key, value in expected_split_metadata.items()
    }

    serving = SimpleNamespace(
        model_manifest=None,
        model_version=None,
        model_name="phishing-model",
        prediction_log_path=tmp_path / "predictions.jsonl",
        error_log_path=tmp_path / "errors.jsonl",
        log_features=True,
        request_id_header="X-Request-ID",
    )
    loaded = LoadedModel(
        model=result.trained_pipeline,
        metadata=ModelMetadata(
            model_name="phishing-model",
            model_alias=None,
            model_version=1,
            model_threshold=result.threshold,
        ),
        use_case=PHISHING_USE_CASE,
    )
    monkeypatch.setattr(api, "ServingSettings", lambda: serving)
    monkeypatch.setattr(api, "load_model", lambda **_: loaded)
    cohort = pd.read_parquet(tmp_path / "splits/s_monitoring.parquet")
    with TestClient(api.create_app(PHISHING_USE_CASE)) as client:
        for record in cohort.drop(columns="label").to_dict("records"):
            response = client.post(
                "/predict", json=record, params={"timestamp": "2024-02-01T00:00:00"}
            )
            assert response.status_code == 200
            assert "timestamp" in response.json()
            assert "reference_date" not in response.json()
    event = json.loads(serving.prediction_log_path.read_text().splitlines()[0])
    assert "timestamp" in event and "reference_date" not in event

    test = pd.read_parquet(tmp_path / "splits/s_test.parquet")
    test["predicted_probability"] = result.trained_pipeline.predict_proba(
        test.loc[:, list(PHISHING_USE_CASE.feature_columns)]
    )[:, 1]
    test["predicted_class"] = (
        test["predicted_probability"] >= result.threshold
    ).astype(int)
    test["model_version"] = 1
    analysis = _event_log_to_df(
        serving.prediction_log_path, "prediction", PHISHING_USE_CASE
    )
    report = build_monitoring_report(
        reference=test,
        analysis=analysis,
        use_case=PHISHING_USE_CASE,
        source_reference="batch",
        source_analysis="api",
        drift=False,
    )
    assert "timestamp" in report.summary.columns
    assert not report.summary.empty

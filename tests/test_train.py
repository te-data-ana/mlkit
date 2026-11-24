import json
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock

import pandas as pd
import pytest

import mlkit
from mlkit import train
from mlkit.config.schemas import ThresholdConfig, TrainingConfig, TuningConfig
from mlkit.data.splitting import create_data_splits, load_split_metadata
from mlkit.training import TuningResult
from mlkit.use_cases import get_use_case

CHURN_USE_CASE = get_use_case("churn")
TEMPORAL_METADATA = {
    "strategy": "temporal",
    "fit_rows": 70,
    "validation_rows": 10,
    "test_rows": 10,
    "monitoring_rows": 10,
    "fit_fraction": 0.7,
    "validation_fraction": 0.1,
    "test_fraction": 0.1,
    "monitoring_fraction": 0.1,
    "fit_actual_fraction": 0.7,
    "validation_actual_fraction": 0.1,
    "test_actual_fraction": 0.1,
    "monitoring_actual_fraction": 0.1,
    "fit_start": "2026-01-01T00:00:00",
    "fit_end": "2026-01-10T00:00:00",
    "validation_start": "2026-01-11T00:00:00",
    "validation_end": "2026-01-31T00:00:00",
    "test_start": "2026-02-01T00:00:00",
    "test_end": "2026-02-02T00:00:00",
    "monitoring_start": "2026-02-03T00:00:00",
    "monitoring_end": "2026-02-04T00:00:00",
}


def write_temporal_metadata(directory: Path, name: str = "temporal") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(TEMPORAL_METADATA), encoding="utf-8"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("registration_args", "expected_registration"),
    [
        (["--register-model"], True),
        (["--register-model", "false"], False),
        (["--no-register-model"], False),
    ],
)
def test_training_cli_passes_run_time_options(
    monkeypatch: pytest.MonkeyPatch,
    registration_args: list[str],
    expected_registration: bool,
) -> None:
    training_job = MagicMock(
        return_value=SimpleNamespace(
            classifier_config={"model_name": "LogisticRegression"},
            metrics={"roc_auc": 0.8},
        )
    )
    monkeypatch.setattr(train, "run_training_job", training_job)

    mlkit.main(
        [
            "train",
            "--use_case",
            "churn",
            "--config",
            "lr.yaml",
            "--calibration_method",
            "isotonic",
            "--n_jobs",
            "2",
            "--tune-model",
            *registration_args,
        ]
    )

    assert training_job.call_args.kwargs["register_model"] is expected_registration
    assert training_job.call_args.kwargs["calibration_method"] == "isotonic"
    assert training_job.call_args.kwargs["n_jobs"] == 2
    assert training_job.call_args.kwargs["tune_model"] is True


@pytest.mark.unit
def test_training_cli_requires_use_case(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        mlkit.main(["train", "--config", "lr.yaml"])

    assert "--use_case" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"calibration_method": "temperature"}, ValueError, "Calibration method"),
        ({"n_jobs": 0}, ValueError, "n_jobs"),
        ({"register_model": "true"}, TypeError, "must be booleans"),
        ({"tune_model": "true"}, TypeError, "must be booleans"),
    ],
)
def test_run_training_job_rejects_invalid_run_time_options(
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, bool | int | str],
    error: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(train, "configure_logging", MagicMock())

    with pytest.raises(error, match=message):
        train.run_training_job(
            use_case="churn",
            config_file="config.yaml",
            split_name="test",
            **options,
        )


@pytest.mark.unit
def test_run_training_job_resolves_required_use_case_before_training(
    monkeypatch: pytest.MonkeyPatch,
    config_factory: Callable[..., TrainingConfig],
    tmp_path: Path,
) -> None:
    config = config_factory(classifier="lr")
    settings = SimpleNamespace(
        config_dir=tmp_path,
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        mlflow_tracking_uri="sqlite:///tracking.db",
        mlflow_experiment_name="test",
    )
    registry_lookup = MagicMock(side_effect=ValueError("stop after use_case lookup"))
    monkeypatch.setattr(train, "configure_logging", MagicMock())
    monkeypatch.setattr(train, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(
        train,
        "load_config",
        lambda **_: (config, tmp_path / "config.yaml"),
    )
    monkeypatch.setattr(train, "get_use_case", registry_lookup)

    with pytest.raises(ValueError, match="stop after use_case lookup"):
        train.run_training_job(
            use_case="alternate",
            config_file="lr.yaml",
            split_name="test",
        )

    registry_lookup.assert_called_once_with("alternate")


@pytest.mark.unit
def test_run_training_job_derives_experiment_from_use_case_and_split(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    caplog.set_level(logging.INFO)
    write_temporal_metadata(tmp_path / "splits")
    settings = SimpleNamespace(
        config_dir=Path("config"),
        data_dir=tmp_path,
        artifact_dir=Path("artifacts"),
        mlflow_tracking_uri="sqlite:///tracking.db",
    )
    config = SimpleNamespace(
        model=SimpleNamespace(classifier="lr"),
        calibration=SimpleNamespace(n_splits=3, n_bins=10),
        threshold=SimpleNamespace(
            strategy="max_f1",
        ),
        promotion=SimpleNamespace(
            metric_weights={
                "roc_auc": 0.2,
                "pr_auc": 0.2,
                "f1": 0.2,
                "brier_score": 0.2,
                "calibration_error": 0.2,
            },
        ),
    )
    use_case = CHURN_USE_CASE.name
    config_file = "train.yaml"
    split_name = "temporal"
    fit_df, validation_df = MagicMock(), MagicMock()
    load_splits = MagicMock(return_value=(fit_df, validation_df, MagicMock()))
    training_result = MagicMock()
    training_result.metrics = {
        "roc_auc": 0.8,
        "pr_auc": 0.8,
        "f1": 0.8,
        "brier_score": 0.2,
        "calibration_error": 0.1,
    }
    train_model_mock = MagicMock(return_value=training_result)
    setup_experiment = MagicMock(return_value="experiment-id")
    mlflow_run = MagicMock()
    get_active_run = train.mlflow.active_run
    assert get_active_run() is None

    monkeypatch.setattr(train, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(train, "configure_logging", lambda: None)
    monkeypatch.setattr(train, "load_config", lambda **_: (config, Path(config_file)))
    monkeypatch.setattr(train, "load_and_validate_training_splits", load_splits)
    monkeypatch.setattr(train, "train_model", train_model_mock)
    monkeypatch.setattr(train, "setup_local_experiment", setup_experiment)
    monkeypatch.setattr(train, "log_experiment_result", MagicMock())
    monkeypatch.setattr(train.mlflow, "get_experiment", MagicMock())
    monkeypatch.setattr(train.mlflow, "start_run", MagicMock(return_value=mlflow_run))
    monkeypatch.setattr(train.mlflow, "log_param", MagicMock())
    monkeypatch.setattr(train.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(
        train.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-id")),
    )
    monkeypatch.setattr(train.mlflow, "set_tags", MagicMock())
    set_tag = MagicMock()
    monkeypatch.setattr(train.mlflow, "set_tag", set_tag)

    train.run_training_job(
        use_case=use_case,
        config_file=config_file,
        split_name=split_name,
        register_model=False,
        write_manifest=False,
    )

    assert (
        setup_experiment.call_args.kwargs["experiment_name"]
        == f"{use_case}-{split_name}"
    )
    assert get_active_run() is None
    set_tag.assert_any_call("git_commit", ANY)
    load_splits.assert_called_once_with(
        split_name=split_name,
        split_dir=tmp_path / "splits",
        use_case=CHURN_USE_CASE,
        metadata=load_split_metadata(tmp_path / "splits/temporal.json"),
    )
    train_model_mock.assert_called_once_with(
        config=config,
        fit_df=fit_df,
        validation_df=validation_df,
        test_df=load_splits.return_value[2],
        use_case=CHURN_USE_CASE,
        calibration_method="none",
        n_jobs=1,
        model_params_override=None,
    )
    log_messages = [
        "Loaded training configuration",
        "Loaded validated split",
        "MLflow experiment",
        "MLflow artifact location",
        "Starting MLflow run",
        "Started MLflow run",
        "Hyperparameter tuning is disabled",
        "Training results logged to MLflow",
        "Model registration is disabled; skipping registry and promotion steps",
        "Training job completed",
    ]
    log_positions = [caplog.text.index(message) for message in log_messages]
    assert log_positions == sorted(log_positions)


@pytest.mark.unit
def test_run_training_job_uses_run_time_options_and_writes_unregistered_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_factory: Callable[..., TrainingConfig],
) -> None:
    use_case = CHURN_USE_CASE.name
    write_temporal_metadata(tmp_path / "splits")
    settings = SimpleNamespace(
        config_dir=tmp_path,
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        mlflow_tracking_uri="sqlite:///tracking.db",
        mlflow_experiment_name="experiment",
    )
    config = config_factory()
    training_result = SimpleNamespace(
        metrics={
            "roc_auc": 0.84,
            "pr_auc": 0.82,
            "f1": 0.75,
            "brier_score": 0.18,
            "calibration_error": 0.09,
        },
        classifier_config={"model_name": "LogisticRegression"},
        feature_names_in=["feature"],
        feature_names_out=["feature"],
        threshold_data="training_validation",
        threshold=0.42,
    )
    train_model_mock = MagicMock(return_value=training_result)
    model_info = SimpleNamespace(
        model_uri="runs:/run-123/model",
        model_uuid="model-123",
        artifact_path="model",
        mlflow_version="3.16.0",
    )
    mlflow_run = MagicMock()
    monkeypatch.setattr(train, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(
        train,
        "load_config",
        lambda **_: (config, tmp_path / "lr.yaml"),
    )
    monkeypatch.setattr(
        train,
        "load_and_validate_training_splits",
        MagicMock(return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())),
    )
    monkeypatch.setattr(train, "train_model", train_model_mock)
    monkeypatch.setattr(train, "setup_local_experiment", MagicMock(return_value="exp"))
    monkeypatch.setattr(
        train,
        "log_experiment_result",
        MagicMock(return_value=model_info),
    )
    model_registry = MagicMock()
    monkeypatch.setattr(train, "ModelRegistry", model_registry)
    monkeypatch.setattr(
        train,
        "ModelCardBuilder",
        lambda: SimpleNamespace(build=MagicMock(return_value="# Model Card")),
    )
    monkeypatch.setattr(train, "log_model_card", MagicMock())
    monkeypatch.setattr(train.mlflow, "get_experiment", MagicMock())
    monkeypatch.setattr(train.mlflow, "get_tracking_uri", MagicMock(return_value="uri"))
    monkeypatch.setattr(train.mlflow, "start_run", MagicMock(return_value=mlflow_run))
    monkeypatch.setattr(train.mlflow, "log_param", MagicMock())
    monkeypatch.setattr(train.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(
        train.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-123")),
    )
    monkeypatch.setattr(train.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(train.mlflow, "set_tag", MagicMock())
    monkeypatch.setattr(train.mlflow, "log_artifact", MagicMock())

    config_file = "lr.yaml"
    split_name = "temporal"
    classifier_alias = config.model.classifier
    run_name = f"{use_case}-{classifier_alias}-{split_name}"

    train.run_training_job(
        use_case=use_case,
        config_file=config_file,
        split_name=split_name,
        register_model=False,
        calibration_method="isotonic",
        n_jobs=3,
    )

    assert train_model_mock.call_args.kwargs["calibration_method"] == "isotonic"
    assert train_model_mock.call_args.kwargs["n_jobs"] == 3
    assert config.calibration.n_splits == 3
    from mlkit.tracking import TrainingManifest

    manifest = TrainingManifest.read(tmp_path / f"manifests/{run_name}.json")
    assert manifest.model.use_case == use_case
    assert manifest.data.time_column == "reference_date"
    expected_split_metadata = {
        f"split_{key}": value for key, value in TEMPORAL_METADATA.items()
    }
    assert manifest.data.split_metadata == expected_split_metadata
    train.mlflow.log_params.assert_any_call(expected_split_metadata)
    assert manifest.training.tuning_enabled is False
    assert manifest.training.calibration == "isotonic"
    assert manifest.training.n_jobs == 3
    assert manifest.lifecycle.registration_enabled is False
    assert manifest.lifecycle.promotion_enabled is False
    assert manifest.model.model_version is None
    assert manifest.model.model_reference.model_uri == "runs:/run-123/model"
    model_registry.assert_not_called()


@pytest.mark.unit
def test_run_training_job_logs_tuning_result_and_fits_final_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    use_case = CHURN_USE_CASE.name
    write_temporal_metadata(tmp_path / "splits")
    config = SimpleNamespace(
        random_state=19,
        model=SimpleNamespace(classifier="rf"),
        calibration=SimpleNamespace(n_splits=3, n_bins=10),
        threshold=SimpleNamespace(strategy="max_f1"),
        promotion=SimpleNamespace(
            metric_weights={
                "roc_auc": 0.2,
                "pr_auc": 0.2,
                "f1": 0.2,
                "brier_score": 0.2,
                "calibration_error": 0.2,
            },
        ),
        tuning=TuningConfig(
            n_trials=3,
            n_splits=2,
            metric="roc_auc",
        ),
    )
    config_file = "train.yaml"
    split_name = "temporal"
    settings = SimpleNamespace(
        config_dir=Path("config"),
        data_dir=tmp_path,
        artifact_dir=Path("artifacts"),
        mlflow_tracking_uri="sqlite:///tracking.db",
        mlflow_experiment_name="experiment",
    )
    train_df = pd.DataFrame(
        {
            "reference_date": pd.date_range("2026-01-01", periods=10),
            "churn": [0, 1] * 5,
            "feature": range(10),
        }
    )
    test_df = pd.DataFrame({"churn": [0, 1], "feature": [10, 11]})
    fit_df, validation_df = train_df.iloc[:8], train_df.iloc[8:]
    load_splits = MagicMock(return_value=(fit_df, validation_df, test_df))
    optimizer = MagicMock(
        return_value=TuningResult(
            best_params={"n_estimators": 150, "max_depth": None},
            best_score=0.83,
            metric="roc_auc",
            n_trials=3,
            n_splits=2,
            time_column="reference_date",
        )
    )
    final_training = MagicMock()
    final_training.metrics = {
        "roc_auc": 0.8,
        "pr_auc": 0.8,
        "f1": 0.8,
        "brier_score": 0.2,
        "calibration_error": 0.1,
    }
    train_model = MagicMock(return_value=final_training)
    mlflow_run = MagicMock()
    log_params = MagicMock()
    log_metric = MagicMock()
    log_experiment_result = MagicMock()

    monkeypatch.setattr(train, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(train, "load_config", lambda **_: (config, Path(config_file)))
    monkeypatch.setattr(train, "load_and_validate_training_splits", load_splits)
    monkeypatch.setattr(train, "optimize_hyperparameters", optimizer)
    monkeypatch.setattr(train, "train_model", train_model)
    monkeypatch.setattr(train, "setup_local_experiment", MagicMock(return_value="id"))
    monkeypatch.setattr(train, "log_experiment_result", log_experiment_result)
    monkeypatch.setattr(train.mlflow, "get_experiment", MagicMock())
    monkeypatch.setattr(train.mlflow, "start_run", MagicMock(return_value=mlflow_run))
    monkeypatch.setattr(train.mlflow, "log_param", MagicMock())
    monkeypatch.setattr(
        train.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-id")),
    )
    monkeypatch.setattr(train.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(train.mlflow, "set_tag", MagicMock())
    monkeypatch.setattr(train.mlflow, "log_params", log_params)
    monkeypatch.setattr(train.mlflow, "log_metric", log_metric)

    result = train.run_training_job(
        use_case=use_case,
        config_file=config_file,
        split_name=split_name,
        tune_model=True,
        register_model=False,
        write_manifest=False,
    )

    assert result is final_training
    load_splits.assert_called_once_with(
        split_name=split_name,
        split_dir=tmp_path / "splits",
        use_case=CHURN_USE_CASE,
        metadata=load_split_metadata(tmp_path / "splits/temporal.json"),
    )
    assert optimizer.call_count == 1
    assert optimizer.call_args.kwargs["config"] is config
    fit_df = optimizer.call_args.kwargs["train_df"]
    validation_df = train_model.call_args.kwargs["validation_df"]
    assert set(fit_df.index).isdisjoint(validation_df.index)
    assert len(fit_df) + len(validation_df) == len(train_df)
    assert optimizer.call_args.kwargs["use_case"] is CHURN_USE_CASE
    assert optimizer.call_args.kwargs["n_jobs"] == 1
    assert log_experiment_result.call_args.kwargs["tune_model"] is True
    train_model.assert_called_once_with(
        config=config,
        fit_df=fit_df,
        validation_df=validation_df,
        test_df=test_df,
        use_case=CHURN_USE_CASE,
        model_params_override={"n_estimators": 150, "max_depth": None},
        calibration_method="none",
        n_jobs=1,
    )
    log_params.assert_any_call(
        {
            "tuning_n_trials": 3,
            "tuning_n_splits": 2,
            "tuning_n_jobs": 1,
            "tuning_metric": "roc_auc",
            "tuning_best_params": '{"max_depth": null, "n_estimators": 150}',
        }
    )
    log_metric.assert_called_once_with("tuning_cv_roc_auc", 0.83)


@pytest.mark.unit
def test_tuning_excludes_threshold_validation(
    monkeypatch: pytest.MonkeyPatch,
    config_factory: Callable[..., TrainingConfig],
    tmp_path: Path,
) -> None:
    config = replace(
        config_factory(classifier="dt"),
        random_state=17,
        threshold=ThresholdConfig(),
        tuning=TuningConfig(n_trials=2, n_splits=2),
    )
    train_df = pd.DataFrame({"churn": [0, 1] * 25})
    fit_df, validation_df = train_df.iloc[:40], train_df.iloc[40:]
    test_df = pd.DataFrame({"churn": [0, 1]}, index=[50, 51])
    random_use_case = replace(
        CHURN_USE_CASE, time_column=None, split_strategy="stratified_random"
    )
    create_data_splits(
        use_case=random_use_case,
        df=pd.DataFrame({"churn": [0, 1] * 50}),
        split_name="s",
        random_state=7,
        settings=SimpleNamespace(data_dir=tmp_path),
    )
    settings = SimpleNamespace(
        config_dir=tmp_path,
        data_dir=tmp_path,
        artifact_dir=tmp_path,
        mlflow_tracking_uri="sqlite:///unused",
    )
    optimizer = MagicMock(
        return_value=TuningResult(
            best_params={},
            best_score=0.8,
            metric="roc_auc",
            n_trials=2,
            n_splits=2,
            time_column="reference_date",
        )
    )
    model = MagicMock()
    model.metrics = {
        "roc_auc": 0.8,
        "pr_auc": 0.8,
        "f1": 0.8,
        "brier_score": 0.2,
        "calibration_error": 0.1,
    }
    monkeypatch.setattr(train, "load_config", lambda **_: (config, tmp_path / "c.yaml"))
    monkeypatch.setattr(
        train,
        "get_use_case",
        lambda _: random_use_case,
    )
    monkeypatch.setattr(
        train,
        "load_and_validate_training_splits",
        lambda **_: (fit_df, validation_df, test_df),
    )
    monkeypatch.setattr(train, "optimize_hyperparameters", optimizer)
    monkeypatch.setattr(train, "train_model", MagicMock(return_value=model))
    monkeypatch.setattr(train, "setup_local_experiment", lambda **_: "id")
    monkeypatch.setattr(train, "log_experiment_result", MagicMock())
    monkeypatch.setattr(train.mlflow, "get_experiment", MagicMock())
    monkeypatch.setattr(train.mlflow, "start_run", MagicMock())
    monkeypatch.setattr(
        train.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="r")),
    )
    for name in ("set_tags", "set_tag", "log_params", "log_metric"):
        monkeypatch.setattr(train.mlflow, name, MagicMock())

    train.run_training_job(
        use_case="churn",
        config_file="c.yaml",
        split_name="s",
        tune_model=True,
        write_manifest=False,
        settings=settings,
    )
    tuned = optimizer.call_args.kwargs["train_df"]
    assert set(tuned.index) == set(fit_df.index)
    assert train.train_model.call_args.kwargs["fit_df"].equals(tuned)
    assert train.train_model.call_args.kwargs["validation_df"].equals(validation_df)
    assert set(tuned.index).isdisjoint(validation_df.index)
    split_params = train.mlflow.log_params.call_args_list[0].args[0]
    assert split_params == {
        f"split_{key}": value
        for key, value in load_split_metadata(tmp_path / "splits/s.json")
        .model_dump(exclude_none=True)
        .items()
    }


@pytest.mark.unit
def test_training_requires_split_metadata_before_experiment_setup(
    monkeypatch: pytest.MonkeyPatch,
    config_factory: Callable[..., TrainingConfig],
    tmp_path: Path,
) -> None:
    config = config_factory()
    monkeypatch.setattr(train, "load_config", lambda **_: (config, tmp_path / "c.yaml"))
    setup = MagicMock()
    monkeypatch.setattr(train, "setup_local_experiment", setup)
    with pytest.raises(FileNotFoundError, match="missing.json"):
        train.run_training_job(
            use_case="churn",
            config_file="c.yaml",
            split_name="missing",
            config_dir=tmp_path,
            split_dir=tmp_path,
        )
    setup.assert_not_called()


@pytest.mark.unit
def test_training_rejects_split_strategy_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    config_factory: Callable[..., TrainingConfig],
    tmp_path: Path,
) -> None:
    create_data_splits(
        use_case=replace(
            CHURN_USE_CASE, time_column=None, split_strategy="stratified_random"
        ),
        df=pd.DataFrame({"churn": [0, 1] * 50}),
        split_name="s",
        settings=SimpleNamespace(data_dir=tmp_path),
    )
    config = config_factory()
    monkeypatch.setattr(train, "load_config", lambda **_: (config, tmp_path / "c.yaml"))
    with pytest.raises(ValueError, match="requires 'temporal'"):
        train.run_training_job(
            use_case="churn",
            config_file="c.yaml",
            split_name="s",
            config_dir=tmp_path,
            split_dir=tmp_path / "splits",
        )


@pytest.mark.unit
@pytest.mark.parametrize("promote_model", [False, True])
def test_registration_uses_derived_name_and_only_promotion_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    promote_model: bool,
) -> None:
    use_case = CHURN_USE_CASE.name
    write_temporal_metadata(tmp_path / "splits")
    config = SimpleNamespace(
        model=SimpleNamespace(classifier="rf"),
        calibration=SimpleNamespace(n_splits=3, n_bins=10),
        threshold=SimpleNamespace(strategy="max_f1"),
        promotion=SimpleNamespace(
            min_delta=0.005,
            metric_weights={
                "roc_auc": 0.2,
                "pr_auc": 0.2,
                "f1": 0.2,
                "brier_score": 0.2,
                "calibration_error": 0.2,
            },
        ),
        tuning=TuningConfig(),
    )
    settings = SimpleNamespace(
        config_dir=tmp_path,
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        mlflow_tracking_uri="sqlite:///tracking.db",
        mlflow_experiment_name="experiment",
    )
    training_result = SimpleNamespace(
        metrics={
            "roc_auc": 0.84,
            "pr_auc": 0.82,
            "f1": 0.75,
            "brier_score": 0.18,
            "calibration_error": 0.09,
        },
        classifier_config={"model_name": "churn-risk"},
        feature_names_in=["feature_in_1", "feature_in_2"],
        feature_names_out=["feature_out_1", "feature_out_2"],
        threshold_data="training_validation",
        threshold=0.42,
    )
    registered_version = SimpleNamespace(version=7, run_id="training-run")
    model_info = SimpleNamespace(
        model_uri="runs:/training-run/model",
        model_uuid="model-456",
        artifact_path="model",
        mlflow_version="3.16.0",
    )
    registry = MagicMock()
    registry.register_model.return_value = registered_version
    registry.get_champion_version.return_value = None
    model_card_builder = MagicMock()
    model_card_builder.build.return_value = "# Model Card"

    config_file = "train.yaml"
    split_name = "temporal"
    classifier_alias = config.model.classifier
    run_name = f"{use_case}-{classifier_alias}-{split_name}"
    model_name = f"{use_case}-model"
    manifest_file = settings.data_dir / f"manifests/{run_name}.json"

    monkeypatch.setattr(train, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(
        train, "load_config", lambda **_: (config, tmp_path / config_file)
    )
    monkeypatch.setattr(
        train,
        "load_and_validate_training_splits",
        MagicMock(return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())),
    )
    monkeypatch.setattr(train, "train_model", MagicMock(return_value=training_result))
    monkeypatch.setattr(train, "setup_local_experiment", MagicMock(return_value="exp"))
    monkeypatch.setattr(
        train,
        "log_experiment_result",
        MagicMock(return_value=model_info),
    )
    monkeypatch.setattr(train, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(train, "ModelCardBuilder", lambda: model_card_builder)
    monkeypatch.setattr(train, "log_model_card", MagicMock())
    monkeypatch.setattr(train.mlflow, "get_experiment", MagicMock())
    monkeypatch.setattr(train.mlflow, "get_tracking_uri", MagicMock(return_value="uri"))
    monkeypatch.setattr(train.mlflow, "start_run", MagicMock())
    monkeypatch.setattr(train.mlflow, "log_param", MagicMock())
    monkeypatch.setattr(train.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(train.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr(
        train.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="training-run")),
    )
    set_tags = MagicMock()
    monkeypatch.setattr(train.mlflow, "set_tags", set_tags)
    set_tag = MagicMock()
    monkeypatch.setattr(train.mlflow, "set_tag", set_tag)

    result = train.run_training_job(
        use_case=use_case,
        config_file=config_file,
        split_name=split_name,
        register_model=True,
        promote_model=promote_model,
        write_manifest=True,
    )

    assert result is training_result
    setup_experiment = train.setup_local_experiment
    setup_experiment.assert_called_once_with(
        experiment_name=f"{use_case}-{split_name}",
        tracking_uri="sqlite:///tracking.db",
        artifact_dir=tmp_path / "artifacts",
    )
    registry.register_model.assert_called_once_with(
        model_info=model_info,
        model_name=model_name,
    )
    if promote_model:
        registry.set_alias.assert_called_once_with(
            model_name=model_name,
            model_alias="champion",
            model_version=7,
        )
        registry.get_champion_version.assert_called_once_with(model_name=model_name)
    else:
        registry.set_alias.assert_not_called()
        registry.get_champion_version.assert_not_called()
    assert manifest_file.is_file()
    from mlkit.tracking import TrainingManifest

    manifest = TrainingManifest.read(manifest_file)
    assert manifest.model.model_name == f"{use_case}-model"
    assert manifest.model.model_version == 7
    assert manifest.run.run_id == "training-run"
    assert manifest.run.run_name == run_name
    assert manifest.model.use_case == use_case
    assert manifest.data.time_column == "reference_date"
    assert manifest.lifecycle.registration_enabled is True
    assert manifest.lifecycle.promotion_enabled is promote_model
    set_tag.assert_any_call("git_commit", ANY)
    train.mlflow.log_artifact.assert_called_once_with(
        local_path=str(manifest_file),
        artifact_path="manifest",
    )

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import pytest

from mlkit.use_cases import get_use_case

CHURN_USE_CASE = get_use_case("churn")


@pytest.mark.smoke
def test_project_package_imports_successfully() -> None:
    import mlkit

    assert mlkit is not None


@pytest.mark.smoke
def test_package_main_dispatches_train_command(monkeypatch) -> None:
    import mlkit

    observed: dict[str, object] = {}
    train_module = importlib.import_module("mlkit.train")

    def fake_run_training_job(**kwargs: object) -> SimpleNamespace:
        observed["kwargs"] = kwargs
        return SimpleNamespace(
            classifier_config={"model_name": "test-model"},
            metrics={"roc_auc": 0.9},
        )

    monkeypatch.setattr(train_module, "run_training_job", fake_run_training_job)

    mlkit.main(
        [
            "train",
            "--use_case",
            "alternate",
            "--config",
            "config.yaml",
        ]
    )

    assert observed["kwargs"] == {
        "use_case": "alternate",
        "config_file": "config.yaml",
        "split_name": "default",
        "calibration_method": "none",
        "n_jobs": 1,
        "tune_model": False,
        "register_model": False,
        "promote_model": False,
        "settings": ANY,
    }


@pytest.mark.smoke
def test_package_main_prints_help_for_empty_args(capsys) -> None:
    import mlkit

    try:
        mlkit.main([])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError(
            "Top-level CLI should exit with code 0 when no subcommand is provided."
        )

    captured = capsys.readouterr()
    assert "usage: mlkit" in captured.out
    assert "train" in captured.out
    assert "batch-predict" in captured.out
    assert "monitor" in captured.out
    assert "serve" in captured.out


@pytest.mark.smoke
def test_package_main_dispatches_monitor_command(monkeypatch) -> None:
    import mlkit

    observed: dict[str, object] = {}

    def fake_run_monitoring(**kwargs: object) -> None:
        observed.update(kwargs)

    monitoring_module = importlib.import_module("mlkit.monitoring.core")
    monkeypatch.setattr(monitoring_module, "run_monitoring", fake_run_monitoring)

    mlkit.main(
        [
            "monitor",
            "--reference_csv",
            "reference.csv",
            "--analysis_csv",
            "analysis.csv",
            "--use_case",
            "churn",
            "--output_dir",
            "reports",
        ]
    )

    assert observed == {
        "reference_csv": Path("reference.csv"),
        "analysis_csv": Path("analysis.csv"),
        "prediction_log": None,
        "error_log": None,
        "use_case": CHURN_USE_CASE,
        "output_dir": Path("reports"),
    }


@pytest.mark.smoke
def test_package_main_builds_report_from_partitioned_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlkit
    from mlkit import data
    from mlkit.monitoring import core

    use_case = SimpleNamespace(time_column="event_date", event_time_column="event_date")
    datasets = {
        "partitioned": object(),
        "api_predictions": object(),
        "api_errors": object(),
    }
    loaded: list[tuple[str, str]] = []
    observed: dict[str, object] = {}

    def fake_read_parquet_dataset(
        dataset_name: str, timestamp_column: str, *, settings: object
    ) -> object:
        loaded.append((dataset_name, timestamp_column))
        return datasets[dataset_name]

    def fake_build_monitoring_report(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(data, "read_parquet_dataset", fake_read_parquet_dataset)
    monkeypatch.setattr(core, "build_monitoring_report", fake_build_monitoring_report)
    monkeypatch.setattr(
        "mlkit.use_cases.get_use_case",
        lambda name: use_case,
    )

    mlkit.main(
        [
            "monitor",
            "--reference_dataset",
            "partitioned",
            "--analysis_dataset",
            "api_predictions",
            "--use_case",
            "alternate",
            "--error_dataset",
            "api_errors",
            "--output_dir",
            "reports",
        ]
    )

    assert loaded == [
        ("partitioned", "event_date"),
        ("api_predictions", "event_date"),
        ("api_errors", "event_date"),
    ]
    assert observed == {
        "reference": datasets["partitioned"],
        "analysis": datasets["api_predictions"],
        "use_case": use_case,
        "source_reference": "batch",
        "source_analysis": "api",
        "errors": datasets["api_errors"],
        "output_dir": Path("reports"),
    }


@pytest.mark.smoke
def test_package_main_uses_monitoring_model_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mlkit
    from mlkit import data
    from mlkit.monitoring import core
    from mlkit.tracking import TrainingManifest

    settings = SimpleNamespace(
        data_dir=tmp_path,
        output_dir=tmp_path / "reports",
        api_host="127.0.0.1",
        api_port=8000,
    )
    manifest = SimpleNamespace(
        model=SimpleNamespace(model_name="churn-risk", model_version=11)
    )
    datasets = {"reference": object(), "analysis": object()}
    observed_report: dict[str, object] = {}
    observed_log: dict[str, object] = {}

    monkeypatch.setattr(mlkit, "RuntimeSettings", lambda: settings)

    def fake_read_parquet_dataset(
        dataset_name: str, timestamp_column: str, *, settings: object
    ) -> object:
        return datasets[dataset_name]

    monkeypatch.setattr(data, "read_parquet_dataset", fake_read_parquet_dataset)
    manifest_read = Mock(return_value=manifest)
    monkeypatch.setattr(TrainingManifest, "read", manifest_read)

    def fake_build_monitoring_report(**kwargs: object) -> str:
        observed_report.update(kwargs)
        return "report"

    def fake_log_monitoring_run(report: object, **kwargs: object) -> None:
        observed_log["report"] = report
        observed_log.update(kwargs)

    monkeypatch.setattr(core, "build_monitoring_report", fake_build_monitoring_report)
    monkeypatch.setattr(core, "log_monitoring_run", fake_log_monitoring_run)

    mlkit.main(
        [
            "monitor",
            "--reference_dataset",
            "reference",
            "--analysis_dataset",
            "analysis",
            "--use_case",
            "churn",
            "--model_manifest",
            "manifests/churn-hgb-OOT_26Q2.json",
            "--mlflow_experiment_name",
            "churn-monitoring",
        ]
    )

    manifest_read.assert_called_once_with(
        tmp_path / "manifests/churn-hgb-OOT_26Q2.json"
    )
    assert observed_report == {
        "reference": datasets["reference"],
        "analysis": datasets["analysis"],
        "use_case": CHURN_USE_CASE,
        "source_reference": "batch",
        "source_analysis": "api",
        "errors": None,
        "output_dir": settings.output_dir,
        "model_version": 11,
    }
    assert observed_log == {
        "report": "report",
        "settings": settings,
        "experiment_name": "churn-monitoring",
        "run_name": "monitoring",
        "tags": {
            "reference_input": "reference",
            "analysis_input": "analysis",
            "reference_name": "reference",
            "analysis_name": "analysis",
            "model_name": "churn-risk",
            "model_version": "11",
        },
    }


@pytest.mark.smoke
def test_package_main_rejects_monitor_manifest_with_explicit_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import mlkit

    with pytest.raises(SystemExit, match="2"):
        mlkit.main(
            [
                "monitor",
                "--reference_dataset",
                "reference",
                "--analysis_dataset",
                "analysis",
                "--use_case",
                "churn",
                "--model_manifest",
                "manifests/model.json",
                "--model_version",
                "11",
            ]
        )

    assert "--model_manifest cannot be combined" in capsys.readouterr().err


@pytest.mark.smoke
def test_package_main_passes_manifest_to_api_for_serving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import os

    import uvicorn

    import mlkit

    settings = SimpleNamespace(
        data_dir=tmp_path,
        api_host="127.0.0.1",
        api_port=8000,
    )
    run = Mock()
    monkeypatch.setattr(mlkit, "ServingSettings", lambda: settings)
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.delenv("MODEL_MANIFEST", raising=False)

    mlkit.main(
        [
            "serve",
            "--model_manifest",
            "manifests/unregistered.json",
            "--use_case",
            "churn",
        ]
    )

    assert os.environ["MODEL_MANIFEST"] == str(tmp_path / "manifests/unregistered.json")
    run.assert_called_once_with(
        "mlkit.serving.api:create_app_from_environment",
        host="127.0.0.1",
        port=8000,
        reload=False,
        factory=True,
    )


@pytest.mark.smoke
def test_cli_port_override_wins_over_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    import mlkit

    monkeypatch.setenv("API_PORT", "not-a-port")
    monkeypatch.delenv("USE_CASE", raising=False)
    monkeypatch.delenv("MODEL_MANIFEST", raising=False)
    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)
    mlkit.main(
        [
            "serve",
            "--use_case",
            "churn",
            "--model_manifest",
            str(tmp_path / "manifest.json"),
            "--port",
            "9123",
        ]
    )
    assert run.call_args.kwargs["port"] == 9123


@pytest.mark.smoke
def test_non_serving_command_does_not_read_serving_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlkit

    monkeypatch.setenv("MODEL_VERSION", "not-a-version")
    train_module = importlib.import_module("mlkit.train")
    monkeypatch.setattr(
        train_module,
        "run_training_job",
        Mock(
            return_value=SimpleNamespace(
                classifier_config={"model_name": "test"}, metrics={"roc_auc": 0.9}
            )
        ),
    )
    mlkit.main(["train", "--use_case", "churn", "--config", "config.yaml"])


@pytest.mark.smoke
def test_package_main_dispatches_api_file_monitoring_with_configured_error_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mlkit
    from mlkit.monitoring import core

    settings = SimpleNamespace(
        output_dir=tmp_path / "reports",
        prediction_log_path=tmp_path / "predictions.jsonl",
        error_log_path=tmp_path / "errors.jsonl",
        api_host="127.0.0.1",
        api_port=8000,
    )
    observed: dict[str, object] = {}

    def fake_run_monitoring(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(mlkit, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(core, "run_monitoring", fake_run_monitoring)

    mlkit.main(
        [
            "monitor",
            "--reference_csv",
            "reference.csv",
            "--prediction_log",
            "custom_predictions.jsonl",
            "--use_case",
            "churn",
        ]
    )

    assert observed == {
        "reference_csv": Path("reference.csv"),
        "analysis_csv": None,
        "prediction_log": Path("custom_predictions.jsonl"),
        "error_log": settings.error_log_path,
        "use_case": CHURN_USE_CASE,
        "output_dir": settings.output_dir,
    }


@pytest.mark.smoke
def test_package_main_dispatches_serve_command(monkeypatch) -> None:
    import mlkit

    called = {}

    class FakeUvicorn:
        @staticmethod
        def run(app: str, host: str, port: int, reload: bool, factory: bool) -> None:
            called["app"] = app
            called["host"] = host
            called["port"] = port
            called["reload"] = reload
            called["factory"] = factory

    fake_uvicorn_module = type("module", (), {"run": staticmethod(FakeUvicorn.run)})
    monkeypatch.setitem(importlib.sys.modules, "uvicorn", fake_uvicorn_module)
    monkeypatch.setenv("MODEL_NAME", "churn-rf-OOT_26Q2")

    mlkit.main(
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--reload",
            "--use_case",
            "churn",
        ]
    )

    assert called == {
        "app": "mlkit.serving.api:create_app_from_environment",
        "host": "0.0.0.0",
        "port": 9000,
        "reload": True,
        "factory": True,
    }


@pytest.mark.smoke
def test_package_main_dispatches_prepare_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mlkit
    from mlkit import data
    from mlkit.data import storage

    raw_data = object()
    validated_data = object()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        mlkit,
        "RuntimeSettings",
        lambda: SimpleNamespace(
            data_dir=tmp_path,
            api_host="127.0.0.1",
            api_port=8000,
        ),
    )

    def fake_load_raw_data(**kwargs: object) -> object:
        observed["load"] = kwargs
        return raw_data

    def fake_validate_data(df: object, use_case: object) -> object:
        observed["validated_input"] = df
        observed["use_case"] = use_case
        return validated_data

    def fake_write_parquet_dataset(**kwargs: object) -> None:
        observed["write"] = kwargs

    monkeypatch.setattr(data, "load_raw_data", fake_load_raw_data)
    monkeypatch.setattr(data, "validate_data", fake_validate_data)
    monkeypatch.setattr(
        storage,
        "write_parquet_dataset",
        fake_write_parquet_dataset,
    )

    mlkit.main(["prepare-data", "--input_csv", "training.csv", "--use_case", "churn"])

    assert observed == {
        "load": {"file_name": "training.csv", "data_dir": tmp_path / "raw"},
        "validated_input": raw_data,
        "use_case": CHURN_USE_CASE,
        "write": {
            "df": validated_data,
            "dataset_name": "partitioned",
            "timestamp_column": "reference_date",
            "overwrite_partitions": True,
            "settings": ANY,
        },
    }


@pytest.mark.smoke
def test_package_main_dispatches_create_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mlkit
    from mlkit.data import splitting

    use_case = SimpleNamespace(time_column="event_date", split_strategy="temporal")
    observed: dict[str, object] = {}

    def fake_create_split(use_case: object, **kwargs: object) -> dict[str, str]:
        observed["use_case"] = use_case
        observed.update(kwargs)
        return {"strategy": "temporal"}

    settings = SimpleNamespace(data_dir=tmp_path)
    monkeypatch.setattr(mlkit, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(
        splitting,
        "create_split",
        fake_create_split,
    )
    monkeypatch.setattr(
        "mlkit.use_cases.get_use_case",
        lambda name: use_case,
    )

    mlkit.main(
        [
            "create-split",
            "--dataset_name",
            "partitioned",
            "--use_case",
            "alternate",
            "--start_date",
            "2026-01-01",
            "--split_date",
            "2026-04-01",
            "--end_date",
            "2026-06-30",
            "--split_name",
            "pytest",
        ]
    )

    assert observed == {
        "use_case": use_case,
        "dataset_name": "partitioned",
        "input_parquet": None,
        "start_date": "2026-01-01",
        "split_date": "2026-04-01",
        "end_date": "2026-06-30",
        "split_name": "pytest",
        "random_state": None,
        "test_fraction": None,
        "monitoring_fraction": None,
        "settings": settings,
    }


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("use_case", "args", "message"),
    [
        ("churn", [], "Temporal splitting requires"),
        ("phishing", [], "Random splitting requires"),
        (
            "phishing",
            ["--input_parquet", "data.parquet", "--start_date", "2024-01-01"],
            "only for temporal splitting",
        ),
        (
            "churn",
            [
                "--dataset_name",
                "data",
                "--start_date",
                "2024-01-01",
                "--split_date",
                "2024-02-01",
                "--end_date",
                "2024-03-01",
                "--random_state",
                "42",
            ],
            "only for random splitting",
        ),
    ],
)
def test_create_split_rejects_incompatible_arguments(
    use_case: str, args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    import mlkit

    with pytest.raises(SystemExit):
        mlkit.main(["create-split", "--use_case", use_case, *args])
    assert message in capsys.readouterr().err


@pytest.mark.smoke
def test_create_split_cli_applies_random_fractions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    import pandas as pd

    import mlkit

    pd.DataFrame({"label": [0, 1] * 50}).to_parquet(
        tmp_path / "prepared.parquet", index=False
    )
    monkeypatch.setattr(
        mlkit, "RuntimeSettings", lambda: SimpleNamespace(data_dir=tmp_path)
    )
    mlkit.main(
        [
            "create-split",
            "--use_case",
            "phishing",
            "--input_parquet",
            "prepared.parquet",
            "--test_fraction",
            "0.2",
            "--monitoring_fraction",
            "0.0",
            "--split_name",
            "s",
        ]
    )
    metadata = json.loads((tmp_path / "splits/s.json").read_text())
    assert (
        metadata["train_rows"],
        metadata["test_rows"],
        metadata["monitoring_rows"],
    ) == (80, 20, 0)
    assert pd.read_parquet(tmp_path / "splits/s_monitoring.parquet").empty


@pytest.mark.smoke
def test_package_main_uses_selected_use_case_time_columns_for_monitoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlkit
    from mlkit import data
    from mlkit.monitoring import core

    use_case = SimpleNamespace(
        name="alternate", time_column="event_date", event_time_column="event_date"
    )
    datasets = {"reference": object(), "analysis": object()}
    loaded: list[tuple[str, str]] = []

    def fake_read_parquet_dataset(
        dataset_name: str, timestamp_column: str, *, settings: object
    ) -> object:
        loaded.append((dataset_name, timestamp_column))
        return datasets[dataset_name]

    monkeypatch.setattr(
        "mlkit.use_cases.get_use_case",
        lambda name: use_case,
    )
    monkeypatch.setattr(data, "read_parquet_dataset", fake_read_parquet_dataset)
    monkeypatch.setattr(core, "build_monitoring_report", lambda **kwargs: None)

    mlkit.main(
        [
            "monitor",
            "--reference_dataset",
            "reference",
            "--analysis_dataset",
            "analysis",
            "--source_analysis",
            "batch",
            "--use_case",
            "alternate",
        ]
    )

    assert loaded == [
        ("reference", "event_date"),
        ("analysis", "event_date"),
    ]


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("override", "expected_reference", "expected_analysis"),
    [(None, None, "timestamp"), ("event_date", "event_date", "event_date")],
)
def test_monitor_loads_undated_reference_and_dated_analysis(
    monkeypatch: pytest.MonkeyPatch,
    override: str | None,
    expected_reference: str | None,
    expected_analysis: str,
) -> None:
    import pandas as pd

    import mlkit
    from mlkit import data
    from mlkit.monitoring import core

    loaded: list[tuple[str, str | None]] = []

    def fake_read(
        dataset_name: str, timestamp_column: str | None, *, settings: object
    ) -> pd.DataFrame:
        loaded.append((dataset_name, timestamp_column))
        return (
            pd.DataFrame({timestamp_column: []}) if timestamp_column else pd.DataFrame()
        )

    monkeypatch.setattr(data, "read_parquet_dataset", fake_read)
    monkeypatch.setattr(core, "build_monitoring_report", lambda **kwargs: None)
    arguments = [
        "monitor",
        "--reference_dataset",
        "reference",
        "--analysis_dataset",
        "analysis",
        "--error_dataset",
        "errors",
        "--use_case",
        "phishing",
    ]
    if override is not None:
        arguments.extend(["--timestamp_column", override])
    mlkit.main(arguments)
    assert loaded == [
        ("reference", expected_reference),
        ("analysis", expected_analysis),
        ("errors", expected_analysis),
    ]


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("command", "function_name", "dataset_name"),
    [
        ("ingest-prediction-logs", "ingest_prediction_logs", "api_predictions"),
        ("ingest-error-logs", "ingest_error_logs", "api_errors"),
    ],
)
def test_package_main_dispatches_log_ingestion(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    function_name: str,
    dataset_name: str,
) -> None:
    import mlkit

    ingestion = importlib.import_module("mlkit.data.ingestion")
    observed: dict[str, str] = {}
    use_case = SimpleNamespace(time_column="event_date", event_time_column="event_date")

    def fake_ingest(
        dataset_name: str, timestamp_column: str, *, settings: object
    ) -> None:
        observed["dataset_name"] = dataset_name
        observed["timestamp_column"] = timestamp_column

    monkeypatch.setattr(ingestion, function_name, fake_ingest)
    monkeypatch.setattr(
        "mlkit.use_cases.get_use_case",
        lambda name: use_case,
    )

    mlkit.main([command, "--use_case", "alternate"])

    assert observed == {
        "dataset_name": dataset_name,
        "timestamp_column": "event_date",
    }

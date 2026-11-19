from pathlib import Path

import pytest
from pydantic import ValidationError

from mlkit.config import RuntimeSettings, ServingSettings


@pytest.mark.unit
def test_runtime_settings_read_container_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tracking_dir = tmp_path / "tracking"
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setenv("TRACKING_DIR", str(tracking_dir))
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    monkeypatch.setenv("API_PORT", "9123")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "container-training")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    settings = RuntimeSettings()

    assert settings.tracking_dir == tracking_dir
    assert settings.artifact_dir == artifact_dir
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 9123
    assert settings.mlflow_experiment_name == "container-training"
    assert settings.mlflow_tracking_uri == f"sqlite:///{tracking_dir / 'mlflow.db'}"


@pytest.mark.unit
def test_explicit_mlflow_uri_overrides_derived_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'custom.db'}"
    monkeypatch.setenv("TRACKING_DIR", str(tmp_path / "tracking"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    settings = RuntimeSettings()

    assert settings.mlflow_tracking_uri == tracking_uri


@pytest.mark.unit
def test_runtime_settings_use_default_mlflow_experiment_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)

    settings = RuntimeSettings()

    assert settings.mlflow_experiment_name == "test"


@pytest.mark.unit
def test_root_and_directory_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLKIT_PROJECT_ROOT", "project")
    monkeypatch.setenv("DATA_DIR", "custom-data")
    monkeypatch.delenv("TRACKING_DIR", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("LOGGING_DIR", raising=False)
    settings = ServingSettings()

    assert settings.project_root == tmp_path / "project"
    assert settings.data_dir == tmp_path / "custom-data"
    assert settings.config_dir == tmp_path / "project/src/config"
    assert settings.logging_dir == tmp_path / "project/logs_local"
    assert settings.mlflow_tracking_uri == (
        f"sqlite:///{tmp_path / 'project/tracking_local/mlflow.db'}"
    )


@pytest.mark.unit
def test_default_root_is_current_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLKIT_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    settings = RuntimeSettings()

    assert settings.project_root == tmp_path
    assert settings.data_dir == tmp_path / "data"


@pytest.mark.unit
def test_explicit_values_win_and_settings_are_immutable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "environment"))
    monkeypatch.setenv("MLKIT_PROJECT_ROOT", str(tmp_path / "environment-root"))
    settings = RuntimeSettings(
        data_dir=tmp_path / "explicit", project_root=tmp_path / "explicit-root"
    )

    assert settings.data_dir == tmp_path / "explicit"
    assert settings.project_root == tmp_path / "explicit-root"
    assert settings.config_dir == tmp_path / "explicit-root/src/config"
    with pytest.raises(ValidationError):
        settings.data_dir = tmp_path / "changed"


@pytest.mark.unit
@pytest.mark.parametrize("port", [0, 8000, 65535])
def test_valid_port(port: int) -> None:
    assert RuntimeSettings(api_port=port).api_port == port


@pytest.mark.unit
@pytest.mark.parametrize("port", [-1, 65536])
def test_invalid_port(port: int) -> None:
    with pytest.raises(ValidationError):
        RuntimeSettings(api_port=port)


@pytest.mark.unit
def test_invalid_port_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PORT", "invalid")
    with pytest.raises(ValidationError):
        RuntimeSettings()


@pytest.mark.unit
def test_feature_logging_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOG_FEATURES", raising=False)
    assert ServingSettings().log_features is False
    monkeypatch.setenv("LOG_FEATURES", "true")
    assert ServingSettings().log_features is True

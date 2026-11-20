from pathlib import Path

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    """Validated snapshot of the environment for a runtime workflow."""

    model_config = SettingsConfigDict(
        extra="ignore", frozen=True, validate_by_name=True
    )

    project_root: Path = Field(
        default_factory=Path.cwd, validation_alias="MLKIT_PROJECT_ROOT"
    )
    data_dir: Path = Field(default_factory=lambda data: data["project_root"] / "data")
    config_dir: Path = Field(
        default_factory=lambda data: data["project_root"] / "src/config"
    )
    artifact_dir: Path = Field(
        default_factory=lambda data: data["project_root"] / "artifacts_local"
    )
    tracking_dir: Path = Field(
        default_factory=lambda data: data["project_root"] / "tracking_local"
    )
    output_dir: Path = Field(
        default_factory=lambda data: data["project_root"] / "output"
    )
    logging_dir: Path = Field(
        default_factory=lambda data: data["project_root"] / "logs_local"
    )
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "test"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=0, le=65535)

    @field_validator(
        "project_root",
        "data_dir",
        "config_dir",
        "artifact_dir",
        "tracking_dir",
        "output_dir",
        "logging_dir",
    )
    @classmethod
    def resolve_directory(cls, value: Path) -> Path:
        """Resolve relative directories against the current working directory."""
        return value.expanduser().resolve()

    @field_validator("mlflow_tracking_uri")
    @classmethod
    def default_tracking_uri(
        cls, value: str | None, info: ValidationInfo
    ) -> str | None:
        """Use a SQLite database in the effective tracking directory by default."""
        if value is not None or "tracking_dir" not in info.data:
            return value
        return f"sqlite:///{info.data['tracking_dir'] / 'mlflow.db'}"

    @property
    def prediction_log_path(self) -> Path:
        return self.logging_dir / "predictions.jsonl"

    @property
    def error_log_path(self) -> Path:
        return self.logging_dir / "prediction_errors.jsonl"


class ServingSettings(RuntimeSettings):
    """Runtime settings for model serving and prediction logging."""

    model_name: str | None = None
    model_alias: str = "champion"
    model_version: int | None = None
    model_manifest: Path | None = None
    request_id_header: str = "X-Request-ID"
    log_features: bool = False

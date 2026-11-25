from pathlib import Path

import yaml

from mlkit.config.schemas import (
    CalibrationConfig,
    ClassifierConfig,
    PreprocessingConfig,
    PromotionConfig,
    ThresholdConfig,
    TrainingConfig,
    TrainingSeedOverrides,
    TuningConfig,
)
from mlkit.config.settings import RuntimeSettings


def load_config(
    file: str,
    path: Path | None = None,
    *,
    settings: RuntimeSettings | None = None,
) -> tuple[TrainingConfig, Path]:
    """Load a YAML training configuration into typed dataclasses.

    Args:
        file: Configuration filename.
        path: Directory containing the configuration file.
        settings: Runtime snapshot used when no path is supplied.

    Returns:
        A tuple containing the parsed ``TrainingConfig`` and the configuration
        file path used to load it.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If a required configuration section is missing.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    resolved_path = (
        path if path is not None else (settings or RuntimeSettings()).config_dir
    )

    config_file_path = resolved_path / file
    with open(config_file_path) as f:
        raw = yaml.safe_load(f)

    return TrainingConfig(
        model=ClassifierConfig(**raw["model"]),
        random_state=raw.get("random_state", 26),
        seed_overrides=TrainingSeedOverrides(**raw.get("seed_overrides", {})),
        preprocessing=PreprocessingConfig(**raw["preprocessing"]),
        tuning=TuningConfig(**raw.get("tuning", {})),
        calibration=CalibrationConfig(**raw.get("calibration", {})),
        threshold=ThresholdConfig(**raw.get("threshold", {})),
        promotion=PromotionConfig(**raw.get("promotion", {})),
    ), config_file_path

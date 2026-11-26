from pathlib import Path

import pytest
import yaml

from mlkit.config import load_config
from mlkit.config.schemas import PromotionConfig


@pytest.mark.unit
def test_load_config_returns_config_and_resolved_path(sample_config_yaml: Path) -> None:
    config_file_name = sample_config_yaml.name
    config_file_dir = sample_config_yaml.parent

    config, path = load_config(
        file=config_file_name,
        path=config_file_dir,
    )

    assert config.tuning.n_trials == 50
    assert config.calibration.n_splits == 3
    assert sum(config.promotion.metric_weights.values()) > 0
    assert path == sample_config_yaml


@pytest.mark.unit
@pytest.mark.parametrize(
    "filename",
    [
        "config.yaml",
        "dc.yaml",
        "dt.yaml",
        "hgb.yaml",
        "lr.yaml",
        "nb.yaml",
        "rf.yaml",
        "sample_training_config.yaml",
    ],
)
def test_shipped_training_configs_have_valid_settings(
    filename: str,
) -> None:
    config_directory = Path(__file__).resolve().parents[2] / "src" / "config"

    config, _ = load_config(file=filename, path=config_directory)
    raw_config = yaml.safe_load(
        (config_directory / filename).read_text(encoding="utf-8")
    )

    assert raw_config["random_state"] == config.random_state
    assert config.tuning.n_trials >= 1
    assert config.calibration.n_splits >= 2
    assert sum(config.promotion.metric_weights.values()) > 0


@pytest.mark.unit
def test_promotion_config_rejects_negative_min_delta() -> None:
    with pytest.raises(ValueError, match="min_delta"):
        PromotionConfig(min_delta=-0.1)


@pytest.mark.unit
def test_training_rejects_classifier_seed_in_model_params(
    sample_config_yaml: Path,
) -> None:
    raw = yaml.safe_load(sample_config_yaml.read_text())
    raw["model"]["classifier_params"]["random_state"] = 3
    sample_config_yaml.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="seed_overrides.classifier"):
        load_config(sample_config_yaml.name, sample_config_yaml.parent)

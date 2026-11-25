import argparse
from dataclasses import replace
from pathlib import Path

import pytest

from mlkit.config import load_config
from mlkit.config.schemas import TrainingSeedOverrides
from mlkit.config.seeds import parse_seed, parse_seed_override, resolve_seed


@pytest.mark.parametrize(
    ("root", "override", "expected"), [(42, None, 42), (42, 0, 0), (0, 17, 17)]
)
def test_seed_resolution(root: int, override: int | None, expected: int) -> None:
    assert resolve_seed(root, override) == expected


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "17"])
def test_seed_resolution_rejects_invalid_values(seed: object) -> None:
    with pytest.raises(ValueError, match="Random seed"):
        resolve_seed(seed)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Random seed"):
        resolve_seed(42, seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["-1", str(2**32), "invalid"])
def test_seed_parser_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Random seed"):
        parse_seed(value)


def test_training_seeds_inherit_and_can_be_overridden(
    sample_config_yaml: Path,
) -> None:
    config, _ = load_config(sample_config_yaml.name, sample_config_yaml.parent)
    assert (config.training_seed, config.classifier_seed) == (26, 42)
    config = replace(
        config, random_state=10, seed_overrides=TrainingSeedOverrides(training=0)
    )
    assert (config.training_seed, config.classifier_seed) == (0, 10)
    assert parse_seed_override("inherit") is None
    assert parse_seed_override("0") == 0

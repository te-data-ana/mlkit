import json
from dataclasses import replace
from pathlib import Path

import pytest

from mlkit.tracking.manifest import TrainingManifest


@pytest.mark.unit
def test_training_manifest_round_trips_json(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    manifest = training_manifest_factory({"promote": False})
    manifest = replace(
        manifest,
        training=replace(manifest.training, tuning_enabled=True),
    )
    output = tmp_path / "manifests" / "run.json"

    manifest.write(output)

    assert TrainingManifest.read(output) == manifest
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert list(payload) == [
        "schema_version",
        "run",
        "model",
        "data",
        "training",
        "evaluation",
        "lifecycle",
    ]
    assert list(payload["training"]) == [
        "classifier",
        "target_classes",
        "features",
        "tuning",
        "n_jobs",
        "calibration",
        "class_selection_strategy",
        "threshold",
    ]
    assert list(payload["lifecycle"]) == ["registration", "promotion"]
    assert payload["schema_version"] == 5
    assert payload["model"]["use_case"] == "binary_uc"
    assert payload["data"]["time_column"] == "reference_date"
    assert payload["training"]["tuning"]["enabled"] is True
    assert payload["training"]["n_jobs"] == 1
    assert payload["training"]["class_selection_strategy"] == "max_f1"
    assert payload["lifecycle"]["registration"]["enabled"] is True
    assert payload["lifecycle"]["promotion"]["enabled"] is True


@pytest.mark.unit
def test_multiclass_manifest_round_trip(
    tmp_path: Path, training_manifest_factory
) -> None:
    manifest = training_manifest_factory()
    manifest = replace(
        manifest,
        schema_version=5,
        training=replace(
            manifest.training,
            class_selection_strategy="argmax",
            threshold_data="not_applicable",
            threshold=None,
            target_classes=(0, 1, 2),
        ),
    )
    path = tmp_path / "multiclass.json"
    manifest.write(path)
    assert TrainingManifest.read(path) == manifest
    payload = json.loads(path.read_text())
    assert payload["training"]["class_selection_strategy"] == "argmax"
    payload["training"]["threshold"]["value"] = 0.5
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="multiclass decision"):
        TrainingManifest.read(path)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("classes", "strategy", "threshold"),
    [
        ([0, 1], "argmax", None),
        ([0, 1, 2], "max_f1", 0.4),
        ([0, 1], "unknown", 0.4),
        ([0, 1], "max_f1", -0.1),
        ([0, 1], "max_f1", None),
    ],
)
def test_manifest_rejects_inconsistent_class_selection(
    tmp_path: Path,
    training_manifest_factory,
    classes: list[int],
    strategy: str,
    threshold: float | None,
) -> None:
    payload = training_manifest_factory().to_dict()
    payload["training"]["target_classes"] = classes
    payload["training"]["class_selection_strategy"] = strategy
    payload["training"]["threshold"]["value"] = threshold
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError), match="strategy|threshold"):
        TrainingManifest.read(path)


@pytest.mark.unit
def test_manifest_requires_class_selection_strategy(
    tmp_path: Path, training_manifest_factory
) -> None:
    payload = training_manifest_factory().to_dict()
    del payload["training"]["class_selection_strategy"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="class_selection_strategy"):
        TrainingManifest.read(path)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("classes", "strategy", "value", "selected_on"),
    [
        ([0, 1], "max_f1", 0.4, "not_applicable"),
        ([0, 1, 2], "argmax", None, "training_validation"),
    ],
)
def test_manifest_rejects_inconsistent_threshold_selection_data(
    tmp_path: Path,
    training_manifest_factory,
    classes: list[int],
    strategy: str,
    value: float | None,
    selected_on: str,
) -> None:
    payload = training_manifest_factory().to_dict()
    payload["training"]["target_classes"] = classes
    payload["training"]["class_selection_strategy"] = strategy
    payload["training"]["threshold"] = {"value": value, "selected_on": selected_on}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="threshold selection data"):
        TrainingManifest.read(path)


@pytest.mark.unit
def test_unregistered_training_manifest_round_trips_logged_model_reference(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    manifest = training_manifest_factory()
    manifest = replace(
        manifest,
        model=replace(manifest.model, model_version=None),
    )
    output = tmp_path / "manifest.json"

    manifest.write(output)

    assert TrainingManifest.read(output) == manifest


@pytest.mark.unit
def test_manifest_round_trips_missing_training_time_column(
    tmp_path: Path, training_manifest_factory: TrainingManifest
) -> None:
    manifest = training_manifest_factory()
    manifest = replace(manifest, data=replace(manifest.data, time_column=None))
    output = tmp_path / "manifest.json"
    manifest.write(output)
    assert TrainingManifest.read(output) == manifest


@pytest.mark.unit
def test_manifest_round_trips_split_fractions(
    tmp_path: Path, training_manifest_factory: TrainingManifest
) -> None:
    manifest = training_manifest_factory()
    metadata = dict(
        manifest.data.split_metadata, test_fraction=0.2, monitoring_fraction=0.0
    )
    manifest = replace(manifest, data=replace(manifest.data, split_metadata=metadata))
    output = tmp_path / "manifest.json"
    manifest.write(output)
    assert TrainingManifest.read(output) == manifest


@pytest.mark.unit
def test_training_manifest_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"run_id": "run-123"}', encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        TrainingManifest.read(path)


@pytest.mark.unit
def test_training_manifest_requires_model_version_or_reference(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    payload = training_manifest_factory().to_dict()
    payload["model"]["version"] = None
    payload["model"]["reference"] = None
    path = tmp_path / "invalid-model-reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TypeError, match="reference"):
        TrainingManifest.read(path)


@pytest.mark.unit
def test_training_manifest_rejects_incomplete_logged_model_reference(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    payload = training_manifest_factory().to_dict()
    payload["model"]["reference"].pop("model_uri")
    path = tmp_path / "invalid-model-info.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="model_uri"):
        TrainingManifest.read(path)


@pytest.mark.unit
def test_training_manifest_requires_current_metadata_fields(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    payload = training_manifest_factory().to_dict()
    del payload["training"]["calibration"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration"):
        TrainingManifest.read(path)


@pytest.mark.unit
@pytest.mark.parametrize("schema_version", [0, 1, 2])
def test_training_manifest_rejects_unsupported_schema_version(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
    schema_version: int,
) -> None:
    payload = training_manifest_factory().to_dict()
    payload["schema_version"] = schema_version
    path = tmp_path / "unsupported-version.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported schema_version"):
        TrainingManifest.read(path)

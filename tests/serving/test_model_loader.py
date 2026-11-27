from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sklearn.linear_model import LogisticRegression

from mlkit.serving import model_loader
from mlkit.tracking.manifest import LoggedModelReference, TrainingManifest
from mlkit.use_cases import UseCase


@pytest.mark.unit
def test_load_model_returns_model_with_registry_metadata(
    binary_use_case: UseCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock MLflow, settings, and registry calls to isolate metadata assembly.
    model = Mock()
    registry = Mock()
    registry.load_registered_model.return_value = SimpleNamespace(
        model=model,
        model_version=SimpleNamespace(version="7"),
        model_threshold=0.6,
    )

    settings = SimpleNamespace(
        model_name="churn-risk",
        model_alias="champion",
        model_version=None,
        mlflow_tracking_uri="sqlite:///mlflow.db",
    )

    monkeypatch.setattr(model_loader, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(model_loader, "ServingSettings", lambda: settings)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", Mock())

    loaded_model = model_loader.load_model(use_case=binary_use_case)

    assert loaded_model.model is model
    assert loaded_model.metadata.model_name == "churn-risk"
    assert loaded_model.metadata.model_alias == "champion"
    assert loaded_model.metadata.model_version == 7
    assert loaded_model.metadata.model_threshold == 0.6
    assert loaded_model.use_case is binary_use_case
    registry.load_registered_model.assert_called_once_with(
        model_name="churn-risk",
        expected_use_case=binary_use_case,
        model_alias="champion",
        model_version=None,
    )


@pytest.mark.unit
def test_load_model_requires_registry_name_without_manifest(
    binary_use_case: UseCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        model_name=None,
        model_alias="champion",
        model_version=None,
        mlflow_tracking_uri="sqlite:///mlflow.db",
    )
    monkeypatch.setattr(model_loader, "ServingSettings", lambda: settings)
    monkeypatch.setattr(model_loader, "ModelRegistry", Mock())
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", Mock())

    with pytest.raises(ValueError, match="Provide a model manifest or set MODEL_NAME"):
        model_loader.load_model(use_case=binary_use_case)


@pytest.mark.unit
def test_load_model_uses_exact_registered_version(
    binary_use_case: UseCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock()
    registry = Mock()
    registry.load_registered_model.return_value = SimpleNamespace(
        model=model,
        model_version=SimpleNamespace(version="9"),
        model_threshold=0.4,
    )
    settings = SimpleNamespace(
        model_name="churn-risk",
        model_alias="champion",
        model_version=None,
        mlflow_tracking_uri="sqlite:///mlflow.db",
    )
    monkeypatch.setattr(model_loader, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(model_loader, "ServingSettings", lambda: settings)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", Mock())

    loaded_model = model_loader.load_model(use_case=binary_use_case, model_version=9)

    assert loaded_model.model is model
    assert loaded_model.metadata.model_version == 9
    assert loaded_model.metadata.model_alias is None
    assert loaded_model.metadata.model_threshold == 0.4
    registry.load_registered_model.assert_called_once_with(
        model_name="churn-risk",
        expected_use_case=binary_use_case,
        model_alias=None,
        model_version=9,
    )


@pytest.mark.unit
def test_load_model_uses_manifest_threshold_for_registered_version(
    binary_use_case: UseCase,
    monkeypatch: pytest.MonkeyPatch,
    training_manifest_factory: Callable[..., TrainingManifest],
) -> None:
    model = Mock()
    registry = Mock()
    registry.load_registered_model.return_value = SimpleNamespace(model=model)
    settings = SimpleNamespace(
        model_name="configured-model",
        model_alias="champion",
        model_version=None,
        mlflow_tracking_uri="sqlite:///mlflow.db",
    )
    monkeypatch.setattr(model_loader, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(model_loader, "ServingSettings", lambda: settings)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", Mock())

    manifest = training_manifest_factory()
    loaded_model = model_loader.load_model(use_case=binary_use_case, manifest=manifest)

    assert loaded_model.model is model
    assert loaded_model.metadata.model_name == manifest.model.model_name
    assert loaded_model.metadata.model_version == manifest.model.model_version
    assert loaded_model.metadata.model_threshold == manifest.training.threshold
    assert loaded_model.metadata.model_reference == manifest.model.model_reference
    registry.load_registered_model.assert_called_once_with(
        model_name=manifest.model.model_name,
        expected_use_case=binary_use_case,
        model_version=manifest.model.model_version,
    )


@pytest.mark.unit
def test_load_model_uses_logged_model_reference_without_registry(
    binary_use_case: UseCase,
    monkeypatch: pytest.MonkeyPatch,
    training_manifest_factory: Callable[..., TrainingManifest],
) -> None:
    model = Mock()
    load_mlflow_model = Mock(return_value=model)
    settings = SimpleNamespace(
        model_name="churn-risk",
        model_alias="champion",
        model_version=None,
        mlflow_tracking_uri="sqlite:///mlflow.db",
    )
    registry = Mock()
    monkeypatch.setattr(model_loader, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(model_loader, "ServingSettings", lambda: settings)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", Mock())
    monkeypatch.setattr(model_loader.mlflow.sklearn, "load_model", load_mlflow_model)
    registry.validate_run_use_case = Mock()

    manifest = training_manifest_factory()
    manifest = replace(
        manifest,
        model=replace(manifest.model, model_version=None),
        training=replace(manifest.training, threshold=0.37),
    )
    loaded_model = model_loader.load_model(use_case=binary_use_case, manifest=manifest)

    assert loaded_model.model is model
    assert loaded_model.metadata.model_version is None
    assert loaded_model.metadata.model_reference == manifest.model.model_reference
    assert loaded_model.metadata.model_alias is None
    assert loaded_model.metadata.model_threshold == 0.37
    load_mlflow_model.assert_called_once_with("runs:/run-123/model")
    registry.validate_run_use_case.assert_called_once_with(
        run_id=manifest.run.run_id,
        expected_use_case="binary_uc",
    )
    registry.load_registered_model.assert_not_called()


@pytest.mark.unit
def test_load_model_rejects_manifest_for_another_use_case(
    binary_use_case: UseCase,
    monkeypatch: pytest.MonkeyPatch,
    training_manifest_factory: Callable[..., TrainingManifest],
) -> None:
    manifest = training_manifest_factory()
    manifest = replace(
        manifest,
        model=replace(manifest.model, use_case="alternate"),
    )
    registry_factory = Mock()
    monkeypatch.setattr(model_loader, "ModelRegistry", registry_factory)

    with pytest.raises(ValueError, match="selects use case 'alternate'"):
        model_loader.load_model(use_case=binary_use_case, manifest=manifest)

    registry_factory.assert_not_called()


@pytest.mark.unit
def test_load_model_rejects_mismatched_manifest_strategy(
    binary_use_case: UseCase,
    monkeypatch: pytest.MonkeyPatch,
    training_manifest_factory: Callable[..., TrainingManifest],
) -> None:
    manifest = training_manifest_factory()
    manifest = replace(
        manifest,
        training=replace(manifest.training, class_selection_strategy="argmax"),
    )
    registry_factory = Mock()
    monkeypatch.setattr(model_loader, "ModelRegistry", registry_factory)
    with pytest.raises(ValueError, match="class selection strategy"):
        model_loader.load_model(use_case=binary_use_case, manifest=manifest)
    registry_factory.assert_not_called()


@pytest.mark.integration
def test_load_model_loads_logged_mlflow_model_using_manifest(
    binary_use_case: UseCase,
    mlflow_test_setup: dict[str, object],
    training_manifest_factory: Callable[..., TrainingManifest],
    tmp_path: Path,
) -> None:
    tracking_uri = str(mlflow_test_setup["tracking_uri"])
    model_loader.mlflow.set_tracking_uri(tracking_uri)
    experiment_id = model_loader.mlflow.create_experiment(
        "unregistered-model-test",
        artifact_location=str(tmp_path / "artifacts"),
    )
    model_loader.mlflow.set_experiment(experiment_id=experiment_id)
    estimator = LogisticRegression().fit([[0], [1], [2], [3]], [0, 0, 1, 1])

    with model_loader.mlflow.start_run(experiment_id=experiment_id) as run:
        model_loader.mlflow.set_tag("use_case_name", "binary_uc")
        model_info = model_loader.mlflow.sklearn.log_model(
            estimator,
            name="model",
        )
        run_id = run.info.run_id

    manifest = training_manifest_factory()
    manifest = replace(
        manifest,
        run=replace(manifest.run, run_id=run_id),
        model=replace(
            manifest.model,
            model_version=None,
            model_reference=LoggedModelReference.from_model_info(model_info),
        ),
    )
    loaded_model = model_loader.load_model(
        use_case=binary_use_case,
        tracking_uri=tracking_uri,
        manifest=manifest,
    )

    assert loaded_model.metadata.model_version is None
    assert loaded_model.metadata.model_reference == manifest.model.model_reference
    assert loaded_model.model.predict_proba([[1]])[0, 1] < 0.5

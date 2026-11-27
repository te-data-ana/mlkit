from collections.abc import Callable
from dataclasses import replace
from typing import Any, Self

import numpy as np
import optuna
import pandas as pd
import pytest

from mlkit.config.schemas import TrainingConfig, TuningConfig
from mlkit.models import MODEL_CATALOG, create_model
from mlkit.training import tuning
from mlkit.training.tuning import (
    _align_decision_scores,
    _suggest_parameters,
    optimize_hyperparameters,
)
from mlkit.use_cases import UseCase


@pytest.mark.unit
def test_optimize_multiclass_pr_auc(
    multiclass_use_case: UseCase,
    multiclass_data_factory: Callable[[int], pd.DataFrame],
    config_factory: Callable[..., TrainingConfig],
) -> None:
    config = replace(
        config_factory(), tuning=TuningConfig(n_trials=1, n_splits=2, metric="pr_auc")
    )
    result = optimize_hyperparameters(
        use_case=multiclass_use_case,
        config=config,
        train_df=multiclass_data_factory(90),
    )
    assert 0 <= result.best_score <= 1


@pytest.mark.unit
def test_align_decision_scores_reorders_classes() -> None:
    scores = _align_decision_scores(
        scores=[[2.0, -1.0, 0.5]],
        model_classes=[2, 0, 1],
        target_classes=(0, 1, 2),
    )
    np.testing.assert_array_equal(scores, [[-1.0, 0.5, 2.0]])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("scores", "model_classes"),
    [
        ([0.1, 0.2, 0.3], [0, 1, 2]),
        ([[0.1, 0.2]], [0, 1, 2]),
        ([[0.1, 0.2, 0.3]], [0, 1, 3]),
    ],
)
def test_align_decision_scores_rejects_mismatched_classes(
    scores: list[float] | list[list[float]], model_classes: list[int]
) -> None:
    with pytest.raises(ValueError, match="one column per target class"):
        _align_decision_scores(scores, model_classes, (0, 1, 2))


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_alias",
    [alias for alias in MODEL_CATALOG if alias != "dc"],
)
def test_catalog_search_space_suggestions_build_each_model(
    model_alias: str,
) -> None:
    search_space = MODEL_CATALOG[model_alias]["search_space"]
    trial = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=7),
    ).ask()

    suggested = _suggest_parameters(
        trial=trial,
        search_space=search_space,
    )
    model, _ = create_model(
        model_alias=model_alias,
        model_params=suggested,
    )

    assert model is not None
    assert set(suggested) == set(search_space)


@pytest.mark.unit
def test_optimize_hyperparameters_returns_reproducible_study_result(
    binary_use_case: UseCase,
    monkeypatch: pytest.MonkeyPatch,
    config_factory: Callable[..., TrainingConfig],
    temporal_sample_training_df: pd.DataFrame,
) -> None:
    config: TrainingConfig = config_factory(classifier="dt")
    config = replace(
        config,
        random_state=13,
        tuning=TuningConfig(
            n_trials=2,
            n_splits=2,
            metric="roc_auc",
        ),
    )
    study_optimize = optuna.study.Study.optimize
    observed_outer_jobs: list[int] = []
    observed_estimator_jobs: list[int] = []

    def optimize_spy(
        study: optuna.study.Study,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        observed_outer_jobs.append(kwargs["n_jobs"])
        return study_optimize(study, *args, **kwargs)

    create_model = tuning.create_model

    def create_model_spy(**kwargs: Any) -> Any:
        observed_estimator_jobs.append(kwargs["n_jobs"])
        return create_model(**kwargs)

    monkeypatch.setattr(optuna.study.Study, "optimize", optimize_spy)
    monkeypatch.setattr(tuning, "create_model", create_model_spy)
    train_df = temporal_sample_training_df.copy()

    result = optimize_hyperparameters(
        config=config,
        train_df=train_df,
        use_case=binary_use_case,
        n_jobs=2,
    )

    assert set(result.best_params) == {
        "max_depth",
        "min_samples_leaf",
    }
    assert 0.0 <= result.best_score <= 1.0
    assert result.n_trials == 2
    assert result.n_splits == 2
    assert result.time_column == "reference_date"
    assert observed_outer_jobs == [2]
    assert observed_estimator_jobs
    assert set(observed_estimator_jobs) == {1}


@pytest.mark.unit
def test_optimize_hyperparameters_uses_declared_feature_columns(
    binary_use_case: UseCase,
    monkeypatch: pytest.MonkeyPatch,
    config_factory: Callable[..., TrainingConfig],
    temporal_sample_training_df: pd.DataFrame,
) -> None:
    config = config_factory(classifier="dt")
    config = replace(
        config,
        random_state=13,
        tuning=TuningConfig(
            n_trials=1,
            n_splits=2,
            metric="roc_auc",
        ),
    )
    train_df = temporal_sample_training_df.copy()
    train_df["customer_id"] = range(len(train_df))
    observed_columns: list[list[str]] = []

    class PipelineSpy:
        def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
            observed_columns.append(list(X.columns))
            return self

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.tile([0.5, 0.5], (len(X), 1))

    def build_pipeline(**_: object) -> PipelineSpy:
        return PipelineSpy()

    monkeypatch.setattr(tuning, "build_classifier_pipeline", build_pipeline)

    optimize_hyperparameters(
        config=config,
        train_df=train_df,
        use_case=binary_use_case,
    )

    assert observed_columns
    assert all(
        columns == list(binary_use_case.feature_columns) for columns in observed_columns
    )


@pytest.mark.unit
def test_optimize_hyperparameters_uses_decision_function_for_svm(
    binary_use_case: UseCase,
    monkeypatch: pytest.MonkeyPatch,
    config_factory: Callable[..., TrainingConfig],
    temporal_sample_training_df: pd.DataFrame,
) -> None:
    config = config_factory(classifier="lsvc")
    config = replace(
        config,
        random_state=13,
        tuning=TuningConfig(
            n_trials=1,
            n_splits=2,
            metric="roc_auc",
        ),
    )
    train_df = temporal_sample_training_df.copy()
    decision_scores: list[np.ndarray] = []

    class DecisionFunctionPipeline:
        def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
            return self

        def decision_function(self, X: pd.DataFrame) -> np.ndarray:
            scores = np.arange(len(X), dtype=float)
            decision_scores.append(scores)
            return scores

    monkeypatch.setattr(
        tuning,
        "build_classifier_pipeline",
        lambda **_: DecisionFunctionPipeline(),
    )

    result = optimize_hyperparameters(
        config=config,
        train_df=train_df,
        use_case=binary_use_case,
    )

    assert result.n_trials == 1
    assert decision_scores


@pytest.mark.unit
def test_optimize_hyperparameters_rejects_classifier_without_search_space(
    binary_use_case: UseCase,
    config_factory: Callable[..., TrainingConfig],
    sample_training_df: pd.DataFrame,
) -> None:
    config: TrainingConfig = config_factory(classifier="dc")
    config = replace(config, tuning=TuningConfig())

    with pytest.raises(ValueError, match="does not define an Optuna search space"):
        optimize_hyperparameters(
            config=config,
            train_df=sample_training_df,
            use_case=binary_use_case,
        )

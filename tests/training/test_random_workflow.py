from collections.abc import Callable
from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from use_cases.churn import CHURN_USE_CASE

from mlkit.config.schemas import TrainingConfig, TuningConfig
from mlkit.training import model_training, tuning
from mlkit.training.calibration import make_cv_splits, make_stratified_random_cv_splits
from mlkit.training.model_training import train_model
from mlkit.training.tuning import optimize_hyperparameters


def test_random_cv_is_seeded_stratified_without_timestamps() -> None:
    data = pd.DataFrame({"label": [0, 1] * 30})
    use_case = replace(
        CHURN_USE_CASE,
        target_column="label",
        time_column=None,
        split_strategy="stratified_random",
    )
    splits = make_cv_splits(use_case, data, n_splits=3, random_state=5)
    again = make_cv_splits(use_case, data, n_splits=3, random_state=5)
    direct = make_stratified_random_cv_splits(
        df=data, target_column="label", n_splits=3, random_state=5
    )
    for (train, test), (other_train, other_test), (direct_train, direct_test) in zip(
        splits, again, direct
    ):
        assert np.array_equal(train, other_train)
        assert np.array_equal(test, other_test)
        assert np.array_equal(train, direct_train)
        assert np.array_equal(test, direct_test)
        assert set(data.iloc[test]["label"]) == {0, 1}
        assert set(train).isdisjoint(test)


@pytest.mark.parametrize("labels", [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, None, 1]])
def test_random_cv_rejects_invalid_labels(labels: list[int | None]) -> None:
    data = pd.DataFrame({"label": labels})
    with pytest.raises(ValueError, match="Stratified CV"):
        make_stratified_random_cv_splits(data, target_column="label", n_splits=2)


def test_training_threshold_holdout_and_random_calibration(
    config_factory: Callable[..., TrainingConfig],
    temporal_sample_training_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = pd.concat([temporal_sample_training_df] * 8, ignore_index=True)
    train, test = data.iloc[:60], data.iloc[60:]
    config = replace(
        config_factory(classifier="dt"),
        random_state=17,
    )
    use_case = replace(
        CHURN_USE_CASE, time_column=None, split_strategy="stratified_random"
    )
    fit_df, validation_df = train.iloc[:48], train.iloc[48:]
    assert set(fit_df.index).isdisjoint(validation_df.index)
    calls = []
    original = model_training.fit_estimator_with_calibration

    def track_fit(**kwargs: Any) -> BaseEstimator:
        calls.append((set(kwargs["train_df"].index), kwargs["random_state"]))
        return original(**kwargs)

    monkeypatch.setattr(model_training, "fit_estimator_with_calibration", track_fit)
    result = train_model(
        use_case=use_case,
        config=config,
        fit_df=fit_df,
        validation_df=validation_df,
        test_df=test,
        calibration_method="sigmoid",
    )
    assert calls == [(set(fit_df.index), 17)]
    assert calls[0][0].isdisjoint(validation_df.index)
    assert result.threshold_data == "training_validation"
    assert 0 <= result.threshold <= 1


def test_random_tuning_without_time_column(
    config_factory: Callable[..., TrainingConfig],
    temporal_sample_training_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = pd.concat([temporal_sample_training_df] * 8, ignore_index=True)
    data = data.drop(columns="reference_date")
    config = replace(
        config_factory(classifier="dt"),
        random_state=17,
        tuning=TuningConfig(n_trials=2, n_splits=2),
    )
    sampler = Mock(wraps=tuning.optuna.samplers.TPESampler)
    cv_splits = Mock(wraps=tuning.make_cv_splits)
    monkeypatch.setattr(tuning.optuna.samplers, "TPESampler", sampler)
    monkeypatch.setattr(tuning, "make_cv_splits", cv_splits)
    use_case = replace(
        CHURN_USE_CASE, time_column=None, split_strategy="stratified_random"
    )
    result = optimize_hyperparameters(use_case, config, data)
    assert result.n_trials == 2
    sampler.assert_called_once_with(seed=17)
    assert cv_splits.call_args.kwargs["random_state"] == 17
    assert np.isfinite(result.best_score)

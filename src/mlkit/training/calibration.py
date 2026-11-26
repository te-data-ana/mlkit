"""Calibration fitting and temporal or stratified fold construction."""

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from mlkit.training.features import select_feature_columns
from mlkit.use_cases import UseCase


def make_temporal_cv_splits(
    df: pd.DataFrame,
    *,
    target_column: str,
    time_column: str,
    n_splits: int,
    target_classes: tuple[int, ...] = (0, 1),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding historical folds while keeping timestamp groups intact.

    Args:
        df: Training dataframe containing the target and timestamp columns.
        target_column: Column containing the target labels.
        time_column: Column defining chronological order.
        n_splits: Number of expanding-window folds.

    Returns:
        Scikit-learn-compatible train/calibration index pairs.

    Raises:
        ValueError: If columns, timestamps, labels, or split count are invalid.
    """
    if target_column not in df.columns:
        raise ValueError(f"Temporal CV target column '{target_column}' is missing.")
    if time_column not in df.columns:
        raise ValueError(f"Temporal CV time column '{time_column}' is missing.")
    if target_column == time_column:
        raise ValueError("Temporal CV target and time columns must be different.")
    timestamp_series = pd.to_datetime(
        df[time_column],
        errors="coerce",
        utc=True,
    )
    if timestamp_series.isna().any():
        raise ValueError("Temporal CV timestamps must be valid and non-missing.")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("Temporal CV n_splits must be at least 2.")
    unique_timestamps = pd.Index(timestamp_series.drop_duplicates()).sort_values()
    if len(unique_timestamps) <= n_splits:
        raise ValueError(
            "Temporal CV requires more distinct timestamps than "
            f"n_splits ({len(unique_timestamps)} timestamps, "
            f"{n_splits} splits)."
        )

    splitter = TimeSeriesSplit(n_splits=n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    timestamp_array = timestamp_series.to_numpy()
    labels = df[target_column].to_numpy()
    for train_times, val_times in splitter.split(unique_timestamps):
        train_unique = unique_timestamps[train_times]
        val_unique = unique_timestamps[val_times]
        train_indices = np.flatnonzero(np.isin(timestamp_array, train_unique))
        val_indices = np.flatnonzero(np.isin(timestamp_array, val_unique))
        if set(labels[train_indices]) != set(target_classes) or set(
            labels[val_indices]
        ) != set(target_classes):
            raise ValueError("Every temporal CV fold must contain all target classes.")
        splits.append((train_indices, val_indices))
    return splits


def make_stratified_random_cv_splits(
    df: pd.DataFrame,
    *,
    target_column: str,
    n_splits: int,
    random_state: int = 42,
    target_classes: tuple[int, ...] = (0, 1),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create seeded stratified cross-validation folds.

    Args:
        df: Training records.
        target_column: Binary target column.
        n_splits: Number of folds.
        random_state: Seed for shuffled folds.

    Returns:
        Positional train and validation indices for each fold.

    Raises:
        ValueError: If the target or folds are invalid.
    """
    if target_column not in df or df[target_column].isna().any():
        raise ValueError("Stratified CV requires a non-missing target.")
    counts = df[target_column].value_counts()
    if (
        set(counts.index) != set(target_classes)
        or n_splits < 2
        or counts.min() < n_splits
    ):
        raise ValueError("Stratified CV requires all classes in every fold.")
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    return list(splitter.split(df, df[target_column]))


def make_cv_splits(
    use_case: UseCase,
    df: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create cross-validation folds using the use-case split strategy.

    Args:
        use_case: Contract defining the target, strategy, and optional time column.
        df: Training records.
        n_splits: Number of folds.
        random_state: Seed used for stratified random folds.

    Returns:
        Positional train and validation indices for each fold.

    Raises:
        ValueError: If the strategy or its required data is invalid.
    """
    if use_case.split_strategy == "temporal":
        if use_case.time_column is None:
            raise ValueError("Temporal CV requires a time column.")
        return make_temporal_cv_splits(
            df,
            target_column=use_case.target_column,
            time_column=use_case.time_column,
            n_splits=n_splits,
            target_classes=use_case.target_classes,
        )
    if use_case.split_strategy == "stratified_random":
        return make_stratified_random_cv_splits(
            df,
            target_column=use_case.target_column,
            n_splits=n_splits,
            random_state=random_state,
            target_classes=use_case.target_classes,
        )
    raise ValueError(f"Unknown cross-validation strategy: {use_case.split_strategy}")


def fit_estimator_with_calibration(
    estimator: BaseEstimator,
    train_df: pd.DataFrame,
    *,
    use_case: UseCase,
    method: str = "none",
    n_splits: int = 4,
    n_jobs: int = 1,
    random_state: int = 42,
) -> BaseEstimator:
    """Fit an estimator, optionally calibrating with the selected fold strategy.

    Args:
        estimator: Unfitted scikit-learn estimator or pipeline.
        train_df: Training data with target and, for temporal CV, timestamps.
        use_case: Contract defining targets, features, and split strategy.
        method: ``none``, ``sigmoid``, or ``isotonic``.
        n_splits: Number of calibration folds.
        n_jobs: Total job budget for calibration folds.
        random_state: Seed used by stratified random folds.

    Returns:
        A fitted estimator, wrapped in ``CalibratedClassifierCV`` when enabled.

    Raises:
        ValueError: If required model features are missing, the method is
            unsupported, or calibration folds are invalid.
    """
    if use_case.target_column not in train_df.columns:
        raise ValueError(
            f"Training target column '{use_case.target_column}' is missing."
        )

    X_train = select_feature_columns(
        df=train_df,
        feature_columns=use_case.feature_columns,
        data_name="Training",
    )
    y_train = train_df[use_case.target_column]

    if method == "none":
        if not hasattr(estimator, "predict_proba"):
            raise ValueError(
                f"Classifier '{estimator.__class__.__name__}' does not expose "
                "predict_proba; set calibration_method to 'sigmoid' or "
                "'isotonic'."
            )
        return estimator.fit(X_train, y_train)
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(
            "Calibration method must be one of 'none', 'sigmoid', or 'isotonic'."
        )

    splits = make_cv_splits(
        use_case,
        df=train_df,
        n_splits=n_splits,
        random_state=random_state,
    )
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=splits,
        ensemble=True,
        n_jobs=n_jobs,
    )
    with parallel_backend("loky", inner_max_num_threads=1):
        return calibrated.fit(X_train, y_train)

from dataclasses import dataclass
from time import perf_counter
from types import TracebackType
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def align_probabilities(
    probabilities: ArrayLike,
    model_classes: ArrayLike,
    target_classes: tuple[int, ...],
) -> np.ndarray:
    """Validate and align model probabilities to the use-case class order.

    Args:
        probabilities: Model output probabilities.
        model_classes: Classes in the model output.
        target_classes: Expected class order.

    Returns:
        Aligned probability array.
    """
    array = np.asarray(probabilities, dtype=float)
    classes = np.asarray(model_classes)
    if (
        classes.ndim != 1
        or len(classes) != len(target_classes)
        or set(classes.tolist()) != set(target_classes)
        or array.ndim != 2
        or array.shape[1] != len(classes)
    ):
        raise ValueError(
            "Model probability columns must match all declared target classes."
        )
    if (
        not np.isfinite(array).all()
        or ((array < 0) | (array > 1)).any()
        or not np.allclose(array.sum(axis=1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError(
            "Model probabilities must be finite, in [0, 1], and sum to one."
        )
    return array[
        :, [int(np.flatnonzero(classes == value)[0]) for value in target_classes]
    ]


@dataclass
class Timer:
    duration: float | None = None

    def __enter__(self) -> Self:
        """Start timing and return this context manager."""
        self._start = perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Store elapsed monotonic time when leaving the context."""
        self.duration = perf_counter() - self._start


@dataclass
class ModelEvaluation:
    """Ranking, classification, calibration, and timing metrics."""

    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    calibration_error: float
    fit_time_sec: float | None
    pred_time_sec: float | None


def evaluate_model(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    threshold: float,
    calibration_bins: int = 10,
    fit_time_sec: float | None = None,
    pred_time_sec: float | None = None,
) -> ModelEvaluation:
    """Evaluate a binary classifier from labels and positive probabilities.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        threshold: Probability threshold used to derive predicted classes.
        calibration_bins: Number of equally sized confidence bins used for
            expected calibration error.
        fit_time_sec: Optional time spent fitting the model, in seconds.
        pred_time_sec: Optional time spent generating predictions, in seconds.

    Returns:
        ModelEvaluation containing ROC AUC, PR AUC, classification metrics,
        Brier score, expected calibration error, and the supplied timing values.

    Raises:
        ValueError: If the supplied labels or probabilities are invalid for a
            scikit-learn metric.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be a positive integer.")

    y_prob = np.asarray(y_prob, dtype=float)
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Predicted probabilities must be finite values in [0, 1].")
    y_pred = (y_prob >= threshold).astype(int)
    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        y_prob,
        n_bins=calibration_bins,
        strategy="uniform",
    )
    bin_edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    bin_indices = np.searchsorted(bin_edges[1:-1], y_prob)
    bin_counts = np.bincount(bin_indices, minlength=calibration_bins)
    nonempty_counts = bin_counts[bin_counts > 0]
    calibration_error = float(
        np.average(
            np.abs(fraction_positive - mean_predicted),
            weights=nonempty_counts,
        )
    )

    return ModelEvaluation(
        roc_auc=roc_auc_score(y_true, y_prob),
        pr_auc=average_precision_score(y_true, y_prob),
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        brier_score=brier_score_loss(y_true, y_prob),
        calibration_error=calibration_error,
        fit_time_sec=fit_time_sec,
        pred_time_sec=pred_time_sec,
    )


def evaluate_multiclass_model(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    classes: tuple[int, ...],
    calibration_bins: int = 10,
    fit_time_sec: float | None = None,
    pred_time_sec: float | None = None,
) -> ModelEvaluation:
    """Evaluate class-aligned multiclass probabilities using macro OVR metrics.

    Args:
        y_true: Ground-truth class labels.
        y_prob: Predicted probabilities for each class.
        classes: Expected class labels.
        calibration_bins: Number of equally sized confidence bins used for
            expected calibration error.
        fit_time_sec: Optional time spent fitting the model, in seconds.
        pred_time_sec: Optional time spent generating predictions, in seconds.
    """
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be a positive integer.")
    probabilities = align_probabilities(
        probabilities=y_prob,
        model_classes=classes,
        target_classes=classes,
    )
    labels = np.asarray(y_true)
    if (
        labels.ndim != 1
        or len(labels) != len(probabilities)
        or set(labels) != set(classes)
    ):
        raise ValueError(
            "Evaluation requires every declared class and one label per row."
        )
    indicator = np.column_stack([labels == value for value in classes]).astype(int)
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    errors = []
    for column in range(len(classes)):
        scores = probabilities[:, column]
        bins = np.searchsorted(np.linspace(0, 1, calibration_bins + 1)[1:-1], scores)
        errors.append(
            sum(
                np.mean(bins == i)
                * abs(scores[bins == i].mean() - indicator[bins == i, column].mean())
                for i in range(calibration_bins)
                if (bins == i).any()
            )
        )
    return ModelEvaluation(
        roc_auc=float(
            roc_auc_score(y_true=indicator, y_score=probabilities, average="macro")
        ),
        pr_auc=float(
            average_precision_score(
                y_true=indicator,
                y_score=probabilities,
                average="macro",
            )
        ),
        accuracy=float(accuracy_score(y_true=labels, y_pred=predicted)),
        precision=float(
            precision_score(
                y_true=labels,
                y_pred=predicted,
                labels=list(classes),
                average="macro",
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y_true=labels,
                y_pred=predicted,
                labels=list(classes),
                average="macro",
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                y_true=labels,
                y_pred=predicted,
                labels=list(classes),
                average="macro",
                zero_division=0,
            )
        ),
        brier_score=float(
            np.mean(np.sum((probabilities - indicator) ** 2, axis=1)) / 2
        ),
        calibration_error=float(np.mean(errors)),
        fit_time_sec=fit_time_sec,
        pred_time_sec=pred_time_sec,
    )

"""Calibration diagnostic helpers."""

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve


def reliability_diagram(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_bins: int,
    classes: tuple[int, ...] | None = None,
) -> Figure:
    """Build a reliability diagram for observed and predicted probabilities.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of equally sized probability bins used to visualize
            calibration quality.

    Returns:
        A Matplotlib figure suitable for logging as an MLflow artifact.
    """
    figure = Figure(figsize=(6, 6))
    axes = figure.subplots()
    axes.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    if classes is None:
        curves = [("Model", y_true, y_prob)]
    else:
        labels, probabilities = np.asarray(y_true), np.asarray(y_prob)
        curves = [
            (str(value), labels == value, probabilities[:, index])
            for index, value in enumerate(classes)
        ]
    for label, actual, probability in curves:
        fraction_positive, mean_predicted = calibration_curve(
            y_true=actual, y_prob=probability, n_bins=n_bins, strategy="uniform"
        )
        axes.plot(mean_predicted, fraction_positive, marker="o", label=label)
    axes.set(
        xlabel="Mean predicted probability",
        ylabel="Fraction of positives",
        title="Reliability Diagram",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure

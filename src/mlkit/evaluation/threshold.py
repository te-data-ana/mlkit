"""Classification threshold selection helpers."""

import numpy as np
from numpy.typing import ArrayLike


def select_f1_threshold(
    y_true: ArrayLike,
    y_prob: ArrayLike,
) -> float:
    """Select the threshold maximizing F1, preferring higher thresholds on ties.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        The selected probability threshold.

    Raises:
        ValueError: If labels do not contain both binary classes or
            probabilities are invalid.
    """
    label_array = np.asarray(y_true)
    prob_array = np.asarray(y_prob, dtype=float)
    if np.unique(label_array).size != 2 or not set(np.unique(label_array)).issubset(
        {0, 1}
    ):
        raise ValueError("Threshold selection data must contain both binary classes.")
    if (
        prob_array.ndim != 1
        or prob_array.size != label_array.size
        or not np.isfinite(prob_array).all()
        or ((prob_array < 0) | (prob_array > 1)).any()
    ):
        raise ValueError("Threshold probabilities must be finite values in [0, 1].")

    thresholds = np.unique(np.concatenate(([0.0], prob_array, [1.0])))
    order = np.argsort(prob_array)
    sorted_probs = prob_array[order]
    positives = np.asarray(label_array[order] == 1, dtype=np.int64)
    suffix_positives = np.concatenate((np.cumsum(positives[::-1])[::-1], [0]))
    indices = np.searchsorted(sorted_probs, thresholds, side="left")
    true_positives = suffix_positives[indices]
    actual_positives = positives.sum()
    predicted_positives = len(order) - indices
    f1_scores = 2 * true_positives / (actual_positives + predicted_positives)
    best_score = float(f1_scores.max())
    best_threshold = float(thresholds[f1_scores == best_score][-1])
    return best_threshold

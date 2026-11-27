import numpy as np
import pytest

from mlkit.evaluation import metrics
from mlkit.evaluation.metrics import (
    evaluate_multiclass_model,
    validate_and_align_probabilities,
)


@pytest.mark.unit
def test_timer_records_elapsed_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    # Freeze the clock so the context manager's measured duration is deterministic.
    clock = iter([10.0, 10.25])
    monkeypatch.setattr(metrics, "perf_counter", lambda: next(clock))

    with metrics.Timer() as timer:
        pass

    assert timer.duration == pytest.approx(0.25)


@pytest.mark.unit
def test_evaluate_model_with_perfect_predictions_returns_perfect_metrics() -> None:
    # Use perfectly separated probabilities so every classification metric is 1.0.
    evaluation = metrics.evaluate_model(
        y_true=np.array([0, 0, 1, 1]),
        y_prob=np.array([0.1, 0.4, 0.6, 0.9]),
        threshold=0.5,
        fit_time_sec=1.2,
        pred_time_sec=0.3,
    )

    # Verify the ranking, classification, calibration, and timing metrics.
    assert evaluation.roc_auc == pytest.approx(1.0)
    assert evaluation.pr_auc == pytest.approx(1.0)
    assert evaluation.accuracy == pytest.approx(1.0)
    assert evaluation.precision == pytest.approx(1.0)
    assert evaluation.recall == pytest.approx(1.0)
    assert evaluation.f1 == pytest.approx(1.0)
    assert evaluation.brier_score == pytest.approx(0.085)
    assert evaluation.calibration_error == pytest.approx(0.25)
    assert evaluation.fit_time_sec == 1.2
    assert evaluation.pred_time_sec == 0.3


@pytest.mark.unit
def test_evaluate_model_uses_threshold_for_classification_metrics() -> None:
    # The custom threshold changes probabilities into two positive predictions.
    evaluation = metrics.evaluate_model(
        y_true=np.array([0, 1, 1, 0]),
        y_prob=np.array([0.2, 0.55, 0.8, 0.9]),
        threshold=0.6,
    )

    # Check metrics that depend on the thresholded class predictions.
    assert evaluation.accuracy == pytest.approx(0.5)
    assert evaluation.precision == pytest.approx(0.5)
    assert evaluation.recall == pytest.approx(0.5)
    assert evaluation.f1 == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("labels", "probabilities", "bins", "expected"),
    [
        ([0, 1, 1, 1], [0.0, 0.5, 0.6, 1.0], 2, 0.225),
        ([0, 1, 0, 1], [0.5, 0.56, 0.31, 0.14], 10, 0.5275),
    ],
)
def test_evaluate_model_calibration_error_on_bin_boundaries(
    labels: list[int], probabilities: list[float], bins: int, expected: float
) -> None:
    evaluation = metrics.evaluate_model(
        y_true=labels, y_prob=probabilities, threshold=0.5, calibration_bins=bins
    )
    assert evaluation.calibration_error == pytest.approx(expected)


@pytest.mark.unit
def test_multiclass_probabilities_and_metrics() -> None:
    probabilities = validate_and_align_probabilities(
        probabilities=[[0.7, 0.1, 0.2]],
        model_classes=[2, 0, 1],
        target_classes=(0, 1, 2),
    )
    np.testing.assert_allclose(probabilities, [[0.1, 0.2, 0.7]])
    with pytest.raises(ValueError, match="sum to one"):
        validate_and_align_probabilities(
            probabilities=[[0.5, 0.5, 0.5]],
            model_classes=[0, 1, 2],
            target_classes=(0, 1, 2),
        )
    metrics = evaluate_multiclass_model(
        y_true=[0, 1, 2],
        y_prob=np.eye(3),
        classes=(0, 1, 2),
    )
    assert metrics.accuracy == metrics.f1 == metrics.roc_auc == metrics.pr_auc == 1
    assert metrics.brier_score == metrics.calibration_error == 0
    metrics = evaluate_multiclass_model(
        y_true=[0, 1, 2],
        y_prob=[[0.8, 0.1, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]],
        classes=(0, 1, 2),
        calibration_bins=2,
    )
    assert metrics.brier_score == pytest.approx(0.58 / 6)
    assert metrics.calibration_error == pytest.approx(7 / 45)


@pytest.mark.unit
@pytest.mark.parametrize(
    "probabilities",
    [
        [[0.5, 0.5]],
        [[0.5, 0.5, 0.5]],
        [[-0.1, 0.5, 0.6]],
        [[float("nan"), 0.5, 0.5]],
    ],
)
def test_aligned_probabilities_are_still_validated(
    probabilities: list[list[float]],
) -> None:
    with pytest.raises(ValueError):
        validate_and_align_probabilities(probabilities, (0, 1, 2), (0, 1, 2))

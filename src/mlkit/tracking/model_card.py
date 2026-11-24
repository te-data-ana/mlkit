import tempfile
from textwrap import dedent

import mlflow

from mlkit.config.schemas import TrainingConfig
from mlkit.tracking.manifest import TrainingManifest


class ModelCardBuilder:
    def build(
        self,
        *,
        config: TrainingConfig,
        context: TrainingManifest,
    ) -> str:
        """Build a Markdown model card for a trained model.

        The generated model card summarizes model metadata, dataset
        characteristics, feature information, preprocessing configuration,
        evaluation metrics, and promotion decision details when available.

        Args:
            config: Training configuration used to build and evaluate the
                model, including preprocessing settings.
            context: Additional model card context containing the registered
                model identity, run-time training options, evaluation, and
                optional promotion decision.

        Returns:
            A Markdown-formatted string representing the model card.

        Raises:
            KeyError: If required metric or promotion-decision keys are missing.
            AttributeError: If any required attributes are missing from
                ``config``, ``context``, or the promotion decision object.
        """

        metrics = context.evaluation.metrics
        decision = context.lifecycle.promotion_decision
        split_metadata = context.data.split_metadata
        strategy = split_metadata["split_strategy"]
        random_split = strategy == "stratified_random"
        detail_rows = [
            f"| Split Strategy | {str(strategy).replace('_', ' ').capitalize()} |"
        ]
        if not random_split and context.data.time_column is not None:
            detail_rows.append(f"| Time Column | {context.data.time_column} |")
        for key, label in (
            ("split_train_start", "Training Start"),
            ("split_train_end", "Training End"),
            ("split_test_start", "Test Start"),
            ("split_test_end", "Test End"),
            ("split_random_state", "Split Seed"),
            ("split_test_fraction", "Test Fraction"),
            ("split_monitoring_fraction", "Monitoring Fraction"),
            ("split_monitoring_rows", "Monitoring Cohort Records"),
        ):
            if key in split_metadata:
                detail_rows.append(f"| {label} | {split_metadata[key]} |")
        split_rows = "\n".join(detail_rows)
        evaluation_name = (
            "random stratified test set" if random_split else "out-of-time test set"
        )
        threshold_note = (
            "The decision threshold was selected by maximizing F1 on a training "
            f"validation holdout (fraction {config.threshold.validation_fraction}). "
            "The estimator was fitted only on the remaining training rows."
        )

        features = "\n".join(
            f"- {feature}" for feature in context.training.feature_names_out
        )

        promotion_section = ""
        if decision:
            champion_metric = (
                f"{decision['champion_score']:.4f}"
                if decision["champion_score"] is not None
                else "N/A"
            )
            metric_delta = (
                f"{decision['score_delta']:.4f}"
                if decision["score_delta"] is not None
                else "N/A"
            )

            promotion_section = dedent(
                f"""
                ## Promotion Decision

                | Property | Value |
                |----------|-------|
                | Weighted Candidate Score | {decision["candidate_score"]:.4f} |
                | Weighted Champion Score | {champion_metric} |
                | Required Delta | {decision["min_delta"]:.4f} |
                | Actual Delta | {metric_delta} |
                | Promoted | {decision["promote"]} |

                Reason: {decision["reason"]}
                """
            ).strip()

        return f"""
# Model Card

## Model Metadata

| Property | Value |
|----------|-------|
| MLflow Run ID | {context.run.run_id} |
| MLflow Run Name | {context.run.run_name} |
| MLflow Model URI | {context.model.model_reference.model_uri} |
| Model Name | {context.model.model_name} |
| Model Version | {context.model.model_version if context.model.model_version is not None else "Not registered"} |
| Classifier Name | {context.training.classifier_name} |
| Classifier Alias | {context.training.classifier_alias} |
| Split Name | {context.data.split_name} |

## Intended Use

Predict {context.model.use_case} probability / class label.

## Training Dataset

Trained on {context.model.use_case} data.

| Property | Value |
|----------|-------|
| Target Column | {context.data.target_column} |
| Training Cohort Records | {split_metadata["split_train_rows"]} |
| Fitting Cohort Records | {split_metadata["split_fit_rows"]} |
| Threshold Cohort Records | {split_metadata["split_validation_rows"]} |
| Test Cohort Records | {split_metadata["split_test_rows"]} |
{split_rows}
| Feature Count | {context.training.feature_count} |

## Model Features

{features}

## Imputation Strategy

| Feature Type | Imputation Strategy |
|--------------|---------------------|
| Numeric | {config.preprocessing.numeric_impute_strategy} |
| Categorical | {config.preprocessing.categorical_impute_strategy} |

## Probability Calibration and Threshold

| Property | Value |
|----------|-------|
| Calibration Method | {context.training.calibration} |
| Threshold Selection Strategy | {context.training.threshold_strategy} |
| Threshold Selection Data | {context.training.threshold_data} |
| Optimal Class Threshold | {context.training.threshold:.6f} |

Note: {threshold_note}

## Evaluation on {evaluation_name}

| Metric | Value |
|--------|-------|
| ROC AUC | {metrics["roc_auc"]:.4f} |
| PR AUC | {metrics["pr_auc"]:.4f} |
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| F1 | {metrics["f1"]:.4f} |
| Brier Score | {metrics["brier_score"]:.4f} |
| Calibration Error (ECE) | {metrics["calibration_error"]:.4f} |

{promotion_section}
"""


def log_model_card(
    markdown: str,
    artifact_path: str = "documentation",
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(markdown)

    mlflow.log_artifact(
        f.name,
        artifact_path=artifact_path,
    )

import shlex
from pathlib import Path

import yaml
from dvc.repo import Repo


def test_dvc_seed_parameters_reach_and_invalidate_consuming_stages() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "pipeline/churn/dvc.yaml:add_reference_dates": ("preparation",),
        "pipeline/churn/dvc.yaml:add_inference_reference_dates": ("preparation",),
        "pipeline/churn/dvc.yaml:train_model": ("training", "classifier"),
        "pipeline/churn/monitoring/dvc.yaml:serve_samples": ("monitoring",),
        "pipeline/phishing/dvc.yaml:create_split": ("split",),
        "pipeline/phishing/dvc.yaml:train_model": ("training", "classifier"),
        "pipeline/phishing/monitoring/dvc.yaml:serve_samples": ("monitoring",),
        "pipeline/diabetes/dvc.yaml:create_split": ("split",),
        "pipeline/diabetes/dvc.yaml:train_model": ("training", "classifier"),
        "pipeline/diabetes/monitoring/dvc.yaml:serve_samples": ("monitoring",),
    }
    with Repo(str(root)) as repo:
        stages = {stage.addressing: stage for stage in repo.index.stages}
        for address, components in expected.items():
            stage = stages[address]
            use_case = address.split("/")[1]
            params_path = f"pipeline/{use_case}.yaml"
            values = yaml.safe_load((root / params_path).read_text())
            params_dep = next(dep for dep in stage.deps if str(dep) == params_path)
            args = shlex.split(stage.cmd)
            assert args[args.index("--random_state") + 1] == str(values["random_state"])
            assert "random_state" in params_dep.params
            for component in components:
                flag = f"--{component}_random_state"
                assert args[args.index(flag) + 1] == str(values["seeds"][component])
                assert f"seeds.{component}" in params_dep.params


def test_diabetes_pipeline_targets_and_outputs() -> None:
    root = Path(__file__).resolve().parents[1]
    with Repo(str(root)) as repo:
        stages = {
            stage.addressing: stage
            for stage in repo.index.stages
            if stage.addressing.startswith("pipeline/diabetes/")
        }
        assert set(stages) == {
            "pipeline/diabetes/dvc.yaml:fetch_and_prepare_uci",
            "pipeline/diabetes/dvc.yaml:create_split",
            "pipeline/diabetes/dvc.yaml:train_model",
            "pipeline/diabetes/dvc.yaml:batch_predict",
            "pipeline/diabetes/monitoring/dvc.yaml:serve_samples",
            "pipeline/diabetes/monitoring/dvc.yaml:ingest_prediction_logs",
            "pipeline/diabetes/monitoring/dvc.yaml:ingest_error_logs",
            "pipeline/diabetes/monitoring/dvc.yaml:monitor",
        }
        fetch = shlex.split(
            stages["pipeline/diabetes/dvc.yaml:fetch_and_prepare_uci"].cmd
        )
        assert fetch[fetch.index("--dataset-id") + 1] == "891"
        assert "--map-target" not in fetch
        serve = shlex.split(
            stages["pipeline/diabetes/monitoring/dvc.yaml:serve_samples"].cmd
        )
        assert serve[serve.index("--drop_columns") + 1] == "diabetes_binary"
        for stage in stages.values():
            assert all(str(out).startswith("data/diabetes/") for out in stage.outs)

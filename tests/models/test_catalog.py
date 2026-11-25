import pytest
from sklearn.svm import SVC, LinearSVC

from mlkit.models import create_model


@pytest.mark.unit
@pytest.mark.parametrize(
    ("alias", "expected_type"),
    [("lsvc", LinearSVC), ("svc", SVC)],
)
def test_svm_catalog_returns_unwrapped_estimators(
    alias: str,
    expected_type: type[LinearSVC] | type[SVC],
) -> None:
    classifier, config = create_model(alias, {})

    assert type(classifier) is expected_type
    assert config["model_name"] == expected_type.__name__
    if alias == "svc":
        assert classifier.kernel == "rbf"


def test_seed_is_applied_only_to_supported_classifiers() -> None:
    tree, tree_config = create_model("dt", {}, random_state=7)
    bayes, bayes_config = create_model("nb", {}, random_state=7)
    assert tree.get_params()["random_state"] == tree_config["random_state"] == 7
    assert "random_state" not in bayes.get_params()
    assert "random_state" not in bayes_config

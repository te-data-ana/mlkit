import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError, field_validator

import mlkit
from mlkit.data import uci
from mlkit.data.uci import fetch_uci_snapshot, prepare_uci_snapshot
from mlkit.models import IdentityFeatureBuilder
from mlkit.use_cases import UseCase


@pytest.mark.unit
def test_uci_snapshot_preserves_dataframes_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    features = pd.DataFrame({"Feature": [1, 2]})
    targets = pd.DataFrame({"target": [0, 1]})
    dataset = SimpleNamespace(
        data=SimpleNamespace(features=features, targets=targets, ids=None),
        metadata={"uci_id": 967},
        variables=pd.DataFrame({"name": ["Feature", "target"]}),
    )
    monkeypatch.setitem(
        sys.modules, "ucimlrepo", SimpleNamespace(fetch_ucirepo=lambda **_: dataset)
    )
    fetch_uci_snapshot(dataset_id=967, snapshot_dir=tmp_path)
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "features.parquet"), features
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "targets.parquet"), targets
    )
    assert '"uci_id": 967' in (tmp_path / "metadata.json").read_text()


@pytest.mark.unit
def test_uci_snapshot_rejects_unaligned_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = SimpleNamespace(
        data=SimpleNamespace(
            features=pd.DataFrame({"x": [1]}, index=[1]),
            targets=pd.DataFrame({"y": [1]}, index=[2]),
            ids=None,
        )
    )
    monkeypatch.setitem(
        sys.modules, "ucimlrepo", SimpleNamespace(fetch_ucirepo=lambda **_: dataset)
    )
    with pytest.raises(ValueError, match="aligned"):
        fetch_uci_snapshot(dataset_id=967, snapshot_dir=tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "ids",
    [
        pd.DataFrame({"ID": [1]}, index=[1]),
        pd.DataFrame({"ID": [1, 2]}, index=[1, 2]),
        pd.DataFrame([[1, 2], [3, 4]], columns=["ID", "ID"]),
        "invalid",
    ],
)
def test_uci_snapshot_rejects_invalid_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ids: object
) -> None:
    dataset = SimpleNamespace(
        data=SimpleNamespace(
            features=pd.DataFrame({"x": [1, 2]}),
            targets=pd.DataFrame({"y": [0, 1]}),
            ids=ids,
        )
    )
    monkeypatch.setitem(
        sys.modules, "ucimlrepo", SimpleNamespace(fetch_ucirepo=lambda **_: dataset)
    )
    with pytest.raises(ValueError, match="identifier"):
        fetch_uci_snapshot(dataset_id=891, snapshot_dir=tmp_path)
    assert not (tmp_path / "features.parquet").exists()


@pytest.mark.unit
def test_uci_snapshot_replaces_stale_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = pd.DataFrame({"ID": [1, 2]})
    dataset = SimpleNamespace(
        data=SimpleNamespace(
            features=pd.DataFrame({"x": [1, 2]}),
            targets=pd.DataFrame({"y": [0, 1]}),
            ids=ids,
        ),
        metadata={},
        variables=pd.DataFrame({"name": ["ID", "x", "y"]}),
    )
    monkeypatch.setitem(
        sys.modules, "ucimlrepo", SimpleNamespace(fetch_ucirepo=lambda **_: dataset)
    )
    fetch_uci_snapshot(891, tmp_path)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "ids.parquet"), ids)
    dataset.data.ids = None
    fetch_uci_snapshot(891, tmp_path)
    assert not (tmp_path / "ids.parquet").exists()


def _write_snapshot(
    directory: Path, features: pd.DataFrame, targets: pd.DataFrame
) -> None:
    features.to_parquet(directory / "features.parquet")
    targets.to_parquet(directory / "targets.parquet")


@pytest.mark.unit
def test_prepare_uci_preserves_multiple_arbitrary_targets(tmp_path: Path) -> None:
    _write_snapshot(
        directory=tmp_path,
        features=pd.DataFrame({"PetalLength": [1.1, 2.2], "ExtraValue": ["a", "b"]}),
        targets=pd.DataFrame(
            {"SpeciesName": ["setosa", "virginica"], "Score": [1.5, None]}
        ),
    )
    output = tmp_path / "out" / "prepared.parquet"
    prepare_uci_snapshot(tmp_path, output)
    result = pd.read_parquet(output)
    expected = pd.DataFrame(
        {
            "petal_length": [1.1, 2.2],
            "extra_value": ["a", "b"],
            "species_name": ["setosa", "virginica"],
            "score": [1.5, None],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.unit
@pytest.mark.parametrize("with_use_case", [False, True])
def test_prepare_uci_deduplicates_using_ids(
    tmp_path: Path, with_use_case: bool
) -> None:
    class Features(BaseModel):
        value: int

    use_case = UseCase(
        name="example",
        target_column="label",
        time_column=None,
        split_strategy="stratified_random",
        feature_schema=Features,
        feature_builder_factory=IdentityFeatureBuilder,
    )
    _write_snapshot(
        tmp_path, pd.DataFrame({"Value": [5, 5, 5]}), pd.DataFrame({"Label": [1, 1, 1]})
    )
    pd.DataFrame({"ID": [10, 11, 10]}).to_parquet(tmp_path / "ids.parquet")
    output = tmp_path / "prepared.parquet"
    prepare_uci_snapshot(
        tmp_path, output, use_case=use_case if with_use_case else None, deduplicate=True
    )
    expected = {"value": [5, 5], "label": [1, 1]}
    if not with_use_case:
        expected["id"] = [10, 11]
    assert pd.read_parquet(output).to_dict("list") == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ids", "error"),
    [
        ({"ID": [1]}, "aligned"),
        ({"IDValue": [1, 2], "ID Value": [3, 4]}, "unique"),
        ({"Value": [1, 2]}, "overlap"),
        ({"Label": [1, 2]}, "overlap"),
    ],
)
def test_prepare_uci_rejects_invalid_ids(
    tmp_path: Path, ids: dict[str, list[int]], error: str
) -> None:
    _write_snapshot(
        tmp_path, pd.DataFrame({"Value": [5, 6]}), pd.DataFrame({"Label": [0, 1]})
    )
    pd.DataFrame(ids).to_parquet(tmp_path / "ids.parquet")
    with pytest.raises(ValueError, match=error):
        prepare_uci_snapshot(tmp_path, tmp_path / "prepared.parquet")
    assert not (tmp_path / "prepared.parquet").exists()


@pytest.mark.unit
def test_prepare_uci_maps_labels_and_validates_use_case(tmp_path: Path) -> None:
    class Features(BaseModel):
        petal_length: float

    use_case = UseCase(
        name="flowers",
        target_column="species",
        time_column=None,
        split_strategy="stratified_random",
        feature_schema=Features,
        feature_builder_factory=IdentityFeatureBuilder,
    )
    _write_snapshot(
        directory=tmp_path,
        features=pd.DataFrame({"PetalLength": [1.1, 2.2], "Ignored": [5, 6]}),
        targets=pd.DataFrame({"Species": ["setosa", "versicolor"]}),
    )
    output = tmp_path / "prepared.parquet"
    prepare_uci_snapshot(
        snapshot_dir=tmp_path,
        output_path=output,
        target_mappings={"species": {"setosa": 0, "versicolor": 1}},
        use_case=use_case,
    )
    result = pd.read_parquet(output)
    assert result.columns.tolist() == ["petal_length", "species"]
    assert result["species"].tolist() == [0, 1]

    _write_snapshot(
        directory=tmp_path,
        features=pd.DataFrame({"PetalLength": [1.1, 2.2]}),
        targets=pd.DataFrame({"Species": [0, 1], "Score": [2, 3]}),
    )
    with pytest.raises(ValueError, match="requires one target"):
        prepare_uci_snapshot(
            snapshot_dir=tmp_path, output_path=output, use_case=use_case
        )


@pytest.mark.unit
@pytest.mark.parametrize("with_use_case", [False, True])
@pytest.mark.parametrize("deduplicate", [False, True])
def test_prepare_uci_deduplicates_source_before_mapping_and_filtering(
    tmp_path: Path, with_use_case: bool, deduplicate: bool
) -> None:
    class Features(BaseModel):
        value: int

    use_case = UseCase(
        name="example",
        target_column="label",
        time_column=None,
        split_strategy="stratified_random",
        feature_schema=Features,
        feature_builder_factory=IdentityFeatureBuilder,
    )
    features = pd.DataFrame({"Value": [1] * 5, "Ignored": ["a", "a", "b", "a", "a"]})
    targets = pd.DataFrame({"Label": ["a", "a", "a", "b", "c"]})
    _write_snapshot(tmp_path, features, targets)
    output = tmp_path / "prepared.parquet"
    prepare_uci_snapshot(
        tmp_path,
        output,
        target_mappings={"label": {"a": 0, "b": 0, "c": 1}},
        use_case=use_case if with_use_case else None,
        deduplicate=deduplicate,
    )
    result = pd.read_parquet(output)
    assert result["label"].tolist() == (
        [0, 0, 0, 1] if deduplicate else [0, 0, 0, 0, 1]
    )
    assert ("ignored" in result.columns) is not with_use_case
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "features.parquet"), features
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "targets.parquet"), targets
    )


@pytest.mark.unit
def test_prepare_uci_validates_all_rows_before_deduplicating(tmp_path: Path) -> None:
    calls = 0

    class Features(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def reject_second_row(cls, value: int) -> int:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("second row rejected")
            return value

    use_case = UseCase(
        name="example",
        target_column="label",
        time_column=None,
        split_strategy="stratified_random",
        feature_schema=Features,
        feature_builder_factory=IdentityFeatureBuilder,
    )
    _write_snapshot(
        tmp_path, pd.DataFrame({"Value": [1, 1]}), pd.DataFrame({"Label": [0, 0]})
    )
    with pytest.raises(ValidationError, match="second row rejected"):
        prepare_uci_snapshot(
            tmp_path, tmp_path / "prepared.parquet", use_case=use_case, deduplicate=True
        )
    assert not (tmp_path / "prepared.parquet").exists()


@pytest.mark.unit
def test_prepare_uci_deduplicates_across_validation_chunks(tmp_path: Path) -> None:
    class Features(BaseModel):
        value: int

    use_case = UseCase(
        name="example",
        target_column="label",
        time_column=None,
        split_strategy="stratified_random",
        feature_schema=Features,
        feature_builder_factory=IdentityFeatureBuilder,
    )
    _write_snapshot(
        tmp_path,
        pd.DataFrame({"Value": [*range(10_000), 0]}),
        pd.DataFrame({"Label": [0] * 10_001}),
    )
    output = tmp_path / "prepared.parquet"
    prepare_uci_snapshot(tmp_path, output, use_case=use_case, deduplicate=True)
    assert pd.read_parquet(output)["value"].tolist() == list(range(10_000))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("features", "targets", "mapping", "error"),
    [
        ({"Feature": [1, 2]}, {"Target": [0]}, None, "aligned"),
        ({"FeatureName": [1], "Feature Name": [2]}, {"Target": [0]}, None, "unique"),
        ({"Feature": [1]}, {"TargetName": [0], "Target Name": [1]}, None, "unique"),
        ({"Label": [1]}, {"label": [0]}, None, "overlap"),
        ({"Feature": [1]}, {"Target": [0]}, {"unknown": {0: 1}}, "Unknown"),
        ({"Feature": [1, 2]}, {"Target": [0, 2]}, {"target": {0: 1}}, "unmapped"),
        ({"Feature": [1, 2]}, {"Target": [0, None]}, {"target": {0: 1}}, "missing"),
        ({"Feature": [1]}, {"Target": [0]}, {"target": {0: None}}, "scalar"),
    ],
)
def test_prepare_uci_rejects_invalid_snapshots_and_mappings(
    tmp_path: Path,
    features: dict,
    targets: dict,
    mapping: dict | None,
    error: str,
) -> None:
    _write_snapshot(tmp_path, pd.DataFrame(features), pd.DataFrame(targets))
    with pytest.raises(ValueError, match=error):
        prepare_uci_snapshot(
            tmp_path, tmp_path / "prepared.parquet", target_mappings=mapping
        )
    assert not (tmp_path / "prepared.parquet").exists()


@pytest.mark.unit
def test_uci_cli_fetches_then_prepares_with_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: list[int] = []
    dataset = SimpleNamespace(
        data=SimpleNamespace(
            features=pd.DataFrame({"Value": [1, 2]}),
            targets=pd.DataFrame({"Label": ["yes", "no"]}),
            ids=None,
        ),
        metadata={"uci_id": 967},
        variables=pd.DataFrame({"name": ["Value", "Label"]}),
    )

    def fake_fetch(*, id: int) -> SimpleNamespace:
        called.append(id)
        return dataset

    monkeypatch.setitem(
        sys.modules, "ucimlrepo", SimpleNamespace(fetch_ucirepo=fake_fetch)
    )
    snapshot = tmp_path / "snapshot"
    output = tmp_path / "prepared.parquet"
    mlkit.main(
        [
            "fetch-and-prepare-uci",
            "--dataset-id",
            "967",
            "--snapshot-dir",
            str(snapshot),
            "--output-path",
            str(output),
            "--map-target",
            "label",
            '"yes"',
            "1",
            "--map-target",
            "label",
            '"no"',
            "0",
        ],
    )
    assert called == [967]
    assert pd.read_parquet(snapshot / "targets.parquet")["Label"].tolist() == [
        "yes",
        "no",
    ]
    assert pd.read_parquet(output)["label"].tolist() == [1, 0]
    assert (snapshot / "metadata.json").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "values",
    [
        ["label", "0", "1", "--map-target", "label", "0", "0"],
        ["label", "not-json", "1"],
        ["label", "null", "1"],
    ],
)
def test_uci_cli_rejects_invalid_mappings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, values: list[str]
) -> None:
    fetch = pytest.fail
    monkeypatch.setattr(uci, "fetch_uci_snapshot", fetch)
    with pytest.raises(SystemExit) as exc:
        mlkit.main(
            [
                "fetch-and-prepare-uci",
                "--dataset-id",
                "967",
                "--snapshot-dir",
                str(tmp_path),
                "--output-path",
                str(tmp_path / "prepared.parquet"),
                "--map-target",
                *values,
            ],
        )
    assert exc.value.code == 2


@pytest.mark.unit
def test_uci_cli_resolves_use_case_before_fetch_and_preparation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import mlkit.use_cases

    use_case = SimpleNamespace(name="example")
    calls = []

    def resolve_use_case(name: str) -> SimpleNamespace:
        calls.append(("resolve", name))
        return use_case

    monkeypatch.setattr(mlkit.use_cases, "get_use_case", resolve_use_case)
    monkeypatch.setattr(
        uci,
        "fetch_uci_snapshot",
        lambda dataset_id, snapshot_dir: calls.append(
            ("fetch", dataset_id, snapshot_dir)
        ),
    )
    monkeypatch.setattr(
        uci,
        "prepare_uci_snapshot",
        lambda snapshot_dir, output_path, **kwargs: calls.append(
            ("prepare", snapshot_dir, output_path, kwargs)
        ),
    )
    output = tmp_path / "prepared.parquet"
    mlkit.main(
        [
            "fetch-and-prepare-uci",
            "--dataset-id",
            "42",
            "--snapshot-dir",
            str(tmp_path),
            "--output-path",
            str(output),
            "--use_case",
            "example",
            "--map-target",
            "label",
            "0",
            "1",
            "--deduplicate",
        ]
    )
    assert calls == [
        ("resolve", "example"),
        ("fetch", 42, tmp_path),
        (
            "prepare",
            tmp_path,
            output,
            {
                "target_mappings": {"label": {0: 1}},
                "use_case": use_case,
                "deduplicate": True,
            },
        ),
    ]


@pytest.mark.unit
def test_uci_cli_rejects_unknown_use_case_before_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(uci, "fetch_uci_snapshot", pytest.fail)
    with pytest.raises(ValueError, match="Unknown use case"):
        mlkit.main(
            [
                "fetch-and-prepare-uci",
                "--dataset-id",
                "42",
                "--snapshot-dir",
                str(tmp_path),
                "--output-path",
                str(tmp_path / "prepared.parquet"),
                "--use_case",
                "nonexistent",
            ]
        )


@pytest.mark.unit
def test_uci_cli_does_not_prepare_if_fetch_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_fetch(dataset_id: int, snapshot_dir: Path) -> None:
        raise RuntimeError("fetch failed")

    monkeypatch.setattr(uci, "fetch_uci_snapshot", fail_fetch)
    monkeypatch.setattr(uci, "prepare_uci_snapshot", pytest.fail)
    with pytest.raises(RuntimeError, match="fetch failed"):
        mlkit.main(
            [
                "fetch-and-prepare-uci",
                "--dataset-id",
                "42",
                "--snapshot-dir",
                str(tmp_path),
                "--output-path",
                str(tmp_path / "prepared.parquet"),
            ]
        )


@pytest.mark.unit
def test_uci_cli_propagates_preparation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_prepare(*args: object, **kwargs: object) -> None:
        raise ValueError("preparation failed")

    monkeypatch.setattr(uci, "fetch_uci_snapshot", lambda *args: None)
    monkeypatch.setattr(uci, "prepare_uci_snapshot", fail_prepare)
    with pytest.raises(ValueError, match="preparation failed"):
        mlkit.main(
            [
                "fetch-and-prepare-uci",
                "--dataset-id",
                "42",
                "--snapshot-dir",
                str(tmp_path),
                "--output-path",
                str(tmp_path / "prepared.parquet"),
            ]
        )

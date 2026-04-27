"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/datasets/views/test_sequence_views.py

Sequence-view sidecar tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.usr.src.contracts import SchemaError
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.registry import ensure_sequence_contract_namespaces
from dnadesign.usr.src.sequence_views import (
    SequenceViewRecord,
    compute_sequence_view_id,
    load_sequence_views,
    write_sequence_views,
)


def _make_dataset(root: Path, name: str, rows: list[dict[str, object]]) -> Dataset:
    ensure_sequence_contract_namespaces(root)
    dataset = Dataset(root, name)
    dataset.init(source="unit-test")
    dataset.import_rows(rows, source="unit-test")
    return dataset


def _first_id(dataset: Dataset) -> str:
    return str(dataset.head(1)["id"].tolist()[0])


def test_sequence_view_id_is_stable_under_human_label_changes(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(
        root,
        "native_refs",
        [{"sequence": "ACGT" * 20, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"}],
    )
    sequence_id = _first_id(dataset)

    first = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="spyP_native",
        aliases=["spyP"],
        product_kind="native_record",
        orientation="unknown",
        analysis_only=False,
        source_dataset_id=dataset.name,
        created_at="2026-04-25T00:00:00.000000Z",
    )
    second = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="spyP_native_renamed",
        aliases=["SpyP", "spyP_MG1655"],
        product_kind="native_record",
        orientation="unknown",
        analysis_only=False,
        source_dataset_id=dataset.name,
        created_at="2026-04-25T00:00:00.000000Z",
    )

    assert first.view_id == second.view_id
    assert first.view_id == compute_sequence_view_id(first.semantic_key())


def test_sequence_view_sidecar_supports_many_views_for_one_sequence_id(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(
        root,
        "reference_views",
        [{"sequence": "ACGT" * 20, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"}],
    )
    sequence_id = _first_id(dataset)

    native = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="spyP_native",
        aliases=["spyP"],
        product_kind="native_record",
        orientation="unknown",
        analysis_only=False,
        source_dataset_id=dataset.name,
        created_at="2026-04-25T00:00:00.000000Z",
    )
    core = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="spyP_core60",
        aliases=["spyP_core"],
        product_kind="analysis_core60",
        context_kind="analysis_core60",
        orientation="forward",
        analysis_only=True,
        source_dataset_id=dataset.name,
        anchor_start_0=0,
        anchor_end_0=60,
        recommended_pooling="core60_mean",
        created_at="2026-04-25T00:00:01.000000Z",
    )

    written = write_sequence_views(dataset, [native, core], conflict_policy="error")
    stored = load_sequence_views(dataset)

    assert written == 2
    assert len(stored) == 2
    assert {row.product_kind for row in stored} == {"native_record", "analysis_core60"}
    assert {row.sequence_id for row in stored} == {sequence_id}


def test_sequence_view_append_alias_merges_case_insensitive_aliases(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(
        root,
        "alias_views",
        [{"sequence": "ACGT" * 20, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"}],
    )
    sequence_id = _first_id(dataset)

    first = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="sulAp_native",
        aliases=["sulAp"],
        product_kind="native_record",
        orientation="unknown",
        analysis_only=False,
        source_dataset_id=dataset.name,
        created_at="2026-04-25T00:00:00.000000Z",
    )
    second = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="sulAp_native",
        aliases=["SulAp", "solA_cipro_control"],
        product_kind="native_record",
        orientation="unknown",
        analysis_only=False,
        source_dataset_id=dataset.name,
        created_at="2026-04-25T00:01:00.000000Z",
    )

    write_sequence_views(dataset, [first], conflict_policy="error")
    written = write_sequence_views(dataset, [second], conflict_policy="append_alias")
    stored = load_sequence_views(dataset)

    assert written == 1
    assert len(stored) == 1
    assert stored[0].aliases == ["sulAp", "solA_cipro_control"]


def test_sequence_view_append_alias_rejects_non_alias_metadata_changes(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(
        root,
        "alias_views",
        [{"sequence": "ACGT" * 20, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"}],
    )
    sequence_id = _first_id(dataset)

    first = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="sulAp_native",
        aliases=["sulAp"],
        product_kind="native_record",
        orientation="unknown",
        analysis_only=False,
        source_dataset_id=dataset.name,
        created_at="2026-04-25T00:00:00.000000Z",
    )
    relabeled = first.model_copy(update={"view_name": "sulAp_native_relabel", "aliases": ["solA_cipro_control"]})

    write_sequence_views(dataset, [first], conflict_policy="error")
    with pytest.raises(SchemaError, match="append_alias can only add human aliases"):
        write_sequence_views(dataset, [relabeled], conflict_policy="append_alias")


def test_sequence_view_source_interval_uses_parent_dataset_bounds(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    parent = _make_dataset(
        root,
        "usr_reference_genbank_native",
        [{"sequence": "ACGT" * 50, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"}],
    )
    child = _make_dataset(
        root,
        "construct_prom_eth_cip_reference_core60",
        [{"sequence": "ACGT" * 15, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"}],
    )
    parent_id = _first_id(parent)
    child_id = _first_id(child)

    valid = SequenceViewRecord(
        sequence_id=child_id,
        view_name="spyP_core60",
        product_kind="analysis_core60",
        context_kind="analysis_core60",
        orientation="forward",
        analysis_only=True,
        source_dataset_id=child.name,
        parent_sequence_id=parent_id,
        parent_dataset_id=parent.name,
        source_interval_start_0=80,
        source_interval_end_0=140,
        anchor_start_0=0,
        anchor_end_0=60,
        recommended_pooling="core60_mean",
        created_at="2026-04-25T00:00:00.000000Z",
    )
    invalid = valid.model_copy(update={"source_interval_end_0": 260})

    write_sequence_views(child, [valid], conflict_policy="error")
    with pytest.raises(SchemaError, match="source_interval bounds exceed"):
        write_sequence_views(child, [invalid], conflict_policy="replace")

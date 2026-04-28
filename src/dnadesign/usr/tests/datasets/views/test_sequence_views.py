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
    SequenceViewContractExpectation,
    SequenceViewRecord,
    ViewSemanticsRecord,
    compute_sequence_view_id,
    load_sequence_view_ids,
    load_sequence_view_index,
    load_sequence_views,
    load_view_semantics_index,
    validate_sequence_view_contract,
    write_sequence_views,
    write_view_semantics,
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
        product_kind="source_record",
        orientation="unknown",
        analysis_only=False,
        source_dataset_id=dataset.name,
        created_at="2026-04-25T00:00:00.000000Z",
    )
    second = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="spyP_native_renamed",
        aliases=["SpyP", "spyP_MG1655"],
        product_kind="source_record",
        orientation="unknown",
        analysis_only=False,
        source_dataset_id=dataset.name,
        created_at="2026-04-25T00:00:00.000000Z",
    )

    assert first.view_id == second.view_id
    assert first.view_id == compute_sequence_view_id(first.semantic_key())


def test_construct_insert_view_is_construct_ready_without_being_analysis_window(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(
        root,
        "usr_prom_eth_cip_anchor",
        [{"sequence": "ACGT" * 15, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"}],
    )
    sequence_id = _first_id(dataset)

    view = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="densegen_construct_insert",
        product_kind="construct_insert",
        context_kind="anchor_only",
        orientation="forward",
        analysis_only=False,
        source_dataset_id=dataset.name,
        recommended_pooling="seq_mean",
        created_at="2026-04-27T00:00:00.000000Z",
    )

    assert view.product_kind == "construct_insert"
    assert view.recommended_pooling == "seq_mean"
    assert view.analysis_only is False


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
        product_kind="source_record",
        orientation="unknown",
        analysis_only=False,
        source_dataset_id=dataset.name,
        created_at="2026-04-25T00:00:00.000000Z",
    )
    core = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="spyP_core60",
        aliases=["spyP_core"],
        product_kind="analysis_window",
        context_kind="analysis_window",
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
    assert {row.product_kind for row in stored} == {"source_record", "analysis_window"}
    assert {row.sequence_id for row in stored} == {sequence_id}
    assert load_sequence_view_ids(dataset) == {native.view_id, core.view_id}
    assert load_sequence_view_index(dataset)[core.view_id]["recommended_pooling"] == "core60_mean"


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
        product_kind="source_record",
        orientation="unknown",
        analysis_only=False,
        source_dataset_id=dataset.name,
        created_at="2026-04-25T00:00:00.000000Z",
    )
    second = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="sulAp_native",
        aliases=["SulAp", "solA_cipro_control"],
        product_kind="source_record",
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
        product_kind="source_record",
        orientation="unknown",
        analysis_only=False,
        source_dataset_id=dataset.name,
        created_at="2026-04-25T00:00:00.000000Z",
    )
    relabeled = first.model_copy(update={"view_name": "sulAp_native_relabel", "aliases": ["solA_cipro_control"]})

    write_sequence_views(dataset, [first], conflict_policy="error")
    with pytest.raises(SchemaError, match="append_alias can only add human aliases"):
        write_sequence_views(dataset, [relabeled], conflict_policy="append_alias")


def test_sequence_view_bulk_write_without_aliases_is_not_quadratic(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(
        root,
        "bulk_anchor_views",
        [
            {
                "sequence": ("A" * 20) + ("C" * index),
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "fixture",
            }
            for index in range(1, 201)
        ],
    )
    ids = [str(row_id) for row_id in dataset.head(200)["id"].tolist()]
    views = [
        SequenceViewRecord(
            sequence_id=sequence_id,
            view_name=f"anchor_{index}",
            aliases=None,
            product_kind="construct_insert",
            context_kind="anchor_only",
            orientation="forward",
            analysis_only=False,
            source_dataset_id=dataset.name,
            recommended_pooling="seq_mean",
            created_at="2026-04-27T00:00:00.000000Z",
        )
        for index, sequence_id in enumerate(ids)
    ]

    assert write_sequence_views(dataset, views, conflict_policy="idempotent") == 200
    assert len(load_sequence_views(dataset)) == 200


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
        product_kind="analysis_window",
        context_kind="analysis_window",
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


def test_view_semantics_sidecar_validates_referenced_view_id(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(
        root,
        "usr_prom_eth_cip_anchor",
        [{"sequence": "ACGT" * 15, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"}],
    )
    sequence_id = _first_id(dataset)
    view = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="densegen_construct_insert",
        product_kind="construct_insert",
        context_kind="anchor_only",
        orientation="forward",
        analysis_only=False,
        source_dataset_id=dataset.name,
        recommended_pooling="seq_mean",
        created_at="2026-04-27T00:00:00.000000Z",
    )
    write_sequence_views(dataset, [view], conflict_policy="error")

    with pytest.raises(SchemaError, match="missing view_id"):
        write_view_semantics(
            dataset,
            [
                ViewSemanticsRecord(
                    view_id="view_missing",
                    sequence_id=sequence_id,
                    source_family="densegen_generated",
                    selection_basis="native_source_length",
                    view_collections=["merged_anchor_handoff"],
                    role_tags=["design_population"],
                    study_id="stress_ethanol_cipro_growth",
                    created_at="2026-04-28T00:00:00.000000Z",
                    created_by="test",
                )
            ],
        )


def test_view_semantics_collection_updates_do_not_change_view_identity(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(
        root,
        "usr_prom_eth_cip_anchor",
        [{"sequence": "ACGT" * 15, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"}],
    )
    sequence_id = _first_id(dataset)
    view = SequenceViewRecord(
        sequence_id=sequence_id,
        view_name="reference_core60_merged_insert",
        product_kind="construct_insert",
        context_kind="anchor_only",
        orientation="forward",
        analysis_only=True,
        source_dataset_id=dataset.name,
        recommended_pooling="seq_mean",
        created_at="2026-04-27T00:00:00.000000Z",
    )
    write_sequence_views(dataset, [view], conflict_policy="error")
    original_view_id = str(view.view_id)

    first = ViewSemanticsRecord(
        view_id=original_view_id,
        sequence_id=sequence_id,
        source_family="construct_derived",
        selection_basis="sigma_site_pair_midpoint",
        view_collections=["merged_anchor_handoff"],
        role_tags=["comparability_view"],
        study_id="stress_ethanol_cipro_growth",
        created_at="2026-04-28T00:00:00.000000Z",
        created_by="test",
    )
    updated = first.model_copy(
        update={
            "view_collections": ["merged_anchor_handoff", "reference_core60_comparison"],
            "created_at": "2026-04-28T00:01:00.000000Z",
        }
    )

    assert write_view_semantics(dataset, [first], conflict_policy="error") == 1
    assert write_view_semantics(dataset, [updated], conflict_policy="replace") == 1

    assert load_sequence_views(dataset)[0].view_id == original_view_id
    semantics = load_view_semantics_index(dataset)[original_view_id]
    assert semantics["view_collections"] == ["merged_anchor_handoff", "reference_core60_comparison"]
    assert semantics["selection_basis"] == "sigma_site_pair_midpoint"


def test_sequence_view_contract_qa_enforces_counts_and_lengths(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(
        root,
        "construct_prom_eth_cip_reference_core60",
        [
            {"sequence": "ACGT" * 15, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"},
            {"sequence": "TGCA" * 15, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"},
        ],
    )
    ids = [str(row_id) for row_id in dataset.head(2)["id"].tolist()]
    write_sequence_views(
        dataset,
        [
            SequenceViewRecord(
                sequence_id=sequence_id,
                view_name=f"reference_core60_{index}",
                product_kind="analysis_window",
                context_kind="analysis_window",
                orientation="forward",
                analysis_only=True,
                source_dataset_id=dataset.name,
                anchor_start_0=0,
                anchor_end_0=60,
                recommended_pooling="core60_mean",
                created_at="2026-04-28T00:00:00.000000Z",
            )
            for index, sequence_id in enumerate(ids)
        ],
        conflict_policy="error",
    )

    report = validate_sequence_view_contract(
        dataset,
        expectation=SequenceViewContractExpectation(
            total_records=2,
            total_views=2,
            counts_by_product_kind={"analysis_window": 2},
            counts_by_orientation={"forward": 2},
            counts_by_recommended_pooling={"core60_mean": 2},
            exact_lengths_by_product_kind={"analysis_window": 60},
        ),
    )

    assert report.ok
    assert report.counts_by_product_kind == {"analysis_window": 2}


def test_sequence_view_contract_qa_fails_missing_reverse_complement_orientation(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = _make_dataset(
        root,
        "construct_prom_eth_cip_context",
        [{"sequence": "A" * 1000, "bio_type": "dna", "alphabet": "dna_4", "source": "fixture"}],
    )
    sequence_id = _first_id(dataset)
    write_sequence_views(
        dataset,
        [
            SequenceViewRecord(
                sequence_id=sequence_id,
                view_name="context_forward",
                product_kind="realized_context",
                context_kind="template_1kb",
                orientation="forward",
                analysis_only=False,
                source_dataset_id=dataset.name,
                anchor_start_0=470,
                anchor_end_0=530,
                recommended_pooling="anchor_mean",
                created_at="2026-04-28T00:00:00.000000Z",
            )
        ],
        conflict_policy="error",
    )

    with pytest.raises(SchemaError, match="orientation=reverse_complement"):
        validate_sequence_view_contract(
            dataset,
            expectation=SequenceViewContractExpectation(
                total_records=1,
                total_views=2,
                counts_by_orientation={"forward": 1, "reverse_complement": 1},
            ),
        )

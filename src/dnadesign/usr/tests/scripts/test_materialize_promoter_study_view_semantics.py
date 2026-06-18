"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/scripts/test_materialize_promoter_study_view_semantics.py

Regression tests for materialize promoter study view semantics USR scripts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dnadesign.usr import (
    Dataset,
    SequenceViewRecord,
    load_sequence_views,
    load_view_semantics,
    write_sequence_views,
)
from dnadesign.usr.scripts.materialize_promoter_study_view_semantics import (
    materialize_promoter_study_view_semantics,
)
from dnadesign.usr.src.contracts import compute_id


def _usr_root(tmp_path: Path) -> Path:
    usr_root = tmp_path / "usr_datasets"
    usr_root.mkdir()
    shutil.copy(Path("src/dnadesign/usr/datasets/registry.yaml"), usr_root / "registry.yaml")
    return usr_root


def _write_rows(dataset: Dataset, rows: list[dict[str, object]]) -> None:
    with dataset.write_session() as session:
        session.init(source="fixture", notes=f"{dataset.name} fixture")
        session.import_rows(rows)


def _view(
    *,
    sequence_id: str,
    product_kind: str,
    orientation: str = "forward",
    context_kind: str | None = None,
    analysis_only: bool = False,
    source_dataset_id: str,
    parent_dataset_id: str | None = None,
    parent_sequence_id: str | None = None,
    anchor_start_0: int | None = None,
    anchor_end_0: int | None = None,
    recommended_pooling: str | None = None,
) -> SequenceViewRecord:
    return SequenceViewRecord(
        sequence_id=sequence_id,
        view_name=f"{sequence_id[:8]}_{product_kind}_{orientation}",
        product_kind=product_kind,
        context_kind=context_kind,
        orientation=orientation,
        analysis_only=analysis_only,
        source_dataset_id=source_dataset_id,
        parent_dataset_id=parent_dataset_id,
        parent_sequence_id=parent_sequence_id,
        anchor_start_0=anchor_start_0,
        anchor_end_0=anchor_end_0,
        recommended_pooling=recommended_pooling,
        created_at="2026-04-28T00:00:00.000000Z",
        created_by="test",
    )


def test_materialize_promoter_study_view_semantics_writes_anchor_addenda(
    tmp_path: Path,
) -> None:
    usr_root = _usr_root(tmp_path)
    dataset = Dataset(usr_root, "usr_prom_eth_cip_anchor")
    densegen_sequence = "A" * 60
    sfxi_sequence = "C" * 60
    core_sequence = "G" * 60
    densegen_id = compute_id("dna", densegen_sequence)
    sfxi_id = compute_id("dna", sfxi_sequence)
    core_id = compute_id("dna", core_sequence)
    _write_rows(
        dataset,
        [
            {
                "id": densegen_id,
                "bio_type": "dna",
                "sequence": densegen_sequence,
                "alphabet": "dna_4",
                "length": 60,
                "source": "plan_pool__ethanol__sig35_f",
            },
            {
                "id": sfxi_id,
                "bio_type": "dna",
                "sequence": sfxi_sequence,
                "alphabet": "dna_4",
                "length": 60,
                "source": "archived/60bp_dual_promoter_cpxR_LexA;reader_sfxi_pdual",
            },
            {
                "id": core_id,
                "bio_type": "dna",
                "sequence": core_sequence,
                "alphabet": "dna_4",
                "length": 60,
                "source": "construct run construct_prom_eth_cip_reference_core60",
            },
        ],
    )
    write_sequence_views(
        dataset,
        [
            _view(
                sequence_id=densegen_id,
                product_kind="construct_insert",
                context_kind="anchor_only",
                source_dataset_id=dataset.name,
                recommended_pooling="seq_mean",
            ),
            _view(
                sequence_id=sfxi_id,
                product_kind="construct_insert",
                context_kind="anchor_only",
                source_dataset_id=dataset.name,
                recommended_pooling="seq_mean",
            ),
            _view(
                sequence_id=core_id,
                product_kind="construct_insert",
                context_kind="anchor_only",
                analysis_only=True,
                source_dataset_id=dataset.name,
                recommended_pooling="seq_mean",
            ),
        ],
        conflict_policy="error",
    )

    dry_run = materialize_promoter_study_view_semantics(usr_root=usr_root, dataset_names=[dataset.name], write=False)

    assert dry_run.datasets == [dataset.name]
    assert dry_run.semantics_planned == 3
    assert dry_run.semantics_written == 0
    assert dry_run.by_source_family == {
        "construct_derived": 1,
        "densegen_generated": 1,
        "sfxi_archive": 1,
    }

    result = materialize_promoter_study_view_semantics(usr_root=usr_root, dataset_names=[dataset.name], write=True)

    assert result.semantics_written == 3
    semantics = {row.sequence_id: row for row in load_view_semantics(dataset)}
    assert semantics[densegen_id].source_family == "densegen_generated"
    assert semantics[densegen_id].selection_basis == "densegen_selected_insert"
    assert semantics[densegen_id].view_collections == ["merged_anchor_handoff"]
    assert semantics[sfxi_id].source_family == "sfxi_archive"
    assert semantics[sfxi_id].view_collections == ["merged_anchor_handoff", "sfxi_archive_handoff"]
    assert semantics[core_id].source_family == "construct_derived"
    assert semantics[core_id].selection_basis == "sigma_site_pair_midpoint"
    assert semantics[core_id].role_tags == ["construct_ready_insert", "comparability_view"]

    rerun = materialize_promoter_study_view_semantics(usr_root=usr_root, dataset_names=[dataset.name], write=True)

    assert rerun.semantics_written == 0
    assert len(load_view_semantics(dataset)) == 3
    assert [view.view_id for view in load_sequence_views(dataset)]


def test_materialize_promoter_study_view_semantics_writes_orientation_collections(
    tmp_path: Path,
) -> None:
    usr_root = _usr_root(tmp_path)
    dataset = Dataset(usr_root, "construct_prom_eth_cip_context")
    forward_sequence = "A" * 1000
    reverse_sequence = "T" * 1000
    forward_id = compute_id("dna", forward_sequence)
    reverse_id = compute_id("dna", reverse_sequence)
    _write_rows(
        dataset,
        [
            {
                "id": forward_id,
                "bio_type": "dna",
                "sequence": forward_sequence,
                "alphabet": "dna_4",
                "length": 1000,
                "source": "construct",
            },
            {
                "id": reverse_id,
                "bio_type": "dna",
                "sequence": reverse_sequence,
                "alphabet": "dna_4",
                "length": 1000,
                "source": "construct",
            },
        ],
    )
    write_sequence_views(
        dataset,
        [
            _view(
                sequence_id=forward_id,
                product_kind="realized_context",
                orientation="forward",
                context_kind="template_1kb",
                source_dataset_id=dataset.name,
                anchor_start_0=470,
                anchor_end_0=530,
                recommended_pooling="anchor_mean",
            ),
            _view(
                sequence_id=reverse_id,
                product_kind="realized_context",
                orientation="reverse_complement",
                context_kind="template_1kb",
                source_dataset_id=dataset.name,
                anchor_start_0=470,
                anchor_end_0=530,
                recommended_pooling="anchor_mean",
            ),
        ],
        conflict_policy="error",
    )

    result = materialize_promoter_study_view_semantics(usr_root=usr_root, dataset_names=[dataset.name], write=True)

    assert result.semantics_written == 2
    semantics = {row.sequence_id: row for row in load_view_semantics(dataset)}
    assert semantics[forward_id].selection_basis == "template_window_center"
    assert semantics[forward_id].view_collections == ["realized_context_forward_all", "merged_context_handoff"]
    assert semantics[reverse_id].selection_basis == "whole_output_reverse_complement"
    assert semantics[reverse_id].view_collections == [
        "realized_context_reverse_complement_all",
        "merged_context_handoff",
    ]
    assert semantics[reverse_id].role_tags == ["context_exposure", "orientation_sensitivity"]


def test_materialize_promoter_study_view_semantics_fails_unknown_dataset(tmp_path: Path) -> None:
    usr_root = _usr_root(tmp_path)

    with pytest.raises(ValueError, match="Unsupported promoter-study semantics dataset"):
        materialize_promoter_study_view_semantics(usr_root=usr_root, dataset_names=["unexpected_dataset"], write=False)

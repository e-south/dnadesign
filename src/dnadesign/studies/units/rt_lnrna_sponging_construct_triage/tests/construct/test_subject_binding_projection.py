"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/construct/test_subject_binding_projection.py

Exact subject-binding traversal into RT-lnRNA Construct views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.construct_materialization import (
    MaterializationContractError,
    materialize_unified_construct_subject_contexts,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.materialization.subjects import (
    _construct_subject_envelope_overlay,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.materialization.usr_io import (
    _write_construct_subject_dataset,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.source_promotions import (
    SourcePromotionContractError,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.subject_bindings import (
    SubjectBindingContractError,
    load_resolved_registered_subject_bindings,
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_resolved_binding_bytes_fail_closed_for_opaque_provider_parts() -> None:
    with pytest.raises(
        SubjectBindingContractError,
        match="RT CDS bytes are not published.*provider:eco1_rt_repack/rt-parts/Eco1RT-G3-D01",
    ):
        load_resolved_registered_subject_bindings(repo_root=_repo_root())


def test_nonsequence_subject_envelope_identity_is_stable_across_row_order(tmp_path: Path) -> None:
    rows = [
        {
            "id": subject_id,
            **_construct_subject_envelope_overlay(),
            "construct_subject__lnrna_sequence": lnrna,
            "construct_subject__rt_cds_sequence": rt_cds,
        }
        for subject_id, lnrna, rt_cds in (
            ("subject-a", "ACGT", "GGCC"),
            ("subject-b", "TGCA", "CCGG"),
        )
    ]

    forward = _write_construct_subject_dataset(usr_root=tmp_path / "forward", rows=rows)
    reverse = _write_construct_subject_dataset(usr_root=tmp_path / "reverse", rows=list(reversed(rows)))

    assert forward == reverse


def test_general_materialization_reports_opaque_provider_parts_and_materializes_resolvable_subjects(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "work"
    report = materialize_unified_construct_subject_contexts(
        repo_root=_repo_root(),
        work_root=work_root,
        allow_partial_byte_resolution=True,
        include_source_promotions=False,
        include_rt_cds_dms=False,
    )

    assert report.subject_binding_requested_subject_count == 49
    assert report.subject_binding_resolved_subject_count == 46
    assert report.subject_binding_blocked_subject_count == 3
    assert report.subject_binding_resolution_complete is False
    assert {block.subject_id for block in report.blocked_subject_bindings} == {
        "rt_lnrna_pair__eco1rt_g3_d01__retron26_lnrna__tetO",
        "rt_lnrna_pair__eco1rt_g3_d02__retron26_lnrna__tetO",
        "rt_lnrna_pair__eco1rt_g3_d01__retron180_lnrna__tetO",
    }
    assert all(block.provider_ref.startswith("provider:eco1_rt_repack/") for block in report.blocked_subject_bindings)
    assert {block.reason for block in report.blocked_subject_bindings} == {"provider_publication_omits_rt_cds_bytes"}
    assert not set(report.input_ids_by_subject_id) & {block.subject_id for block in report.blocked_subject_bindings}


def test_general_partial_materialization_requires_explicit_opt_in_before_writes(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    with pytest.raises(MaterializationContractError, match="allow_partial_byte_resolution=True"):
        materialize_unified_construct_subject_contexts(
            repo_root=_repo_root(),
            work_root=work_root,
        )
    assert not work_root.exists()


def test_exact_blocked_subject_projection_fails_before_writes(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    with pytest.raises(MaterializationContractError, match="exact subject projection is byte-blocked"):
        materialize_unified_construct_subject_contexts(
            repo_root=_repo_root(),
            work_root=work_root,
            subject_binding_subject_ids=("rt_lnrna_pair__eco1rt_g3_d01__retron26_lnrna__tetO",),
        )
    assert not work_root.exists()


def test_exact_resolvable_subject_reports_complete_resolution(tmp_path: Path) -> None:
    report = materialize_unified_construct_subject_contexts(
        repo_root=_repo_root(),
        work_root=tmp_path / "work",
        subject_binding_subject_ids=("rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO",),
    )

    assert report.subject_binding_requested_subject_count == 1
    assert report.subject_binding_resolved_subject_count == 1
    assert report.subject_binding_blocked_subject_count == 0
    assert report.subject_binding_resolution_complete is True
    assert report.blocked_subject_bindings == ()


def test_optional_promotions_are_opt_in_and_operational_recipe_is_explicit() -> None:
    parameters = inspect.signature(materialize_unified_construct_subject_contexts).parameters
    for name in (
        "include_source_promotions",
        "include_msd_compiler_promotions",
        "include_rt_cds_dms",
    ):
        assert parameters[name].default is False

    pipeline_path = (
        _repo_root() / "docs/studies/rt_lnrna_sponging_construct_triage/operations/runtime/command-groups/pipeline.yaml"
    )
    payload = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    group = next(item for item in payload["command_groups"] if item["id"] == "construct_projection")
    sequence = group["operational_sequence"]
    assert [item["id"] for item in sequence] == [
        "hairpin_compiler_primitive_preflight",
        "unified_construct_materialization",
    ]
    assert sequence[1]["kwargs"] == {
        "allow_partial_byte_resolution": True,
        "include_source_promotions": True,
        "include_msd_compiler_promotions": True,
        "include_rt_cds_dms": True,
    }


def test_explicit_compiler_opt_in_fails_before_writes_when_pool_inputs_are_missing(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    with pytest.raises(SourcePromotionContractError, match="MSD compiler pool spec is missing"):
        materialize_unified_construct_subject_contexts(
            repo_root=_repo_root(),
            work_root=work_root,
            allow_partial_byte_resolution=True,
            include_source_promotions=False,
            include_msd_compiler_promotions=True,
            include_rt_cds_dms=False,
            msd_variant_pool_spec_paths=(tmp_path / "missing-pool.yaml",),
        )
    assert not work_root.exists()

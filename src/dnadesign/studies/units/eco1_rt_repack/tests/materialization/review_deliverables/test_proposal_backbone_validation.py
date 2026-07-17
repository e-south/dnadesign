"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_proposal_backbone_validation.py

Proposal-scene and structure-set validation contracts for Eco1 RT review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals.proposal_backbone_cycle import (  # noqa: E501
    build_proposal_backbone_scenes,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals.structure_set import (  # noqa: E501
    read_foldcheck_structure_set,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.foldcheck_structure_set_fixtures import (  # noqa: E501
    write_foldcheck_full_structure_set,
)


def _triage_row(
    *,
    candidate_id: str,
    policy_id: str,
    retained: bool,
    mutation_count: int,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "policy_id": policy_id,
        "local_structure_gate_status": "passed" if retained else "threshold_exceeded",
        "mutation_count_total": mutation_count,
        "sequence_distance_to_wt": mutation_count,
    }


def test_proposal_scenes_include_only_local_geometry_retained_candidates(tmp_path: Path) -> None:
    foldcheck_root = tmp_path / "foldcheck_review"
    write_foldcheck_full_structure_set(foldcheck_root)
    structure_set = read_foldcheck_structure_set(foldcheck_root / "foldcheck_full_structure_set.yaml")
    rows = [
        _triage_row(
            candidate_id="thread_candidate_beta",
            policy_id=NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
            retained=False,
            mutation_count=3,
        ),
        _triage_row(
            candidate_id="thread_candidate_alpha",
            policy_id=DISTAL_SCAFFOLD_POLICY_ID,
            retained=True,
            mutation_count=2,
        ),
    ]

    scenes = build_proposal_backbone_scenes(triage_rows=rows, structure_set=structure_set)

    assert [scene.candidate_id for scene in scenes] == ["thread_candidate_alpha"]
    assert [scene.chapter_label for scene in scenes] == ["Distal redesign"]
    assert [scene.chapter_position for scene in scenes] == [1]
    assert [scene.chapter_size for scene in scenes] == [1]
    assert [scene.mutation_count for scene in scenes] == [2]
    assert [scene.wt_sequence_identity_percent for scene in scenes] == [99.375]


def test_proposal_scene_filter_is_independent_of_panel_eligibility(tmp_path: Path) -> None:
    foldcheck_root = tmp_path / "foldcheck_review"
    write_foldcheck_full_structure_set(foldcheck_root)
    structure_set = read_foldcheck_structure_set(foldcheck_root / "foldcheck_full_structure_set.yaml")
    rows = [
        {
            **_triage_row(
                candidate_id="thread_candidate_alpha",
                policy_id=DISTAL_SCAFFOLD_POLICY_ID,
                retained=True,
                mutation_count=2,
            ),
            "selection_contract_pass": False,
        },
        _triage_row(
            candidate_id="thread_candidate_beta",
            policy_id=NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
            retained=False,
            mutation_count=3,
        ),
    ]

    scenes = build_proposal_backbone_scenes(triage_rows=rows, structure_set=structure_set)

    assert [scene.candidate_id for scene in scenes] == ["thread_candidate_alpha"]


def test_structure_set_rejects_duplicate_candidate_ids(tmp_path: Path) -> None:
    foldcheck_root = tmp_path / "foldcheck_review"
    write_foldcheck_full_structure_set(foldcheck_root)
    manifest_path = foldcheck_root / "foldcheck_full_structure_set.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["structures"].append(dict(payload["structures"][1]))
    payload["structure_count"] += 1
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate candidate_id.*thread_candidate_alpha"):
        read_foldcheck_structure_set(manifest_path)


def test_structure_set_rejects_missing_model_file(tmp_path: Path) -> None:
    foldcheck_root = tmp_path / "foldcheck_review"
    write_foldcheck_full_structure_set(foldcheck_root)
    missing_path = foldcheck_root / "structures" / "full_fold_set" / "thread_candidate_beta.pdb"
    missing_path.unlink()

    with pytest.raises(FileNotFoundError, match="thread_candidate_beta"):
        read_foldcheck_structure_set(foldcheck_root / "foldcheck_full_structure_set.yaml")


def test_structure_set_rejects_model_content_that_does_not_match_its_digest(tmp_path: Path) -> None:
    foldcheck_root = tmp_path / "foldcheck_review"
    write_foldcheck_full_structure_set(foldcheck_root)
    model_path = foldcheck_root / "structures" / "full_fold_set" / "thread_candidate_beta.pdb"
    model_path.write_text(model_path.read_text(encoding="utf-8") + "REMARK changed after manifest\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch for thread_candidate_beta"):
        read_foldcheck_structure_set(foldcheck_root / "foldcheck_full_structure_set.yaml")


def test_proposal_scenes_reject_unmatched_triage_and_structure_sets(tmp_path: Path) -> None:
    foldcheck_root = tmp_path / "foldcheck_review"
    write_foldcheck_full_structure_set(foldcheck_root)
    structure_set = read_foldcheck_structure_set(foldcheck_root / "foldcheck_full_structure_set.yaml")
    rows = [
        _triage_row(
            candidate_id="thread_candidate_alpha",
            policy_id=DISTAL_SCAFFOLD_POLICY_ID,
            retained=True,
            mutation_count=2,
        )
    ]

    with pytest.raises(ValueError, match="triage and structure-set candidate IDs differ"):
        build_proposal_backbone_scenes(triage_rows=rows, structure_set=structure_set)


def test_proposal_scenes_reject_identity_that_disagrees_with_mutation_count(tmp_path: Path) -> None:
    foldcheck_root = tmp_path / "foldcheck_review"
    write_foldcheck_full_structure_set(foldcheck_root)
    structure_set = read_foldcheck_structure_set(foldcheck_root / "foldcheck_full_structure_set.yaml")
    rows = [
        _triage_row(
            candidate_id="thread_candidate_alpha",
            policy_id=DISTAL_SCAFFOLD_POLICY_ID,
            retained=True,
            mutation_count=4,
        ),
        _triage_row(
            candidate_id="thread_candidate_beta",
            policy_id=NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
            retained=False,
            mutation_count=3,
        ),
    ]

    with pytest.raises(ValueError, match="WT identity and mutation count disagree.*thread_candidate_alpha"):
        build_proposal_backbone_scenes(triage_rows=rows, structure_set=structure_set)

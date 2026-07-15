"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_proposal_backbone_cycle.py

ProteinMPNN proposal-backbone movie contracts for Eco1 RT review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.communication_visuals.proposal_backbone_cycle import (  # noqa: E501
    PROPOSAL_RAW_FRAMES_PER_FULL_ROTATION,
    build_proposal_backbone_scenes,
    proposal_backbone_raw_frame_count,
    write_proposal_backbone_cycle_script,
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


def test_proposal_script_streams_one_centered_rotating_retained_model(tmp_path: Path) -> None:
    foldcheck_root = tmp_path / "foldcheck_review"
    write_foldcheck_full_structure_set(foldcheck_root)
    structure_set = read_foldcheck_structure_set(foldcheck_root / "foldcheck_full_structure_set.yaml")
    rows = [
        _triage_row(
            candidate_id="thread_candidate_alpha",
            policy_id=COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
            retained=True,
            mutation_count=2,
        ),
        _triage_row(
            candidate_id="thread_candidate_beta",
            policy_id=NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
            retained=False,
            mutation_count=3,
        ),
    ]
    scenes = build_proposal_backbone_scenes(triage_rows=rows, structure_set=structure_set)
    script_path = tmp_path / "proposal_cycle.cxc"

    write_proposal_backbone_cycle_script(
        script_path=script_path,
        reference_backbone_path=foldcheck_root / "structures" / "ec86kit_chain_a_backbone_reference.pdb",
        scenes=scenes,
        frame_directory=tmp_path / "frames",
    )

    script = script_path.read_text(encoding="utf-8")
    assert script.count("\nopen ") == 2
    assert "id #1000 name cryoem_reference" in script
    assert script.count("id #1001 name retained_model") == 1
    assert "outside_cutoff" not in script
    assert "#2000" not in script
    assert script.count("\nclose #1001") == 1
    assert script.count("align #1001/A:3-311@CA toAtoms #1000/A:1-309@CA") == 1
    assert "cutoffDistance" in script
    assert "cutoffDistance none" not in script
    assert "matchmaker" not in script.lower()
    assert "surface" not in script.lower()
    assert "/D" not in script and "/E" not in script and "/F" not in script
    assert script.count("hide #1001/A:1-2,312-320 cartoons") == 1
    assert "cartoon #1000/A:1-309" in script
    assert "cartoon #1001/A:3-311" in script
    assert "color #1001/A:3-311 #4F6270 target c" in script
    assert "show (#1001/A:3-311 & sidechain) atoms" in script
    assert "style (#1001/A:3-311 & sidechain) stick" in script
    assert "size (#1001/A:3-311 & sidechain) stickRadius 0.08" in script
    assert "color (#1001/A:3-311 & sidechain) #4F6270 target a" in script
    assert "show (#1000/A:1-309 & sidechain) atoms" not in script
    assert "view #1000 pad 0.08" in script
    assert "zoom 1.05" in script
    assert "move x" not in script
    assert "turn y 180.000000 models #1000 center #1000" in script
    assert "move y -4.0 models #1000" in script
    exact_turn_step = 360.0 / (PROPOSAL_RAW_FRAMES_PER_FULL_ROTATION - 1)
    retained_turn = f"turn y {exact_turn_step:.6f} models #1000,1001 center #1000"
    expected_turns_per_chapter = PROPOSAL_RAW_FRAMES_PER_FULL_ROTATION - 1
    assert script.count(retained_turn) == expected_turns_per_chapter
    assert "Outside local-geometry cutoff" not in script
    assert "Combined redesign | 1 model" in script
    assert script.count("frame-") == proposal_backbone_raw_frame_count(scenes)
    assert "Model 1/1" in script
    assert "WT identity 99.4% | 2 substitutions" in script
    assert "ypos 0.900 size 18" in script
    assert script.count("Local geometry retained") == 0
    assert '2dlabels text "Cryo-EM reference"' in script
    assert '2dlabels text "ColabFold model"' not in script
    assert '2dlabels text "thread_candidate_' not in script


def test_proposal_chapter_distributes_rendered_frames_across_models(tmp_path: Path) -> None:
    foldcheck_root = tmp_path / "foldcheck_review"
    write_foldcheck_full_structure_set(foldcheck_root)
    structure_set = read_foldcheck_structure_set(foldcheck_root / "foldcheck_full_structure_set.yaml")
    rows = [
        _triage_row(
            candidate_id="thread_candidate_alpha",
            policy_id=DISTAL_SCAFFOLD_POLICY_ID,
            retained=True,
            mutation_count=2,
        ),
        _triage_row(
            candidate_id="thread_candidate_beta",
            policy_id=DISTAL_SCAFFOLD_POLICY_ID,
            retained=True,
            mutation_count=3,
        ),
    ]
    scenes = build_proposal_backbone_scenes(triage_rows=rows, structure_set=structure_set)
    script_path = tmp_path / "proposal_cycle.cxc"

    write_proposal_backbone_cycle_script(
        script_path=script_path,
        reference_backbone_path=foldcheck_root / "structures" / "ec86kit_chain_a_backbone_reference.pdb",
        scenes=scenes,
        frame_directory=tmp_path / "frames",
    )

    script = script_path.read_text(encoding="utf-8")
    alpha_frames = script.count("WT identity 99.4% | 2 substitutions")
    beta_frames = script.count("WT identity 99.1% | 3 substitutions")
    assert alpha_frames > 0
    assert abs(alpha_frames - beta_frames) <= 1


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

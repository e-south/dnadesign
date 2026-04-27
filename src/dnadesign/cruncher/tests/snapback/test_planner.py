"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/snapback/test_planner.py

Planner tests for v2 explicit snapback workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.cruncher.app.snapback_workflow import validate_snapback_spec


def _write_workspace(
    tmp_path: Path,
    *,
    spec_payload: dict[str, object],
    catalog_entries: list[dict[str, object]],
) -> Path:
    workspace = tmp_path / "workspaces" / "demo_snapback"
    spec_path = workspace / "configs" / "snapback" / "demo.snapback.yaml"
    catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        yaml.safe_dump({"nickases": {"schema_version": 1, "entries": catalog_entries}}, sort_keys=False),
        encoding="utf-8",
    )
    spec_path.write_text(yaml.safe_dump(spec_payload, sort_keys=False), encoding="utf-8")
    return spec_path


def _base_payload() -> dict[str, object]:
    return {
        "snapback": {
            "schema_version": 2,
            "contract": "single_nick_snapback_v2",
            "name": "demo_snapback",
        },
        "input": {
            "canonical_top_strand": {
                "sequence": "CCTCAGCA",
                "protected_region": {"start": 0, "end": 8},
                "pre_nick_duplex_window": {"start": 0, "end": 8},
            }
        },
        "design": {
            "nickase": {
                "variant_id": "Nt.Bpu10I",
                "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
            },
            "orientation_policy": {
                "normalize_to_top_strand_nick": True,
                "release_direction": "left_to_right_from_nick",
            },
            "single_nick_goal": {"nick_boundary_window": {"min": 2, "max": 2}},
            "topology": {
                "retained_homology_window": {"start": 2, "end": 6},
                "cap_sequence": "T",
                "foldback_arm": "CTGA",
                "homology_policy": {"max_mismatches": 0, "min_paired_bp": 4, "max_paired_bp": 4},
            },
            "constraints": {
                "terminal_ligatable_duplex_bp": {"min": 4, "max": 4},
                "max_uninterrupted_duplex_bp": 4,
                "max_added_nt": 5,
                "forbid_additional_target_strand_nicks": False,
                "forbid_any_additional_nicks": False,
            },
            "sequence_quality": {
                "gc_fraction": {"min": 0.25, "max": 0.75},
                "max_homopolymer_run": 2,
            },
        },
        "output": {"run_dir": "outputs/design", "emit_visual_contracts": True},
    }


def _catalog_entries() -> list[dict[str, object]]:
    return [
        {
            "id": "Nt.Bpu10I",
            "specificity_id": "Bpu10I",
            "motif_top_5to3": "CCTNAGC",
            "top_cut_offset": 2,
            "source": "demo",
        }
    ]


def test_validate_snapback_spec_returns_satisfied_v2_candidate(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path, spec_payload=_base_payload(), catalog_entries=_catalog_entries())

    report = validate_snapback_spec(spec_path)

    assert report.status == "satisfied"
    assert report.candidate is not None
    assert report.candidate.designed_sequence == "CCTCAGCATCTGA"
    assert report.candidate.nick_boundary == 2
    assert report.candidate.nick_boundary_from_left == 2
    assert report.candidate.released_prefix_nt == 0
    assert report.candidate.retained_start_from_nick == 0
    assert report.candidate.source_cap_sequence == "CA"
    assert report.candidate.effective_cap_sequence == "CAT"
    assert report.candidate.cap_nt == 3
    assert report.candidate.cap_extension_nt == 1
    assert report.candidate.paired_bp == 4
    assert report.candidate.mismatch_count == 0
    assert report.candidate.terminal_ligatable_duplex_bp == 4
    assert report.candidate.max_uninterrupted_duplex_bp == 4
    assert report.candidate.extra_nick_event_count == 0


def test_validate_snapback_spec_allows_one_internal_mismatch_when_policy_permits_it(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 0, "end": 2}
    payload["design"]["topology"]["foldback_arm"] = "CTAA"
    payload["design"]["topology"]["homology_policy"]["max_mismatches"] = 1
    payload["design"]["constraints"]["terminal_ligatable_duplex_bp"] = {"min": 1, "max": 4}
    payload["design"]["constraints"]["max_uninterrupted_duplex_bp"] = 2
    payload["design"]["sequence_quality"]["gc_fraction"] = {"min": 0.0, "max": 0.75}
    spec_path = _write_workspace(tmp_path, spec_payload=payload, catalog_entries=_catalog_entries())

    report = validate_snapback_spec(spec_path)

    assert report.status == "satisfied"
    assert report.candidate is not None
    assert report.candidate.mismatch_count == 1
    assert report.candidate.mismatch_positions == [1]
    assert report.candidate.terminal_ligatable_duplex_bp == 1
    assert report.candidate.max_uninterrupted_duplex_bp == 2


def test_validate_snapback_spec_reports_unsatisfied_when_retained_homology_starts_before_nick(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["design"]["topology"]["retained_homology_window"] = {"start": 1, "end": 5}
    spec_path = _write_workspace(tmp_path, spec_payload=payload, catalog_entries=_catalog_entries())

    report = validate_snapback_spec(spec_path)

    assert report.status == "unsatisfied"
    assert report.candidate is None
    assert [issue.code for issue in report.issues] == ["RETAINED_HOMOLOGY_MUST_START_AT_NICK"]


def test_validate_snapback_spec_supports_boundary_zero(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["input"]["canonical_top_strand"]["sequence"] = "ATGACGT"
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 0, "end": 7}
    payload["input"]["canonical_top_strand"]["pre_nick_duplex_window"] = {"start": 0, "end": 7}
    payload["design"]["nickase"]["variant_id"] = "Nt.Zero"
    payload["design"]["single_nick_goal"]["nick_boundary_window"] = {"min": 0, "max": 0}
    payload["design"]["topology"]["retained_homology_window"] = {"start": 0, "end": 4}
    payload["design"]["topology"]["cap_sequence"] = ""
    payload["design"]["topology"]["foldback_arm"] = "TCAT"
    payload["design"]["constraints"]["terminal_ligatable_duplex_bp"] = {"min": 4, "max": 4}
    payload["design"]["constraints"]["max_added_nt"] = 4
    payload["design"]["sequence_quality"]["gc_fraction"] = {"min": 0.0, "max": 0.75}
    payload["design"]["sequence_quality"]["max_homopolymer_run"] = 2
    spec_path = _write_workspace(
        tmp_path,
        spec_payload=payload,
        catalog_entries=[{"id": "Nt.Zero", "specificity_id": "Zero", "motif_top_5to3": "ATG", "top_cut_offset": 0}],
    )

    report = validate_snapback_spec(spec_path)

    assert report.status == "satisfied"
    assert report.candidate is not None
    assert report.candidate.nick_boundary == 0
    assert report.candidate.released_prefix_nt == 0
    assert report.candidate.retained_start_from_nick == 0


def test_validate_snapback_spec_normalizes_reverse_orientation_to_top_strand(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["input"]["canonical_top_strand"]["sequence"] = "GCTGAGGATTA"
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 0, "end": 11}
    payload["input"]["canonical_top_strand"]["pre_nick_duplex_window"] = {"start": 0, "end": 11}
    payload["design"]["nickase"]["variant_id"] = "Nb.TopNorm"
    payload["design"]["single_nick_goal"]["nick_boundary_window"] = {"min": 7, "max": 7}
    payload["design"]["topology"]["retained_homology_window"] = {"start": 7, "end": 11}
    payload["design"]["topology"]["cap_sequence"] = "ATT"
    payload["design"]["topology"]["foldback_arm"] = "TAAT"
    payload["design"]["constraints"]["max_added_nt"] = 7
    payload["design"]["sequence_quality"]["gc_fraction"] = {"min": 0.0, "max": 0.75}
    payload["design"]["sequence_quality"]["max_homopolymer_run"] = 4
    spec_path = _write_workspace(
        tmp_path,
        spec_payload=payload,
        catalog_entries=[
            {"id": "Nb.TopNorm", "specificity_id": "TopNorm", "motif_top_5to3": "CCTCAGC", "bottom_cut_offset": 0}
        ],
    )

    report = validate_snapback_spec(spec_path)

    assert report.status == "satisfied"
    assert report.candidate is not None
    assert report.candidate.intended_site.orientation == "reverse"
    assert report.candidate.intended_nick.strand == "primary"


def test_validate_snapback_spec_can_forbid_any_additional_nicks(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["input"]["canonical_top_strand"]["sequence"] = "CCTCAGCCCTCAGCAG"
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 0, "end": 16}
    payload["input"]["canonical_top_strand"]["pre_nick_duplex_window"] = {"start": 0, "end": 16}
    payload["design"]["single_nick_goal"]["nick_boundary_window"] = {"min": 9, "max": 9}
    payload["design"]["topology"]["retained_homology_window"] = {"start": 9, "end": 13}
    payload["design"]["topology"]["cap_sequence"] = ""
    payload["design"]["topology"]["foldback_arm"] = "CTGA"
    payload["design"]["constraints"]["max_added_nt"] = 4
    payload["design"]["constraints"]["forbid_any_additional_nicks"] = True
    spec_path = _write_workspace(tmp_path, spec_payload=payload, catalog_entries=_catalog_entries())

    report = validate_snapback_spec(spec_path)

    assert report.status == "unsatisfied"
    assert report.candidate is None
    assert [issue.code for issue in report.issues] == ["EXTRA_NICKS_FOUND"]


def test_validate_snapback_spec_can_forbid_additional_target_strand_nicks(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["input"]["canonical_top_strand"]["sequence"] = "CCTCAGCCCTCAGCAG"
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 0, "end": 16}
    payload["input"]["canonical_top_strand"]["pre_nick_duplex_window"] = {"start": 0, "end": 16}
    payload["design"]["single_nick_goal"]["nick_boundary_window"] = {"min": 9, "max": 9}
    payload["design"]["topology"]["retained_homology_window"] = {"start": 9, "end": 13}
    payload["design"]["topology"]["cap_sequence"] = ""
    payload["design"]["topology"]["foldback_arm"] = "CTGA"
    payload["design"]["constraints"]["max_added_nt"] = 4
    payload["design"]["constraints"]["forbid_additional_target_strand_nicks"] = True
    spec_path = _write_workspace(tmp_path, spec_payload=payload, catalog_entries=_catalog_entries())

    report = validate_snapback_spec(spec_path)

    assert report.status == "unsatisfied"
    assert report.candidate is None
    assert [issue.code for issue in report.issues] == ["EXTRA_TARGET_STRAND_NICKS_FOUND"]


def test_validate_snapback_spec_rejects_mismatch_inside_protected_overlap(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 3, "end": 5}
    payload["design"]["topology"]["foldback_arm"] = "CTAA"
    payload["design"]["topology"]["homology_policy"]["max_mismatches"] = 1
    payload["design"]["constraints"]["terminal_ligatable_duplex_bp"] = {"min": 1, "max": 4}
    payload["design"]["constraints"]["max_uninterrupted_duplex_bp"] = 2
    payload["design"]["sequence_quality"]["gc_fraction"] = {"min": 0.0, "max": 0.75}
    spec_path = _write_workspace(tmp_path, spec_payload=payload, catalog_entries=_catalog_entries())

    report = validate_snapback_spec(spec_path)

    assert report.status == "unsatisfied"
    assert report.candidate is None
    assert [issue.code for issue in report.issues] == ["PROTECTED_REGION_MISMATCH_OVERLAP"]

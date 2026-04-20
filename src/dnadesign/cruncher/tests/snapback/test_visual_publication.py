"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/snapback/test_visual_publication.py

Visual publication tests for snapback QA and public snapback visual artifacts.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.contracts.visual import SnapbackVisualV1
from dnadesign.cruncher.app.snapback_workflow import validate_snapback_spec
from dnadesign.cruncher.snapback.view_contracts import (
    build_post_nick_exposed_snapback_visual,
    build_post_nick_exposed_view,
    build_post_nick_foldback_snapback_visual,
    build_post_nick_foldback_view,
    build_pre_nick_duplex_view,
    build_pre_nick_snapback_visual,
)


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


def _base_payload() -> dict[str, object]:
    return {
        "snapback": {
            "schema_version": 2,
            "contract": "single_nick_snapback_v2",
            "name": "demo_snapback",
        },
        "input": {
            "canonical_top_strand": {
                "sequence": "CCTCAGCAGTC",
                "protected_region": {"start": 0, "end": 11},
                "pre_nick_duplex_window": {"start": 0, "end": 11},
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
                "retained_homology_window": {"start": 7, "end": 11},
                "cap_sequence": "TT",
                "foldback_arm": "GACT",
                "homology_policy": {"max_mismatches": 0, "min_paired_bp": 4, "max_paired_bp": 4},
            },
            "constraints": {
                "terminal_ligatable_duplex_bp": {"min": 4, "max": 4},
                "max_uninterrupted_duplex_bp": 4,
                "max_added_nt": 6,
                "forbid_additional_target_strand_nicks": False,
                "forbid_any_additional_nicks": False,
            },
            "sequence_quality": {
                "gc_fraction": {"min": 0.25, "max": 0.75},
                "max_homopolymer_run": 2,
            },
        },
        "output": {
            "run_dir": "outputs/snapback",
            "emit_visual_contracts": True,
            "emit_baserender_jobs": True,
        },
    }


def test_snapback_visual_builders_publish_three_consistent_states(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path, spec_payload=_base_payload(), catalog_entries=_catalog_entries())
    report = validate_snapback_spec(spec_path)

    pre = build_pre_nick_duplex_view(report=report, solution_id="demo", title="Pre")
    exposed = build_post_nick_exposed_view(report=report, solution_id="demo", title="Exposed")
    foldback = build_post_nick_foldback_view(report=report, solution_id="demo", title="Foldback")

    assert pre["nick_boundary"] == 2
    assert pre["ligation_junction_boundary"] == 7
    assert exposed["nick_boundary"] == 2
    assert exposed["ligation_junction_boundary"] == 7
    assert foldback["source_nick_boundary"] == 2
    assert foldback["ligation_junction_boundary"] == 5

    pre_public = SnapbackVisualV1.model_validate(
        build_pre_nick_snapback_visual(report=report, solution_id="demo", title="Pre")
    )
    exposed_public = SnapbackVisualV1.model_validate(
        build_post_nick_exposed_snapback_visual(report=report, solution_id="demo", title="Exposed")
    )
    foldback_public = SnapbackVisualV1.model_validate(
        build_post_nick_foldback_snapback_visual(report=report, solution_id="demo", title="Foldback")
    )

    assert pre_public.state_kind == "pre_nick_duplex"
    assert exposed_public.state_kind == "post_nick_exposed"
    assert foldback_public.state_kind == "post_nick_foldback"
    assert pre_public.pairings == []
    assert exposed_public.pairings == []
    assert pre_public.nick_boundary == 2
    assert exposed_public.exposed_complement_span is not None
    assert foldback_public.nick_boundary is None
    assert foldback_public.ligation_junction_boundary == 5
    assert any(pair.left_index == 5 for pair in foldback_public.pairings)


def test_snapback_foldback_visuals_publish_absolute_mismatch_positions(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 0, "end": 8}
    payload["design"]["topology"]["foldback_arm"] = "GAGT"
    payload["design"]["topology"]["homology_policy"]["max_mismatches"] = 1
    payload["design"]["constraints"]["terminal_ligatable_duplex_bp"] = {"min": 1, "max": 4}
    payload["design"]["constraints"]["max_uninterrupted_duplex_bp"] = 2
    spec_path = _write_workspace(tmp_path, spec_payload=payload, catalog_entries=_catalog_entries())
    report = validate_snapback_spec(spec_path)
    assert report.candidate is not None

    foldback = build_post_nick_foldback_view(report=report, solution_id="demo", title="Foldback")
    public = SnapbackVisualV1.model_validate(
        build_post_nick_foldback_snapback_visual(report=report, solution_id="demo", title="Foldback")
    )

    candidate = report.candidate
    expected_primary = [candidate.post_nick_retained_homology_span.start + 1]
    expected_foldback = [candidate.post_nick_foldback_arm_span.end - 1 - 1]

    assert foldback["primary_mismatch_positions"] == expected_primary
    assert foldback["foldback_partner_mismatch_positions"] == expected_foldback
    assert public.primary_mismatch_positions == expected_primary
    assert public.complement_mismatch_positions == expected_foldback

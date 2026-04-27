"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/snapback/test_visual_publication.py

Visual publication tests for snapback QA and public snapback visual artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dnadesign.contracts.visual import SnapbackVisualV1
from dnadesign.cruncher.app.snapback_publish import build_publication_bundle, write_publication_bundle
from dnadesign.cruncher.app.snapback_workflow import validate_snapback_spec
from dnadesign.cruncher.nickases.models import reverse_complement
from dnadesign.cruncher.snapback.artifacts import snapback_triptych_visual_contracts_path, views_manifest_path
from dnadesign.cruncher.snapback.public_visuals import (
    build_post_nick_exposed_snapback_visual,
    build_post_nick_foldback_snapback_visual,
    build_pre_nick_snapback_visual,
)
from dnadesign.cruncher.snapback.view_contracts import (
    build_post_nick_exposed_view,
    build_post_nick_foldback_view,
    build_pre_nick_duplex_view,
)
from dnadesign.cruncher.snapback.view_models import SnapbackPostNickExposedViewV1, SnapbackPostNickFoldbackViewV1


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
        "output": {
            "run_dir": "outputs/design",
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
    assert pre["ligation_junction_boundary"] == 2
    assert pre["source_cap_window"] == {"start": 6, "end": 8}
    assert pre["effective_cap_window"] == {"start": 6, "end": 9}
    assert exposed["nick_boundary"] == 2
    assert exposed["ligation_junction_boundary"] == 2
    assert foldback["source_nick_boundary"] == 2
    assert foldback["ligation_junction_boundary"] == 0

    pre_public = SnapbackVisualV1.model_validate(
        build_pre_nick_snapback_visual(report=report, solution_id="demo", title="Pre")
    )
    exposed_public = SnapbackVisualV1.model_validate(
        build_post_nick_exposed_snapback_visual(report=report, solution_id="demo", title="Exposed")
    )
    foldback_public = SnapbackVisualV1.model_validate(
        build_post_nick_foldback_snapback_visual(report=report, solution_id="demo", title="Foldback")
    )
    assert report.candidate is not None

    assert pre_public.state_kind == "pre_nick_duplex"
    assert exposed_public.state_kind == "post_nick_exposed"
    assert foldback_public.state_kind == "post_nick_foldback"
    assert pre_public.pairings == []
    assert exposed_public.pairings == []
    assert pre_public.nick_boundary == 2
    assert pre_public.ligation_junction_boundary == 2
    assert exposed_public.exposed_complement_span is not None
    assert foldback_public.nick_boundary is None
    assert foldback_public.ligation_junction_boundary == 0
    assert foldback_public.loop_geometry is not None
    assert foldback_public.loop_geometry.kind == "hairpin_corner_triloop_v1"
    assert foldback_public.loop_geometry.source_cap_span.model_dump(mode="json") == {"start": 4, "end": 6}
    assert foldback_public.loop_geometry.cap_extension_span.model_dump(mode="json") == {"start": 6, "end": 7}
    assert foldback_public.primary_sequence == report.candidate.post_nick_sequence
    assert foldback_public.complement_sequence == reverse_complement(report.candidate.post_nick_sequence)[::-1]
    assert foldback_public.complement_sequence != foldback_public.primary_sequence
    assert any(pair.left_index == 0 for pair in foldback_public.pairings)


def test_snapback_foldback_visuals_publish_absolute_mismatch_positions(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 0, "end": 2}
    payload["design"]["topology"]["foldback_arm"] = "CTAA"
    payload["design"]["topology"]["homology_policy"]["max_mismatches"] = 1
    payload["design"]["constraints"]["terminal_ligatable_duplex_bp"] = {"min": 1, "max": 4}
    payload["design"]["constraints"]["max_uninterrupted_duplex_bp"] = 2
    payload["design"]["sequence_quality"]["gc_fraction"] = {"min": 0.0, "max": 0.75}
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


def test_snapback_publication_bundle_manifest_matches_emitted_files(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path, spec_payload=_base_payload(), catalog_entries=_catalog_entries())
    report = validate_snapback_spec(spec_path)
    run_dir = tmp_path / "outputs" / "design"

    bundle = build_publication_bundle(report=report, solution_id="demo", include_jobs=True)
    write_publication_bundle(run_dir, bundle=bundle)

    manifest_path = views_manifest_path(run_dir)
    manifest_dir = manifest_path.parent
    for entry in bundle.manifest["views"]:
        assert (run_dir / entry["path"]).exists()
    triptych_jsonl = snapback_triptych_visual_contracts_path(run_dir)
    assert triptych_jsonl.exists()
    assert len(triptych_jsonl.read_text(encoding="utf-8").strip().splitlines()) == 3
    assert len(bundle.manifest["recommended_jobs"]) == 1
    assert bundle.manifest["recommended_jobs"][0]["name"] == "snapback_triptych"
    for job in bundle.manifest["recommended_jobs"]:
        assert (manifest_dir / job["path"]).resolve().exists()


def test_snapback_exposed_view_rejects_non_adjacent_cap_partition(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path, spec_payload=_base_payload(), catalog_entries=_catalog_entries())
    report = validate_snapback_spec(spec_path)
    payload = build_post_nick_exposed_view(report=report, solution_id="demo", title="Exposed")
    payload["topology"]["cap_extension_span"]["start"] += 1

    with pytest.raises(ValidationError, match="source_cap_span must end at cap_extension_span.start."):
        SnapbackPostNickExposedViewV1.model_validate(payload)


def test_snapback_foldback_view_rejects_non_adjacent_cap_partition(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path, spec_payload=_base_payload(), catalog_entries=_catalog_entries())
    report = validate_snapback_spec(spec_path)
    payload = build_post_nick_foldback_view(report=report, solution_id="demo", title="Foldback")
    payload["topology"]["cap_extension_span"]["start"] += 1

    with pytest.raises(ValidationError, match="source_cap_span must end at cap_extension_span.start."):
        SnapbackPostNickFoldbackViewV1.model_validate(payload)

"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/released_snapback/test_target_search.py

Target-search tests for released-product snapback workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import dnadesign.cruncher.snapback.released_target_search as released_target_search
from dnadesign.cruncher.app.snapback_released_target_search_workflow import run_released_snapback_target_search
from dnadesign.cruncher.nickases.models import NickaseCatalogEntry
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeEntry
from dnadesign.cruncher.snapback.models import CatalogSources
from dnadesign.cruncher.snapback.released_models import (
    ReleaseCatalogSources,
    ReleasedFinalTargetGeometry,
    ReleasedTargetSearchConfig,
    SingleNickReleasedTargetSearchRequest,
)


def _write_nick_catalog(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nx.Exact7",
                            "specificity_id": "Nx.Exact7",
                            "motif_top_5to3": "AACGTTG",
                            "top_cut_offset": 0,
                        },
                        {
                            "id": "Nx.ExactAlt7",
                            "specificity_id": "Nx.ExactAlt7",
                            "motif_top_5to3": "AAAGTTT",
                            "top_cut_offset": 0,
                        },
                        {
                            "id": "Nx.Near7",
                            "specificity_id": "Nx.Near7",
                            "motif_top_5to3": "TAACGTT",
                            "top_cut_offset": 1,
                        },
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_release_catalog(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "release_enzymes": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "variant_id": "Re.Exact",
                            "display_name": "Re.Exact",
                            "recognition_sequence": "CCAA",
                            "top_cut_offset": 0,
                            "bottom_cut_offset": 1,
                            "class_label": "other_ds_re",
                            "commercial_confidence": "primary_vendor_current",
                            "source_catalog_id": "local_release",
                        },
                        {
                            "variant_id": "Re.Overlap",
                            "display_name": "Re.Overlap",
                            "recognition_sequence": "GGGG",
                            "top_cut_offset": 12,
                            "bottom_cut_offset": 13,
                            "class_label": "other_ds_re",
                            "commercial_confidence": "primary_vendor_current",
                            "source_catalog_id": "local_release",
                        },
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_released_target_search_reports_exact_hits_near_hits_blockers_and_pre_post_truncation_counts(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspaces" / "demo_released"
    nick_catalog_path = workspace_root / "inputs" / "nickases" / "local.nickases.yaml"
    release_catalog_path = workspace_root / "inputs" / "release_enzymes" / "local.release.yaml"
    nick_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    release_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    _write_nick_catalog(nick_catalog_path)
    _write_release_catalog(release_catalog_path)

    report = run_released_snapback_target_search(
        request=SingleNickReleasedTargetSearchRequest(
            target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
            nick_sources=CatalogSources(additional_paths=[Path("inputs/nickases/local.nickases.yaml")]),
            release_sources=ReleaseCatalogSources(additional_paths=[Path("inputs/release_enzymes/local.release.yaml")]),
            search=ReleasedTargetSearchConfig(max_results=1, near_boundary_search_limit=3),
        ),
        workspace_root=workspace_root,
    )

    assert report.status == "exact_hits_found"
    assert report.metadata.pre_truncation_exact_hit_count == 4
    assert report.metadata.post_truncation_exact_hit_count == 1
    assert report.metadata.pre_truncation_near_hit_count >= 2
    assert report.metadata.post_truncation_near_hit_count == 1
    assert report.exact_hits[0].nickase_variant_id in {"Nx.Exact7", "Nx.ExactAlt7"}
    assert report.exact_hits[0].release_variant_id == "Re.Exact"
    assert report.exact_hits[0].nick_boundary_from_left == 0
    assert report.exact_hits[0].retained_input_length_nt == 6
    assert report.near_hits[0].nickase_variant_id == "Nx.Near7"
    assert report.near_hits[0].nick_boundary_from_left == 1
    assert report.metadata.blocker_counts["RELEASE_OVERLAPS_REQUIRED_RETAINED_REGION"] >= 1


def test_search_pair_collects_all_near_hits_within_bounded_window(monkeypatch: pytest.MonkeyPatch) -> None:
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=2, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="test_nick"),
        release_sources=ReleaseCatalogSources(preset="test_release"),
        search=ReleasedTargetSearchConfig(near_boundary_search_limit=2),
    )
    nick_placement = released_target_search._NickPlacement(
        entry=NickaseCatalogEntry(
            id="Nx.Test",
            specificity_id="Nx.Test",
            motif_top_5to3="AAAA",
            top_cut_offset=0,
        ),
        orientation="forward",
        motif="AAAA",
        site_start_at_boundary_zero=0,
    )
    release_placement = released_target_search._ReleasePlacement(
        entry=ReleaseEnzymeEntry(
            variant_id="Re.Test",
            display_name="Re.Test",
            recognition_sequence="TTTT",
            top_cut_offset=0,
            bottom_cut_offset=0,
            class_label="other_ds_re",
            commercial_confidence="primary_vendor_current",
            source_catalog_id="test_release",
        ),
        orientation="forward",
        motif="TTTT",
        retained_length_offset=9,
        site_shift_from_boundary=0,
        bottom_cut_shift_from_boundary=0,
    )

    def fake_build_precursor_sequence(**_: object) -> str:
        return "A" * 12

    def fake_evaluate_released_precursor(*, target: ReleasedFinalTargetGeometry, **_: object) -> SimpleNamespace:
        status = "satisfied" if target.nick_boundary_from_left in {1, 3, 4} else "unsatisfied"
        return SimpleNamespace(
            status=status,
            issues=[],
            candidate=object(),
            projection=object(),
            pre_nick_match=object(),
            release_match=object(),
        )

    def fake_hit_from_evaluation(*, boundary: int, hit_kind: str, **_: object) -> tuple[int, str]:
        return (boundary, hit_kind)

    monkeypatch.setattr(released_target_search, "_build_precursor_sequence", fake_build_precursor_sequence)
    monkeypatch.setattr(released_target_search, "evaluate_released_precursor", fake_evaluate_released_precursor)
    monkeypatch.setattr(released_target_search, "_hit_from_evaluation", fake_hit_from_evaluation)

    exact_hit, near_hits = released_target_search._search_pair(
        request=request,
        nick_placement=nick_placement,
        release_placement=release_placement,
        blocker_counts={},
    )

    assert exact_hit is None
    assert near_hits == [(1, "nearest"), (3, "nearest"), (4, "nearest")]

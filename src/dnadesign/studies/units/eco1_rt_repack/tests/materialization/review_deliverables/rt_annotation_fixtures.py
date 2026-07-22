"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/rt_annotation_fixtures.py

RT annotation fixture sources for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_rt_annotation_context_sources(output_root: Path) -> tuple[Path, Path]:
    """Write compact RT annotation context sources for visual-rendering tests."""

    annotation_tracks_path = output_root / "rt-annotation-tracks.yaml"
    manual_authority_path = output_root / "manual-mask-authority.yaml"
    target_hash = "sha256:" + "a" * 64
    annotation_tracks_path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "dnadesign.aligner.msa.visualization.annotation_tracks",
                "schema_version": 1,
                "study_id": "eco1_rt_repack",
                "status": "fixture_rt_interval_authority_v1",
                "coordinate_space": "target_ungapped_position",
                "target_row_id": "eco1_rt_ec86kit_reference",
                "target_sequence_hash": target_hash,
                "source_basis": [],
                "tracks": [
                    {
                        "id": "retron_rt_context_spans",
                        "label": "Mask-context spans",
                        "features": [
                            {
                                "id": "retron_x_context",
                                "label": "Region X local context",
                                "start": 2,
                                "end": 4,
                            },
                            {
                                "id": "catalytic_context",
                                "label": "Catalytic YADD local context",
                                "start": 3,
                                "end": 5,
                            },
                        ],
                    },
                    {
                        "id": "retron_rt_core_intervals",
                        "label": "RT1-RT7 core intervals",
                        "features": [
                            {
                                "id": "rt1_interval",
                                "label": "RT1",
                                "start": 2,
                                "end": 3,
                            },
                            {
                                "id": "rt2_interval",
                                "label": "RT2",
                                "start": 4,
                                "end": 5,
                            },
                        ],
                    },
                    {
                        "id": "retron_rt_motif_anchors",
                        "label": "RT motif anchors",
                        "features": [
                            {
                                "id": "retron_x_naxxh",
                                "label": "NAxxH",
                                "start": 3,
                                "end": 3,
                            },
                            {
                                "id": "catalytic_yadd",
                                "label": "YADD",
                                "start": 4,
                                "end": 4,
                            },
                        ],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manual_authority_path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt_repack.manual_mask_authority_source",
                "schema_version": 1,
                "study_id": "eco1_rt_repack",
                "status": "fixture_motif_review_labels_v1",
                "coordinate_space": "canonical_position",
                "target_row_id": "eco1_rt_ec86kit_reference",
                "target_sequence_hash": target_hash,
                "mask_policy_id": "eco1_rt_manual_motif_wang_direct_contact_v1",
                "source_basis": [],
                "authority_sets": [
                    _authority_set(
                        "ec86_rt1_interval", "rt_core_interval", "review_label", "rt1_interval", "RT1", 2, 3
                    ),
                    _authority_set(
                        "ec86_rt2_interval", "rt_core_interval", "review_label", "rt2_interval", "RT2", 4, 5
                    ),
                    _authority_set(
                        "ec86_retron_x_region",
                        "retron_x_motif_anchor",
                        "fixed",
                        "retron_x_naxxh",
                        "NAxxH",
                        3,
                        3,
                    ),
                    _authority_set(
                        "ec86_active_site_geometry",
                        "catalytic_core_motif_anchor",
                        "fixed",
                        "catalytic_yadd",
                        "YADD",
                        4,
                        4,
                    ),
                ],
                "context_only_spans": [
                    {
                        "id": "retron_x_context",
                        "label": "Region X local context",
                        "start": 2,
                        "end": 4,
                    },
                    {
                        "id": "catalytic_context",
                        "label": "Catalytic YADD local context",
                        "start": 3,
                        "end": 5,
                    },
                ],
                "deferred_authority": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return annotation_tracks_path, manual_authority_path


def _authority_set(
    set_id: str,
    authority_type: str,
    policy: str,
    feature_id: str,
    label: str,
    start: int,
    end: int,
) -> dict[str, Any]:
    return {
        "id": set_id,
        "label": label,
        "authority_type": authority_type,
        "policy": policy,
        "features": [
            {
                "id": feature_id,
                "label": label,
                "start": start,
                "end": end,
                "reason": feature_id,
                "source_locator": "fixture",
                "evidence_basis": ["fixture"],
            }
        ],
    }

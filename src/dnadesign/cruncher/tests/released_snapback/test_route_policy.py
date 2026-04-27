"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/released_snapback/test_route_policy.py

Route-policy contract tests for released-product Snapback.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.cruncher.snapback.released_route_policy import (
    infer_released_search_final_geometry_source,
    released_route_semantics,
    released_search_route_policy,
)
from dnadesign.cruncher.snapback.released_search_models import ReleasedTargetSearchConfig, ReleasedTargetSearchMetadata
from dnadesign.cruncher.snapback.released_solve_models import ReleasedSolveReportMetadata
from dnadesign.cruncher.snapback.released_spec_models import ReleasedFinalTargetGeometry


def _target_metadata_kwargs(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "target": ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        "nick_catalog_source": "preset:nick",
        "release_catalog_source": "preset:release",
        "allowed_active_strands": ["bottom"],
        "allowed_route_families": ["bottom_active_from_top_nick"],
        "evaluated_pair_count": 0,
        "pre_truncation_exact_hit_count": 0,
        "post_truncation_exact_hit_count": 0,
        "pre_truncation_near_hit_count": 0,
        "post_truncation_near_hit_count": 0,
    }
    payload.update(overrides)
    return payload


def _solve_metadata_kwargs(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "target": ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        "nick_catalog_source": "preset:nick",
        "release_catalog_source": "preset:release",
        "allowed_active_strands": ["bottom"],
        "allowed_route_families": ["bottom_active_from_top_nick"],
        "evaluated_pair_count": 0,
        "available_exact_hit_count": 0,
        "available_near_hit_count": 0,
        "materialized_hit_count": 0,
        "requested_materialize_top_k": 1,
        "render_format": "pdf",
    }
    payload.update(overrides)
    return payload


def test_released_route_semantics_keeps_bottom_active_contract_together() -> None:
    semantics = released_route_semantics("bottom_active_from_top_nick")

    assert semantics.final_geometry_source == "exposed_bottom_strand"
    assert semantics.active_strand == "bottom"
    assert semantics.retained_partner_strand == "top"
    assert semantics.physical_nicked_strand == "top"


def test_released_route_semantics_keeps_top_active_contract_together() -> None:
    semantics = released_route_semantics("top_active_from_bottom_nick")

    assert semantics.final_geometry_source == "retained_active_strand"
    assert semantics.active_strand == "top"
    assert semantics.retained_partner_strand == "bottom"
    assert semantics.physical_nicked_strand == "bottom"


def test_released_search_route_policy_defaults_to_exposed_bottom_lane() -> None:
    policy = released_search_route_policy(allow_top_active_routes=False)

    assert policy.final_geometry_source == "exposed_bottom_strand"
    assert policy.allowed_active_strands == ("bottom",)
    assert policy.allowed_route_families == ("bottom_active_from_top_nick",)


def test_released_search_route_policy_broadens_to_retained_active_audit() -> None:
    policy = released_search_route_policy(allow_top_active_routes=True)

    assert policy.final_geometry_source == "retained_active_strand"
    assert policy.allowed_active_strands == ("top", "bottom")
    assert policy.allowed_route_families == (
        "bottom_active_from_top_nick",
        "top_active_from_bottom_nick",
    )


def test_infer_released_search_final_geometry_source_handles_mixed_route_policy() -> None:
    assert (
        infer_released_search_final_geometry_source(
            allowed_active_strands=["top", "bottom"],
            allowed_route_families=["bottom_active_from_top_nick", "top_active_from_bottom_nick"],
        )
        == "retained_active_strand"
    )


def test_infer_released_search_final_geometry_source_uses_executed_route_families() -> None:
    assert (
        infer_released_search_final_geometry_source(
            allowed_active_strands=["top", "bottom"],
            allowed_route_families=["bottom_active_from_top_nick"],
        )
        == "exposed_bottom_strand"
    )


def test_infer_released_search_final_geometry_source_rejects_incompatible_route_policy() -> None:
    with pytest.raises(ValueError, match="compatible with allowed_active_strands"):
        infer_released_search_final_geometry_source(
            allowed_active_strands=["bottom"],
            allowed_route_families=["top_active_from_bottom_nick"],
        )


def test_released_target_search_config_carries_canonical_route_policy_geometry_source() -> None:
    config = ReleasedTargetSearchConfig(
        allowed_active_strands=["top", "bottom"],
        allowed_route_families=["top_active_from_bottom_nick"],
    )

    assert config.route_policy_final_geometry_source == "retained_active_strand"


def test_released_target_search_config_rejects_policy_geometry_drift() -> None:
    with pytest.raises(ValueError, match="search.route_policy_final_geometry_source must match"):
        ReleasedTargetSearchConfig(
            route_policy_final_geometry_source="retained_active_strand",
            allowed_active_strands=["bottom"],
            allowed_route_families=["bottom_active_from_top_nick"],
        )


def test_released_target_search_config_rejects_legacy_policy_geometry_field() -> None:
    with pytest.raises(ValueError, match="final_geometry_source"):
        ReleasedTargetSearchConfig(
            final_geometry_source="exposed_bottom_strand",
        )


def test_released_target_search_metadata_rejects_policy_geometry_drift() -> None:
    with pytest.raises(ValueError, match="metadata.route_policy_final_geometry_source must match"):
        ReleasedTargetSearchMetadata(
            **_target_metadata_kwargs(route_policy_final_geometry_source="retained_active_strand")
        )


def test_released_target_search_metadata_rejects_empty_route_policy_lists() -> None:
    with pytest.raises(ValueError, match="metadata.allowed_active_strands must not be empty"):
        ReleasedTargetSearchMetadata(**_target_metadata_kwargs(allowed_active_strands=[]))
    with pytest.raises(ValueError, match="metadata.allowed_route_families must not be empty"):
        ReleasedTargetSearchMetadata(**_target_metadata_kwargs(allowed_route_families=[]))


def test_released_target_search_metadata_rejects_legacy_policy_geometry_field() -> None:
    with pytest.raises(ValueError, match="final_geometry_source"):
        ReleasedTargetSearchMetadata(**_target_metadata_kwargs(final_geometry_source="exposed_bottom_strand"))


def test_released_solve_metadata_rejects_policy_geometry_drift() -> None:
    with pytest.raises(ValueError, match="metadata.route_policy_final_geometry_source must match"):
        ReleasedSolveReportMetadata(
            **_solve_metadata_kwargs(route_policy_final_geometry_source="retained_active_strand")
        )


def test_released_solve_metadata_rejects_empty_route_policy_lists() -> None:
    with pytest.raises(ValueError, match="metadata.allowed_active_strands must not be empty"):
        ReleasedSolveReportMetadata(**_solve_metadata_kwargs(allowed_active_strands=[]))
    with pytest.raises(ValueError, match="metadata.allowed_route_families must not be empty"):
        ReleasedSolveReportMetadata(**_solve_metadata_kwargs(allowed_route_families=[]))


def test_released_solve_metadata_rejects_legacy_policy_geometry_field() -> None:
    with pytest.raises(ValueError, match="final_geometry_source"):
        ReleasedSolveReportMetadata(**_solve_metadata_kwargs(final_geometry_source="exposed_bottom_strand"))

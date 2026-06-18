"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_solve_snapshot.py

Snapshot helpers for preserved-site Snapback solve materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.cruncher.snapback.solve_models import SingleNickSnapbackSolveSpec, SnapbackSolveHit


def build_snapback_explicit_spec_payload_for_hit(
    spec: SingleNickSnapbackSolveSpec,
    *,
    hit: SnapbackSolveHit,
    workspace_root: Path,
    hit_run_dir: Path,
) -> dict[str, object]:
    candidate = hit.explicit_report
    resolved_terminal_ligatable_duplex_bp = spec.resolved_terminal_ligatable_duplex_bp()
    resolved_max_uninterrupted_duplex_bp = spec.resolved_max_uninterrupted_duplex_bp()
    materialized_run_dir = hit_run_dir.resolve().relative_to(workspace_root.resolve())
    return {
        "snapback": {
            "schema_version": 2,
            "contract": "single_nick_snapback_v2",
            "name": f"{spec.name}__hit_{hit.rank:02d}",
        },
        "input": {
            "canonical_top_strand": {
                "sequence": candidate.input_sequence,
                "protected_region": candidate.protected_region.model_dump(mode="json"),
                "pre_nick_duplex_window": candidate.pre_nick_duplex_window.model_dump(mode="json"),
            },
        },
        "design": {
            "nickase": {
                "variant_id": hit.variant_id,
                "catalog": {
                    "preset": spec.catalog.preset,
                    "additional_presets": list(spec.catalog.additional_presets),
                    "additional_paths": [str(path) for path in spec.catalog.additional_paths],
                },
            },
            "orientation_policy": {
                "normalize_to_top_strand_nick": spec.orientation_policy.normalize_to_top_strand_nick,
                "release_direction": "left_to_right_from_nick",
            },
            "single_nick_goal": {
                "nick_boundary_window": {
                    "min": candidate.nick_boundary,
                    "max": candidate.nick_boundary,
                }
            },
            "topology": {
                "retained_homology_window": candidate.retained_homology_window.model_dump(mode="json"),
                "cap_sequence": candidate.cap_sequence,
                "foldback_arm": candidate.foldback_arm,
                "homology_policy": {
                    "max_mismatches": spec.search.max_mismatches,
                    "min_paired_bp": candidate.paired_bp,
                    "max_paired_bp": candidate.paired_bp,
                },
            },
            "constraints": {
                "terminal_ligatable_duplex_bp": resolved_terminal_ligatable_duplex_bp.model_dump(mode="json"),
                "max_uninterrupted_duplex_bp": resolved_max_uninterrupted_duplex_bp,
                "max_added_nt": spec.search.max_added_nt,
                "forbid_additional_target_strand_nicks": spec.constraints.forbid_additional_target_strand_nicks,
                "forbid_any_additional_nicks": spec.constraints.forbid_any_additional_nicks,
            },
            "sequence_quality": spec.sequence_quality.model_dump(mode="json"),
        },
        "output": {
            "run_dir": str(materialized_run_dir),
            "emit_visual_contracts": spec.output.emit_visual_contracts,
            "render_format": spec.output.render_format,
            "emit_baserender_jobs": spec.output.emit_baserender_jobs,
        },
    }


def dump_snapback_explicit_spec_yaml_for_hit(
    spec: SingleNickSnapbackSolveSpec,
    *,
    hit: SnapbackSolveHit,
    workspace_root: Path,
    hit_run_dir: Path,
) -> str:
    return yaml.safe_dump(
        build_snapback_explicit_spec_payload_for_hit(
            spec,
            hit=hit,
            workspace_root=workspace_root,
            hit_run_dir=hit_run_dir,
        ),
        sort_keys=False,
    )


__all__ = [
    "build_snapback_explicit_spec_payload_for_hit",
    "dump_snapback_explicit_spec_yaml_for_hit",
]

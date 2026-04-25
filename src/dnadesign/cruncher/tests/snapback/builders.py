"""
Shared builders for preserved-site Snapback tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SnapbackWorkspaceFixture:
    workspace_root: Path
    explicit_path: Path
    solve_path: Path
    catalog_path: Path


def write_snapback_workspace(tmp_path: Path) -> SnapbackWorkspaceFixture:
    workspace_root = tmp_path / "workspaces" / "demo_snapback"
    explicit_path = workspace_root / "configs" / "snapback" / "demo.snapback.yaml"
    solve_path = workspace_root / "configs" / "snapback" / "demo.snapback.solve.yaml"
    catalog_path = workspace_root / "inputs" / "nickases" / "local.nickases.yaml"
    explicit_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nt.Bpu10I",
                            "specificity_id": "Bpu10I",
                            "motif_top_5to3": "CCTNAGC",
                            "top_cut_offset": 2,
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    explicit_path.write_text(
        yaml.safe_dump(
            {
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    solve_path.write_text(
        yaml.safe_dump(
            {
                "snapback_solve": {
                    "schema_version": 3,
                    "contract": "single_nick_snapback_solve_v3",
                    "name": "demo_snapback_solve",
                },
                "input": {
                    "canonical_top_strand": {
                        "sequence": "CCTCAGCA",
                        "protected_region": {"start": 0, "end": 8},
                        "pre_nick_duplex_window": {"start": 0, "end": 8},
                    }
                },
                "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
                "orientation_policy": {"normalize_to_top_strand_nick": True},
                "goal": {"nick_boundary_window": {"min": 2, "max": 2}},
                "search": {
                    "retained_homology_length": {"min": 4, "max": 4},
                    "max_added_nt": 5,
                    "max_mismatches": 0,
                    "max_enumerated_candidates": 64,
                    "max_search_nodes": 64,
                    "max_hits": 4,
                    "materialize_top_k": 2,
                },
                "constraints": {
                    "terminal_ligatable_duplex_bp": {"min": 4, "max": 4},
                    "max_uninterrupted_duplex_bp": 4,
                    "forbid_additional_target_strand_nicks": False,
                    "forbid_any_additional_nicks": False,
                },
                "sequence_quality": {
                    "gc_fraction": {"min": 0.0, "max": 0.75},
                    "max_homopolymer_run": 3,
                },
                "output": {
                    "run_dir": "outputs/solve",
                    "emit_visual_contracts": True,
                    "emit_baserender_jobs": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return SnapbackWorkspaceFixture(
        workspace_root=workspace_root,
        explicit_path=explicit_path,
        solve_path=solve_path,
        catalog_path=catalog_path,
    )


__all__ = ["SnapbackWorkspaceFixture", "write_snapback_workspace"]

"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/released_snapback/builders.py

Shared builders for released-product snapback tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ReleasedWorkspaceFixture:
    workspace_root: Path
    spec_path: Path
    nick_catalog_path: Path
    release_catalog_path: Path


def write_released_workspace(
    tmp_path: Path,
    *,
    precursor_top_strand: str = "AACGTTGTTCCAA",
) -> ReleasedWorkspaceFixture:
    workspace_root = tmp_path / "workspaces" / "demo_released"
    spec_path = workspace_root / "configs" / "snapback" / "demo.released.snapback.yaml"
    nick_catalog_path = workspace_root / "inputs" / "nickases" / "local.nickases.yaml"
    release_catalog_path = workspace_root / "inputs" / "release_enzymes" / "local.release.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    release_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog_path.write_text(
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
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    release_catalog_path.write_text(
        yaml.safe_dump(
            {
                "release_enzymes": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "variant_id": "Re.Exact",
                            "display_name": "Re.Exact",
                            "recognition_sequence": "CCAA",
                            "top_cut_offset": 1,
                            "bottom_cut_offset": 0,
                            "class_label": "other_ds_re",
                            "commercial_confidence": "primary_vendor_current",
                            "source_catalog_id": "local_release",
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    spec_path.write_text(
        yaml.safe_dump(
            {
                "released_snapback": {
                    "schema_version": 1,
                    "kind": "single_nick_released_snapback_v1",
                    "name": "demo_released",
                },
                "input": {
                    "precursor_top_strand": precursor_top_strand,
                },
                "nick_stage": {
                    "nickase_variant_id": "Nx.Exact7",
                    "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
                },
                "release_stage": {
                    "release_variant_id": "Re.Exact",
                    "catalog": {"additional_paths": ["inputs/release_enzymes/local.release.yaml"]},
                    "retained_side": "upstream",
                    "stage_order": "nick_then_release",
                },
                "final_target": {
                    "nick_boundary_from_left": 0,
                    "paired_bp": 3,
                    "cap_nt": 3,
                },
                "constraints": {
                    "allow_post_release_loss_of_nickase_site": True,
                    "allow_post_release_loss_of_release_site": True,
                    "require_release_site_downstream_of_nick": True,
                    "require_complete_downstream_fragment_separation": True,
                },
                "output": {"run_dir": "outputs/released_design"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return ReleasedWorkspaceFixture(
        workspace_root=workspace_root,
        spec_path=spec_path,
        nick_catalog_path=nick_catalog_path,
        release_catalog_path=release_catalog_path,
    )


__all__ = ["ReleasedWorkspaceFixture", "write_released_workspace"]

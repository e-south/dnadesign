"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_workspace_service.py

Scaffold v2 explicit and solve snapback workspaces.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json

_WORKSPACE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MANIFEST_NAME = "snapback_workspace_manifest.json"


@dataclass(frozen=True)
class SnapbackWorkspaceScaffoldResult:
    workspace_root: Path
    manifest_path: Path
    readme_path: Path
    example_spec_path: Path
    example_solve_spec_path: Path
    catalog_path: Path


def _repo_root_from(start: Path) -> Path | None:
    try:
        cursor = start.resolve()
    except Exception:
        cursor = start
    for root in [cursor, *cursor.parents]:
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return None


def default_snapback_workspaces_root() -> Path:
    repo_root = _repo_root_from(Path(__file__).resolve())
    if repo_root is not None:
        return (repo_root / "src" / "dnadesign" / "cruncher" / "workspaces").resolve()
    raise ValueError("Unable to determine the standard Cruncher workspaces root. Pass --root or --output explicitly.")


def _validate_workspace_name(name: str) -> str:
    raw = str(name).strip()
    if raw == "":
        raise ValueError("Snapback workspace name must be non-empty.")
    if "/" in raw or "\\" in raw:
        raise ValueError(
            f"Invalid snapback workspace name: {raw!r}. "
            "Use a simple workspace name with --root, or pass --output for an explicit path."
        )
    if _WORKSPACE_NAME_RE.fullmatch(raw) is None:
        raise ValueError(f"Invalid snapback workspace name: {raw!r}.")
    return raw


def snapback_workspace_path(name: str, *, root: Path | None = None) -> Path:
    workspace_name = _validate_workspace_name(name)
    parent = default_snapback_workspaces_root() if root is None else Path(root).expanduser().resolve()
    return parent / workspace_name


def _manifest_path(workspace_root: Path) -> Path:
    return workspace_root / _MANIFEST_NAME


def _catalog_payload() -> dict[str, object]:
    return {
        "nickases": {
            "schema_version": 1,
            "entries": [
                {
                    "id": "Nt.Bpu10I",
                    "specificity_id": "Bpu10I",
                    "motif_top_5to3": "CCTNAGC",
                    "top_cut_offset": 2,
                    "source": "local_overlay",
                    "metadata": {
                        "vendor": "thermofisher",
                        "vendor_catalog_number": "ER2011",
                        "raw_geometry_note": "CC↓TNAGC",
                    },
                }
            ],
        }
    }


def _example_spec_payload() -> dict[str, object]:
    return {
        "snapback": {
            "schema_version": 2,
            "contract": "single_nick_snapback_v2",
            "name": "demo_teto_bpu10i_cap",
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
        "output": {"run_dir": "outputs/snapback", "emit_visual_contracts": True, "emit_baserender_jobs": True},
    }


def _example_solve_payload() -> dict[str, object]:
    return {
        "snapback_solve": {
            "schema_version": 2,
            "contract": "single_nick_snapback_solve_v2",
            "name": "demo_teto_bpu10i_cap_solve",
        },
        "input": {
            "canonical_top_strand": {
                "sequence": "CCTCAGCAGTC",
                "protected_region": {"start": 0, "end": 11},
                "pre_nick_duplex_window": {"start": 0, "end": 11},
            }
        },
        "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
        "nickase_policy": {
            "allowed_variant_ids": ["Nt.Bpu10I"],
            "normalize_to_top_strand_nick": True,
        },
        "goal": {
            "nick_boundary_window": {"min": 2, "max": 2},
            "retained_start_from_nick": {"min": 5, "max": 5},
        },
        "search": {
            "retained_homology_length": {"min": 4, "max": 4},
            "cap_nt": {"min": 1, "max": 2},
            "max_added_nt": 6,
            "max_mismatches": 0,
            "max_enumerated_candidates": 256,
            "max_search_nodes": 256,
            "max_hits": 8,
            "materialize_top_k": 3,
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
            "run_dir": "outputs/snapback_solves",
            "emit_visual_contracts": True,
            "emit_baserender_jobs": True,
        },
    }


def init_snapback_workspace(target_root: Path, *, force_overwrite: bool = False) -> SnapbackWorkspaceScaffoldResult:
    workspace_root = Path(target_root).expanduser().resolve()
    if workspace_root.exists():
        if not force_overwrite:
            raise ValueError(f"Snapback workspace already exists: {workspace_root}")
        shutil.rmtree(workspace_root)
    configs_dir = workspace_root / "configs" / "snapback"
    inputs_dir = workspace_root / "inputs" / "nickases"
    (workspace_root / "outputs" / "snapback").mkdir(parents=True, exist_ok=True)
    (workspace_root / "outputs" / "snapback_solves").mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = inputs_dir / "local.nickases.yaml"
    example_spec_path = configs_dir / "demo_teto_bpu10i_cap.snapback.yaml"
    example_solve_spec_path = configs_dir / "demo_teto_bpu10i_cap.snapback.solve.yaml"
    readme_path = workspace_root / "README.md"
    manifest_path = _manifest_path(workspace_root)

    catalog_path.write_text(yaml.safe_dump(_catalog_payload(), sort_keys=False), encoding="utf-8")
    example_spec_path.write_text(yaml.safe_dump(_example_spec_payload(), sort_keys=False), encoding="utf-8")
    example_solve_spec_path.write_text(yaml.safe_dump(_example_solve_payload(), sort_keys=False), encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            [
                "# Snapback Workspace",
                "",
                "Scaffolded by `cruncher snapback init-workspace`.",
                "",
                "Included files:",
                "- `configs/snapback/demo_teto_bpu10i_cap.snapback.yaml`",
                "- `configs/snapback/demo_teto_bpu10i_cap.snapback.solve.yaml`",
                "- `inputs/nickases/local.nickases.yaml`",
                "",
                "Suggested next steps:",
                "1. `uv run cruncher snapback validate --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml`",
                "2. `uv run cruncher snapback design --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml`",
                "3. `uv run cruncher snapback solve --spec configs/snapback/demo_teto_bpu10i_cap.snapback.solve.yaml`",
                "",
                "Design bundles emit a three-state QA triptych:",
                "- producer-owned QA JSON views under `views/`",
                "- shared `snapback_visual_v1` contracts under `views/`",
                "- BaseRender job files under `baserender_jobs/`",
                "- rendered PNGs under `renders/` after `uv run baserender job run ...`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    atomic_write_json(
        manifest_path,
        {
            "workflow": "snapback_workspace_scaffold",
            "workspace_root": str(workspace_root),
            "example_spec": str(example_spec_path),
            "example_solve_spec": str(example_solve_spec_path),
            "catalog": str(catalog_path),
        },
    )
    return SnapbackWorkspaceScaffoldResult(
        workspace_root=workspace_root,
        manifest_path=manifest_path,
        readme_path=readme_path,
        example_spec_path=example_spec_path,
        example_solve_spec_path=example_solve_spec_path,
        catalog_path=catalog_path,
    )

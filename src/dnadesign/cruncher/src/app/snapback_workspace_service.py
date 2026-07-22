"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_workspace_service.py

Scaffold v2 explicit and v3 co-design solve snapback workspaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

_WORKSPACE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class SnapbackWorkspaceScaffoldResult:
    workspace_root: Path
    readme_path: Path
    runbook_path: Path
    runbook_config_path: Path
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
                    "vendor": "Thermo Fisher Scientific",
                    "vendor_catalog_number": "ER2011",
                    "selection": {
                        "outside_site": False,
                        "snapback_tier": "tier3",
                        "commercial_confidence": "primary_vendor_current",
                        "warning_codes": ["NONSPECIFIC_NICKING_ASSAY_SIGNAL"],
                    },
                    "operational": {"incubation_temp_c": 37, "buffer_family": "R"},
                    "notes": ["Thermo manual raw geometry: CC↓TNAGC."],
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
            "render_format": "png",
        },
    }


def _example_solve_payload() -> dict[str, object]:
    return {
        "snapback_solve": {
            "schema_version": 3,
            "contract": "single_nick_snapback_solve_v3",
            "name": "demo_teto_catalog_scan",
        },
        "input": {
            "canonical_top_strand": {
                "sequence": "CCTCAGCA",
                "protected_region": {"start": 0, "end": 8},
                "pre_nick_duplex_window": {"start": 0, "end": 8},
            }
        },
        "catalog": {
            "preset": "neb_nicking_v1",
            "additional_presets": ["thermo_nicking_v1"],
            "additional_paths": [],
        },
        "orientation_policy": {"normalize_to_top_strand_nick": True},
        "search": {
            "min_paired_bp": 3,
            "max_added_nt": 5,
            "max_mismatches": 0,
            "max_enumerated_candidates": 4096,
            "max_search_nodes": 4096,
            "max_hits": 8,
            "materialize_top_k": 3,
        },
        "constraints": {},
        "sequence_quality": {
            "gc_fraction": {"min": 0.0, "max": 0.75},
            "max_homopolymer_run": 3,
        },
        "output": {
            "run_dir": "outputs/solve",
            "emit_visual_contracts": True,
            "emit_baserender_jobs": True,
            "render_format": "png",
        },
    }


def _runbook_payload(*, workspace_name: str) -> dict[str, object]:
    return {
        "runbook": {
            "schema_version": 1,
            "name": workspace_name,
            "steps": [
                {
                    "id": "snapback_validate",
                    "description": "Validate the checked-in explicit snapback demo spec.",
                    "run": [
                        "snapback",
                        "validate",
                        "--spec",
                        "configs/snapback/demo_teto_bpu10i_cap.snapback.yaml",
                    ],
                },
                {
                    "id": "snapback_design",
                    "description": "Materialize the checked-in explicit snapback demo bundle.",
                    "run": [
                        "snapback",
                        "design",
                        "--spec",
                        "configs/snapback/demo_teto_bpu10i_cap.snapback.yaml",
                        "--force-overwrite",
                    ],
                },
                {
                    "id": "snapback_show_design",
                    "description": "Inspect the explicit snapback bundle and integrity checks.",
                    "run": ["snapback", "show", "--run", "outputs/design"],
                },
                {
                    "id": "snapback_solve",
                    "description": "Run the broader catalog-scan solve demo.",
                    "run": [
                        "snapback",
                        "solve",
                        "--spec",
                        "configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml",
                        "--force-overwrite",
                    ],
                },
                {
                    "id": "snapback_show_solve",
                    "description": "Inspect the solve bundle, frontier, and materialized hit paths.",
                    "run": ["snapback", "show", "--run", "outputs/solve"],
                },
            ],
        }
    }


def _runbook_markdown(*, workspace_name: str) -> str:
    return "\n".join(
        [
            f"## {workspace_name} Snapback Runbook",
            "",
            "**Workspace Path**",
            f"- src/dnadesign/cruncher/workspaces/{workspace_name}/",
            "",
            "**Purpose**",
            "- Checked-in snapback demo for one explicit Bpu10I design and one broader v3 catalog-scan solve workflow.",
            "- Uses stable workspace output roots under `outputs/design` and `outputs/solve`.",
            (
                "- The explicit lane uses the local `Nt.Bpu10I` overlay; "
                "the solve lane searches built-in `neb_nicking_v1` plus "
                "`thermo_nicking_v1`."
            ),
            (
                "- Keeps materialized solve hits inside "
                "`outputs/solve/analysis/materialized_hits/` instead of nested run-id bundles."
            ),
            "",
            "**Run This Single Command**",
            "",
            f"    uv run cruncher workspaces run --workspace {workspace_name} --runbook configs/runbook.yaml",
            "",
            "### Step-by-Step Commands",
            "",
            "    set -euo pipefail",
            f"    cd src/dnadesign/cruncher/workspaces/{workspace_name}",
            '    cruncher() { uv run cruncher "$@"; }',
            "",
            "    # Standard machine-runbook sequence (matches configs/runbook.yaml).",
            "    cruncher snapback validate --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml",
            "    cruncher snapback design --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml --force-overwrite",
            "    cruncher snapback show --run outputs/design",
            (
                "    cruncher snapback solve --spec "
                "configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml --force-overwrite"
            ),
            "    cruncher snapback show --run outputs/solve",
            "",
            "### Optional follow-up commands",
            "",
            "    uv run baserender job run outputs/design/baserender_jobs/snapback_triptych.job.yaml",
            (
                "    uv run baserender job run "
                "outputs/solve/analysis/materialized_hits/hit_01/"
                "baserender_jobs/snapback_triptych.job.yaml"
            ),
            "    uv run cruncher snapback show --run outputs/design --json",
            "    uv run cruncher snapback show --run outputs/solve --json",
            "",
        ]
    )


def init_snapback_workspace(target_root: Path, *, force_overwrite: bool = False) -> SnapbackWorkspaceScaffoldResult:
    workspace_root = Path(target_root).expanduser().resolve()
    workspace_name = workspace_root.name
    if workspace_root.exists():
        if not force_overwrite:
            raise ValueError(f"Snapback workspace already exists: {workspace_root}")
        shutil.rmtree(workspace_root)
    configs_dir = workspace_root / "configs" / "snapback"
    runbook_config_path = workspace_root / "configs" / "runbook.yaml"
    inputs_dir = workspace_root / "inputs" / "nickases"
    configs_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = inputs_dir / "local.nickases.yaml"
    example_spec_path = configs_dir / "demo_teto_bpu10i_cap.snapback.yaml"
    example_solve_spec_path = configs_dir / "demo_teto_catalog_scan.snapback.solve.yaml"
    readme_path = workspace_root / "README.md"
    runbook_path = workspace_root / "runbook.md"

    catalog_path.write_text(yaml.safe_dump(_catalog_payload(), sort_keys=False), encoding="utf-8")
    example_spec_path.write_text(yaml.safe_dump(_example_spec_payload(), sort_keys=False), encoding="utf-8")
    example_solve_spec_path.write_text(yaml.safe_dump(_example_solve_payload(), sort_keys=False), encoding="utf-8")
    runbook_config_path.write_text(
        yaml.safe_dump(_runbook_payload(workspace_name=workspace_name), sort_keys=False), encoding="utf-8"
    )
    runbook_path.write_text(_runbook_markdown(workspace_name=workspace_name), encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            [
                "# Snapback Workspace",
                "",
                "Scaffolded by `cruncher snapback init-workspace`.",
                "",
                "Included files:",
                "- `configs/runbook.yaml`",
                "- `configs/snapback/demo_teto_bpu10i_cap.snapback.yaml`",
                "- `configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml`",
                "- `inputs/nickases/local.nickases.yaml`",
                "- `runbook.md`",
                "",
                "Canonical single-command refresh:",
                f"- `uv run cruncher workspaces run --workspace {workspace_name} --runbook configs/runbook.yaml`",
                "",
                "Suggested next steps:",
                "1. `uv run cruncher snapback validate --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml`",
                "2. `uv run cruncher snapback design --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml`",
                (
                    "3. `uv run cruncher snapback solve --spec "
                    "configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml`"
                ),
                "",
                "Canonical refresh:",
                (
                    "1. `uv run cruncher snapback design --spec "
                    "configs/snapback/demo_teto_bpu10i_cap.snapback.yaml --force-overwrite`"
                ),
                (
                    "2. `uv run cruncher snapback solve --spec "
                    "configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml --force-overwrite`"
                ),
                "",
                "Workspace-scoped output roots:",
                "- explicit design bundle under `outputs/design/`",
                "- solve summary bundle under `outputs/solve/`",
                "- materialized top hits under `outputs/solve/analysis/materialized_hits/hit_<rank>/`",
                "",
                "Snapback invariants in this scaffold:",
                "- solve uses co-design by default across the resolved nickase catalog",
                "- the solve scaffold resolves built-in `neb_nicking_v1` plus `thermo_nicking_v1`",
                "- the local `Nt.Bpu10I` overlay remains the explicit design example only",
                "- omitted solve boundary and retained-length windows resolve to compact-first defaults",
                "- retained homology starts exactly at the resolved nick boundary",
                "- the effective cap loop is fixed at 3 nt",
                "- pre-nick and exposed visuals use the nick as the single snapback origin boundary",
                "",
                "Design bundles emit a three-state QA triptych:",
                "- producer-owned QA JSON views under `analysis/views/`",
                "- shared `snapback_visual_v1` contracts under `analysis/views/`",
                (
                    "- one composite JSONL triptych contract under "
                    "`analysis/views/snapback_triptych.snapback_visual.v1.jsonl`"
                ),
                "- one BaseRender job under `baserender_jobs/snapback_triptych.job.yaml`",
                "- one rendered `png|svg|pdf` triptych under `plots/` after `uv run baserender job run ...`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return SnapbackWorkspaceScaffoldResult(
        workspace_root=workspace_root,
        readme_path=readme_path,
        runbook_path=runbook_path,
        runbook_config_path=runbook_config_path,
        example_spec_path=example_spec_path,
        example_solve_spec_path=example_solve_spec_path,
        catalog_path=catalog_path,
    )

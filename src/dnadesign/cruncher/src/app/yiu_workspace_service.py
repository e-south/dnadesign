"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workspace_service.py

Scaffold YIU workspaces with a runbook-only explicit workflow family layout.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_WORKSPACE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class YiuWorkspaceScaffoldResult:
    workspace_root: Path
    runbook_path: Path
    spec_path: Path
    restriction_catalog_path: Path
    nickase_catalog_path: Path
    adapter_catalog_path: Path


def _repo_root_from(start: Path) -> Path | None:
    cursor = start.resolve()
    for root in [cursor, *cursor.parents]:
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return None


def default_cruncher_workspaces_root() -> Path:
    repo_root = _repo_root_from(Path(__file__).resolve())
    if repo_root is None:
        raise ValueError(
            "Unable to determine the standard Cruncher workspaces root. Pass --root or --output explicitly."
        )
    return (repo_root / "src" / "dnadesign" / "cruncher" / "workspaces").resolve()


def _validate_workspace_name(name: str) -> str:
    raw = str(name).strip()
    if not raw:
        raise ValueError("YIU workspace name must be non-empty.")
    if "/" in raw or "\\" in raw:
        raise ValueError("YIU workspace name must be a simple directory name or use --output.")
    if _WORKSPACE_NAME_RE.fullmatch(raw) is None:
        raise ValueError(f"Invalid YIU workspace name: {raw!r}.")
    return raw


def yiu_workspace_path(name: str, *, root: Path | None = None) -> Path:
    workspace_name = _validate_workspace_name(name)
    parent = default_cruncher_workspaces_root() if root is None else Path(root).expanduser().resolve()
    return parent / workspace_name


def init_yiu_workspace(workspace_root: Path, *, force_overwrite: bool = False) -> YiuWorkspaceScaffoldResult:
    resolved_root = workspace_root.expanduser().resolve()
    if resolved_root.exists() and any(resolved_root.iterdir()) and not force_overwrite:
        raise ValueError(f"YIU workspace root already exists and is not empty: {resolved_root}")
    if resolved_root.exists() and force_overwrite:
        for child in list(resolved_root.iterdir()):
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
    (resolved_root / "configs" / "yiu").mkdir(parents=True, exist_ok=True)
    (resolved_root / "catalogs").mkdir(parents=True, exist_ok=True)
    (resolved_root / "outputs" / "yiu" / "explicit").mkdir(parents=True, exist_ok=True)
    (resolved_root / "published" / "views").mkdir(parents=True, exist_ok=True)
    (resolved_root / "published" / "jobs").mkdir(parents=True, exist_ok=True)

    runbook_path = resolved_root / "configs" / "runbook.yaml"
    runbook_payload = {
        "runbook": {
            "schema_version": 1,
            "name": resolved_root.name,
            "steps": [
                {"id": "yiu_validate", "run": ["yiu", "validate", "--spec", "configs/yiu/example.yiu.yaml"]},
                {"id": "yiu_design", "run": ["yiu", "design", "--spec", "configs/yiu/example.yiu.yaml"]},
                {"id": "yiu_trace", "run": ["yiu", "trace", "--spec", "configs/yiu/example.yiu.yaml"]},
            ],
        }
    }
    runbook_path.write_text(yaml.safe_dump(runbook_payload, sort_keys=False), encoding="utf-8")

    spec_path = resolved_root / "configs" / "yiu" / "example.yiu.yaml"
    spec_payload = {
        "yiu": {
            "schema_version": 1,
            "protocol": "yiu_v1",
            "name": "example_yiu",
            "source_oligo": {
                "sequence": "AAAAGGTCTCACGTTTAAGGGGCCGGGGTCTCACGTTTTT",
                "primer_sites": [
                    {"id": "fwd_primer", "start": 0, "end": 4, "strand": "primary"},
                    {"id": "rev_primer", "start": 36, "end": 40, "strand": "complement"},
                ],
                "restriction_sites": [
                    {
                        "id": "left_digest",
                        "enzyme": "BsaI",
                        "recognition_sequence": "GGTCTC",
                        "start": 4,
                        "orientation": "forward",
                        "top_cut_offset": 6,
                        "bottom_cut_offset": 10,
                    },
                    {
                        "id": "right_digest",
                        "enzyme": "BsaI",
                        "recognition_sequence": "GGTCTC",
                        "start": 26,
                        "orientation": "forward",
                        "top_cut_offset": 6,
                        "bottom_cut_offset": 10,
                    },
                ],
                "nickase_sites": [
                    {
                        "id": "nick_1",
                        "enzyme": "Nt.Mock",
                        "recognition_sequence": "GGGG",
                        "start": 18,
                        "orientation": "forward",
                        "top_cut_offset": 2,
                    }
                ],
                "payload_windows": [
                    {"id": "left_half", "start": 14, "end": 18},
                    {"id": "right_half", "start": 22, "end": 26},
                ],
                "homology_windows": [
                    {"id": "left_fold", "start": 10, "end": 14},
                    {"id": "right_fold", "start": 32, "end": 36},
                ],
                "retained_regions": [
                    {"id": "retained_left", "start": 14, "end": 18},
                    {"id": "retained_right", "start": 22, "end": 26},
                ],
                "sacrificial_regions": [{"id": "sacrificial_center", "start": 18, "end": 22}],
            },
            "step_graph": {
                "steps": [
                    {
                        "kind": "pcr",
                        "id": "pcr_linear_duplex",
                        "forward_primer_site": "fwd_primer",
                        "reverse_primer_site": "rev_primer",
                    },
                    {
                        "kind": "restriction_digest",
                        "id": "digested_linear_duplex",
                        "left_site": "left_digest",
                        "right_site": "right_digest",
                        "expected_left_overhang": "ACGT",
                        "expected_right_overhang": "ACGT",
                    },
                    {"kind": "circularization", "id": "circularization_candidate", "compatibility": "exact_complement"},
                    {"kind": "exonuclease_selection", "id": "post_exonuclease_enriched_pool"},
                    {
                        "kind": "nickase_digest",
                        "id": "post_nickase_fragmentation",
                        "site_ids": ["nick_1"],
                        "sacrificial_region_ids": ["sacrificial_center"],
                        "retained_region_ids": ["retained_left", "retained_right"],
                    },
                    {"kind": "size_selection", "id": "post_size_selection"},
                    {
                        "kind": "foldback",
                        "id": "foldback_or_cap_intermediate",
                        "left_homology_window": "left_fold",
                        "right_homology_window": "right_fold",
                        "min_complementary_bases": 4,
                    },
                    {"kind": "adapter_ligation", "id": "y_adapter_ligated_product", "adapter_sequence": "AGATCGGA"},
                    {
                        "kind": "amplification",
                        "id": "downstream_amplifiable_product",
                        "forward_primer_requirement": "AGAT",
                        "reverse_primer_requirement": "CCGG",
                    },
                ]
            },
            "payload_goal": {
                "assembled_payload": "TTAACCGG",
                "left_half_ref": "left_half",
                "right_half_ref": "right_half",
                "junction_rule": "contiguous_after_ligation",
            },
            "cleanup_policy": {
                "linear_depletion": {"enabled": True, "enzyme": "T5 exonuclease"},
                "size_selection": {
                    "max_retained_sacrificial_fragment_nt": 4,
                    "min_retained_product_nt": 8,
                },
            },
            "adapter_policy": {
                "adapter_sequence": "AGATCGGA",
                "primer_binding_requirements": [
                    {"id": "amp_fwd", "sequence": "AGAT"},
                    {"id": "amp_rev", "sequence": "CCGG"},
                ],
            },
            "catalogs": {
                "restriction_enzymes": "catalogs/restriction_enzymes.yaml",
                "nickases": "catalogs/nickases.yaml",
                "adapters": "catalogs/adapters.yaml",
            },
            "output": {"run_dir": "outputs/yiu/explicit", "emit_view_contracts": True},
        }
    }
    spec_path.write_text(yaml.safe_dump(spec_payload, sort_keys=False), encoding="utf-8")

    restriction_catalog_path = resolved_root / "catalogs" / "restriction_enzymes.yaml"
    restriction_catalog_path.write_text(
        yaml.safe_dump({"restriction_enzymes": {"entries": [{"id": "BsaI", "recognition_sequence": "GGTCTC"}]}}),
        encoding="utf-8",
    )
    nickase_catalog_path = resolved_root / "catalogs" / "nickases.yaml"
    nickase_catalog_path.write_text(
        yaml.safe_dump({"nickases": {"entries": [{"id": "Nt.Mock", "recognition_sequence": "GGGG"}]}}),
        encoding="utf-8",
    )
    adapter_catalog_path = resolved_root / "catalogs" / "adapters.yaml"
    adapter_catalog_path.write_text(
        yaml.safe_dump({"adapters": {"entries": [{"id": "demo_y_adapter", "sequence": "AGATCGGA"}]}}),
        encoding="utf-8",
    )

    return YiuWorkspaceScaffoldResult(
        workspace_root=resolved_root,
        runbook_path=runbook_path,
        spec_path=spec_path,
        restriction_catalog_path=restriction_catalog_path,
        nickase_catalog_path=nickase_catalog_path,
        adapter_catalog_path=adapter_catalog_path,
    )

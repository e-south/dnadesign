"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workspace_service.py

Scaffold YIU workspaces with the split-payload circularized explicit + solve layout.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

_WORKSPACE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DEMO_EXPLICIT_SPEC_NAME = "example_split_payload_circularized"
_DEMO_EXPLICIT_SPEC_FILENAME = f"{_DEMO_EXPLICIT_SPEC_NAME}.yiu.yaml"
_DEMO_SOLVE_SPEC_FILENAME = f"{_DEMO_EXPLICIT_SPEC_NAME}.yiu.solve.yaml"
_DEMO_EXPLICIT_SPEC_RELATIVE_PATH = f"configs/yiu/{_DEMO_EXPLICIT_SPEC_FILENAME}"
_DEMO_SOLVE_SPEC_RELATIVE_PATH = f"configs/yiu/{_DEMO_SOLVE_SPEC_FILENAME}"
_DEMO_EXPLICIT_RUN_ROOT = f"outputs/yiu/explicit/{_DEMO_EXPLICIT_SPEC_NAME}"
_DEMO_SOLVE_RUN_ROOT = f"outputs/yiu/solve/{_DEMO_EXPLICIT_SPEC_NAME}"


@dataclass(frozen=True)
class YiuWorkspaceScaffoldResult:
    workspace_root: Path
    runbook_path: Path
    runbook_doc_path: Path
    spec_path: Path
    solve_spec_path: Path
    compat_spec_paths: tuple[Path, ...]
    enzyme_catalog_path: Path
    oligo_parts_catalog_path: Path
    backbone_catalog_path: Path


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


def _split_example_spec_payload() -> dict[str, object]:
    return {
        "yiu": {
            "schema_version": 2,
            "family": "yiu",
            "protocol_template": "yiu_circularized_payload_v1",
            "workflow_scope": "core_insert_generation",
            "name": _DEMO_EXPLICIT_SPEC_NAME,
            "source_oligo": {
                "authored_sequence": "ccgatgTCCCTATCAaacgttGTGATAGAGAGGGGGGGGGGGGCCTCAGCCCGCTGA",
                "annotations": {
                    "primer_binding_cores": [
                        {"id": "source_fwd_core", "start": 0, "end": 6, "strand": "primary"},
                        {"id": "source_rev_core", "start": 51, "end": 57, "strand": "complement"},
                    ],
                    "restriction_sites": [
                        {
                            "id": "split_payload_digest",
                            "enzyme": "TypeIIS.Mock",
                            "recognition_sequence": "AACGTT",
                            "start": 15,
                            "orientation": "forward",
                            "top_cut_offset": 1,
                            "bottom_cut_offset": 5,
                        }
                    ],
                    "nickase_sites": [
                        {
                            "id": "nick_1",
                            "enzyme": "Nb.BssSI",
                            "recognition_sequence": "GGGG",
                            "start": 31,
                            "orientation": "forward",
                            "bottom_cut_offset": 4,
                        },
                        {
                            "id": "nick_2",
                            "enzyme": "Nb.BssSI",
                            "recognition_sequence": "GGGG",
                            "start": 35,
                            "orientation": "forward",
                            "bottom_cut_offset": 4,
                        },
                    ],
                    "payload_windows": [
                        {
                            "id": "payload_left",
                            "start": 6,
                            "end": 15,
                            "projection_mode": "compound_required",
                            "annotation_class": "payload_half_left",
                        },
                        {
                            "id": "payload_right",
                            "start": 21,
                            "end": 31,
                            "projection_mode": "compound_required",
                            "annotation_class": "payload_half_right",
                        },
                    ],
                    "retained_regions": [
                        {
                            "id": "retained_payload_left",
                            "start": 6,
                            "end": 15,
                            "projection_mode": "compound_allowed",
                            "annotation_class": "retained_region",
                        },
                        {
                            "id": "retained_payload_right",
                            "start": 21,
                            "end": 31,
                            "projection_mode": "compound_allowed",
                            "annotation_class": "retained_region",
                        },
                    ],
                    "sacrificial_regions": [
                        {
                            "id": "sacrificial_tract",
                            "start": 31,
                            "end": 43,
                            "projection_mode": "atomic_required",
                            "annotation_class": "sacrificial_region",
                        }
                    ],
                    "named_regions": [
                        {
                            "id": "snapback_seed",
                            "start": 43,
                            "end": 57,
                            "projection_mode": "compound_allowed",
                            "annotation_class": "snapback_seed",
                        }
                    ],
                },
            },
            "steps": {
                "source_pcr": {"forward_primer_id": "oES790", "reverse_primer_id": "oES791"},
                "type_iis_digest": {"enzyme_id": "TypeIIS.Mock", "site_ids": ["split_payload_digest"]},
                "circularization": {
                    "ligation_rule": {
                        "mode": "exact_complement",
                        "min_contiguous_core_bp": 4,
                        "max_left_tail_nt": 0,
                        "max_right_tail_nt": 0,
                        "max_bulge_nt": 0,
                        "min_left_flank_bp": 0,
                        "min_right_flank_bp": 0,
                        "bulge_owner": "either",
                    }
                },
                "exonuclease_cleanup": {"enabled": True, "enzyme": "T5 exonuclease"},
                "sacrificial_digest": {
                    "enzyme_ids": ["Nb.BssSI"],
                    "site_ids": ["nick_1", "nick_2"],
                    "sacrificial_region_ids": ["sacrificial_tract"],
                    "retained_region_ids": ["snapback_seed", "retained_payload_left", "retained_payload_right"],
                },
                "fragment_cleanup": {"enabled": True, "max_fragment_nt": 6, "min_retained_nt": 20},
                "snapback_adapter_engagement": {
                    "adapter_id": "oES792",
                    "ligation_rule": {
                        "mode": "partial_complement",
                        "min_contiguous_core_bp": 6,
                        "max_left_tail_nt": 0,
                        "max_right_tail_nt": 0,
                        "max_bulge_nt": 0,
                        "min_left_flank_bp": 0,
                        "min_right_flank_bp": 0,
                        "bulge_owner": "either",
                    },
                },
                "hairpin_ligation": {
                    "ligase": "T4_DNA_ligase",
                    "require_5p_phosphate": True,
                    "ligation_rule": {
                        "mode": "partial_complement",
                        "min_contiguous_core_bp": 6,
                        "max_left_tail_nt": 0,
                        "max_right_tail_nt": 0,
                        "max_bulge_nt": 0,
                        "min_left_flank_bp": 0,
                        "min_right_flank_bp": 0,
                        "bulge_owner": "either",
                    },
                },
                "hairpin_pcr": {"forward_primer_id": "oES793", "reverse_primer_id": "oES794"},
                "insert_cleanup": {"enabled": False},
                "backbone_pcr": {"enabled": False},
                "golden_gate_assembly": {"enabled": False},
            },
            "payload_goal": {
                "assembled_payload_pattern": "TCCCTATCAGTGATAGAGA",
                "left_half_ref": "payload_left",
                "right_half_ref": "payload_right",
                "assembly_space": "circularized_payload_junction",
                "evidence_policy": "require_guaranteed",
            },
            "template_bindings": {
                "source_forward_primer_core_ref": "source_fwd_core",
                "source_reverse_primer_core_ref": "source_rev_core",
                "snapback_seed_region_ref": "snapback_seed",
                "retained_left_region_ref": "retained_payload_left",
                "retained_right_region_ref": "retained_payload_right",
                "primary_sacrificial_region_refs": ["sacrificial_tract"],
                "circularization_left_overhang_ref": "split_payload_digest",
                "circularization_right_overhang_ref": "split_payload_digest",
            },
            "compound_regions": [
                {
                    "id": "assembled_payload",
                    "segments": [
                        {
                            "source_state": "source_oligo_ssdna",
                            "source_region_ref": "payload_left",
                            "orientation": "forward",
                        },
                        {
                            "source_state": "source_oligo_ssdna",
                            "source_region_ref": "payload_right",
                            "orientation": "forward",
                        },
                    ],
                    "join_policy": "junction_assemble",
                }
            ],
            "hard_invariants": [
                {
                    "id": "payload_assembly",
                    "class": "payload_assembly",
                    "transform_ref": "circularization",
                    "space_kind": "assembly_junction",
                    "region_ref": "assembled_payload",
                    "evidence_policy": "require_guaranteed",
                    "params": {"expected_pattern": "TCCCTATCAGTGATAGAGA"},
                },
                {
                    "id": "sacrificial_fragmentation",
                    "class": "sacrificial_fragmentation",
                    "transform_ref": "sacrificial_digest",
                    "space_kind": "fragment_pool",
                    "region_ref": "sacrificial_tract",
                    "strand_scope": "complement",
                    "evidence_policy": "require_guaranteed",
                    "params": {
                        "enzyme_id": "Nb.BssSI",
                        "max_fragment_nt": 6,
                        "min_site_count": 2,
                        "require_full_region_cover": True,
                    },
                },
                {
                    "id": "snapback_exposure",
                    "class": "snapback_exposure",
                    "state_ref": "post_fragment_cleanup",
                    "space_kind": "state_sequence",
                    "region_ref": "snapback_seed",
                    "strand_scope": "primary",
                    "evidence_policy": "require_guaranteed",
                    "params": {
                        "sequence_pattern": "CCTCAGCCCGCTGA",
                        "require_free_five_prime_end": True,
                    },
                },
            ],
            "catalogs": {
                "enzymes": "catalogs/enzymes.yaml",
                "oligo_parts": "catalogs/oligo_parts.yaml",
                "backbones": "catalogs/backbones.yaml",
            },
            "output": {
                "run_dir": "outputs/yiu/explicit",
                "emit_view_contracts": True,
                "emit_baserender_jobs": True,
                "publish_contract_version": 3,
            },
        }
    }


def _split_solve_spec_payload() -> dict[str, object]:
    return {
        "yiu_solve": {
            "schema_version": 1,
            "base_spec": _DEMO_EXPLICIT_SPEC_RELATIVE_PATH,
            "search": {
                "max_hits": 4,
                "materialize_top_k": 2,
                "max_search_nodes": 128,
                "max_enumerated_candidates": 32,
            },
            "variables": {
                "source_windows": [
                    {
                        "id": "payload_left",
                        "span_ref": "payload_left",
                        "alphabet": "iupac_dna",
                        "allowed_patterns": ["TCCCTATCA"],
                    },
                    {
                        "id": "payload_right",
                        "span_ref": "payload_right",
                        "alphabet": "iupac_dna",
                        "allowed_patterns": ["GTGATAGAGA"],
                    },
                ]
            },
            "candidate_policy": {
                "require_guaranteed_hard_invariants": True,
                "forbid_possible_hits": True,
            },
            "output": {
                "run_dir": "outputs/yiu/solve",
                "emit_view_contracts": True,
                "emit_baserender_jobs": True,
                "publish_contract_version": 3,
            },
        }
    }


def _compat_v2_spec_payload() -> dict[str, object]:
    return {
        "yiu": {
            "schema_version": 2,
            "family": "yiu",
            "protocol_template": "yiu_adapter_hairpin_v1",
            "workflow_scope": "core_insert_generation",
            "name": "example_adapter_hairpin",
            "source_oligo": {
                "sequence": "AAAAGGTCTCACGTTTAAGGGGCCGGGGTCTCACGTTTTT",
                "annotations": {
                    "primer_binding_cores": [
                        {"id": "source_fwd_core", "start": 0, "end": 4, "strand": "primary"},
                        {"id": "source_rev_core", "start": 36, "end": 40, "strand": "complement"},
                    ],
                    "primer_tails": [
                        {"id": "source_fwd_tail", "primer_binding_core_id": "source_fwd_core", "sequence": "GG"},
                        {"id": "source_rev_tail", "primer_binding_core_id": "source_rev_core", "sequence": "CC"},
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
                        {"id": "payload_left", "start": 14, "end": 18, "projection_mode": "compound_required"},
                        {"id": "payload_right", "start": 22, "end": 26, "projection_mode": "compound_required"},
                    ],
                    "homology_windows": [
                        {"id": "stem_left", "start": 14, "end": 18, "projection_mode": "compound_allowed"},
                        {"id": "stem_right", "start": 22, "end": 26, "projection_mode": "compound_allowed"},
                    ],
                    "retained_regions": [
                        {"id": "retained_left", "start": 14, "end": 18, "projection_mode": "compound_allowed"},
                        {"id": "retained_right", "start": 22, "end": 26, "projection_mode": "compound_allowed"},
                    ],
                    "sacrificial_regions": [
                        {"id": "sacrificial_center", "start": 18, "end": 22, "projection_mode": "atomic_required"}
                    ],
                },
            },
            "steps": {
                "source_pcr": {"forward_primer_id": "oES790", "reverse_primer_id": "oES791"},
                "double_nicking_digest": {"enzymes": ["Nt.Mock"]},
                "heat_cleanup": {"enabled": True, "min_retained_nt": 8},
                "adapter_anneal": {
                    "adapter_id": "oES792",
                    "compatibility_mode": "partial_complement",
                    "partial_complement": {
                        "min_paired_nt": 4,
                        "allow_left_tail": True,
                        "allow_right_tail": True,
                    },
                },
                "hairpin_ligation": {
                    "ligase": "T4_DNA_ligase",
                    "require_5p_phosphate": True,
                    "compatibility_mode": "partial_complement",
                    "partial_complement": {
                        "min_paired_nt": 4,
                        "allow_left_tail": True,
                        "allow_right_tail": True,
                    },
                },
                "hairpin_pcr": {"forward_primer_id": "oES793", "reverse_primer_id": "oES794"},
                "insert_cleanup": {"enabled": False},
                "backbone_pcr": {"enabled": False},
                "golden_gate_assembly": {"enabled": False},
            },
            "payload_goal": {
                "assembled_payload_pattern": "TTAACCGG",
                "left_half_ref": "payload_left",
                "right_half_ref": "payload_right",
                "assembly_space": "post_ligation",
                "evidence_policy": "require_guaranteed",
            },
            "catalogs": {
                "enzymes": "catalogs/enzymes.yaml",
                "oligo_parts": "catalogs/oligo_parts.yaml",
                "backbones": "catalogs/backbones.yaml",
            },
            "output": {
                "run_dir": "outputs/yiu/explicit",
                "emit_view_contracts": True,
                "publish_contract_version": 2,
            },
        }
    }


def _compat_v1_spec_payload() -> dict[str, object]:
    return {
        "yiu": {
            "schema_version": 1,
            "protocol": "yiu_v1",
            "name": "example_legacy_v1",
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
                    {"id": "left_fold", "start": 14, "end": 18},
                    {"id": "right_fold", "start": 14, "end": 18},
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
            "output": {"run_dir": "outputs/yiu/explicit", "emit_view_contracts": True},
        }
    }


def _catalog_payloads() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return (
        {
            "enzymes": {
                "entries": [
                    {
                        "id": "TypeIIS.Mock",
                        "recognition_sequence": "AACGTT",
                        "top_cut_offset": 1,
                        "bottom_cut_offset": 5,
                    },
                    {"id": "Nb.BssSI", "recognition_sequence": "GGGG", "bottom_cut_offset": 4},
                    {"id": "Nt.Mock", "recognition_sequence": "GGGG", "top_cut_offset": 2},
                ]
            }
        },
        {
            "oligo_parts": {
                "entries": [
                    {"id": "oES790", "part_kind": "primer", "sequence": "GGAAAAAA"},
                    {"id": "oES791", "part_kind": "primer", "sequence": "TTTTTTCC"},
                    {"id": "oES792", "part_kind": "adapter", "sequence": "TCAGCGGGCTGAGG", "phosphorylated_5p": True},
                    {"id": "oES793", "part_kind": "primer", "sequence": "TCCCTA"},
                    {"id": "oES794", "part_kind": "primer", "sequence": "TCAGCG"},
                ]
            }
        },
        {"backbones": {"entries": []}},
    )


def _runbook_steps() -> list[dict[str, object]]:
    return [
        {
            "id": "yiu_validate",
            "description": "Validate the checked-in split-payload circularized YIU demo spec.",
            "run": ["yiu", "validate", "--spec", _DEMO_EXPLICIT_SPEC_RELATIVE_PATH],
        },
        {
            "id": "yiu_design",
            "description": "Materialize the explicit YIU bundle and published visual contracts.",
            "run": [
                "yiu",
                "design",
                "--spec",
                _DEMO_EXPLICIT_SPEC_RELATIVE_PATH,
                "--force-overwrite",
            ],
        },
        {
            "id": "yiu_trace",
            "description": "Re-materialize the explicit bundle under trace intent for QA parity.",
            "run": [
                "yiu",
                "trace",
                "--spec",
                _DEMO_EXPLICIT_SPEC_RELATIVE_PATH,
                "--force-overwrite",
            ],
        },
        {
            "id": "yiu_solve",
            "description": "Run the paired solve spec and materialize the top accepted hit bundles.",
            "run": [
                "yiu",
                "solve",
                "--spec",
                _DEMO_SOLVE_SPEC_RELATIVE_PATH,
                "--force-overwrite",
            ],
        },
    ]


def _write_runbook_markdown(workspace_root: Path, *, runbook_path: Path) -> Path:
    workspace_name = workspace_root.name
    repo_root = _repo_root_from(workspace_root)
    if repo_root is not None:
        try:
            workspace_display_path = workspace_root.relative_to(repo_root).as_posix()
        except ValueError:
            workspace_display_path = workspace_root.as_posix()
    else:
        workspace_display_path = workspace_root.as_posix()
    workspace_root_arg = shlex.quote(workspace_display_path)
    workspace_name_arg = shlex.quote(workspace_name)
    runbook_arg = shlex.quote(runbook_path.relative_to(workspace_root).as_posix())
    lines = [
        f"## {workspace_name} YIU Runbook",
        "",
        "**Workspace Path**",
        f"- {workspace_display_path}/",
        "",
        "**Purpose**",
        "- Checked-in YIU demo for the split-payload circularized flow.",
        "- Covers validate, explicit materialization, trace-alias materialization, and solve from one repo workspace.",
        "",
        "**Run This Single Command**",
        "",
        (f"    uv run cruncher workspaces run --workspace {workspace_name_arg} --runbook {runbook_arg}"),
        "",
        "### Step-by-Step Commands",
        "",
        "    set -euo pipefail",
        f"    cd {workspace_root_arg}",
        '    cruncher() { uv run cruncher "$@"; }',
        "",
        "    # Standard machine-runbook sequence (matches configs/runbook.yaml).",
        f"    cruncher yiu validate --spec {_DEMO_EXPLICIT_SPEC_RELATIVE_PATH}",
        f"    cruncher yiu design --spec {_DEMO_EXPLICIT_SPEC_RELATIVE_PATH} --force-overwrite",
        f"    cruncher yiu trace --spec {_DEMO_EXPLICIT_SPEC_RELATIVE_PATH} --force-overwrite",
        f"    cruncher yiu solve --spec {_DEMO_SOLVE_SPEC_RELATIVE_PATH} --force-overwrite",
        "",
        "### Optional follow-up commands",
        "",
        f'    DESIGN_ID="$(ls -1 {_DEMO_EXPLICIT_RUN_ROOT} | tail -n 1)"',
        f'    SOLVE_ID="$(ls -1 {_DEMO_SOLVE_RUN_ROOT} | tail -n 1)"',
        f'    uv run cruncher yiu show --run "{_DEMO_EXPLICIT_RUN_ROOT}/$DESIGN_ID"',
        f'    uv run cruncher yiu show --run "{_DEMO_SOLVE_RUN_ROOT}/$SOLVE_ID"',
        (
            f"    uv run cruncher visuals validate --job "
            f'"{_DEMO_EXPLICIT_RUN_ROOT}/$DESIGN_ID/'
            'published/baserender_jobs/circularized_payload_candidate.job.yaml"'
        ),
        (
            f"    uv run cruncher visuals run --job "
            f'"{_DEMO_EXPLICIT_RUN_ROOT}/$DESIGN_ID/'
            'published/baserender_jobs/circularized_payload_candidate.job.yaml"'
        ),
        "",
    ]
    runbook_doc_path = workspace_root / "runbook.md"
    runbook_doc_path.write_text("\n".join(lines), encoding="utf-8")
    return runbook_doc_path


def init_yiu_workspace(workspace_root: Path, *, force_overwrite: bool = False) -> YiuWorkspaceScaffoldResult:
    resolved_root = workspace_root.expanduser().resolve()
    if resolved_root.exists() and any(resolved_root.iterdir()) and not force_overwrite:
        raise ValueError(f"YIU workspace root already exists and is not empty: {resolved_root}")
    if resolved_root.exists() and force_overwrite:
        shutil.rmtree(resolved_root)

    (resolved_root / "configs" / "yiu" / "compat").mkdir(parents=True, exist_ok=True)
    (resolved_root / "catalogs").mkdir(parents=True, exist_ok=True)
    (resolved_root / "outputs" / "yiu" / "explicit").mkdir(parents=True, exist_ok=True)
    (resolved_root / "outputs" / "yiu" / "solve").mkdir(parents=True, exist_ok=True)

    runbook_path = resolved_root / "configs" / "runbook.yaml"
    runbook_payload = {
        "runbook": {
            "schema_version": 1,
            "name": resolved_root.name,
            "steps": _runbook_steps(),
        }
    }
    runbook_path.write_text(yaml.safe_dump(runbook_payload, sort_keys=False), encoding="utf-8")
    runbook_doc_path = _write_runbook_markdown(resolved_root, runbook_path=runbook_path)

    spec_path = resolved_root / "configs" / "yiu" / _DEMO_EXPLICIT_SPEC_FILENAME
    spec_path.write_text(yaml.safe_dump(_split_example_spec_payload(), sort_keys=False), encoding="utf-8")

    solve_spec_path = resolved_root / "configs" / "yiu" / _DEMO_SOLVE_SPEC_FILENAME
    solve_spec_path.write_text(yaml.safe_dump(_split_solve_spec_payload(), sort_keys=False), encoding="utf-8")

    compat_v2_path = resolved_root / "configs" / "yiu" / "compat" / "example_adapter_hairpin.yiu.yaml"
    compat_v2_path.write_text(yaml.safe_dump(_compat_v2_spec_payload(), sort_keys=False), encoding="utf-8")

    compat_v1_path = resolved_root / "configs" / "yiu" / "compat" / "example_legacy_v1.yiu.yaml"
    compat_v1_path.write_text(yaml.safe_dump(_compat_v1_spec_payload(), sort_keys=False), encoding="utf-8")

    enzymes_payload, oligo_parts_payload, backbones_payload = _catalog_payloads()
    enzyme_catalog_path = resolved_root / "catalogs" / "enzymes.yaml"
    enzyme_catalog_path.write_text(yaml.safe_dump(enzymes_payload, sort_keys=False), encoding="utf-8")
    oligo_parts_catalog_path = resolved_root / "catalogs" / "oligo_parts.yaml"
    oligo_parts_catalog_path.write_text(yaml.safe_dump(oligo_parts_payload, sort_keys=False), encoding="utf-8")
    backbone_catalog_path = resolved_root / "catalogs" / "backbones.yaml"
    backbone_catalog_path.write_text(yaml.safe_dump(backbones_payload, sort_keys=False), encoding="utf-8")

    return YiuWorkspaceScaffoldResult(
        workspace_root=resolved_root,
        runbook_path=runbook_path,
        runbook_doc_path=runbook_doc_path,
        spec_path=spec_path,
        solve_spec_path=solve_spec_path,
        compat_spec_paths=(compat_v2_path, compat_v1_path),
        enzyme_catalog_path=enzyme_catalog_path,
        oligo_parts_catalog_path=oligo_parts_catalog_path,
        backbone_catalog_path=backbone_catalog_path,
    )

"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/yiu/test_solve.py

Solve-surface contracts for the YIU workflow family.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from dnadesign.cruncher.app.yiu_solve_workflow import run_yiu_solve, solve_yiu_spec
from dnadesign.cruncher.app.yiu_workflow import yiu_show_payload


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _split_explicit_payload() -> dict[str, object]:
    return {
        "yiu": {
            "schema_version": 2,
            "family": "yiu",
            "protocol_template": "yiu_split_payload_circularized_v1",
            "workflow_scope": "core_insert_generation",
            "name": "demo_yiu_split_v2",
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
                "hairpin_pcr": {
                    "forward_primer_id": "oES793",
                    "reverse_primer_id": "oES794",
                },
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


def _solve_payload(
    *,
    emit_view_contracts: bool = True,
    emit_baserender_jobs: bool = True,
) -> dict[str, object]:
    return {
        "yiu_solve": {
            "schema_version": 1,
            "base_spec": "configs/yiu/example.yiu.yaml",
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
                "emit_view_contracts": emit_view_contracts,
                "emit_baserender_jobs": emit_baserender_jobs,
                "publish_contract_version": 3,
            },
        }
    }


def _write_workspace(
    tmp_path: Path,
    *,
    emit_view_contracts: bool = True,
    emit_baserender_jobs: bool = True,
) -> tuple[Path, Path]:
    workspace = tmp_path / "workspaces" / "demo_yiu_split_solve"
    explicit_spec_path = workspace / "configs" / "yiu" / "example.yiu.yaml"
    _write_yaml(explicit_spec_path, _split_explicit_payload())
    _write_yaml(
        workspace / "configs" / "yiu" / "example.yiu.solve.yaml",
        _solve_payload(
            emit_view_contracts=emit_view_contracts,
            emit_baserender_jobs=emit_baserender_jobs,
        ),
    )
    _write_yaml(
        workspace / "catalogs" / "enzymes.yaml",
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
                ]
            }
        },
    )
    _write_yaml(
        workspace / "catalogs" / "oligo_parts.yaml",
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
    )
    _write_yaml(workspace / "catalogs" / "backbones.yaml", {"backbones": {"entries": []}})
    return workspace, workspace / "configs" / "yiu" / "example.yiu.solve.yaml"


_PRESSURE_PATTERNS = [
    "AA",
    "AC",
    "AG",
    "AT",
    "CA",
    "CC",
    "CG",
    "CT",
    "GA",
    "GC",
    "GG",
    "GT",
    "TA",
    "TC",
    "TG",
    "TT",
]


def _write_pressure_workspace(
    tmp_path: Path,
    *,
    max_hits: int,
    materialize_top_k: int,
    max_search_nodes: int = 256,
    max_enumerated_candidates: int = 256,
    emit_view_contracts: bool = False,
    emit_baserender_jobs: bool = False,
) -> tuple[Path, Path]:
    workspace, solve_spec_path = _write_workspace(
        tmp_path,
        emit_view_contracts=emit_view_contracts,
        emit_baserender_jobs=emit_baserender_jobs,
    )
    explicit_spec_path = workspace / "configs" / "yiu" / "example.yiu.yaml"
    explicit_payload = copy.deepcopy(_split_explicit_payload())
    explicit_payload["yiu"]["name"] = "demo_yiu_split_pressure"
    explicit_payload["yiu"]["source_oligo"]["annotations"]["named_regions"].append(
        {
            "id": "neutral_prefix",
            "start": 0,
            "end": 2,
            "projection_mode": "compound_allowed",
            "annotation_class": "neutral_region",
        }
    )
    _write_yaml(explicit_spec_path, explicit_payload)

    solve_payload = _solve_payload(
        emit_view_contracts=emit_view_contracts,
        emit_baserender_jobs=emit_baserender_jobs,
    )
    solve_payload["yiu_solve"]["search"] = {
        "max_hits": max_hits,
        "materialize_top_k": materialize_top_k,
        "max_search_nodes": max_search_nodes,
        "max_enumerated_candidates": max_enumerated_candidates,
    }
    solve_payload["yiu_solve"]["variables"]["source_windows"].append(
        {
            "id": "neutral_prefix",
            "span_ref": "neutral_prefix",
            "alphabet": "iupac_dna",
            "allowed_patterns": list(_PRESSURE_PATTERNS),
        }
    )
    _write_yaml(solve_spec_path, solve_payload)
    return workspace, solve_spec_path


def test_run_yiu_solve_materializes_top_hit_bundle_and_visual_inventory(tmp_path: Path) -> None:
    workspace, solve_spec_path = _write_workspace(tmp_path)

    run_dir, report = run_yiu_solve(solve_spec_path)

    assert run_dir is not None
    assert report.status == "solved"
    assert report.solve_id is not None
    assert len(report.hits) == 1
    assert report.metadata.accepted_candidate_count == 1
    assert report.metadata.materialized_hit_count == 1
    assert run_dir == Path(report.run_dir)
    assert (run_dir / "yiu_solve_report.json").exists()
    assert (run_dir / "yiu_solve_status.json").exists()
    assert (run_dir / "yiu_solve_manifest.json").exists()
    assert (run_dir / "accepted_hits.jsonl").exists()
    assert (run_dir / "hits.csv").exists()
    assert (run_dir / "published" / "visual_manifest.json").exists()
    assert (run_dir / "published" / "baserender_jobs").is_dir()
    assert (run_dir / "published" / "renders").is_dir()

    hit_dir = run_dir / "hits" / "hit_0001"
    assert hit_dir.is_dir()
    assert (hit_dir / "yiu_manifest.json").exists()
    assert (hit_dir / "published" / "visual_manifest.json").exists()
    accepted_hits = [
        json.loads(line) for line in (run_dir / "accepted_hits.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert accepted_hits[0]["rank"] == 1
    assert accepted_hits[0]["materialized_run_dir"].endswith("hits/hit_0001")

    solve_payload = yiu_show_payload(run_dir)
    explicit_payload = yiu_show_payload(hit_dir)

    assert solve_payload["bundle_kind"] == "solve"
    assert solve_payload["visual_manifest_path"].endswith("published/visual_manifest.json")
    assert solve_payload["published_jobs_dir"].endswith("published/baserender_jobs")
    assert explicit_payload["bundle_kind"] == "explicit"
    assert explicit_payload["visual_manifest_path"].endswith("published/visual_manifest.json")
    assert str(run_dir).startswith(str(workspace / "outputs" / "yiu" / "solve"))


def test_run_yiu_solve_without_view_publication_skips_visual_surface(tmp_path: Path) -> None:
    workspace, solve_spec_path = _write_workspace(
        tmp_path,
        emit_view_contracts=False,
        emit_baserender_jobs=False,
    )

    run_dir, report = run_yiu_solve(solve_spec_path)

    assert run_dir is not None
    assert report.status == "solved"
    assert not (run_dir / "published" / "visual_manifest.json").exists()
    assert not (run_dir / "published" / "views").exists()
    assert not (run_dir / "published" / "baserender_jobs").exists()
    assert not (run_dir / "published" / "renders").exists()
    hit_dir = run_dir / "hits" / "hit_0001"
    assert hit_dir.is_dir()
    assert not (hit_dir / "published" / "visual_manifest.json").exists()

    solve_payload = yiu_show_payload(run_dir)

    assert solve_payload["bundle_kind"] == "solve"
    assert solve_payload["visual_manifest_path"] is None
    assert solve_payload["published_views_dir"] is None
    assert solve_payload["published_jobs_dir"] is None
    assert solve_payload["published_renders_dir"] is None
    assert str(run_dir).startswith(str(workspace / "outputs" / "yiu" / "solve"))


def test_run_yiu_solve_without_baserender_jobs_keeps_views_only(tmp_path: Path) -> None:
    _workspace, solve_spec_path = _write_workspace(
        tmp_path,
        emit_view_contracts=True,
        emit_baserender_jobs=False,
    )

    run_dir, report = run_yiu_solve(solve_spec_path)

    assert run_dir is not None
    assert report.status == "solved"
    assert (run_dir / "published" / "visual_manifest.json").exists()
    assert (run_dir / "published" / "views").is_dir()
    assert not (run_dir / "published" / "baserender_jobs").exists()
    assert not (run_dir / "published" / "renders").exists()

    solve_payload = yiu_show_payload(run_dir)

    assert solve_payload["visual_manifest_path"].endswith("published/visual_manifest.json")
    assert solve_payload["published_views_dir"].endswith("published/views")
    assert solve_payload["published_jobs_dir"] is None
    assert solve_payload["published_renders_dir"] is None


def test_solve_yiu_spec_retains_global_top_hits_when_acceptance_frontier_exceeds_max_hits(tmp_path: Path) -> None:
    _workspace, solve_spec_path = _write_pressure_workspace(
        tmp_path,
        max_hits=16,
        materialize_top_k=0,
    )

    exhaustive_report, *_ = solve_yiu_spec(solve_spec_path)

    solve_payload = yaml.safe_load(solve_spec_path.read_text(encoding="utf-8"))
    solve_payload["yiu_solve"]["search"]["max_hits"] = 4
    solve_payload["yiu_solve"]["search"]["materialize_top_k"] = 0
    _write_yaml(solve_spec_path, solve_payload)

    bounded_report, *_ = solve_yiu_spec(solve_spec_path)

    assert exhaustive_report.status == "solved"
    assert bounded_report.status == "solved"
    assert exhaustive_report.metadata.accepted_candidate_count == len(_PRESSURE_PATTERNS)
    assert exhaustive_report.metadata.returned_hit_count == len(_PRESSURE_PATTERNS)
    assert bounded_report.metadata.accepted_candidate_count == len(_PRESSURE_PATTERNS)
    assert bounded_report.metadata.returned_hit_count == 4
    assert bounded_report.metadata.accepted_pool_truncated is True
    assert [hit.source_sequence for hit in bounded_report.hits] == [
        hit.source_sequence for hit in exhaustive_report.hits[:4]
    ]
    assert [tuple(hit.score) for hit in bounded_report.hits] == sorted(tuple(hit.score) for hit in bounded_report.hits)


def test_run_yiu_solve_status_surface_marks_truncated_search_and_pool_bounding(tmp_path: Path) -> None:
    _workspace, solve_spec_path = _write_pressure_workspace(
        tmp_path,
        max_hits=4,
        materialize_top_k=2,
        max_enumerated_candidates=8,
    )

    run_dir, report = run_yiu_solve(solve_spec_path)

    assert run_dir is not None
    assert report.status == "solved"
    assert report.metadata.search_truncated is True
    assert report.metadata.accepted_pool_truncated is True
    assert report.metadata.accepted_candidate_count == 8
    assert report.metadata.returned_hit_count == 4
    assert report.metadata.materialized_hit_count == 2
    assert report.metadata.warning_codes == ["MAX_ENUMERATED_CANDIDATES_REACHED"]
    assert len(report.hits) == 4

    status_payload = json.loads((run_dir / "yiu_solve_status.json").read_text(encoding="utf-8"))
    assert status_payload["search_truncated"] is True
    assert status_payload["accepted_pool_truncated"] is True
    assert status_payload["accepted_candidate_count"] == 8
    assert status_payload["returned_hit_count"] == 4
    assert status_payload["materialized_hit_count"] == 2
    assert status_payload["warning_codes"] == ["MAX_ENUMERATED_CANDIDATES_REACHED"]

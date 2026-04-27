"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/snapback/test_solver.py

Bounded solve tests for v3 co-design snapback workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from dnadesign.cruncher.app.snapback_solve_workflow import run_snapback_solve


def _write_workspace(
    tmp_path: Path,
    *,
    spec_payload: dict[str, object],
    catalog_entries: list[dict[str, object]],
) -> Path:
    workspace = tmp_path / "workspaces" / "demo_snapback"
    spec_path = workspace / "configs" / "snapback" / "demo.snapback.solve.yaml"
    catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        yaml.safe_dump({"nickases": {"schema_version": 1, "entries": catalog_entries}}, sort_keys=False),
        encoding="utf-8",
    )
    spec_path.write_text(yaml.safe_dump(spec_payload, sort_keys=False), encoding="utf-8")
    return spec_path


def _catalog_entries() -> list[dict[str, object]]:
    return [
        {
            "id": "Nt.Bpu10I",
            "specificity_id": "Bpu10I",
            "motif_top_5to3": "CCTNAGC",
            "top_cut_offset": 2,
            "source": "demo",
        }
    ]


def _base_payload() -> dict[str, object]:
    return {
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
        "output": {"run_dir": "outputs/solve", "emit_visual_contracts": True},
    }


def test_run_snapback_solve_finds_ranked_hits_and_materializes_top_k(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path, spec_payload=_base_payload(), catalog_entries=_catalog_entries())

    run_dir, report = run_snapback_solve(spec_path)

    assert report.status == "satisfied"
    assert len(report.hits) == 4
    assert report.hits[0].variant_id == "Nt.Bpu10I"
    assert report.hits[0].nickase.nicked_strand == "top"
    assert report.hits[0].nickase.active_cut_offset == 2
    assert report.hits[0].intended_site_sequence == "CCTCAGC"
    assert report.hits[0].cap_sequence == "A"
    assert report.hits[0].nick_boundary_from_left == 2
    assert report.hits[0].site_mutation_count == 0
    assert report.hits[0].retained_start_from_nick == 0
    assert report.hits[0].cap_nt == 3
    assert report.hits[0].cap_extension_nt == 1
    assert report.metadata.materialized_hit_count == 2
    assert (run_dir / "analysis" / "reports" / "solve_report.json").exists()
    assert (run_dir / "analysis" / "reports" / "solve_report.md").exists()
    assert (run_dir / "export" / "table__hits.csv").exists()
    assert (run_dir / "export" / "table__frontier.csv").exists()
    assert (run_dir / "meta" / "solve_manifest.json").exists()
    assert (run_dir / "meta" / "solve_status.json").exists()
    assert (run_dir / "provenance" / "input_solve_spec.yaml").exists()
    assert (run_dir / "provenance" / "resolved_catalog.yaml").exists()
    assert run_dir == spec_path.parent.parent.parent / "outputs" / "solve"
    materialized_dirs = sorted((run_dir / "analysis" / "materialized_hits").iterdir())
    assert len(materialized_dirs) == 2
    assert materialized_dirs[0].name == "hit_01"
    assert materialized_dirs[1].name == "hit_02"
    first_payload = json.loads(
        (materialized_dirs[0] / "analysis" / "reports" / "report.json").read_text(encoding="utf-8")
    )
    assert first_payload["candidate"]["cap_sequence"] == "A"
    assert first_payload["candidate"]["foldback_arm"] == report.hits[0].foldback_arm
    views_manifest = json.loads(
        (materialized_dirs[0] / "analysis" / "views" / "views_manifest.v1.json").read_text(encoding="utf-8")
    )
    pre_visual = json.loads(
        (materialized_dirs[0] / "analysis" / "views" / "pre_nick_duplex.snapback_visual.v1.json").read_text(
            encoding="utf-8"
        )
    )
    foldback_visual = json.loads(
        (materialized_dirs[0] / "analysis" / "views" / "post_nick_foldback.snapback_visual.v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert pre_visual["state_id"].startswith(f"{views_manifest['solution_id']}.")
    assert foldback_visual["state_id"] == f"{views_manifest['solution_id']}.post_nick_foldback"
    assert foldback_visual["primary_sequence"] == first_payload["candidate"]["post_nick_sequence"]
    assert foldback_visual["meta"]["cap_extension_nt"] == report.hits[0].cap_extension_nt
    assert foldback_visual["meta"]["terminal_ligatable_duplex_bp"] == report.hits[0].terminal_ligatable_duplex_bp
    hits_table = (run_dir / "export" / "table__hits.csv").read_text(encoding="utf-8")
    assert "intended_site_orientation" in hits_table
    assert ",forward," in hits_table
    report_md = (materialized_dirs[0] / "analysis" / "reports" / "report.md").read_text(encoding="utf-8")
    assert "nick_boundary_from_left: 2" in report_md
    assert "extra_nick_event_count: 0" in report_md


def test_run_snapback_solve_defaults_to_compact_ranges_and_reports_frontier(tmp_path: Path) -> None:
    payload = _base_payload()
    payload.pop("goal")
    payload["search"].pop("retained_homology_length")
    payload["search"]["min_paired_bp"] = 3
    payload["search"]["max_enumerated_candidates"] = 4096
    payload["search"]["max_search_nodes"] = 4096
    payload["constraints"].pop("terminal_ligatable_duplex_bp")
    payload["constraints"].pop("max_uninterrupted_duplex_bp")
    spec_path = _write_workspace(tmp_path, spec_payload=payload, catalog_entries=_catalog_entries())

    run_dir, report = run_snapback_solve(spec_path)

    assert report.status == "satisfied"
    assert report.hits[0].nick_boundary == 2
    assert report.hits[0].paired_bp == 3
    assert report.hits[0].cap_extension_nt == 0
    assert report.metadata.resolved_search_space.nick_boundary_window.min == 0
    assert report.metadata.resolved_search_space.nick_boundary_window.max == 8
    assert report.metadata.resolved_search_space.retained_homology_length.min == 3
    assert report.metadata.resolved_search_space.retained_homology_length.max == 8
    assert report.metadata.frontier_row_count >= 1
    assert report.metadata.first_satisfied_frontier is not None
    assert report.metadata.first_satisfied_frontier.nick_boundary_from_left == 2
    assert report.metadata.first_satisfied_frontier.paired_bp == 3
    frontier_table = (run_dir / "export" / "table__frontier.csv").read_text(encoding="utf-8")
    assert "nick_boundary_from_left,paired_bp,cap_extension_nt" in frontier_table
    assert "2,3,0," in frontier_table


def test_run_snapback_solve_marks_truncation_when_bounds_are_hit(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["search"]["max_enumerated_candidates"] = 1
    payload["search"]["max_search_nodes"] = 1
    spec_path = _write_workspace(tmp_path, spec_payload=payload, catalog_entries=_catalog_entries())

    run_dir, report = run_snapback_solve(spec_path)

    assert report.status == "search_truncated"
    assert report.metadata.search_truncated is True
    assert report.metadata.enumerated_candidate_count == 1
    status_payload = json.loads((run_dir / "meta" / "solve_status.json").read_text(encoding="utf-8"))
    assert status_payload["status"] == "search_truncated"


def test_run_snapback_solve_prefers_lower_gc_when_gc_bounds_are_omitted(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["sequence_quality"]["gc_fraction"] = None
    spec_path = _write_workspace(tmp_path, spec_payload=payload, catalog_entries=_catalog_entries())

    _run_dir, report = run_snapback_solve(spec_path)

    assert report.status == "satisfied"
    assert [hit.cap_sequence for hit in report.hits] == ["A", "T", "G", "C"]


def test_run_snapback_solve_does_not_bias_in_range_gc_tie_breaks_to_the_lower_bound(tmp_path: Path) -> None:
    omitted_bounds = _base_payload()
    omitted_bounds["sequence_quality"]["gc_fraction"] = None
    omitted_path = _write_workspace(
        tmp_path / "omitted",
        spec_payload=omitted_bounds,
        catalog_entries=_catalog_entries(),
    )

    explicit_bounds = _base_payload()
    explicit_bounds["sequence_quality"]["gc_fraction"] = {"min": 0.0, "max": 1.0}
    explicit_path = _write_workspace(
        tmp_path / "explicit",
        spec_payload=explicit_bounds,
        catalog_entries=_catalog_entries(),
    )

    _omitted_run_dir, omitted_report = run_snapback_solve(omitted_path)
    _explicit_run_dir, explicit_report = run_snapback_solve(explicit_path)

    assert omitted_report.status == "satisfied"
    assert explicit_report.status == "satisfied"
    assert [hit.cap_sequence for hit in explicit_report.hits] == [hit.cap_sequence for hit in omitted_report.hits]
    assert [hit.cap_sequence for hit in explicit_report.hits] == ["A", "T", "G", "C"]


def test_run_snapback_solve_codesigns_recognition_site_across_catalog_entries(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["input"]["canonical_top_strand"]["sequence"] = "AAAAAAAT"
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 0, "end": 8}
    payload["input"]["canonical_top_strand"]["pre_nick_duplex_window"] = {"start": 0, "end": 8}
    payload["goal"]["nick_boundary_window"] = {"min": 1, "max": 3}
    payload["search"]["max_hits"] = 2
    payload["search"]["materialize_top_k"] = 1
    spec_path = _write_workspace(
        tmp_path,
        spec_payload=payload,
        catalog_entries=[
            {
                "id": "Nt.Late",
                "specificity_id": "Late",
                "motif_top_5to3": "GGGG",
                "top_cut_offset": 3,
                "source": "demo",
            },
            {
                "id": "Nt.Early",
                "specificity_id": "Early",
                "motif_top_5to3": "ATGC",
                "top_cut_offset": 1,
                "source": "demo",
            },
        ],
    )

    run_dir, report = run_snapback_solve(spec_path)

    assert report.status == "satisfied"
    assert report.hits[0].variant_id == "Nt.Early"
    assert report.hits[0].nick_boundary == 1
    assert report.hits[0].intended_site_orientation == "forward"
    assert report.hits[0].intended_site_sequence == "ATGC"
    assert report.hits[0].site_mutation_count == 3
    assert report.hits[0].materialized_run_dir == "outputs/solve/analysis/materialized_hits/hit_01"
    materialized_run_dir = spec_path.parent.parent.parent / report.hits[0].materialized_run_dir
    materialized_spec = (materialized_run_dir / "provenance" / "spec_used.yaml").read_text(encoding="utf-8")
    assert "sequence: ATGCAAAT" in materialized_spec
    assert "variant_id: Nt.Early" in materialized_spec
    assert "run_dir: outputs/solve/analysis/materialized_hits/hit_01" in materialized_spec


def test_run_snapback_solve_broad_preset_catalog_respects_outside_site_feasibility(tmp_path: Path) -> None:
    payload = _base_payload()
    payload.pop("goal")
    payload["search"].pop("retained_homology_length")
    payload["search"]["min_paired_bp"] = 3
    payload["search"]["max_enumerated_candidates"] = 4096
    payload["search"]["max_search_nodes"] = 4096
    payload["search"]["max_hits"] = 8
    payload["search"]["materialize_top_k"] = 1
    payload["constraints"] = {}
    payload["catalog"] = {
        "preset": "neb_nicking_v1",
        "additional_presets": ["thermo_nicking_v1"],
        "additional_paths": [],
    }
    spec_path = _write_workspace(tmp_path, spec_payload=payload, catalog_entries=_catalog_entries())

    _run_dir, report = run_snapback_solve(spec_path)

    assert report.status == "satisfied"
    assert report.metadata.catalog_presets == ["neb_nicking_v1", "thermo_nicking_v1"]
    assert report.metadata.first_satisfied_frontier is not None
    assert report.metadata.first_satisfied_frontier.nick_boundary_from_left == 0
    assert report.metadata.first_satisfied_frontier.paired_bp == 5
    assert report.metadata.first_satisfied_frontier.cap_extension_nt == 0
    assert report.hits[0].variant_id == "Nb.BsrDI"
    assert report.hits[0].nick_boundary_from_left == 0
    assert report.hits[0].paired_bp == 5
    assert report.hits[0].site_mutation_count == 4
    assert report.hits[0].nickase.selection is not None
    assert report.hits[0].nickase.selection.outside_site is False
    assert all(hit.variant_id != "Nt.BspQI" for hit in report.hits)
    assert any(hit.variant_id == "Nt.Bpu10I" for hit in report.hits)


def test_run_snapback_solve_truncation_searches_shortest_frontier_before_later_catalog_entries(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["input"]["canonical_top_strand"]["sequence"] = "AAAAAAAT"
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 0, "end": 8}
    payload["input"]["canonical_top_strand"]["pre_nick_duplex_window"] = {"start": 0, "end": 8}
    payload["goal"]["nick_boundary_window"] = {"min": 1, "max": 3}
    payload["search"]["max_hits"] = 1
    payload["search"]["materialize_top_k"] = 0
    payload["search"]["max_enumerated_candidates"] = 1
    payload["search"]["max_search_nodes"] = 64
    spec_path = _write_workspace(
        tmp_path,
        spec_payload=payload,
        catalog_entries=[
            {
                "id": "A.Late",
                "specificity_id": "Late",
                "motif_top_5to3": "GGGG",
                "top_cut_offset": 3,
                "source": "demo",
            },
            {
                "id": "Z.Early",
                "specificity_id": "Early",
                "motif_top_5to3": "ATGC",
                "top_cut_offset": 1,
                "source": "demo",
            },
        ],
    )

    _run_dir, report = run_snapback_solve(spec_path)

    assert report.status == "search_truncated"
    assert len(report.hits) == 1
    assert report.hits[0].variant_id == "Z.Early"
    assert report.hits[0].nick_boundary == 1
    assert report.metadata.first_satisfied_frontier is not None
    assert report.metadata.first_satisfied_frontier.nick_boundary_from_left == 1


def test_run_snapback_solve_truncation_prefers_lower_site_mutation_count_within_frontier(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["input"]["canonical_top_strand"]["sequence"] = "ATGCAAAT"
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 0, "end": 8}
    payload["input"]["canonical_top_strand"]["pre_nick_duplex_window"] = {"start": 0, "end": 8}
    payload["goal"]["nick_boundary_window"] = {"min": 1, "max": 1}
    payload["search"]["max_hits"] = 1
    payload["search"]["materialize_top_k"] = 0
    payload["search"]["max_enumerated_candidates"] = 1
    payload["search"]["max_search_nodes"] = 64
    spec_path = _write_workspace(
        tmp_path,
        spec_payload=payload,
        catalog_entries=[
            {
                "id": "A.MoreMutations",
                "specificity_id": "MoreMutations",
                "motif_top_5to3": "AAAA",
                "top_cut_offset": 1,
                "source": "demo",
            },
            {
                "id": "Z.FewerMutations",
                "specificity_id": "FewerMutations",
                "motif_top_5to3": "ATGC",
                "top_cut_offset": 1,
                "source": "demo",
            },
        ],
    )

    _run_dir, report = run_snapback_solve(spec_path)

    assert report.status == "search_truncated"
    assert len(report.hits) == 1
    assert report.hits[0].variant_id == "Z.FewerMutations"
    assert report.hits[0].site_mutation_count == 0


def test_run_snapback_solve_truncation_prefers_better_nickase_profile_within_frontier(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["input"]["canonical_top_strand"]["sequence"] = "ATGCAAAT"
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 0, "end": 8}
    payload["input"]["canonical_top_strand"]["pre_nick_duplex_window"] = {"start": 0, "end": 8}
    payload["goal"]["nick_boundary_window"] = {"min": 1, "max": 1}
    payload["search"]["max_hits"] = 1
    payload["search"]["materialize_top_k"] = 0
    payload["search"]["max_enumerated_candidates"] = 1
    payload["search"]["max_search_nodes"] = 64
    spec_path = _write_workspace(
        tmp_path,
        spec_payload=payload,
        catalog_entries=[
            {
                "id": "A.Weak",
                "specificity_id": "Weak",
                "motif_top_5to3": "ATGC",
                "top_cut_offset": 1,
                "selection": {"outside_site": False, "snapback_tier": "tier3"},
            },
            {
                "id": "Z.Strong",
                "specificity_id": "Strong",
                "motif_top_5to3": "ATGC",
                "top_cut_offset": 1,
                "selection": {"outside_site": True, "snapback_tier": "tier1"},
            },
        ],
    )

    _run_dir, report = run_snapback_solve(spec_path)

    assert report.status == "search_truncated"
    assert len(report.hits) == 1
    assert report.hits[0].variant_id == "Z.Strong"
    assert report.hits[0].nickase.selection is not None
    assert report.hits[0].nickase.selection.snapback_tier == "tier1"


def test_run_snapback_solve_prefers_better_typed_nickase_profile_when_geometry_ties(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["input"]["canonical_top_strand"]["sequence"] = "ATGCAAAT"
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 0, "end": 8}
    payload["input"]["canonical_top_strand"]["pre_nick_duplex_window"] = {"start": 0, "end": 8}
    payload["goal"]["nick_boundary_window"] = {"min": 1, "max": 1}
    payload["search"]["max_hits"] = 2
    payload["search"]["materialize_top_k"] = 0
    spec_path = _write_workspace(
        tmp_path,
        spec_payload=payload,
        catalog_entries=[
            {
                "id": "Nt.Weak",
                "specificity_id": "Weak",
                "motif_top_5to3": "ATGC",
                "top_cut_offset": 1,
                "selection": {"outside_site": False, "snapback_tier": "tier3"},
            },
            {
                "id": "Nt.Strong",
                "specificity_id": "Strong",
                "motif_top_5to3": "ATGC",
                "top_cut_offset": 1,
                "selection": {"outside_site": True, "snapback_tier": "tier1"},
            },
        ],
    )

    _run_dir, report = run_snapback_solve(spec_path)

    assert report.status == "satisfied"
    assert [hit.variant_id for hit in report.hits] == ["Nt.Strong", "Nt.Weak"]
    assert report.hits[0].nickase.selection is not None
    assert report.hits[0].nickase.selection.snapback_tier == "tier1"

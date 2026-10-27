"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/cassette/test_solver.py

Solve/search tests for the cassette workflow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pytest
import yaml

from dnadesign.cruncher.app.cassette_solve_workflow import run_cassette_solve, solve_cassette_spec
from dnadesign.cruncher.cassette.errors import CassetteSpecError
from dnadesign.cruncher.cassette.load import load_cassette_solve_spec
from dnadesign.cruncher.cassette.selection import (
    CandidateHitRecord,
    build_accepted_candidate_pool,
    hamming_distance,
    select_hits,
    select_hits_mmr,
)
from dnadesign.cruncher.cassette.solve_models import CandidateScoreBreakdown, SearchSettingsSpec
from dnadesign.cruncher.cassette.solver import _candidate_hit_id


def _write_solve_spec(tmp_path: Path, *, payload: dict[str, Any]) -> Path:
    workspace = tmp_path / "workspaces" / "demo_cassette"
    spec_path = workspace / "configs" / "cassettes" / "demo_hairpin.cassette.solve.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump({"cassette_solve": payload}, sort_keys=False), encoding="utf-8")
    return spec_path


def _base_solve_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "topology": {
            "stem5p_arm_pattern": "NNNNNCCTCAGC",
            "loop_pattern": "TTT",
        },
        "construct_context": {
            "left_flank": "",
            "right_flank": "",
            "evaluation_scope": "cassette_plus_flanks",
        },
        "nick_goal": {
            "target_strand": "primary",
            "left_nick_window": {"start": 7, "end": 7},
            "right_nick_window": {"start": 17, "end": 17},
            "bounded_segment_length": {"min": 10, "max": 10},
        },
        "assignment_policy": {
            "allowed_left_variant_ids": ["Nt.BbvCI"],
            "allowed_right_variant_ids": ["Nb.BbvCI"],
            "forbidden_intended_variant_ids": [],
            "forbidden_intended_specificity_ids": [],
            "allow_same_variant": True,
            "allow_same_specificity_opposite_variant": True,
        },
        "site_blacklist": {
            "forbidden_any_site_specificity_ids": [],
            "forbidden_unintended_site_specificity_ids": [],
            "forbidden_any_site_variant_ids": [],
            "scope": "evaluation_context",
        },
        "sequence_blacklist": {
            "forbidden_literals": [],
            "forbidden_iupac_motifs": [],
            "forbid_reverse_complements": True,
            "scope": "evaluation_context",
        },
        "sequence_quality": {},
        "catalog": {
            "preset": "neb_nicking_v1",
            "additional_paths": [],
        },
        "search": {
            "max_hits": 3,
            "max_enumerated_candidates": 256,
            "min_pairwise_hamming_distance": 2,
            "bounded_segment_target": 10,
            "gc_target": 0.5,
            "materialize_top_k": 2,
        },
        "output": {
            "run_dir": "outputs/cassette_solves",
            "emit_visual_contracts": True,
            "emit_baserender_jobs": True,
            "baserender_profiles": [
                "duplex_qa",
                "hairpin_qa",
                "top_hits_duplex_qa",
                "top_hits_hairpin_qa",
            ],
        },
    }


def _selection_config(
    *,
    policy: str,
    pool_size: int,
    min_pairwise_distance: int,
    diversity_weight: float | None = None,
) -> dict[str, Any]:
    selection: dict[str, Any] = {
        "policy": policy,
        "pool_size": pool_size,
        "distance_metric": "hamming",
        "min_pairwise_distance": min_pairwise_distance,
    }
    if diversity_weight is not None:
        selection["diversity_weight"] = diversity_weight
    return selection


def test_candidate_hit_id_discriminates_variant_assignments_and_target_strand() -> None:
    base = {
        "cassette_sequence": "AAACGCCTCAGCTTTGCTGAGGCGTTT",
        "left_variant_id": "Nt.BbvCI",
        "right_variant_id": "Nb.BbvCI",
        "left_boundary": 0,
        "right_boundary": 24,
    }

    primary = _candidate_hit_id(target_strand="primary", **base)
    complement = _candidate_hit_id(target_strand="complement", **base)
    alternate_variant = _candidate_hit_id(
        target_strand="primary",
        cassette_sequence=base["cassette_sequence"],
        left_variant_id="Nt.AlwI",
        right_variant_id=base["right_variant_id"],
        left_boundary=base["left_boundary"],
        right_boundary=base["right_boundary"],
    )

    assert len(primary) == 12
    assert primary != complement
    assert primary != alternate_variant


def _policy_payload(
    *,
    policy: str,
    pool_size: int,
    min_pairwise_distance: int,
    diversity_weight: float | None = None,
) -> dict[str, Any]:
    payload = _base_solve_payload()
    payload["search"]["max_hits"] = 5
    payload["search"]["materialize_top_k"] = 0
    payload["search"]["selection"] = _selection_config(
        policy=policy,
        pool_size=pool_size,
        min_pairwise_distance=min_pairwise_distance,
        diversity_weight=diversity_weight,
    )
    payload["search"].pop("min_pairwise_hamming_distance", None)
    return payload


def _candidate_record(
    sequence: str,
    *,
    base_penalty_vector: tuple[int | float, ...] = (0, 0.0, 0.0, 0),
) -> CandidateHitRecord:
    return CandidateHitRecord(
        hit_id=f"hit_{sequence}",
        left_variant_id="Nt.BbvCI",
        right_variant_id="Nb.BbvCI",
        explicit_spec=cast(Any, None),
        report=object(),
        cassette_sequence=sequence,
        stem5p_arm=sequence,
        loop="",
        gc_fraction=0.5,
        extra_site_count=int(base_penalty_vector[0]),
        score_breakdown=CandidateScoreBreakdown(
            extra_site_count=int(base_penalty_vector[0]),
            bounded_segment_distance=float(base_penalty_vector[1]),
            gc_distance=float(base_penalty_vector[2]),
            homopolymer_penalty=int(base_penalty_vector[3]),
        ),
        base_penalty_vector=base_penalty_vector,
        score_tuple=(*base_penalty_vector, sequence),
        left_nick_boundary=0,
        right_nick_boundary=len(sequence),
        bounded_segment_length=len(sequence),
    )


def test_load_cassette_solve_spec_requires_target_strand(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["nick_goal"].pop("target_strand")
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(CassetteSpecError, match="target_strand"):
        load_cassette_solve_spec(spec_path)


def test_load_cassette_solve_spec_rejects_duplicate_assignment_ids(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["assignment_policy"]["allowed_left_variant_ids"] = ["Nt.BbvCI", "Nt.BbvCI"]
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(CassetteSpecError, match="must not repeat values"):
        load_cassette_solve_spec(spec_path)


def test_load_cassette_solve_spec_rejects_oversized_assignment_pair_space(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["assignment_policy"]["allowed_left_variant_ids"] = [f"left_{index}" for index in range(1, 18)]
    payload["assignment_policy"]["allowed_right_variant_ids"] = [f"right_{index}" for index in range(1, 18)]
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(CassetteSpecError, match="assignment pair space"):
        load_cassette_solve_spec(spec_path)


def test_load_cassette_solve_spec_rejects_oversized_topology(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["topology"]["stem5p_arm_pattern"] = "N" * 65
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(CassetteSpecError, match="stem5p_arm_pattern"):
        load_cassette_solve_spec(spec_path)


def test_load_cassette_solve_spec_rejects_unsafe_search_caps(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["search"]["max_hits"] = 129
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(CassetteSpecError, match="max_hits"):
        load_cassette_solve_spec(spec_path)


def test_load_cassette_solve_spec_normalizes_legacy_selection_defaults(tmp_path: Path) -> None:
    spec_path = _write_solve_spec(tmp_path, payload=_base_solve_payload())

    spec, _resolved_spec_path, _workspace_root = load_cassette_solve_spec(spec_path)

    assert spec.search.selection.policy == "greedy_hamming"
    assert spec.search.selection.distance_metric == "hamming"
    assert spec.search.selection.min_pairwise_distance == 2
    assert spec.search.selection.pool_size == 64
    assert spec.search.selection.diversity_weight is None
    assert spec.search.selection_policy_defaulted is True


def test_load_cassette_solve_spec_rejects_mmr_without_diversity_weight(tmp_path: Path) -> None:
    payload = _policy_payload(policy="mmr", pool_size=8, min_pairwise_distance=2)
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(CassetteSpecError, match="diversity_weight"):
        load_cassette_solve_spec(spec_path)


def test_load_cassette_solve_spec_rejects_greedy_hamming_diversity_weight(tmp_path: Path) -> None:
    payload = _policy_payload(policy="greedy_hamming", pool_size=8, min_pairwise_distance=2, diversity_weight=0.2)
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(CassetteSpecError, match="policy=mmr"):
        load_cassette_solve_spec(spec_path)


def test_load_cassette_solve_spec_rejects_score_only_min_pairwise_distance(tmp_path: Path) -> None:
    payload = _policy_payload(policy="score_only", pool_size=8, min_pairwise_distance=1)
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(CassetteSpecError, match="min_pairwise_distance"):
        load_cassette_solve_spec(spec_path)


def test_solve_returns_multiple_hits_and_materializes_top_k(tmp_path: Path) -> None:
    spec_path = _write_solve_spec(tmp_path, payload=_base_solve_payload())

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "solved"
    assert report.solve_id is not None
    assert len(report.hits) == 3
    assert report.metadata.materialized_hit_count == 2
    assert report.hits[0].materialized_run_dir is not None
    assert (run_dir / "solve_report.json").exists()
    assert (run_dir / "solve_manifest.json").exists()
    assert (run_dir / "solve_status.json").exists()
    assert (run_dir / "table__hits.csv").exists()
    assert (run_dir / "views" / "top_hits.linear_duplex.v1.jsonl").exists()
    assert (run_dir / "views" / "top_hits.ssdna_hairpin.v1.jsonl").exists()
    assert (run_dir / "baserender_jobs" / "top_hits_duplex.job.yaml").exists()
    assert (run_dir / "baserender_jobs" / "top_hits_hairpin.job.yaml").exists()
    assert (run_dir / "specs" / "resolved_catalog.yaml").exists()
    hit_dirs = sorted((run_dir / "hits").iterdir())
    assert len(hit_dirs) == 2
    first_hit_report = json.loads((hit_dirs[0] / "explicit" / "report.json").read_text(encoding="utf-8"))
    assert first_hit_report["status"] == "satisfied"
    assert (hit_dirs[0] / "explicit" / "resolved_candidate.cassette.yaml").exists()
    assert (hit_dirs[0] / "views" / "linear_duplex.v1.json").exists()
    assert (hit_dirs[0] / "views" / "ssdna_hairpin.v1.json").exists()
    assert (hit_dirs[0] / "views" / "views_manifest.v1.json").exists()
    assert (hit_dirs[0] / "baserender_jobs" / "linear_duplex.job.yaml").exists()
    assert (hit_dirs[0] / "baserender_jobs" / "ssdna_hairpin.job.yaml").exists()


def test_solve_reports_no_hits_when_blacklist_blocks_required_site(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["sequence_blacklist"]["forbidden_literals"] = ["CCTCAGC"]
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "no_hits"
    assert report.hits == []
    assert (run_dir / "solve_report.json").exists()
    hit_table = (run_dir / "table__hits.csv").read_text(encoding="utf-8")
    assert "rank,solution_id,score,score_tuple,base_penalty_vector,hit_id" in hit_table


def test_solve_reports_invalid_spec_when_assignment_policy_eliminates_all_pairs(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["assignment_policy"]["forbidden_intended_variant_ids"] = ["Nt.BbvCI", "Nb.BbvCI"]
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    report = solve_cassette_spec(spec_path)

    assert report.status == "invalid_spec"
    assert report.issues[0].code == "NO_ALLOWED_VARIANT_PAIRS"


def test_solve_site_blacklist_operates_at_specificity_level(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["site_blacklist"]["forbidden_any_site_specificity_ids"] = ["BbvCI"]
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    report = solve_cassette_spec(spec_path)

    assert report.status == "no_hits"


def test_solve_search_node_guardrail_surfaces_warning(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["search"]["max_search_nodes"] = 1
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    report = solve_cassette_spec(spec_path)

    assert report.status == "no_hits"
    assert "search.max_search_nodes reached before exhausting the solve search tree." in report.metadata.warnings


def test_run_cassette_solve_persists_invalid_spec_bundle(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["nick_goal"].pop("target_strand")
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "invalid_spec"
    assert report.run_dir == str(run_dir.resolve())
    assert (run_dir / "solve_report.json").exists()
    assert (run_dir / "solve_manifest.json").exists()
    assert (run_dir / "solve_status.json").exists()
    assert (run_dir / "table__hits.csv").exists()
    assert (run_dir / "specs" / "input_solve_spec.yaml").exists()
    assert not (run_dir / "specs" / "resolved_catalog.yaml").exists()


def test_run_cassette_solve_persists_invalid_spec_bundle_when_output_shape_is_malformed(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["output"] = []
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "invalid_spec"
    assert report.run_dir == str(run_dir.resolve())
    assert (run_dir / "solve_report.json").exists()
    assert (run_dir / "solve_status.json").exists()
    assert (run_dir / "table__hits.csv").exists()


def test_run_cassette_solve_persists_invalid_catalog_bundle(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["catalog"]["additional_paths"] = ["inputs/catalogs/missing.yaml"]
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "invalid_catalog"
    assert report.run_dir == str(run_dir.resolve())
    assert (run_dir / "solve_report.json").exists()
    assert (run_dir / "solve_manifest.json").exists()
    assert (run_dir / "solve_status.json").exists()
    assert (run_dir / "table__hits.csv").exists()
    assert (run_dir / "specs" / "input_solve_spec.yaml").exists()
    assert not (run_dir / "specs" / "resolved_catalog.yaml").exists()


def test_run_cassette_solve_rejects_symlinked_output_root_outside_workspace(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    workspace_root = tmp_path / "workspaces" / "demo_cassette"
    external_root = tmp_path / "external_outputs"
    external_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "outputs").parent.mkdir(parents=True, exist_ok=True)
    (workspace_root / "outputs").symlink_to(external_root, target_is_directory=True)
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(ValueError, match="must stay inside workspace"):
        run_cassette_solve(spec_path)


def test_run_cassette_solve_status_surface_marks_truncated_search(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["search"]["max_enumerated_candidates"] = 32
    payload["search"]["materialize_top_k"] = 1
    payload["search"]["selection"] = _selection_config(
        policy="greedy_hamming",
        pool_size=8,
        min_pairwise_distance=2,
    )
    payload["search"].pop("min_pairwise_hamming_distance", None)
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "solved"
    status_payload = json.loads((run_dir / "solve_status.json").read_text(encoding="utf-8"))
    assert status_payload["search_truncated"] is True
    assert status_payload["accepted_pool_truncated"] is True
    assert "MAX_ENUMERATED_CANDIDATES_REACHED" in status_payload["warning_codes"]
    assert "ACCEPTED_POOL_TRUNCATED" in status_payload["warning_codes"]
    assert "SELECTION_RESULTS_SEARCH_BOUNDED" in status_payload["warning_codes"]
    assert "SELECTION_RESULTS_POOL_BOUNDED" in status_payload["warning_codes"]
    assert status_payload["selection"]["policy"] == "greedy_hamming"
    assert status_payload["selection"]["accepted_candidate_count"] == report.selection_summary.accepted_candidate_count
    assert status_payload["selection"]["accepted_pool_size"] == report.selection_summary.accepted_pool_size
    assert (
        status_payload["selection"]["accepted_pool_admitted_count"]
        == report.selection_summary.accepted_pool_admitted_count
    )
    assert (
        status_payload["selection"]["accepted_pool_rejected_count"]
        == report.selection_summary.accepted_pool_rejected_count
    )
    assert (
        status_payload["selection"]["accepted_pool_worst_score_at_close"]
        == report.selection_summary.accepted_pool_worst_score_at_close
    )
    assert status_payload["selection"]["policy_underfilled"] is False
    assert status_payload["selection"]["policy_limited_hit_count"] == 0
    assert status_payload["top_hits_linear_duplex_jsonl"].endswith("views/top_hits.linear_duplex.v1.jsonl")
    assert status_payload["top_hits_hairpin_jsonl"].endswith("views/top_hits.ssdna_hairpin.v1.jsonl")
    assert status_payload["top_hits_duplex_job"].endswith("baserender_jobs/top_hits_duplex.job.yaml")
    assert status_payload["top_hits_hairpin_job"].endswith("baserender_jobs/top_hits_hairpin.job.yaml")
    solve_report_md = (run_dir / "solve_report.md").read_text(encoding="utf-8")
    assert "warning[MAX_ENUMERATED_CANDIDATES_REACHED]" in solve_report_md
    assert "warning[SELECTION_RESULTS_POOL_BOUNDED]" in solve_report_md


def test_select_hits_mmr_uses_ascending_sequence_tiebreak_for_equal_utility() -> None:
    selected = select_hits_mmr(
        [
            _candidate_record("AAAA"),
            _candidate_record("AATT"),
            _candidate_record("TTAA"),
        ],
        max_hits=2,
        distance_metric="hamming",
        diversity_weight=0.5,
        min_pairwise_distance=0,
    )

    assert [hit.record.cassette_sequence for hit in selected] == ["AAAA", "AATT"]


def test_selection_summary_reports_hamming_pairwise_distances_correctly() -> None:
    accepted_pool = build_accepted_candidate_pool(pool_size=4)
    for sequence in ("AAAA", "AATT", "TTTT"):
        accepted_pool.consider(_candidate_record(sequence))

    outcome = select_hits(
        accepted_pool=accepted_pool,
        search_settings=SearchSettingsSpec.model_validate(
            {
                "max_hits": 3,
                "materialize_top_k": 0,
                "selection": {
                    "policy": "score_only",
                    "pool_size": 4,
                    "distance_metric": "hamming",
                    "min_pairwise_distance": 0,
                },
            }
        ),
        accepted_candidate_count=3,
        search_truncated=False,
    )

    assert hamming_distance("AAAA", "AATT") == 2
    assert outcome.summary.pairwise_distance_summary.min == 2.0
    assert outcome.summary.pairwise_distance_summary.max == 4.0
    assert outcome.summary.pairwise_distance_summary.mean == pytest.approx((2.0 + 4.0 + 2.0) / 3.0)


def test_select_hits_mmr_reuses_pairwise_hamming_computations(monkeypatch: pytest.MonkeyPatch) -> None:
    import dnadesign.cruncher.cassette.selection as selection_module

    sequences = ["AAAAAA", "AAAACA", "AAACAA", "AACAAA", "ACAAAA", "CAAAAA"]
    accepted_pool = build_accepted_candidate_pool(pool_size=len(sequences))
    for sequence in sequences:
        accepted_pool.consider(_candidate_record(sequence))

    call_count = 0
    original = selection_module.hamming_distance

    def _counted(left: str, right: str) -> int:
        nonlocal call_count
        call_count += 1
        return original(left, right)

    monkeypatch.setattr(selection_module, "hamming_distance", _counted)

    outcome = select_hits(
        accepted_pool=accepted_pool,
        search_settings=SearchSettingsSpec.model_validate(
            {
                "max_hits": 4,
                "materialize_top_k": 0,
                "selection": {
                    "policy": "mmr",
                    "pool_size": len(sequences),
                    "distance_metric": "hamming",
                    "min_pairwise_distance": 0,
                    "diversity_weight": 0.35,
                },
            }
        ),
        accepted_candidate_count=len(sequences),
        search_truncated=False,
    )

    unique_pool_pairs = (len(sequences) * (len(sequences) - 1)) // 2
    assert len(outcome.selected_hits) == 4
    assert call_count <= unique_pool_pairs


def test_accepted_candidate_pool_replacement_pressure_preserves_pool_summary() -> None:
    metrics_by_pool_size: dict[int, dict[str, object]] = {}

    for pool_size in (64, 128, 256, 1024):
        accepted_pool = build_accepted_candidate_pool(pool_size=pool_size)
        total_candidates = pool_size * 4
        started_at = perf_counter()
        for penalty in range(total_candidates, 0, -1):
            accepted_pool.consider(
                _candidate_record(
                    f"{penalty:08d}",
                    base_penalty_vector=(penalty, 0.0, 0.0, 0),
                )
            )
        elapsed_seconds = perf_counter() - started_at
        summary = accepted_pool.summary()
        metrics_by_pool_size[pool_size] = {
            "elapsed_seconds": elapsed_seconds,
            "final_size": summary.final_size,
            "admitted_count": summary.admitted_count,
            "rejected_count": summary.rejected_count,
            "truncated": summary.truncated,
            "worst_score_at_close": summary.worst_score_at_close,
        }

    for pool_size, metrics in metrics_by_pool_size.items():
        assert metrics["elapsed_seconds"] >= 0.0
        assert metrics["final_size"] == pool_size
        assert metrics["admitted_count"] == pool_size * 4
        assert metrics["rejected_count"] == 0
        assert metrics["truncated"] is True
        assert metrics["worst_score_at_close"] == (pool_size, 0.0, 0.0, 0, f"{pool_size:08d}")


def test_solve_tracks_total_accepted_candidates_and_pool_telemetry_when_pool_truncates(tmp_path: Path) -> None:
    payload = _policy_payload(policy="greedy_hamming", pool_size=8, min_pairwise_distance=2)
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    report = solve_cassette_spec(spec_path)

    assert report.status == "solved"
    assert report.selection_summary is not None
    assert report.selection_summary.accepted_candidate_count > report.selection_summary.accepted_pool_size
    assert report.selection_summary.accepted_pool_size == 8
    assert report.selection_summary.accepted_pool_truncated is True
    assert report.selection_summary.accepted_pool_rejected_count > 0
    assert report.selection_summary.accepted_pool_worst_score_at_close is not None


def test_solve_warns_when_selection_policy_underfills_without_search_or_pool_truncation(tmp_path: Path) -> None:
    payload = _policy_payload(policy="greedy_hamming", pool_size=1024, min_pairwise_distance=9)
    payload["search"]["max_enumerated_candidates"] = 20000
    payload["search"]["max_search_nodes"] = 500000
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "solved"
    assert report.selection_summary is not None
    assert report.selection_summary.accepted_pool_truncated is False
    assert report.selection_summary.search_truncated is False
    assert report.selection_summary.policy_underfilled is True
    assert report.selection_summary.policy_limited_hit_count == 1
    assert report.selection_summary.policy_underfilled_reason == "selection_policy_constraints_filtered_pool"
    assert report.selection_summary.accepted_pool_size == 1024
    assert report.selection_summary.selected_hit_count == 4
    assert report.metadata.warning_codes == ["SELECTION_POLICY_LIMITED_HITS"]
    status_payload = json.loads((run_dir / "solve_status.json").read_text(encoding="utf-8"))
    assert status_payload["selection"]["policy_underfilled"] is True
    assert status_payload["selection"]["policy_limited_hit_count"] == 1
    assert status_payload["selection"]["policy_underfilled_reason"] == "selection_policy_constraints_filtered_pool"


def test_solve_policy_comparison_reports_selection_summary_and_changes_hit_set(tmp_path: Path) -> None:
    payloads = {
        "score_only": _policy_payload(policy="score_only", pool_size=64, min_pairwise_distance=0),
        "greedy_hamming": _policy_payload(policy="greedy_hamming", pool_size=64, min_pairwise_distance=2),
        "mmr": _policy_payload(policy="mmr", pool_size=64, min_pairwise_distance=2, diversity_weight=0.35),
    }

    metrics_by_policy: dict[str, dict[str, object]] = {}
    for policy, payload in payloads.items():
        started_at = perf_counter()
        report = solve_cassette_spec(_write_solve_spec(tmp_path / policy, payload=payload))
        elapsed_seconds = perf_counter() - started_at
        assert report.status == "solved"
        assert report.selection_summary is not None
        metrics_by_policy[policy] = {
            "hit_ids": [hit.hit_id for hit in report.hits],
            "elapsed_seconds": elapsed_seconds,
            "visited_search_nodes": report.metadata.visited_search_node_count,
            "enumerated_candidate_count": report.metadata.enumerated_candidate_count,
            "accepted_candidate_count": report.metadata.accepted_candidate_count,
            "pool_truncated": report.selection_summary.accepted_pool_truncated,
            "returned_hit_count": report.selection_summary.selected_hit_count,
            "pairwise_mean": report.selection_summary.pairwise_distance_summary.mean,
            "pairwise_min": report.selection_summary.pairwise_distance_summary.min,
            "warning_codes": report.metadata.warning_codes,
        }

    assert metrics_by_policy["score_only"]["hit_ids"] != metrics_by_policy["mmr"]["hit_ids"]
    assert metrics_by_policy["score_only"]["elapsed_seconds"] >= 0.0
    assert metrics_by_policy["greedy_hamming"]["elapsed_seconds"] >= 0.0
    assert metrics_by_policy["mmr"]["elapsed_seconds"] >= 0.0
    assert metrics_by_policy["mmr"]["pairwise_mean"] >= metrics_by_policy["score_only"]["pairwise_mean"]
    assert metrics_by_policy["greedy_hamming"]["pairwise_min"] >= 2
    assert metrics_by_policy["mmr"]["returned_hit_count"] == 5

"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/scar_nick/artifacts.py

Artifact paths and persistence helpers for scar-nick runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_text, atomic_write_yaml
from dnadesign.cruncher.scar_nick.models import ScarNickEvaluationReport
from dnadesign.cruncher.utils.hashing import sha256_path

RUN_META_DIR = "meta"
RUN_PROVENANCE_DIR = "provenance"
RUN_ANALYSIS_DIR = "analysis"
RUN_ANALYSIS_VIEWS_DIR = "views"
RUN_MATERIALIZED_CANDIDATES_DIR = "materialized_candidates"
RUN_EXPORT_DIR = "export"
RUN_PLOTS_DIR = "plots"
RUN_BASERENDER_JOBS_DIR = "baserender_jobs"


def _scoped_run_dir(workspace_root: Path, run_dir: Path) -> Path:
    resolved_workspace_root = workspace_root.resolve()
    candidate = resolved_workspace_root.joinpath(run_dir).resolve()
    try:
        candidate.relative_to(resolved_workspace_root)
    except ValueError as exc:
        raise ValueError(
            f"Scar-nick run directory must stay inside workspace {resolved_workspace_root}: {candidate}"
        ) from exc
    return candidate


def build_run_dir(*, workspace_root: Path, run_dir: Path) -> Path:
    return _scoped_run_dir(workspace_root, run_dir)


def ensure_run_dirs(run_dir: Path) -> None:
    (run_dir / RUN_META_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_PROVENANCE_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)
    views_dir(run_dir).mkdir(parents=True, exist_ok=True)
    materialized_candidates_dir(run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_EXPORT_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    baserender_jobs_dir(run_dir).mkdir(parents=True, exist_ok=True)


def ensure_visual_run_dirs(run_dir: Path) -> None:
    (run_dir / RUN_META_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)
    views_dir(run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    baserender_jobs_dir(run_dir).mkdir(parents=True, exist_ok=True)


def manifest_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "scar_nick_manifest.json"


def status_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "scar_nick_status.json"


def spec_snapshot_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR / "spec.snapshot.yaml"


def nickase_catalog_snapshot_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR / "nickase_catalog.yaml"


def release_catalog_snapshot_path(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR / "release_catalog.yaml"


def report_json_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / "report.json"


def report_md_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / "report.md"


def candidate_profiles_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / "candidate_profiles.json"


def nickase_geometry_audit_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / "nickase_geometry_audit.json"


def candidate_json_path(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR / "candidate.json"


def candidate_table_path(run_dir: Path) -> Path:
    return run_dir / RUN_EXPORT_DIR / "table__scar_nick_candidates.csv"


def candidate_pair_call_table_path(run_dir: Path) -> Path:
    return run_dir / RUN_EXPORT_DIR / "table__scar_nick_candidate_pair_calls.csv"


def nickase_geometry_audit_table_path(run_dir: Path) -> Path:
    return run_dir / RUN_EXPORT_DIR / "table__scar_nick_nickase_geometry_audit.csv"


def analysis_dir(run_dir: Path) -> Path:
    return run_dir / RUN_ANALYSIS_DIR


def views_dir(run_dir: Path) -> Path:
    return analysis_dir(run_dir) / RUN_ANALYSIS_VIEWS_DIR


def materialized_candidates_dir(run_dir: Path) -> Path:
    return analysis_dir(run_dir) / RUN_MATERIALIZED_CANDIDATES_DIR


def materialized_candidate_dir(run_dir: Path, *, rank: int) -> Path:
    return materialized_candidates_dir(run_dir) / f"candidate_{rank:02d}"


def candidate_manifest_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "scar_nick_candidate_manifest.json"


def baserender_jobs_dir(run_dir: Path) -> Path:
    return run_dir / RUN_BASERENDER_JOBS_DIR


def pre_terminal_nick_view_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "pre_terminal_nick.v1.json"


def post_terminal_nick_view_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "post_terminal_nick.v1.json"


def pre_terminal_nick_visual_contract_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "pre_terminal_nick.scar_nick_visual.v1.json"


def post_terminal_nick_visual_contract_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "post_terminal_nick.scar_nick_visual.v1.json"


def scar_nick_terminal_nick_visual_contracts_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "scar_nick_terminal_nick.scar_nick_visual.v1.jsonl"


def views_manifest_path(run_dir: Path) -> Path:
    return views_dir(run_dir) / "views_manifest.v1.json"


def scar_nick_terminal_nick_job_path(run_dir: Path) -> Path:
    return baserender_jobs_dir(run_dir) / "scar_nick_terminal_nick.job.yaml"


def snapshot_inputs(
    run_dir: Path,
    *,
    spec_path: Path,
    release_catalog_yaml: str,
    nickase_catalog_yaml: str,
) -> None:
    shutil.copyfile(spec_path, spec_snapshot_path(run_dir))
    release_catalog_snapshot_path(run_dir).write_text(release_catalog_yaml, encoding="utf-8")
    nickase_catalog_snapshot_path(run_dir).write_text(nickase_catalog_yaml, encoding="utf-8")


def build_manifest(
    *,
    run_dir: Path,
    workspace_root: Path,
    spec_path: Path,
    report: ScarNickEvaluationReport,
) -> dict[str, Any]:
    artifacts = [
        {"name": "report_json", "path": str(report_json_path(run_dir).relative_to(run_dir))},
        {"name": "report_md", "path": str(report_md_path(run_dir).relative_to(run_dir))},
        {"name": "candidate_profiles", "path": str(candidate_profiles_path(run_dir).relative_to(run_dir))},
        {"name": "nickase_geometry_audit", "path": str(nickase_geometry_audit_path(run_dir).relative_to(run_dir))},
        {"name": "candidate_table", "path": str(candidate_table_path(run_dir).relative_to(run_dir))},
        {
            "name": "candidate_pair_call_table",
            "path": str(candidate_pair_call_table_path(run_dir).relative_to(run_dir)),
        },
        {
            "name": "nickase_geometry_audit_table",
            "path": str(nickase_geometry_audit_table_path(run_dir).relative_to(run_dir)),
        },
        {"name": "spec_snapshot", "path": str(spec_snapshot_path(run_dir).relative_to(run_dir))},
        {
            "name": "nickase_catalog_snapshot",
            "path": str(nickase_catalog_snapshot_path(run_dir).relative_to(run_dir)),
        },
        {
            "name": "release_catalog_snapshot",
            "path": str(release_catalog_snapshot_path(run_dir).relative_to(run_dir)),
        },
    ]
    optional_visuals = [
        ("terminal_nick_view", post_terminal_nick_view_path(run_dir)),
        ("terminal_nick_visual_contract", post_terminal_nick_visual_contract_path(run_dir)),
        ("scar_nick_terminal_nick_visual_contracts", scar_nick_terminal_nick_visual_contracts_path(run_dir)),
        ("views_manifest", views_manifest_path(run_dir)),
        ("scar_nick_terminal_nick_job", scar_nick_terminal_nick_job_path(run_dir)),
    ]
    artifacts.extend(
        {"name": name, "path": str(path.relative_to(run_dir))} for name, path in optional_visuals if path.exists()
    )
    candidate_manifests = sorted(
        materialized_candidates_dir(run_dir).glob("candidate_*/meta/scar_nick_candidate_manifest.json")
    )
    artifacts.extend(
        {
            "name": f"{candidate_manifest.parent.parent.name}_manifest",
            "path": str(candidate_manifest.relative_to(run_dir)),
        }
        for candidate_manifest in candidate_manifests
    )
    return {
        "stage": "scar_nick",
        "workflow": "scar_nick_design",
        "run_dir": str(run_dir.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "spec_name": report.spec_name,
        "status": report.status,
        "spec_path": str(spec_path.resolve()),
        "spec_snapshot_sha256": sha256_path(spec_snapshot_path(run_dir)),
        "nickase_catalog_sha256": sha256_path(nickase_catalog_snapshot_path(run_dir)),
        "release_catalog_sha256": sha256_path(release_catalog_snapshot_path(run_dir)),
        "report_json_sha256": sha256_path(report_json_path(run_dir)),
        "report_md_sha256": sha256_path(report_md_path(run_dir)),
        "candidate_profiles_sha256": sha256_path(candidate_profiles_path(run_dir)),
        "nickase_geometry_audit_sha256": sha256_path(nickase_geometry_audit_path(run_dir)),
        "candidate_table_sha256": sha256_path(candidate_table_path(run_dir)),
        "candidate_pair_call_table_sha256": sha256_path(candidate_pair_call_table_path(run_dir)),
        "nickase_geometry_audit_table_sha256": sha256_path(nickase_geometry_audit_table_path(run_dir)),
        "artifacts": artifacts,
    }


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    atomic_write_json(manifest_path(run_dir), manifest)
    return manifest_path(run_dir)


def write_status(run_dir: Path, *, report: ScarNickEvaluationReport, status_message: str) -> Path:
    payload = {
        "stage": "scar_nick",
        "status": "completed" if report.status == "satisfied" else "unsatisfied",
        "status_message": status_message,
        "run_dir": str(run_dir.resolve()),
        "spec_name": report.spec_name,
        "candidate_count": len(report.candidates),
        "issue_count": len(report.issues),
    }
    atomic_write_json(status_path(run_dir), payload)
    return status_path(run_dir)


def write_report(run_dir: Path, report: ScarNickEvaluationReport, *, markdown: str) -> None:
    atomic_write_json(report_json_path(run_dir), report.model_dump(mode="json"))
    report_md_path(run_dir).write_text(markdown, encoding="utf-8")
    atomic_write_json(
        candidate_profiles_path(run_dir),
        {
            "accepted": [candidate.model_dump(mode="json") for candidate in report.candidates],
            "reserve": [candidate.model_dump(mode="json") for candidate in report.reserve_candidates],
            "rejected_reference_candidates": [
                candidate.model_dump(mode="json") for candidate in report.rejected_reference_candidates
            ],
        },
    )
    atomic_write_json(
        nickase_geometry_audit_path(run_dir),
        {
            "version": 1,
            "kind": "scar_nick_nickase_geometry_audit_v1",
            "entries": [entry.model_dump(mode="json") for entry in report.nickase_geometry_audit],
        },
    )


def write_nickase_geometry_audit_table(run_dir: Path, report: ScarNickEvaluationReport) -> None:
    fieldnames = [
        "variant_id",
        "specificity_id",
        "orientation",
        "strand",
        "terminal_candidate",
        "motif_top_5to3",
        "source_site_start",
        "source_site_end",
        "boundary",
        "terminal_boundary",
        "exact_terminal",
        "policy_rejection_reasons",
        "rejection_reasons",
        "release_overlap_conflicts",
        "downstream_symbols",
        "retained_scar_domains",
        "feasible_scar_count",
        "upstream_flank_sequence",
        "type_iis_offset_sequence",
        "compatible",
    ]
    with nickase_geometry_audit_table_path(run_dir).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in report.nickase_geometry_audit:
            writer.writerow(
                {
                    "variant_id": entry.variant_id,
                    "specificity_id": entry.specificity_id,
                    "orientation": entry.orientation or "",
                    "strand": entry.strand or "",
                    "terminal_candidate": entry.terminal_candidate,
                    "motif_top_5to3": entry.motif_top_5to3,
                    "source_site_start": "" if entry.source_site_start is None else entry.source_site_start,
                    "source_site_end": "" if entry.source_site_end is None else entry.source_site_end,
                    "boundary": "" if entry.boundary is None else entry.boundary,
                    "terminal_boundary": entry.terminal_boundary,
                    "exact_terminal": entry.exact_terminal,
                    "policy_rejection_reasons": json.dumps(entry.policy_rejection_reasons),
                    "rejection_reasons": json.dumps(entry.rejection_reasons),
                    "release_overlap_conflicts": json.dumps(
                        [conflict.model_dump(mode="json") for conflict in entry.release_overlap_conflicts]
                    ),
                    "downstream_symbols": json.dumps(
                        [symbol.model_dump(mode="json") for symbol in entry.downstream_symbols]
                    ),
                    "retained_scar_domains": json.dumps(
                        [domain.model_dump(mode="json") for domain in entry.retained_scar_domains]
                    ),
                    "feasible_scar_count": entry.feasible_scar_count,
                    "upstream_flank_sequence": entry.upstream_flank_sequence,
                    "type_iis_offset_sequence": entry.type_iis_offset_sequence,
                    "compatible": entry.compatible,
                }
            )


def write_candidate_table(run_dir: Path, report: ScarNickEvaluationReport) -> None:
    fieldnames = [
        "rank",
        "candidate_id",
        "left_base",
        "right_base",
        "profile_order",
        "profile_policy_status",
        "profile_policy_reason",
        "profile_s3s2s1s0",
        "profile_payload_outward",
        "s0_match_required",
        "s3_pair_identity",
        "s2_pair_identity",
        "s1_pair_identity",
        "s0_pair_identity",
        "s3_pair_type",
        "s2_pair_type",
        "s1_pair_type",
        "s0_pair_type",
        "pair_classes",
        "m_count",
        "w_count",
        "x_count",
        "non_watson_crick_count",
        "middle_hard_count",
        "middle_wobble_count",
        "worst_hard_mismatch_tier",
        "hard_mismatch_tier_sum",
        "middle_hard_mismatch_tier_sum",
        "edge_hard_mismatch_tier_sum",
        "ligation_support",
        "effective_disruption",
        "tnna_flag",
        "release_placement",
        "retained_scar",
        "nickase_site",
        "nicked_strand",
        "surviving_strand",
        "retained_scar_source",
        "discarded_strand_enzyme_burden",
        "nickase_vendor",
        "nickase_source_url",
        "nickase_commercial_confidence",
        "nickase_warning_codes",
        "nick_boundary",
        "gc_fraction",
        "reference_control_distance",
        "rejection_reasons",
    ]
    with candidate_table_path(run_dir).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in report.candidates:
            release = candidate.release_placement
            pair_type_by_site = {pair.site: pair.class_label for pair in candidate.pair_classes}
            release_placement = (
                ""
                if release is None
                else (
                    f"{release.variant_id}:{release.orientation}"
                    f"[{release.recognition_site_start},{release.recognition_site_end})"
                    f" top={release.top_cut_boundary} bottom={release.bottom_cut_boundary}"
                )
            )
            nickase = candidate.nickase_placement
            writer.writerow(
                {
                    "rank": candidate.rank,
                    "candidate_id": candidate.candidate_id,
                    "left_base": candidate.left_base,
                    "right_base": candidate.right_base,
                    "profile_order": candidate.profile_order,
                    "profile_policy_status": candidate.profile_policy_status,
                    "profile_policy_reason": candidate.profile_policy_reason,
                    "profile_s3s2s1s0": candidate.profile_s3s2s1s0,
                    "profile_payload_outward": candidate.profile_payload_outward,
                    "s0_match_required": candidate.s0_match_required,
                    "s3_pair_identity": candidate.s3_pair_identity,
                    "s2_pair_identity": candidate.s2_pair_identity,
                    "s1_pair_identity": candidate.s1_pair_identity,
                    "s0_pair_identity": candidate.s0_pair_identity,
                    "s3_pair_type": pair_type_by_site["S3"],
                    "s2_pair_type": pair_type_by_site["S2"],
                    "s1_pair_type": pair_type_by_site["S1"],
                    "s0_pair_type": pair_type_by_site["S0"],
                    "pair_classes": json.dumps([entry.model_dump(mode="json") for entry in candidate.pair_classes]),
                    "m_count": candidate.m_count,
                    "w_count": candidate.w_count,
                    "x_count": candidate.x_count,
                    "non_watson_crick_count": candidate.non_watson_crick_count,
                    "middle_hard_count": candidate.middle_hard_count,
                    "middle_wobble_count": candidate.middle_wobble_count,
                    "worst_hard_mismatch_tier": candidate.worst_hard_mismatch_tier,
                    "hard_mismatch_tier_sum": candidate.hard_mismatch_tier_sum,
                    "middle_hard_mismatch_tier_sum": candidate.middle_hard_mismatch_tier_sum,
                    "edge_hard_mismatch_tier_sum": candidate.edge_hard_mismatch_tier_sum,
                    "ligation_support": f"{candidate.ligation_support:.6f}",
                    "effective_disruption": f"{candidate.effective_disruption:.6f}",
                    "tnna_flag": candidate.tnna_flag,
                    "release_placement": release_placement,
                    "retained_scar": candidate.retained_scar,
                    "nickase_site": candidate.nickase_site,
                    "nicked_strand": candidate.nicked_strand or "",
                    "surviving_strand": candidate.surviving_strand or "",
                    "retained_scar_source": candidate.retained_scar_source,
                    "discarded_strand_enzyme_burden": candidate.discarded_strand_enzyme_burden or "",
                    "nickase_vendor": "" if nickase is None else nickase.vendor,
                    "nickase_source_url": "" if nickase is None else nickase.source_url,
                    "nickase_commercial_confidence": "" if nickase is None else nickase.commercial_confidence,
                    "nickase_warning_codes": json.dumps([] if nickase is None else nickase.warning_codes),
                    "nick_boundary": candidate.nick_boundary,
                    "gc_fraction": f"{candidate.gc_fraction:.6f}",
                    "reference_control_distance": candidate.reference_control_distance,
                    "rejection_reasons": json.dumps(candidate.rejection_reasons),
                }
            )


def write_candidate_pair_call_table(run_dir: Path, report: ScarNickEvaluationReport) -> None:
    fieldnames = [
        "rank",
        "candidate_id",
        "left_base",
        "right_base",
        "profile_order",
        "profile_policy_status",
        "profile_policy_reason",
        "profile_s3s2s1s0",
        "site",
        "position",
        "position_class",
        "left_offset_5to3",
        "right_offset_5to3",
        "left_nt",
        "right_nt",
        "aligned_right_nt",
        "pair_identity",
        "class_label",
        "is_watson_crick",
        "is_wobble",
        "is_hard_mismatch",
        "canonical_mismatch_class",
        "class_tier_t4",
        "m_count",
        "w_count",
        "x_count",
        "non_watson_crick_count",
        "middle_hard_count",
        "middle_wobble_count",
        "worst_hard_mismatch_tier",
        "hard_mismatch_tier_sum",
        "middle_hard_mismatch_tier_sum",
        "edge_hard_mismatch_tier_sum",
        "ligation_support",
        "effective_disruption",
        "nicked_strand",
        "surviving_strand",
        "retained_scar",
        "nickase_site",
    ]
    with candidate_pair_call_table_path(run_dir).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in report.candidates:
            for pair in candidate.pair_classes:
                writer.writerow(
                    {
                        "rank": candidate.rank,
                        "candidate_id": candidate.candidate_id,
                        "left_base": candidate.left_base,
                        "right_base": candidate.right_base,
                        "profile_order": candidate.profile_order,
                        "profile_policy_status": candidate.profile_policy_status,
                        "profile_policy_reason": candidate.profile_policy_reason,
                        "profile_s3s2s1s0": candidate.profile_s3s2s1s0,
                        "site": pair.site,
                        "position": pair.position,
                        "position_class": pair.position_class,
                        "left_offset_5to3": pair.source_offset,
                        "right_offset_5to3": 3 - pair.source_offset,
                        "left_nt": pair.left_base,
                        "right_nt": pair.right_base,
                        "aligned_right_nt": pair.aligned_right_base,
                        "pair_identity": f"{pair.left_base}:{pair.right_base}",
                        "class_label": pair.class_label,
                        "is_watson_crick": pair.class_label == "M",
                        "is_wobble": pair.class_label == "W",
                        "is_hard_mismatch": pair.class_label == "X",
                        "canonical_mismatch_class": pair.canonical_mismatch_class or "",
                        "class_tier_t4": pair.class_tier_t4,
                        "m_count": candidate.m_count,
                        "w_count": candidate.w_count,
                        "x_count": candidate.x_count,
                        "non_watson_crick_count": candidate.non_watson_crick_count,
                        "middle_hard_count": candidate.middle_hard_count,
                        "middle_wobble_count": candidate.middle_wobble_count,
                        "worst_hard_mismatch_tier": candidate.worst_hard_mismatch_tier,
                        "hard_mismatch_tier_sum": candidate.hard_mismatch_tier_sum,
                        "middle_hard_mismatch_tier_sum": candidate.middle_hard_mismatch_tier_sum,
                        "edge_hard_mismatch_tier_sum": candidate.edge_hard_mismatch_tier_sum,
                        "ligation_support": f"{candidate.ligation_support:.6f}",
                        "effective_disruption": f"{candidate.effective_disruption:.6f}",
                        "nicked_strand": candidate.nicked_strand or "",
                        "surviving_strand": candidate.surviving_strand or "",
                        "retained_scar": candidate.retained_scar,
                        "nickase_site": candidate.nickase_site,
                    }
                )


def write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> Path:
    payload = "\n".join(json.dumps(record) for record in records)
    if payload:
        payload += "\n"
    atomic_write_text(path, payload)
    return path


def write_visual_bundle(
    run_dir: Path,
    *,
    terminal_nick_view: dict[str, Any],
    terminal_nick_visual_contract: dict[str, Any],
    terminal_nick_visual_contracts: list[dict[str, Any]],
    views_manifest: dict[str, Any],
    baserender_job: dict[str, Any],
) -> None:
    atomic_write_json(post_terminal_nick_view_path(run_dir), terminal_nick_view)
    atomic_write_json(post_terminal_nick_visual_contract_path(run_dir), terminal_nick_visual_contract)
    write_jsonl_records(scar_nick_terminal_nick_visual_contracts_path(run_dir), terminal_nick_visual_contracts)
    atomic_write_json(views_manifest_path(run_dir), views_manifest)
    atomic_write_yaml(scar_nick_terminal_nick_job_path(run_dir), baserender_job, sort_keys=False)


def build_materialized_candidate_manifest_payload(
    *,
    candidate_payload: dict[str, Any],
    views_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "scar_nick",
        "workflow": "scar_nick_candidate_visual_qa",
        "candidate_id": candidate_payload.get("candidate_id"),
        "rank": candidate_payload.get("rank"),
        "artifacts": [
            {"name": "candidate", "path": "analysis/candidate.json"},
            {"name": "terminal_nick_view", "path": "analysis/views/post_terminal_nick.v1.json"},
            {
                "name": "terminal_nick_visual_contract",
                "path": "analysis/views/post_terminal_nick.scar_nick_visual.v1.json",
            },
            {
                "name": "scar_nick_terminal_nick_visual_contracts",
                "path": "analysis/views/scar_nick_terminal_nick.scar_nick_visual.v1.jsonl",
            },
            {"name": "views_manifest", "path": "analysis/views/views_manifest.v1.json"},
            {
                "name": "scar_nick_terminal_nick_job",
                "path": "baserender_jobs/scar_nick_terminal_nick.job.yaml",
            },
        ],
        "views": views_manifest,
    }


def write_materialized_candidate_manifest(
    run_dir: Path,
    *,
    candidate_payload: dict[str, Any],
    views_manifest: dict[str, Any],
) -> Path:
    payload = build_materialized_candidate_manifest_payload(
        candidate_payload=candidate_payload,
        views_manifest=views_manifest,
    )
    atomic_write_json(candidate_json_path(run_dir), candidate_payload)
    atomic_write_json(candidate_manifest_path(run_dir), payload)
    return candidate_manifest_path(run_dir)


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing scar-nick manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_status(run_dir: Path) -> dict[str, Any]:
    path = status_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing scar-nick status: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_provenance_current(run_dir: Path, manifest: dict[str, Any]) -> None:
    checks = {
        "spec_snapshot_sha256": spec_snapshot_path(run_dir),
        "nickase_catalog_sha256": nickase_catalog_snapshot_path(run_dir),
        "release_catalog_sha256": release_catalog_snapshot_path(run_dir),
    }
    for key, path in checks.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing scar-nick provenance file: {path}")
        expected = manifest.get(key)
        observed = sha256_path(path)
        if expected != observed:
            raise ValueError(f"Scar-nick provenance drift for {path.name}: expected {expected}, observed {observed}")


def assert_manifest_hashes_current(run_dir: Path, manifest: dict[str, Any]) -> None:
    checks = {
        "report_json_sha256": report_json_path(run_dir),
        "report_md_sha256": report_md_path(run_dir),
        "candidate_profiles_sha256": candidate_profiles_path(run_dir),
        "nickase_geometry_audit_sha256": nickase_geometry_audit_path(run_dir),
        "candidate_table_sha256": candidate_table_path(run_dir),
        "candidate_pair_call_table_sha256": candidate_pair_call_table_path(run_dir),
        "nickase_geometry_audit_table_sha256": nickase_geometry_audit_table_path(run_dir),
    }
    for key, path in checks.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing scar-nick artifact: {path}")
        expected = manifest.get(key)
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"Scar-nick manifest missing artifact hash: {key}")
        observed = sha256_path(path)
        if expected != observed:
            raise ValueError(f"Scar-nick artifact drift for {path.name}: expected {expected}, observed {observed}")


def _manifest_artifact_path(root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Scar-nick manifest artifact path drift detected.")
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or any(part == ".." for part in relative_path.parts):
        raise ValueError(f"Scar-nick manifest artifact path must stay inside run: {raw_path}")
    return root / relative_path


def _is_visual_artifact_name(name: str) -> bool:
    return "visual" in name or name in {"views_manifest", "scar_nick_terminal_nick_job"}


def _assert_manifest_artifacts_present(root: Path, artifacts: object) -> None:
    if not isinstance(artifacts, list):
        raise ValueError("Scar-nick manifest artifact inventory drift detected.")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Scar-nick manifest artifact inventory drift detected.")
        path = _manifest_artifact_path(root, artifact.get("path"))
        name = str(artifact.get("name") or "")
        if path.exists():
            if path.name == "scar_nick_candidate_manifest.json":
                nested_manifest = json.loads(path.read_text(encoding="utf-8"))
                _assert_manifest_artifacts_present(path.parent.parent, nested_manifest.get("artifacts", []))
            continue
        name = str(artifact.get("name") or "")
        if _is_visual_artifact_name(name):
            raise FileNotFoundError(f"Missing scar-nick visual artifact: {path}")
        raise FileNotFoundError(f"Missing scar-nick artifact: {path}")


def assert_manifest_artifacts_present(run_dir: Path, manifest: dict[str, Any]) -> None:
    _assert_manifest_artifacts_present(run_dir, manifest.get("artifacts", []))


__all__ = [
    "assert_manifest_hashes_current",
    "assert_provenance_current",
    "assert_manifest_artifacts_present",
    "baserender_jobs_dir",
    "build_materialized_candidate_manifest_payload",
    "build_manifest",
    "build_run_dir",
    "candidate_manifest_path",
    "candidate_json_path",
    "candidate_pair_call_table_path",
    "candidate_profiles_path",
    "candidate_table_path",
    "ensure_run_dirs",
    "ensure_visual_run_dirs",
    "load_manifest",
    "load_status",
    "manifest_path",
    "materialized_candidate_dir",
    "materialized_candidates_dir",
    "nickase_catalog_snapshot_path",
    "nickase_geometry_audit_path",
    "nickase_geometry_audit_table_path",
    "post_terminal_nick_view_path",
    "post_terminal_nick_visual_contract_path",
    "pre_terminal_nick_view_path",
    "pre_terminal_nick_visual_contract_path",
    "release_catalog_snapshot_path",
    "report_json_path",
    "report_md_path",
    "scar_nick_terminal_nick_job_path",
    "scar_nick_terminal_nick_visual_contracts_path",
    "snapshot_inputs",
    "spec_snapshot_path",
    "status_path",
    "views_dir",
    "views_manifest_path",
    "write_candidate_pair_call_table",
    "write_candidate_table",
    "write_jsonl_records",
    "write_manifest",
    "write_materialized_candidate_manifest",
    "write_nickase_geometry_audit_table",
    "write_report",
    "write_status",
    "write_visual_bundle",
]

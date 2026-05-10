"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/scar_nick/reporting.py

Markdown reporting for scar_nick evaluation bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.scar_nick.models import ScarNickEvaluationReport


def _workspace_relative_run_dir(report: ScarNickEvaluationReport) -> str | None:
    if not report.run_dir:
        return None
    try:
        return str(Path(report.run_dir).resolve().relative_to(Path(report.workspace_root).resolve()))
    except ValueError:
        return Path(report.run_dir).name


def render_markdown_report(report: ScarNickEvaluationReport) -> str:
    run_dir = _workspace_relative_run_dir(report)
    lines = [
        f"# scar_nick report: {report.spec_name}",
        "",
        f"- status: {report.status}",
        f"- workflow: {report.workflow}",
        f"- terminal_boundary: {report.metadata.terminal_boundary}",
        f"- release_variant: {report.metadata.release_variant_id}",
        f"- accepted_candidates: {len(report.candidates)}",
        f"- compatible_nickase_placements: {report.metadata.compatible_nickase_placement_count}",
        f"- enzyme_compatible_scars: {report.metadata.enzyme_compatible_scar_count}",
    ]
    if run_dir:
        lines.append(f"- run_dir: {run_dir}")
    lines.extend(
        [
            "",
            "## Handoff Tables",
            "",
            "- candidate_table: `export/table__scar_nick_candidates.csv`",
            "- candidate_pair_call_table: `export/table__scar_nick_candidate_pair_calls.csv`",
            "- nickase_geometry_audit_table: `export/table__scar_nick_nickase_geometry_audit.csv`",
        ]
    )
    if report.candidates:
        lines.extend(["", "## Candidates"])
        for candidate in report.candidates:
            lines.append(
                f"- rank {candidate.rank}: `{candidate.left_base}/{candidate.right_base}` "
                f"profile={candidate.profile_s3s2s1s0} "
                f"policy={candidate.profile_policy_status}:{candidate.profile_policy_reason} "
                f"non_wc={candidate.non_watson_crick_count} "
                f"middle_hard={candidate.middle_hard_count} "
                f"hard_tier={candidate.hard_mismatch_tier_sum} "
                f"middle_hard_tier={candidate.middle_hard_mismatch_tier_sum} "
                f"nick={candidate.nickase_site}"
            )
    if report.reserve_candidates:
        lines.extend(["", "## Reserve Profile Examples"])
        for candidate in report.reserve_candidates:
            lines.append(
                f"- `{candidate.left_base}/{candidate.right_base}` "
                f"profile={candidate.profile_s3s2s1s0} "
                f"policy={candidate.profile_policy_status}:{candidate.profile_policy_reason} "
                f"non_wc={candidate.non_watson_crick_count} "
                f"nick={candidate.nickase_site}"
            )
    if report.issues:
        lines.extend(["", "## Issues"])
        for issue in report.issues:
            lines.append(f"- {issue.code}: {issue.message}")
    return "\n".join(lines) + "\n"


__all__ = ["render_markdown_report"]

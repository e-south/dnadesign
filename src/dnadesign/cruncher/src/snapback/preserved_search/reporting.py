"""
Reporting helpers for preserved-site target search.
"""

from __future__ import annotations

from dnadesign.cruncher.snapback.target_models import SnapbackTargetSearchReport


def render_target_search_markdown_report(report: SnapbackTargetSearchReport) -> str:
    lines = [
        "# Snapback Target Search Report",
        "",
        f"- status: {report.status}",
        f"- catalog_source: {report.metadata.catalog_source}",
        f"- target_boundary: {report.metadata.target.nick_boundary_from_left}",
        f"- target_paired_bp: {report.metadata.target.paired_bp}",
        f"- target_cap_nt: {report.metadata.target.cap_nt}",
        f"- exact_hit_count: {report.metadata.exact_hit_count}",
        f"- near_hit_count: {report.metadata.near_hit_count}",
        f"- evaluated_orientation_count: {report.metadata.evaluated_orientation_count}",
    ]
    if report.exact_hits:
        lines.extend(["", "## Exact Hits"])
        for hit in report.exact_hits:
            outside_site = hit.nickase.selection.outside_site if hit.nickase.selection is not None else "unknown"
            lines.append(
                f"- rank {hit.rank}: {hit.variant_id} boundary={hit.nick_boundary_from_left} "
                f"site={hit.intended_site_orientation}:{hit.intended_site_sequence} "
                f"input_nt={hit.input_length_nt} extra_target_nicks={hit.extra_target_strand_nick_count} "
                f"extra_nicks={hit.extra_nick_event_count} outside_site={outside_site}"
            )
    if report.near_hits:
        lines.extend(["", "## Near Hits"])
        for hit in report.near_hits:
            outside_site = hit.nickase.selection.outside_site if hit.nickase.selection is not None else "unknown"
            lines.append(
                f"- rank {hit.rank}: {hit.variant_id} boundary={hit.nick_boundary_from_left} "
                f"site={hit.intended_site_orientation}:{hit.intended_site_sequence} "
                f"input_nt={hit.input_length_nt} extra_target_nicks={hit.extra_target_strand_nick_count} "
                f"extra_nicks={hit.extra_nick_event_count} outside_site={outside_site}"
            )
    if report.feasibility:
        lines.extend(["", "## Feasibility"])
        for row in report.feasibility:
            blockers = ",".join(row.exact_boundary_blockers) if row.exact_boundary_blockers else "-"
            lines.append(
                f"- {row.variant_id} {row.orientation} exact={row.exact_boundary_hit_possible} "
                f"target_site_start={row.site_start_at_target_boundary} earliest_boundary="
                f"{row.earliest_feasible_boundary if row.earliest_feasible_boundary is not None else '-'} "
                f"blockers={blockers}"
            )
    return "\n".join(lines).strip() + "\n"

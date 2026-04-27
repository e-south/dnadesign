"""
Presentation helpers for Snapback CLI commands.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()


def print_report(report) -> None:
    console.print(f"Snapback spec -> {report.spec_name}")
    console.print(f"Status -> {report.status}")
    if report.candidate is not None:
        console.print(
            "Intended nick -> "
            f"{report.candidate.intended_nick.variant_id}@{report.candidate.nick_boundary} "
            f"({report.candidate.intended_site.orientation})"
        )
        console.print(f"Released prefix nt -> {report.candidate.released_prefix_nt}")
        console.print(f"Cap nt -> {report.candidate.cap_nt}")
        console.print(f"Added nt -> {report.candidate.added_nt}")
        console.print(f"Terminal ligatable duplex bp -> {report.candidate.terminal_ligatable_duplex_bp}")
        console.print(f"Max uninterrupted duplex bp -> {report.candidate.max_uninterrupted_duplex_bp}")
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


def print_solve_report(report) -> None:
    console.print(f"Snapback solve spec -> {report.spec_path}")
    console.print(f"Status -> {report.status}")
    if report.solve_id:
        console.print(f"Solve id -> {report.solve_id}")
    if report.run_dir:
        console.print(f"Outputs -> {report.run_dir}")
    resolved = report.metadata.resolved_search_space
    console.print(
        "Resolved compact search -> "
        f"boundary={resolved.nick_boundary_window.min}..{resolved.nick_boundary_window.max}, "
        f"paired_bp={resolved.retained_homology_length.min}..{resolved.retained_homology_length.max}, "
        f"terminal_duplex={resolved.terminal_ligatable_duplex_bp.min}..{resolved.terminal_ligatable_duplex_bp.max}, "
        f"max_uninterrupted={resolved.max_uninterrupted_duplex_bp}"
    )
    console.print(
        "Search -> "
        f"nodes={report.metadata.visited_search_node_count}, "
        f"enumerated={report.metadata.enumerated_candidate_count}, "
        f"accepted={report.metadata.accepted_candidate_count}, "
        f"frontier_rows={report.metadata.frontier_row_count}, "
        f"materialized={report.metadata.materialized_hit_count}"
    )
    if report.metadata.first_satisfied_frontier is not None:
        frontier = report.metadata.first_satisfied_frontier
        console.print(
            "First satisfied frontier -> "
            f"boundary={frontier.nick_boundary_from_left}, "
            f"paired_bp={frontier.paired_bp}, "
            f"cap_ext_nt={frontier.cap_extension_nt}, "
            f"accepted={frontier.accepted_candidate_count}"
        )
    for code, warning in zip(report.metadata.warning_codes, report.metadata.warnings, strict=False):
        console.print(f"Warning -> {code}: {warning}")
    if report.hits:
        console.print("Hits:")
        for hit in report.hits:
            line = (
                f"  - rank {hit.rank}: {hit.hit_id} "
                f"{hit.variant_id}@{hit.nick_boundary} "
                f"site={hit.intended_site_orientation}:{hit.intended_site_sequence} "
                f"site_mutations={hit.site_mutation_count} cap={hit.cap_sequence} added_nt={hit.added_nt}"
            )
            if hit.materialized_run_dir is not None:
                line += f" bundle={hit.materialized_run_dir}"
            console.print(line)
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


def print_target_search_report(report) -> None:
    console.print("Snapback target search")
    console.print(f"Status -> {report.status}")
    console.print(
        "Target -> "
        f"boundary={report.metadata.target.nick_boundary_from_left}, "
        f"paired_bp={report.metadata.target.paired_bp}, "
        f"cap_nt={report.metadata.target.cap_nt}"
    )
    console.print(f"Catalog -> {report.metadata.catalog_source}")
    console.print(f"Orientations evaluated -> {report.metadata.evaluated_orientation_count}")
    console.print(f"Exact hits -> {report.metadata.exact_hit_count}")
    console.print(f"Near hits -> {report.metadata.near_hit_count}")
    if report.exact_hits:
        console.print("Exact hits:")
        for hit in report.exact_hits:
            outside_site = hit.nickase.selection.outside_site if hit.nickase.selection is not None else "unknown"
            console.print(
                "  - "
                f"rank {hit.rank}: {hit.variant_id} "
                f"boundary={hit.nick_boundary_from_left} "
                f"site={hit.intended_site_orientation}:{hit.intended_site_sequence} "
                f"input_nt={hit.input_length_nt} "
                f"extra_target_nicks={hit.extra_target_strand_nick_count} "
                f"extra_nicks={hit.extra_nick_event_count} "
                f"outside_site={outside_site}"
            )
    if report.near_hits:
        console.print("Near hits:")
        for hit in report.near_hits:
            outside_site = hit.nickase.selection.outside_site if hit.nickase.selection is not None else "unknown"
            console.print(
                "  - "
                f"rank {hit.rank}: {hit.variant_id} "
                f"boundary={hit.nick_boundary_from_left} "
                f"site={hit.intended_site_orientation}:{hit.intended_site_sequence} "
                f"input_nt={hit.input_length_nt} "
                f"extra_target_nicks={hit.extra_target_strand_nick_count} "
                f"extra_nicks={hit.extra_nick_event_count} "
                f"outside_site={outside_site}"
            )
    if report.feasibility:
        console.print("Feasibility:")
        for row in report.feasibility[:8]:
            blockers = ",".join(row.exact_boundary_blockers) if row.exact_boundary_blockers else "-"
            console.print(
                "  - "
                f"{row.variant_id} {row.orientation} "
                f"exact={row.exact_boundary_hit_possible} "
                f"target_site_start={row.site_start_at_target_boundary} "
                "earliest_boundary="
                f"{row.earliest_feasible_boundary if row.earliest_feasible_boundary is not None else '-'} "
                f"blockers={blockers}"
            )


def print_released_report(report) -> None:
    console.print(f"Released-product snapback spec -> {report.spec_name}")
    console.print(f"Status -> {report.status}")
    console.print(f"Nick catalog -> {report.metadata.nick_catalog_source}")
    console.print(f"Release catalog -> {report.metadata.release_catalog_source}")
    if report.candidate is not None and report.projection is not None:
        console.print(
            f"Active {report.projection.active_strand} -> "
            f"route={report.candidate.route_family} "
            f"input_nt={report.candidate.active_product_input_length_nt} "
            f"product_nt={report.candidate.active_product_length_nt} "
            f"nick={report.candidate.nick_boundary_from_left} "
            f"paired_bp={report.candidate.paired_bp} cap_nt={report.candidate.cap_nt}"
        )
        console.print(
            "Release cuts -> "
            f"top={report.projection.release_top_cut_precursor} "
            f"bottom={report.projection.release_bottom_cut_precursor}"
        )
        console.print(
            "Site survival -> "
            f"nickase={report.projection.nickase_site_survives_post_release} "
            f"release={report.projection.release_site_survives_post_release}"
        )
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


def print_released_target_search_report(report) -> None:
    console.print("Released-product snapback target search")
    console.print(f"Status -> {report.status}")
    console.print(
        "Target -> "
        f"boundary={report.metadata.target.nick_boundary_from_left}, "
        f"paired_bp={report.metadata.target.paired_bp}, "
        f"cap_nt={report.metadata.target.cap_nt}"
    )
    console.print(f"Nick catalog -> {report.metadata.nick_catalog_source}")
    console.print(f"Release catalog -> {report.metadata.release_catalog_source}")
    console.print(
        "Route policy -> "
        f"policy_final_geometry={report.metadata.route_policy_final_geometry_source} "
        f"active={','.join(report.metadata.allowed_active_strands)} "
        f"routes={','.join(report.metadata.allowed_route_families)}"
    )
    console.print(f"Pairs evaluated -> {report.metadata.evaluated_pair_count}")
    console.print(
        "Hits -> "
        f"exact={report.metadata.post_truncation_exact_hit_count}/"
        f"{report.metadata.pre_truncation_exact_hit_count}, "
        f"near={report.metadata.post_truncation_near_hit_count}/"
        f"{report.metadata.pre_truncation_near_hit_count}"
    )
    if report.metadata.blocker_counts:
        console.print("Blockers:")
        for code, count in sorted(report.metadata.blocker_counts.items()):
            console.print(f"  - {code}: {count}")
    if report.exact_hits:
        console.print("Exact hits:")
        for hit in report.exact_hits:
            console.print(
                "  - "
                f"rank {hit.rank}: {hit.nickase_variant_id}+{hit.release_variant_id} "
                f"route={hit.route_family} "
                f"active={hit.active_strand} "
                f"hit_final_geometry={hit.projection.final_geometry_source} "
                f"boundary={hit.nick_boundary_from_left} "
                f"active_input_nt={hit.active_product_input_length_nt} "
                f"precursor_nt={hit.precursor_length_nt} "
                f"tail_nt={hit.sacrificial_downstream_tail_nt}"
            )
    if report.near_hits:
        console.print("Near hits:")
        for hit in report.near_hits:
            console.print(
                "  - "
                f"rank {hit.rank}: {hit.nickase_variant_id}+{hit.release_variant_id} "
                f"route={hit.route_family} "
                f"active={hit.active_strand} "
                f"hit_final_geometry={hit.projection.final_geometry_source} "
                f"boundary={hit.nick_boundary_from_left} "
                f"active_input_nt={hit.active_product_input_length_nt} "
                f"precursor_nt={hit.precursor_length_nt} "
                f"tail_nt={hit.sacrificial_downstream_tail_nt}"
            )


def print_snapback_screen_report(report, *, emit_mechanism_ledger: bool = True) -> None:
    console.print("Snapback screen")
    console.print(f"Status -> {report.status}")
    console.print(
        "Target topology -> "
        f"origin={report.target_topology.logical_origin}, "
        f"stem_bp={report.target_topology.stem_bp}, "
        f"cap_nt={report.target_topology.cap_nt}, "
        f"retained={','.join(report.target_topology.retained_product_strands)}"
    )
    console.print(f"Exact hits -> {report.exact_hit_count}")
    console.print(f"Near hits -> {report.near_hit_count}")
    if emit_mechanism_ledger and report.mechanism_ledger:
        console.print("Mechanism ledger:")
        for entry in report.mechanism_ledger:
            provenance = ",".join(f"{key}={count}" for key, count in entry.provenance_counts.items()) or "-"
            console.print(
                "  - "
                f"rank {entry.rank}: {entry.nickase_variant_id}+{entry.release_variant_id} "
                f"kind={entry.hit_kind} "
                f"route={entry.route_family} "
                f"retained={entry.retained_product_strand} "
                f"physical_nick={entry.physical_nicked_strand} "
                f"origin={entry.logical_origin} "
                f"stem={entry.logical_stem_bp} "
                f"cap={entry.cap_nt} "
                f"effective_pairing={entry.effective_foldback_pairing_bp} "
                f"class={entry.mechanism_class} "
                f"provenance={provenance}"
            )


def print_released_solve_report(report) -> None:
    console.print("Released-product snapback solve")
    console.print(f"Status -> {report.status}")
    if report.run_dir:
        console.print(f"Outputs -> {report.run_dir}")
    console.print(
        "Target -> "
        f"boundary={report.metadata.target.nick_boundary_from_left}, "
        f"paired_bp={report.metadata.target.paired_bp}, "
        f"cap_nt={report.metadata.target.cap_nt}"
    )
    console.print(f"Nick catalog -> {report.metadata.nick_catalog_source}")
    console.print(f"Release catalog -> {report.metadata.release_catalog_source}")
    console.print(
        "Route policy -> "
        f"policy_final_geometry={report.metadata.route_policy_final_geometry_source} "
        f"active={','.join(report.metadata.allowed_active_strands)} "
        f"routes={','.join(report.metadata.allowed_route_families)}"
    )
    console.print(f"Pairs evaluated -> {report.metadata.evaluated_pair_count}")
    console.print(
        "Available hits -> "
        f"exact={report.metadata.available_exact_hit_count}, "
        f"near={report.metadata.available_near_hit_count}, "
        f"selected={report.metadata.selected_hit_kind or '-'}"
    )
    console.print(
        "Materialized -> "
        f"{report.metadata.materialized_hit_count}/{report.metadata.requested_materialize_top_k} "
        f"render_format={report.metadata.render_format} "
        f"emit_renders={report.metadata.emit_renders}"
    )
    if report.hits:
        console.print("Hits:")
        for hit in report.hits:
            line = (
                f"  - rank {hit.rank}: {hit.nickase_variant_id}+{hit.release_variant_id} "
                f"kind={hit.hit_kind} "
                f"route={hit.target_search_hit.route_family} "
                f"active={hit.target_search_hit.active_strand} "
                f"hit_final_geometry={hit.target_search_hit.projection.final_geometry_source} "
                f"bundle={hit.materialized_run_dir}"
            )
            if hit.rendered_plot_path is not None:
                line += f" plot={hit.rendered_plot_path}"
            console.print(line)
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


def echo_scaffold_line(label: str, value: str | Path) -> None:
    console.print(f"{label} -> {value}")


__all__ = [
    "console",
    "echo_scaffold_line",
    "print_released_report",
    "print_released_solve_report",
    "print_released_target_search_report",
    "print_report",
    "print_snapback_screen_report",
    "print_solve_report",
    "print_target_search_report",
]

"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli_render.py

Shared Rich table builders for CLI output.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

from .utils.rich_style import make_table


def stage_a_plan_table(plan_rows: Iterable[dict[str, str]]):
    table = make_table()
    table.add_column("input", overflow="fold")
    table.add_column("TF", overflow="fold")
    table.add_column("retain")
    table.add_column("budget")
    table.add_column("eligibility")
    table.add_column("selection")
    table.add_column("uniqueness")
    table.add_column("length")
    for row in plan_rows:
        table.add_row(
            str(row.get("input", "-")),
            str(row.get("tf", "-")),
            str(row.get("retain", "-")),
            str(row.get("budget", "-")),
            str(row.get("eligibility", "-")),
            str(row.get("selection", "-")),
            str(row.get("uniqueness", "-")),
            str(row.get("length", "-")),
        )
    return table


def stage_a_recap_tables(
    recap_rows: Iterable[dict[str, object]],
    *,
    display_map_by_input: dict[str, dict[str, str]],
    show_motif_ids: bool,
) -> list[tuple[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in recap_rows:
        grouped.setdefault(str(row["input_name"]), []).append(row)

    tables: list[tuple[str, object]] = []
    for input_name in sorted(grouped):
        recap_table = make_table()
        recap_table.add_column("TF", overflow="fold")
        recap_table.add_column("generated")
        recap_table.add_column("eligible_raw")
        recap_table.add_column("eligible_unique")
        recap_table.add_column("retained")
        recap_table.add_column("tier target")
        recap_table.add_column("tier fill")
        recap_table.add_column("selection")
        recap_table.add_column("k(pool/target)")
        recap_table.add_column("div(k5)")
        recap_table.add_column("Δdiv(k5)")
        recap_table.add_column("set_overlap")
        recap_table.add_column("set_swaps")
        recap_table.add_column("Δscore(p10)")
        recap_table.add_column("Δscore(med)")
        recap_table.add_column("score(min/med/avg/max)")
        recap_table.add_column("len(n/min/med/avg/max)")
        for row in sorted(grouped[input_name], key=lambda item: str(item["regulator"])):
            reg_label = str(row["regulator"])
            if not show_motif_ids:
                reg_label = display_map_by_input.get(input_name, {}).get(reg_label, reg_label)
            recap_table.add_row(
                reg_label,
                str(row["generated"]),
                str(row["eligible_raw"]),
                str(row["eligible_unique"]),
                str(row["retained"]),
                str(row.get("tier_target", "-")),
                str(row["tier_fill"]),
                str(row.get("selection", "-")),
                str(row.get("diversity_pool", "-")),
                str(row.get("diversity_med", "-")),
                str(row.get("diversity_delta", "-")),
                str(row.get("set_overlap", "-")),
                str(row.get("set_swaps", "-")),
                str(row.get("diversity_score_p10_delta", "-")),
                str(row.get("diversity_score_med_delta", "-")),
                str(row["score"]),
                str(row["length"]),
            )
        tables.append((f"Input: {input_name}", recap_table))

        tier_rows = [
            row
            for row in grouped[input_name]
            if row.get("tier0_score") is not None
            or row.get("tier1_score") is not None
            or row.get("tier2_score") is not None
        ]
        if tier_rows:
            boundary_table = make_table("TF", "tier0.1% score", "tier1% score", "tier9% score")
            for row in sorted(tier_rows, key=lambda item: str(item["regulator"])):
                reg_label = str(row["regulator"])
                if not show_motif_ids:
                    reg_label = display_map_by_input.get(input_name, {}).get(reg_label, reg_label)
                t0 = row.get("tier0_score")
                t1 = row.get("tier1_score")
                t2 = row.get("tier2_score")
                boundary_table.add_row(
                    reg_label,
                    f"{float(t0):.2f}" if t0 is not None else "-",
                    f"{float(t1):.2f}" if t1 is not None else "-",
                    f"{float(t2):.2f}" if t2 is not None else "-",
                )
            tables.append(("", boundary_table))
    return tables

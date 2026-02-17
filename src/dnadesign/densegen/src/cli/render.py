"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli/render.py

Shared Rich table builders for CLI output.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

from ..utils.rich_style import make_table


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
            str(row["input"]),
            str(row["tf"]),
            str(row["retain"]),
            str(row["budget"]),
            str(row["eligibility"]),
            str(row["selection"]),
            str(row["uniqueness"]),
            str(row["length"]),
        )
    return table


def stage_a_recap_tables(
    recap_rows: Iterable[dict[str, object]],
    *,
    display_map_by_input: dict[str, dict[str, str]],
    show_motif_ids: bool,
    verbose: bool = False,
) -> list[tuple[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in recap_rows:
        grouped.setdefault(str(row["input_name"]), []).append(row)

    tables: list[tuple[str, object]] = []
    for input_name in sorted(grouped):
        recap_table = make_table()
        recap_table.add_column("TF", overflow="fold")
        recap_table.add_column("generated")
        if verbose:
            recap_table.add_column("has_hit")
            recap_table.add_column("eligible_raw")
        recap_table.add_column("eligible_unique")
        recap_table.add_column("pool")
        recap_table.add_column("retained")
        if verbose:
            recap_table.add_column("tier target")
        recap_table.add_column("tier fill")
        recap_table.add_column("selection")
        recap_table.add_column("overlap")
        if verbose:
            recap_table.add_column("set_swaps")
        recap_table.add_column("pairwise top")
        recap_table.add_column("pairwise div")
        recap_table.add_column("score_norm top (min/med/max)")
        recap_table.add_column("score_norm div (min/med/max)")
        recap_table.add_column("fimo(min/med/avg/max)")
        recap_table.add_column("len(n/min/med/avg/max)")
        for row in sorted(grouped[input_name], key=lambda item: str(item["regulator"])):
            reg_label = str(row["regulator"])
            if not show_motif_ids:
                reg_label = display_map_by_input.get(input_name, {}).get(reg_label, reg_label)
            pool_label = str(row["diversity_pool"])
            pool_source = str(row["diversity_pool_source"])
            if verbose and pool_source not in {"-", ""}:
                pool_label = f"{pool_label} ({pool_source})"
            recap_row = [reg_label, str(row["generated"])]
            if verbose:
                recap_row.extend([str(row["has_hit"]), str(row["eligible_raw"])])
            recap_row.extend([str(row["eligible_unique"]), pool_label, str(row["retained"])])
            if verbose:
                recap_row.append(str(row["tier_target"]))
            recap_row.extend(
                [
                    str(row["tier_fill"]),
                    str(row["selection"]),
                    str(row["set_overlap"]),
                ]
            )
            if verbose:
                recap_row.extend([str(row["set_swaps"])])
            recap_row.extend(
                [
                    str(row["pairwise_top"]),
                    str(row["pairwise_div"]),
                    str(row["score_norm_top"]),
                    str(row["score_norm_div"]),
                    str(row["score"]),
                    str(row["length"]),
                ]
            )
            recap_table.add_row(*recap_row)
        tables.append((f"Input: {input_name}", recap_table))

        tier_rows = [
            row
            for row in grouped[input_name]
            if row["tier0_score"] is not None or row["tier1_score"] is not None or row["tier2_score"] is not None
        ]
        if tier_rows:
            boundary_table = make_table("TF", "tier0.1% score", "tier1% score", "tier9% score")
            for row in sorted(tier_rows, key=lambda item: str(item["regulator"])):
                reg_label = str(row["regulator"])
                if not show_motif_ids:
                    reg_label = display_map_by_input.get(input_name, {}).get(reg_label, reg_label)
                t0 = row["tier0_score"]
                t1 = row["tier1_score"]
                t2 = row["tier2_score"]
                boundary_table.add_row(
                    reg_label,
                    f"{float(t0):.2f}" if t0 is not None else "n/a",
                    f"{float(t1):.2f}" if t1 is not None else "n/a",
                    f"{float(t2):.2f}" if t2 is not None else "n/a",
                )
            tables.append(("", boundary_table))
    return tables

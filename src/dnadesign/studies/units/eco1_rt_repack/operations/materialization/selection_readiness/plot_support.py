"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/plot_support.py

Shared helpers for Eco1 selection-readiness plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)


def plot_row(
    *,
    plot_id: str,
    title: str,
    path: Path,
    input_hashes: dict[str, str | None],
    alt_text: str,
    description: str,
    interpretation_limit: str,
    render_mode: str,
) -> dict[str, Any]:
    return {
        "plot_id": plot_id,
        "title": title,
        "artifact_kind": "svg",
        "status": "rendered",
        "path": str(path),
        "data_sources": [
            "design_classes/selection/feasibility_report.parquet",
            "design_classes/selection/candidate_triage_table.parquet",
            "design_classes/selection/candidate_selection_panel.parquet",
        ],
        "input_hashes": {key: value for key, value in input_hashes.items() if value is not None},
        "alt_text": alt_text,
        "description": description,
        "interpretation_limit": interpretation_limit,
        "role": "manuscript_facing",
        "render_mode": render_mode,
    }


def tie_break_trace(row: dict[str, object]) -> dict[str, object]:
    loaded = json.loads(str(row["tie_break_trace_json"]))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected tie_break_trace_json to contain a JSON object for {row['candidate_id']}")
    return {"candidate_id": row["candidate_id"], **loaded}


def ordered_panel_rows(panel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_class = {str(row["design_class_id"]): row for row in panel_rows}
    ordered = [by_class[spec.design_class_id] for spec in ALL_SPECS if spec.design_class_id in by_class]
    known_classes = {spec.design_class_id for spec in ALL_SPECS}
    extra = sorted(
        [row for row in panel_rows if str(row["design_class_id"]) not in known_classes],
        key=lambda row: (str(row["design_class_id"]), str(row["candidate_id"])),
    )
    return ordered + extra


def canonical_mutations(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(entry) for entry in value]
    if isinstance(value, tuple):
        return [str(entry) for entry in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(loaded, (list, tuple)):
        return [str(entry) for entry in loaded]
    return [str(loaded)]


def parse_mutation(mutation: str) -> dict[str, int | str]:
    if len(mutation) < 3:
        raise ValueError(f"Malformed canonical mutation in selected panel sequence plot: {mutation!r}")
    position_text = mutation[1:-1]
    if not position_text.isdigit():
        raise ValueError(f"Malformed canonical mutation in selected panel sequence plot: {mutation!r}")
    return {"wt": mutation[0], "position": int(position_text), "alt": mutation[-1]}


def mutation_category(wt: str, alt: str) -> int:
    if alt in {"P", "G"} and wt != alt:
        return 6
    if alt in {"K", "R", "H"} and wt not in {"K", "R", "H"}:
        return 2
    if wt in {"K", "R", "H"} and alt not in {"K", "R", "H"}:
        return 3
    if alt in {"D", "E"} and wt not in {"D", "E"}:
        return 4
    if wt in {"D", "E"} and alt not in {"D", "E"}:
        return 5
    return 1


def position_tick_indices(position_count: int) -> list[int]:
    if position_count <= 80:
        return list(range(position_count))
    step = 20 if position_count <= 340 else 25
    ticks = list(range(0, position_count, step))
    if ticks[-1] != position_count - 1:
        ticks.append(position_count - 1)
    return ticks


def class_label(class_id: str) -> str:
    for spec in ALL_SPECS:
        if spec.design_class_id == class_id:
            denominator = "clade 9" if spec.conservation_profile_id.endswith("clade9_conservation_v1") else "subtype"
            threshold = int(round(spec.conservation_threshold * 100))
            contact = int(round(spec.contact_threshold_angstrom))
            return f"{denominator} p{threshold}, {contact} A"
    raise ValueError(f"Unknown Eco1 design class id for plot label: {class_id}")


def legend_sizes(values: list[int]) -> list[int]:
    if not values:
        return []
    candidates = [min(values), int(round(sum(values) / len(values))), max(values)]
    return sorted(set(candidates))


def short_candidate(candidate_id: str) -> str:
    prefix = "thread_candidate_"
    return candidate_id.replace(prefix, "") if candidate_id.startswith(prefix) else candidate_id


__all__ = [
    "canonical_mutations",
    "class_label",
    "legend_sizes",
    "mutation_category",
    "ordered_panel_rows",
    "parse_mutation",
    "plot_row",
    "position_tick_indices",
    "short_candidate",
    "tie_break_trace",
]

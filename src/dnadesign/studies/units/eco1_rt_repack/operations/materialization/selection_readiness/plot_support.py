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

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    GENERATION_POLICY_VERSION,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    VARIANT_ID_PREFIX,
)

_GENERATION_POLICY_LABELS = {
    DISTAL_SCAFFOLD_POLICY_ID: "distal scaffold",
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID: "near DNA/RNA",
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID: "near DNA/RNA plus distal",
}


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
    role: str = "manuscript_facing",
) -> dict[str, Any]:
    return {
        "plot_id": plot_id,
        "title": title,
        "artifact_kind": "svg",
        "status": "rendered",
        "path": str(path),
        "data_sources": [
            "selection/candidate_triage_table.parquet",
            "selection/candidate_selection_panel.parquet",
        ],
        "input_hashes": {key: value for key, value in input_hashes.items() if value is not None},
        "alt_text": alt_text,
        "description": description,
        "interpretation_limit": interpretation_limit,
        "role": role,
        "render_mode": render_mode,
    }


def tie_break_trace(row: dict[str, object]) -> dict[str, object]:
    loaded = json.loads(str(row["tie_break_trace_json"]))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected tie_break_trace_json to contain a JSON object for {row['candidate_id']}")
    return {"candidate_id": row["candidate_id"], **loaded}


def ordered_panel_rows(panel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        panel_rows,
        key=lambda row: (
            int(row.get("selection_rank") or 9999),
            str(row.get("selection_slot") or ""),
            str(row["policy_id"]),
            str(row["candidate_id"]),
        ),
    )


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


def policy_label(policy_id: str) -> str:
    try:
        return _GENERATION_POLICY_LABELS[policy_id]
    except KeyError as exc:
        raise ValueError(f"Unknown Eco1 generation policy id for plot label: {policy_id}") from exc


def legend_sizes(values: list[int]) -> list[int]:
    if not values:
        return []
    candidates = [min(values), int(round(sum(values) / len(values))), max(values)]
    return sorted(set(candidates))


def short_candidate(candidate_id: str) -> str:
    prefix = "thread_candidate_"
    return candidate_id.replace(prefix, "") if candidate_id.startswith(prefix) else candidate_id


def short_selected_variant(row: dict[str, object]) -> str:
    """Return a compact stable alias for one selected row."""

    variant_id = str(row.get("variant_id") or "")
    if variant_id:
        return variant_id.removeprefix(f"{VARIANT_ID_PREFIX}-G{GENERATION_POLICY_VERSION}-")
    return short_candidate(str(row.get("candidate_id") or ""))


def matrix_text_color(value: float, *, max_value: float) -> str:
    return "#ffffff" if max_value > 0 and value >= max_value * 0.55 else "#24292f"


__all__ = [
    "canonical_mutations",
    "legend_sizes",
    "matrix_text_color",
    "mutation_category",
    "ordered_panel_rows",
    "parse_mutation",
    "plot_row",
    "policy_label",
    "position_tick_indices",
    "short_candidate",
    "short_selected_variant",
    "tie_break_trace",
]

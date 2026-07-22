"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/selection_plot_fixtures.py

Panel-selection plot fixture helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    SELECTION_PLOT_METADATA,
)


def plot_row(
    *,
    plot_id: str,
    title: str,
    path: str,
    alt_text: str,
    description: str,
    interpretation_limit: str,
    input_hash_tail: str,
) -> dict[str, object]:
    row = {
        "plot_id": plot_id,
        "title": title,
        "artifact_kind": "svg",
        "status": "rendered",
        "path": path,
        "data_sources": ["selection/candidate_selection_panel.parquet"],
        "input_hashes": {"candidate_selection_panel": "sha256:" + input_hash_tail * 64},
        "alt_text": alt_text,
        "description": description,
        "interpretation_limit": interpretation_limit,
        "role": "manuscript_facing",
        "render_mode": "wide_visual",
    }
    row.update(SELECTION_PLOT_METADATA.get(plot_id, {}))
    return row


def write_svg(path: Path, *, plot_id: str, title: str) -> None:
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" role="img" width="320" height="180">
<title>{title}</title>
<desc>Fixture panel-selection visual for review-deliverable linking.</desc>
<text x="20" y="90">{plot_id}</text>
</svg>
""",
        encoding="utf-8",
    )

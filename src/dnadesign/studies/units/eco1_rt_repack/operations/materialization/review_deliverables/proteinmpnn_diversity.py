"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/proteinmpnn_diversity.py

ProteinMPNN candidate-diversity panels for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import pyarrow.parquet as pq
from matplotlib.lines import Line2D

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    save_accessible_svg,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_MUTATION_RE = re.compile(r"^[A-Z*](?P<position>[0-9]+)[A-Z*]$")
_TEMPERATURE_COLORS = {
    0.1: "#386c55",
    0.3: "#d19a33",
}


def write_proteinmpnn_diversity_panels(
    *,
    panel_root: Path,
    candidate_table_path: Path,
) -> list[dict[str, Any]]:
    """Render compact ProteinMPNN diversity panels."""

    rows = pq.read_table(
        candidate_table_path,
        columns=[
            "candidate_id",
            "score",
            "global_score",
            "seq_recovery",
            "seed",
            "temperature",
            "mutation_count",
            "canonical_mutations",
            "status",
        ],
    ).to_pylist()
    accepted_rows = [row for row in rows if str(row.get("status")) == "accepted"]
    if not accepted_rows:
        raise ValueError(f"No accepted candidate rows found in {candidate_table_path}")
    return [
        _write_score_mutation_burden(panel_root, accepted_rows, candidate_table_path),
        _write_mutation_density(panel_root, accepted_rows, candidate_table_path),
    ]


def _write_score_mutation_burden(
    panel_root: Path,
    rows: list[dict[str, Any]],
    candidate_table_path: Path,
) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    temperatures = [float(row["temperature"]) for row in rows]
    colors = [_temperature_color(temperature) for temperature in temperatures]
    axes[0].scatter(
        [int(row["mutation_count"]) for row in rows],
        [float(row["score"]) for row in rows],
        c=colors,
        s=40,
        edgecolors="#ffffff",
        linewidths=0.35,
    )
    axes[0].set_xlabel("Mutation count")
    axes[0].set_ylabel("ProteinMPNN score (lower is better)")
    axes[0].set_title("ProteinMPNN score varies with mutation burden.", fontsize=12)
    axes[0].grid(alpha=0.25)
    axes[1].scatter(
        [float(row["seq_recovery"]) * 100.0 for row in rows],
        [float(row["global_score"]) for row in rows],
        c=colors,
        s=40,
        edgecolors="#ffffff",
        linewidths=0.35,
    )
    axes[1].set_xlabel("Sequence identity to Ec86 WT (%)")
    axes[1].set_ylabel("ProteinMPNN global score (lower is better)")
    axes[1].set_title("Accepted designs span sequence identity to Ec86 WT.", fontsize=12)
    axes[1].grid(alpha=0.25)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
    axes[1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=_temperature_color(temperature),
                markeredgecolor="#ffffff",
                label=f"Temperature {temperature:g}",
            )
            for temperature in sorted(set(temperatures))
        ],
        frameon=False,
        fontsize=10,
        loc="best",
    )
    fig.tight_layout()

    path = panel_root / "proteinmpnn_score_mutation_burden.svg"
    alt = (
        f"ProteinMPNN diversity panel for {len(rows)} accepted candidates, showing "
        "mutation count versus score and WT sequence recovery versus global score."
    )
    save_accessible_svg(fig, path, title="ProteinMPNN sampled two temperature settings.", description=alt)
    return make_deliverable_row(
        deliverable_id="proteinmpnn_score_mutation_burden",
        section="proteinmpnn",
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=["candidate_table.parquet"],
        input_hashes=file_hashes({"candidate_table": candidate_table_path}),
        alt_text=alt,
        description=(
            "Summarizes ProteinMPNN proposal scores, global scores, mutation burden, "
            "and sequence identity for accepted candidates."
        ),
        interpretation_limit=(
            "Sequence recovery and ProteinMPNN scores are descriptive proposal metrics. "
            "They are not fold, synthesis, or activity acceptance criteria."
        ),
    )


def _write_mutation_density(
    panel_root: Path,
    rows: list[dict[str, Any]],
    candidate_table_path: Path,
) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    for row in rows:
        for mutation in row.get("canonical_mutations") or []:
            match = _MUTATION_RE.match(str(mutation))
            if match:
                counts[int(match.group("position"))] += 1
    positions = sorted(counts)
    fig, ax = plt.subplots(figsize=(12.0, 3.8))
    ax.bar(positions, [counts[position] for position in positions], width=0.9, color="#4f7f9f")
    ax.set_xlabel("Ec86 canonical residue position", fontsize=11)
    ax.set_ylabel("Candidate mutation count", fontsize=11)
    ax.set_title("ProteinMPNN mutations concentrate in the mutable design canvas.", fontsize=13, pad=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()

    path = panel_root / "proteinmpnn_mutation_density.svg"
    alt = (
        f"Bar chart of mutation density across Ec86 residue positions for {len(rows)} accepted ProteinMPNN candidates."
    )
    save_accessible_svg(fig, path, title="ProteinMPNN mutation density across Ec86 positions", description=alt)
    return make_deliverable_row(
        deliverable_id="proteinmpnn_mutation_density",
        section="proteinmpnn",
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=["candidate_table.parquet"],
        input_hashes=file_hashes({"candidate_table": candidate_table_path}),
        alt_text=alt,
        description="Shows where ProteinMPNN sampled mutations under the current mask.",
        interpretation_limit=(
            "Mutation density describes sampled design variation and does not imply "
            "residue importance or biochemical effect."
        ),
    )


def _temperature_color(temperature: float) -> str:
    return _TEMPERATURE_COLORS.get(round(float(temperature), 3), "#5b7fa6")

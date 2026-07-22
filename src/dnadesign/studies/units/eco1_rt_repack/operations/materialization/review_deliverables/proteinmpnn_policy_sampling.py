"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/proteinmpnn_policy_sampling.py

ProteinMPNN proposal-distribution visuals for active generation policies.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pyarrow.parquet as pq
from matplotlib.lines import Line2D

from ..shared.rendering import LABEL_SIZE, LEGEND_SIZE, TITLE_SIZE, save_accessible_svg, style_open_axes
from .communication_visuals.style import POLICY_ORDER, policy_color, policy_label
from .constants import SECTION_DESIGNS_AND_FOLD_TRIAGE
from .manifest import file_hashes, make_deliverable_row
from .proteinmpnn_policy_frequency import write_policy_residue_frequency_bundle

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_proteinmpnn_policy_sampling(
    *,
    panel_root: Path,
    candidate_pool_path: Path,
    policy_positions_path: Path,
) -> list[dict[str, Any]]:
    """Write proposal-distribution and residue-frequency evidence for v3 policies."""

    return [
        _write_score_and_mutation_spread(panel_root=panel_root, candidate_pool_path=candidate_pool_path),
        write_policy_residue_frequency_bundle(
            panel_root=panel_root,
            candidate_pool_path=candidate_pool_path,
            policy_positions_path=policy_positions_path,
        ),
    ]


def _write_score_and_mutation_spread(*, panel_root: Path, candidate_pool_path: Path) -> dict[str, Any]:
    rows = pq.read_table(
        candidate_pool_path,
        columns=["candidate_id", "policy_id", "score", "global_score", "seq_recovery", "mutation_count", "status"],
    ).to_pylist()
    rows = [row for row in rows if str(row.get("status") or "") == "accepted"]
    if not rows:
        raise ValueError(f"No accepted ProteinMPNN candidates found in {candidate_pool_path}")
    panel_root.mkdir(parents=True, exist_ok=True)
    title = "ProteinMPNN proposals span three declared design spaces"
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.15))
    for policy_id in POLICY_ORDER:
        policy_rows = [row for row in rows if str(row.get("policy_id") or "") == policy_id]
        axes[0].scatter(
            [int(row["mutation_count"]) for row in policy_rows],
            [float(row["score"]) for row in policy_rows],
            s=25,
            alpha=0.65,
            color=policy_color(policy_id),
            edgecolors="white",
            linewidths=0.25,
        )
        axes[1].scatter(
            [float(row["seq_recovery"]) * 100.0 for row in policy_rows],
            [float(row["global_score"]) for row in policy_rows],
            s=25,
            alpha=0.65,
            color=policy_color(policy_id),
            edgecolors="white",
            linewidths=0.25,
        )
    axes[0].set_xlabel("Substitutions from WT", fontsize=LABEL_SIZE)
    axes[0].set_ylabel(
        "Mean NLL = -log P(aa | backbone)\nOpen positions",
        fontsize=LABEL_SIZE - 1,
    )
    axes[0].set_title("Score and mutation burden", fontsize=LABEL_SIZE)
    axes[1].set_xlabel("WT retained at designable positions (%)", fontsize=LABEL_SIZE)
    axes[1].set_ylabel(
        "Mean NLL = -log P(aa | backbone)\nAll modeled positions",
        fontsize=LABEL_SIZE - 1,
    )
    axes[1].set_title("Global score and open-site WT retention", fontsize=LABEL_SIZE)
    for ax in axes:
        style_open_axes(ax)
        ax.set_box_aspect(1)
    axes[1].legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=policy_color(policy_id),
                markeredgecolor="white",
                label=policy_label(policy_id),
            )
            for policy_id in POLICY_ORDER
        ],
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="center left",
        ncol=1,
        bbox_to_anchor=(1.02, 0.50),
    )
    fig.suptitle(title, fontsize=TITLE_SIZE + 1, y=0.97, x=0.47)
    fig.subplots_adjust(left=0.105, right=0.84, top=0.83, bottom=0.17, wspace=0.12)
    path = panel_root / "proteinmpnn_policy_proposal_spread.svg"
    alt = (
        f"Two-panel scatter plot for {len(rows)} accepted ProteinMPNN sequences, colored by distal, peripheral, "
        "or combined generation policy. The left panel compares mutation count with mean negative log probability "
        "over open positions; the right compares the fraction of designable positions retaining WT identity with "
        "the same quantity over all modeled positions. WT-retention values are discrete because each policy opens "
        "25, 59, or 84 positions."
    )
    save_accessible_svg(fig, path, title=title, description=alt, dpi=220)
    return make_deliverable_row(
        deliverable_id="proteinmpnn_policy_proposal_spread",
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=["generation_policies_v3/candidate_pool.parquet"],
        input_hashes=file_hashes({"candidate_pool": candidate_pool_path}),
        title=title,
        alt_text=alt,
        description=(
            "Shows proposal distributions under the three complete ProteinMPNN policies. Score is the mean "
            "negative log probability assigned to the sampled amino acids; lower values indicate greater model "
            "compatibility with the input backbone, not measured function. Vertical WT-retention bands are the "
            "expected one-residue increments from fixed designable-set sizes, not a sampling error."
        ),
        interpretation_limit=(
            "ProteinMPNN scores and WT recovery describe generated sequences. They do not establish fold quality, "
            "expression, RT activity, processivity, or strand displacement."
        ),
        role="manuscript_facing",
        render_mode="standard_visual",
        evidence_summary={
            "candidate_count": len(rows),
            "policy_counts": {
                policy_id: sum(str(row.get("policy_id") or "") == policy_id for row in rows)
                for policy_id in POLICY_ORDER
            },
        },
    )

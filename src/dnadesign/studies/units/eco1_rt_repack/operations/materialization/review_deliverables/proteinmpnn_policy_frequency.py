"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/proteinmpnn_policy_frequency.py

ProteinMPNN residue-frequency views for active generation policies.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pyarrow.parquet as pq
from matplotlib.lines import Line2D

from ..shared.rendering import LABEL_SIZE, LEGEND_SIZE, TITLE_SIZE, save_accessible_svg
from .communication_visuals.style import POLICY_ORDER, policy_label
from .constants import SECTION_DESIGNS_AND_FOLD_TRIAGE
from .manifest import file_hashes, make_deliverable_row

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
_MUTATION_RE = re.compile(r"^(?P<wt>[A-Z*])(?P<position>[0-9]+)(?P<alt>[A-Z*])$")
_TITLE = "ProteinMPNN samples distinct residue choices under each policy"


def write_policy_residue_frequency_bundle(
    *,
    panel_root: Path,
    candidate_pool_path: Path,
    policy_positions_path: Path,
) -> dict[str, Any]:
    """Write one selectable residue-frequency heatmap per generation policy."""

    candidates = _accepted_candidates(candidate_pool_path)
    position_rows = pq.read_table(policy_positions_path).to_pylist()
    wt_by_position = _validated_wt_by_position(position_rows)
    positions = sorted(wt_by_position)
    counts_by_policy = _mutation_counts_by_policy(candidates)
    candidate_counts = Counter(str(row["policy_id"]) for row in candidates)
    max_count = max(
        (count for policy_counts in counts_by_policy.values() for count in policy_counts.values()),
        default=1,
    )
    panel_root.mkdir(parents=True, exist_ok=True)
    views: list[dict[str, Any]] = []
    for policy_id in POLICY_ORDER:
        policy_rows = [row for row in position_rows if str(row["policy_id"]) == policy_id]
        if not policy_rows:
            raise ValueError(f"No generation-policy position rows found for {policy_id}")
        open_positions = {int(row["eco1_position"]) for row in policy_rows if bool(row["is_open_position"])}
        path = panel_root / f"proteinmpnn_residue_frequency_{_policy_slug(policy_id)}.svg"
        _render_policy_frequency(
            path=path,
            policy_id=policy_id,
            positions=positions,
            wt_by_position=wt_by_position,
            open_positions=open_positions,
            counts=counts_by_policy.get(policy_id, Counter()),
            max_count=max_count,
            candidate_count=int(candidate_counts[policy_id]),
        )
        manifest_path = path.relative_to(panel_root.parent)
        views.append(
            {
                "label": policy_label(policy_id),
                "policy_id": policy_id,
                "path": str(manifest_path),
                "candidate_count": int(candidate_counts[policy_id]),
                "open_position_count": len(open_positions),
            }
        )

    source_paths = {
        "candidate_pool": candidate_pool_path,
        "generation_policy_positions": policy_positions_path,
    }
    return make_deliverable_row(
        deliverable_id="proteinmpnn_policy_residue_frequency",
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="proteinmpnn_policy_residue_frequency_bundle",
        status="rendered",
        path=panel_root.parent / str(views[0]["path"]),
        source_tables=[
            "generation_policies_v3/candidate_pool.parquet",
            "generation_policies_v3/generation_policy_positions.parquet",
        ],
        input_hashes=file_hashes(source_paths),
        title=_TITLE,
        alt_text=(
            "Three selectable amino-acid frequency heatmaps, one for each ProteinMPNN generation policy. "
            "Columns are Eco1 positions, rows are amino acids, color is candidate count, x marks show the "
            "wild-type residue, and pale bands mark positions fixed by that policy."
        ),
        description=(
            "Shows which substitutions ProteinMPNN repeatedly sampled under the distal, peripheral, and combined "
            "fixed/open/alphabet contracts."
        ),
        interpretation_limit=(
            "Sampling frequency reflects the model and generation policy. It does not measure folding, expression, "
            "RT activity, processivity, or strand displacement."
        ),
        role="manuscript_facing",
        render_mode="wide_visual",
        evidence_summary={
            "candidate_count": len(candidates),
            "position_count": len(positions),
            "policy_views": views,
        },
    )


def _render_policy_frequency(
    *,
    path: Path,
    policy_id: str,
    positions: list[int],
    wt_by_position: dict[int, str],
    open_positions: set[int],
    counts: Counter[tuple[int, str]],
    max_count: int,
    candidate_count: int,
) -> None:
    fig, ax = plt.subplots(figsize=(18.0, 6.05))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    position_index = {position: index for index, position in enumerate(positions)}
    residue_index = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
    for position in positions:
        if position not in open_positions:
            index = position_index[position]
            ax.axvspan(index - 0.5, index + 0.5, color="#EEF0F2", alpha=0.92, linewidth=0, zorder=0)
    matrix = np.zeros((len(AMINO_ACIDS), len(positions)), dtype=float)
    for (position, residue), count in counts.items():
        if position in position_index and residue in residue_index:
            matrix[residue_index[residue], position_index[position]] = float(count)
    image = ax.imshow(
        np.ma.masked_where(matrix == 0, matrix),
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        vmin=0,
        vmax=max(1, max_count),
        zorder=2,
    )
    ax.scatter(
        range(len(positions)),
        [residue_index[wt_by_position[position]] for position in positions],
        marker="x",
        s=7.5,
        linewidths=0.45,
        color="#24292F",
        zorder=3,
    )
    tick_indexes = [0, *[index for index, position in enumerate(positions) if position % 20 == 0], len(positions) - 1]
    tick_indexes = sorted(set(tick_indexes))
    ax.set_xticks(tick_indexes, [str(positions[index]) for index in tick_indexes])
    ax.set_yticks(range(len(AMINO_ACIDS)), AMINO_ACIDS)
    ax.set_xlabel("Eco1 residue position", fontsize=LABEL_SIZE)
    ax.set_ylabel("Sampled amino acid", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=12.0)
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_ylim(len(AMINO_ACIDS) - 0.5, -0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(_TITLE, fontsize=TITLE_SIZE + 1, loc="center", pad=32)
    ax.text(
        0.5,
        1.045,
        f"{policy_label(policy_id)} | {candidate_count} complete sequences | {len(open_positions)} open positions",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=LABEL_SIZE,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.012)
    colorbar.set_label("Sequences with this amino acid", fontsize=LEGEND_SIZE)
    colorbar.ax.tick_params(labelsize=11.5)
    fig.legend(
        handles=[
            Line2D([], [], marker="x", linestyle="none", color="#24292F", label="WT residue"),
            Line2D([], [], marker="s", linestyle="none", color="#EEF0F2", label="Fixed position"),
        ],
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.subplots_adjust(left=0.058, right=0.955, bottom=0.19, top=0.76)
    save_accessible_svg(
        fig,
        path,
        title=f"{_TITLE} | {policy_label(policy_id)}",
        description=(
            f"ProteinMPNN amino-acid frequency heatmap for the {policy_label(policy_id).lower()} generation policy. "
            "Columns are Eco1 residue positions, rows are sampled amino acids, color is candidate count, x marks "
            "show wild-type residues, and pale vertical bands show fixed positions."
        ),
        dpi=240,
    )


def _accepted_candidates(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path, columns=["candidate_id", "policy_id", "canonical_mutations", "status"]).to_pylist()
    accepted = [row for row in rows if str(row.get("status") or "") == "accepted"]
    if not accepted:
        raise ValueError(f"No accepted ProteinMPNN candidates found in {path}")
    return accepted


def _mutation_counts_by_policy(rows: list[dict[str, Any]]) -> dict[str, Counter[tuple[int, str]]]:
    counts: dict[str, Counter[tuple[int, str]]] = {}
    for row in rows:
        policy_id = str(row.get("policy_id") or "")
        if policy_id not in POLICY_ORDER:
            raise ValueError(f"Candidate {row.get('candidate_id')} has unknown policy_id: {policy_id!r}")
        policy_counts = counts.setdefault(policy_id, Counter())
        for token in row.get("canonical_mutations") or []:
            match = _MUTATION_RE.fullmatch(str(token))
            if match is None:
                raise ValueError(f"Malformed mutation for {row.get('candidate_id')}: {token!r}")
            alt = match.group("alt")
            if alt in AMINO_ACIDS:
                policy_counts[(int(match.group("position")), alt)] += 1
    return counts


def _validated_wt_by_position(rows: list[dict[str, Any]]) -> dict[int, str]:
    wt_by_position: dict[int, str] = {}
    for row in rows:
        position = int(row["eco1_position"])
        wt_aa = str(row["wt_aa"]).strip().upper()
        if len(wt_aa) != 1:
            raise ValueError(f"Generation-policy position {position} has invalid WT amino acid: {wt_aa!r}")
        prior = wt_by_position.setdefault(position, wt_aa)
        if prior != wt_aa:
            raise ValueError(f"Generation policies disagree on WT amino acid at position {position}")
    return wt_by_position


def _policy_slug(policy_id: str) -> str:
    return {
        POLICY_ORDER[0]: "distal",
        POLICY_ORDER[1]: "peripheral",
        POLICY_ORDER[2]: "combined",
    }[policy_id]

"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/matrix_annotations.py

Semantic annotations for response-metastudy matrix columns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_vocabulary import representation_label, representation_role


def add_column_group_brackets(
    axis: plt.Axes,
    groups: list[tuple[float, float, str]],
    *,
    baseline: float = 1.03,
) -> None:
    """Draw labeled top brackets over contiguous matrix-column groups."""

    if not groups:
        raise ValueError("column-group brackets require at least one group.")
    transform = axis.get_xaxis_transform()
    for start, end, label in groups:
        if not start < end or not str(label).strip():
            raise ValueError(f"invalid column-group bracket: {(start, end, label)!r}")
        (line,) = axis.plot(
            [start, start, end, end],
            [baseline, baseline + 0.025, baseline + 0.025, baseline],
            color="#4b5563",
            linewidth=0.9,
            transform=transform,
            clip_on=False,
            zorder=6,
        )
        line.set_gid(f"column-group-bracket:{label}")
        text = axis.text(
            (start + end) / 2.0,
            baseline + 0.04,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111827",
            transform=transform,
            clip_on=False,
        )
        text.set_gid(f"column-group-label:{label}")


def label_representation_axis(axis: plt.Axes, columns: pd.Index) -> None:
    """Label a representation matrix and bracket its semantic roles."""

    column_ids = [str(value) for value in columns]
    roles = [representation_role(value) for value in column_ids]
    axis.set_xticks(
        np.arange(len(column_ids)),
        [representation_label(value) for value in column_ids],
        ha="center",
        fontsize=7,
    )
    role_order = list(dict.fromkeys(roles))
    groups = []
    for role in role_order:
        indices = [index for index, value in enumerate(roles) if value == role]
        if indices != list(range(min(indices), max(indices) + 1)):
            raise ValueError(f"representation role {role!r} is not contiguous.")
        groups.append((min(indices) - 0.45, max(indices) + 0.45, role))
    for _, end, _ in groups[:-1]:
        axis.axvline(end + 0.05, color="#d1d5db", linewidth=0.9, zorder=5)
    add_column_group_brackets(axis, groups, baseline=1.04)


__all__ = ["add_column_group_brackets", "label_representation_axis"]

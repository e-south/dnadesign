"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/pwm/panel_metadata.py

SVG metadata helpers for bidirectional TetR PWM trim review panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from html import escape
from typing import Sequence

from ..contracts.plan import PwmTrimPanel
from .panel_labels import compact_panel_subtitle, panel_title
from .sequence_rows import PwmLogoColumn, observed_sequence_for_panel
from .trim_annotations import BOUNDARY_TICK_FONT_SIZE


def panel_subtitle(panel: PwmTrimPanel) -> str:
    return compact_panel_subtitle(panel)


def panel_metadata_attributes(columns: Sequence[PwmLogoColumn], panel: PwmTrimPanel) -> str:
    retained_nt = panel.retained_end_0 - panel.retained_start_0
    return " ".join(
        [
            f'data-payload-trim-id="{escape(panel.payload_trim_id)}"',
            f'data-display-title="{escape(panel_title(panel))}"',
            f'data-observed-sequence-5to3="{escape(observed_sequence_for_panel(columns, panel))}"',
            f'data-retained-feature-span-0="{panel.retained_start_0}..{panel.retained_end_0}"',
            f'data-retained-feature-label-5to3="{escape(retained_feature_label(columns, panel))}"',
            f'data-trim-5p-nt="{panel.trim_5p_nt}"',
            f'data-trim-3p-nt="{panel.trim_3p_nt}"',
            f'data-retained-nt="{retained_nt}"',
            f'data-retained-information-fraction="{panel.retained_information_fraction:.6f}"',
            f'data-compact-subtitle="{escape(compact_panel_subtitle(panel))}"',
            f'data-boundary-ticks-0="{panel.retained_start_0},{panel.retained_end_0}"',
            f'data-boundary-tick-font-size-px="{BOUNDARY_TICK_FONT_SIZE}"',
            f'data-visible-trim-summary="{escape(visible_trim_summary(panel))}"',
        ]
    )


def trim_state_elements(columns: Sequence[PwmLogoColumn], panel: PwmTrimPanel) -> list[str]:
    elements: list[str] = []
    for column in columns:
        state = "included" if panel.retained_start_0 <= column.parent_position_0 < panel.retained_end_0 else "excluded"
        elements.append(
            f'<pwm-column data-parent-position="{column.parent_position_0}" data-trim-state="{state}" '
            f'data-parent-base="{escape(column.parent_base)}" '
            f'data-information-bits="{column.information_bits:.6f}"/>'
        )
    return elements


def visible_trim_summary(panel: PwmTrimPanel) -> str:
    retained_nt = panel.retained_end_0 - panel.retained_start_0
    retained_percent = panel.retained_information_fraction * 100.0
    return (
        f"removed {panel.trim_5p_nt}+{panel.trim_3p_nt} nt; "
        f"retained {retained_nt} nt; retained PWM information {retained_percent:.1f}%"
    )


def retained_feature_label(columns: Sequence[PwmLogoColumn], panel: PwmTrimPanel) -> str:
    start, end = panel.retained_start_0, panel.retained_end_0
    return "".join(column.parent_base for column in columns if start <= column.parent_position_0 < end)


__all__ = [
    "observed_sequence_for_panel",
    "panel_metadata_attributes",
    "panel_subtitle",
    "retained_feature_label",
    "trim_state_elements",
    "visible_trim_summary",
]

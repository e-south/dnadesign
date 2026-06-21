"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/pwm/panel_labels.py

Display labels for bidirectional TetR PWM review panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..contracts.plan import PwmTrimPanel

PANEL_DISPLAY_TITLES = {
    "TetR_w00_19": "Full site",
    "TetR_w02_17": "Trim 02-17",
    "TetR_w03_16": "Trim 03-16",
}


def panel_title(panel: PwmTrimPanel) -> str:
    return PANEL_DISPLAY_TITLES.get(panel.payload_trim_id, panel.label)


def compact_panel_subtitle(panel: PwmTrimPanel) -> str:
    retained_nt = panel.retained_end_0 - panel.retained_start_0
    retained_percent = round(panel.retained_information_fraction * 100.0)
    return f"{retained_nt} nt | [{panel.retained_start_0},{panel.retained_end_0}) | {retained_percent:.0f}% IC"


__all__ = ["compact_panel_subtitle", "panel_title"]

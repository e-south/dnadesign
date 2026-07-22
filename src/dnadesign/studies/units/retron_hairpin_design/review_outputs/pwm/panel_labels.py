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


def panel_title(panel: PwmTrimPanel) -> str:
    if panel.payload_trim_id.endswith("_w00_19") or panel.payload_trim_id.endswith("-w00-19"):
        return "Full site"
    if panel.payload_trim_id.endswith("_w02_17") or panel.payload_trim_id.endswith("-w02-17"):
        return "Trim 02-17"
    if panel.payload_trim_id.endswith("_w03_16") or panel.payload_trim_id.endswith("-w03-16"):
        return "Trim 03-16"
    return panel.label


def compact_panel_subtitle(panel: PwmTrimPanel) -> str:
    retained_nt = panel.retained_end_0 - panel.retained_start_0
    retained_percent = round(panel.retained_information_fraction * 100.0)
    return f"{retained_nt} nt | [{panel.retained_start_0},{panel.retained_end_0}) | {retained_percent:.0f}% IC"


__all__ = ["compact_panel_subtitle", "panel_title"]

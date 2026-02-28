"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/trajectory_video_contract.py

Build BaseRender sequence-rows job mappings for chain-trajectory videos.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from dnadesign.baserender import cruncher_showcase_style_overrides
from dnadesign.cruncher.config.schema_v3 import AnalysisTrajectoryVideoConfig


def _trajectory_video_style_overrides() -> dict[str, object]:
    overrides = dict(cruncher_showcase_style_overrides())
    overrides["figure_scale"] = 0.98
    overrides["padding_y"] = 3.0
    overrides["font_size_label"] = 15
    overrides["overlay_align"] = "center"
    overrides["show_reverse_complement"] = True
    overrides["baseline_spacing"] = 36.0
    overrides["track_spacing"] = 10.0

    layout = dict(overrides.get("layout", {}))
    layout["outer_pad_cells"] = 0.0
    overrides["layout"] = layout

    motif_logo = dict(overrides.get("motif_logo", {}))
    motif_logo["layout"] = "overlay"
    motif_logo["bits_to_cells"] = 0.70
    overrides["motif_logo"] = motif_logo
    return overrides


def build_sequence_rows_video_job(
    *,
    records_path: Path,
    out_path: Path,
    config: AnalysisTrajectoryVideoConfig,
    pauses: Mapping[str, float],
    title_text: str,
) -> dict[str, object]:
    return {
        "version": 3,
        "input": {
            "kind": "parquet",
            "path": str(records_path),
            "adapter": {
                "kind": "generic_features",
                "columns": {
                    "id": "id",
                    "sequence": "sequence",
                    "features": "features",
                    "effects": "effects",
                    "display": "display",
                },
                "policies": {},
            },
            "alphabet": "DNA",
        },
        "render": {
            "renderer": "sequence_rows",
            "style": {"overrides": dict(_trajectory_video_style_overrides())},
        },
        "outputs": [
            {
                "kind": "video",
                "path": str(out_path),
                "fmt": "mp4",
                "fps": int(config.playback.fps),
                "frames_per_record": 1,
                "pauses": dict(pauses),
                "total_duration": float(config.playback.target_duration_sec),
                "height_px": 820,
                "title_text": str(title_text),
                "title_font_size": 12,
                "title_align": "center",
            }
        ],
        "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
    }

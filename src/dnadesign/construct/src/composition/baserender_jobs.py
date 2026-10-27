"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition/baserender_jobs.py

BaseRender job emission for linear ssDNA composition bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..contracts.errors import ValidationError
from .models import ComposedLinearSsdna
from .visual import SEQUENCE_EVIDENCE_MAP_PATH


def baserender_job_artifacts(composed: ComposedLinearSsdna) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for fmt in _baserender_export_formats(composed):
        artifacts[f"baserender_component_span_{fmt}_job"] = f"baserender_jobs/component_span_qa_{fmt}.yaml"
    return artifacts


def write_baserender_jobs(artifact_bundle: Path, composed: ComposedLinearSsdna) -> None:
    for fmt in _baserender_export_formats(composed):
        job_path = artifact_bundle / "baserender_jobs" / f"component_span_qa_{fmt}.yaml"
        payload = {
            "version": 4,
            "contract": {"kind": "nucleotide_evidence_map_render_v3"},
            "bundle": {"path": f"../visual/renders/component_span_qa_{fmt}.render-v1"},
            "input": {
                "kind": "json",
                "path": f"../{SEQUENCE_EVIDENCE_MAP_PATH.as_posix()}",
                "adapter": {
                    "kind": "sequence_evidence_map_v1",
                    "columns": {},
                    "policies": {},
                },
                "alphabet": "DNA",
            },
            "render": {
                "renderer": "nucleotide_evidence_map",
                "style": {
                    "preset": "presentation_default",
                    "overrides": {
                        "figure_scale": 1.15,
                        "padding_y": 16.0,
                        "overlay_align": "center",
                        "baseline_spacing": 28.0,
                        "layout": {"outer_pad_cells": 0.35},
                        "sequence": {"strand_gap_cells": 0.08},
                        "connectors": True,
                        "color_ticks": "#CBD5E1",
                        "connector_alpha": 0.42,
                        "connector_width": 0.5,
                        "connector_dash": [],
                        "show_reverse_complement": True,
                        "legend": False,
                        "legend_mode": "none",
                        "legend_height_px": 48.0,
                        "uniform_display_font_size": True,
                        "legend_font_size": 6,
                        "font_size_seq": 6,
                        "font_size_label": 6,
                        "font_size_feature_label": 6,
                        "font_size_annotation_label": 6,
                        "font_size_span_link_label": 6,
                        "span_link_line_width": 1.4,
                        "span_link_tick_line_width": 1.0,
                    },
                },
            },
            "outputs": [
                {
                    "kind": "images",
                    "path": f"component_span_qa.{fmt}",
                    "fmt": fmt,
                }
            ],
            "run": {
                "strict": True,
                "fail_on_skips": True,
            },
        }
        job_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _baserender_export_formats(composed: ComposedLinearSsdna) -> list[str]:
    requested = composed.config.visual.render_exports.formats or ["svg"]
    formats: list[str] = []
    allowed = {"svg", "pdf", "png"}
    for raw_format in requested:
        fmt = raw_format.strip().lower()
        if fmt not in allowed:
            raise ValidationError(f"Unsupported BaseRender export format '{raw_format}'. Expected svg, pdf, or png.")
        if fmt not in formats:
            formats.append(fmt)
    return formats

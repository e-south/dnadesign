"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/reporting/review/rendering/markdown.py

Markdown renderer entrypoint for DenseGen axis probe reviews.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

from .markdown_document import render_probe_review_markdown as _render_probe_review_markdown


def render_probe_review_markdown(review_manifest: Mapping[str, Any], metrics_payload: Mapping[str, Any]) -> str:
    return _render_probe_review_markdown(review_manifest, metrics_payload)

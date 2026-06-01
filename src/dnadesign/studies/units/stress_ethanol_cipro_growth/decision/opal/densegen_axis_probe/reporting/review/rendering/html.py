"""HTML renderer entrypoint for DenseGen axis probe reviews."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .html_document import render_probe_review_html as _render_probe_review_html


def render_probe_review_html(
    review_manifest: Mapping[str, Any],
    metrics_payload: Mapping[str, Any],
    *,
    base_dir: Path,
) -> str:
    return _render_probe_review_html(review_manifest, metrics_payload, base_dir=base_dir)

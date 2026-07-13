"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/reader_evidence_visual.py

Visual rendering for Reader evidence notebook artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .reader_evidence_media import select_reader_media_artifact
from .reader_evidence_preview import reader_pdf_preview_path
from .reader_evidence_triptych import is_reader_sfxi_triptych_artifact, render_reader_sfxi_triptych_visual
from .reader_promoter_evidence import (
    ReaderPromoterEvidenceIntegrityError,
    is_reader_promoter_evidence_artifact,
    verify_reader_promoter_evidence_artifact,
)
from .zoomable_visual import render_notebook_zoomable_image


def render_notebook_reader_evidence_artifact_visual(
    surface: Mapping[str, Any],
    *,
    selected_plot_type_label: str | None,
    selected_artifact_label: str | None,
    mo: Any,
    selected_time_h: float | None = None,
) -> Any:
    """Render the selected plot artifact or live Reader triptych."""

    if not selected_plot_type_label:
        return mo.md("No plot type selected.")
    if not selected_artifact_label:
        return mo.md("No plot instance selected.")
    selected = select_reader_media_artifact(
        surface,
        selected_plot_type_label=selected_plot_type_label,
        selected_artifact_label=selected_artifact_label,
    )
    if selected is None:
        return mo.md("Selected plot artifact is no longer available.")
    if selected_time_h is not None and is_reader_sfxi_triptych_artifact(selected):
        return render_reader_sfxi_triptych_visual(selected, selected_time_h=selected_time_h, mo=mo)
    return _render_static_reader_artifact(selected, mo=mo)


def _render_static_reader_artifact(selected: Mapping[str, Any], *, mo: Any) -> Any:
    path = Path(str(selected.get("path") or ""))
    media_type = str(selected.get("media_type") or "")
    if is_reader_promoter_evidence_artifact(selected):
        try:
            path = verify_reader_promoter_evidence_artifact(selected)
        except ReaderPromoterEvidenceIntegrityError as exc:
            return mo.md(f"Promoter-response evidence verification failed: `{exc}`")
    if not path.exists():
        return mo.md(f"Plot artifact missing: `{path}`")
    if media_type == "application/pdf" or path.suffix.lower() == ".pdf":
        try:
            preview_path = reader_pdf_preview_path(path)
        except RuntimeError as exc:
            return mo.md(f"PDF plot artifact could not be rendered as an image: `{path}`\n\n{exc}")
        return _render_zoomable_artifact(
            path=preview_path,
            source_path=path,
            selected=selected,
            mime_type="image/png",
            mo=mo,
        )
    if media_type.startswith("image/") or path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return _render_zoomable_artifact(
            path=path,
            source_path=path,
            selected=selected,
            mime_type=_image_mime_type(path=path, media_type=media_type),
            mo=mo,
        )
    return mo.md(f"Plot artifact: `{path}`")


def _render_zoomable_artifact(
    *,
    path: Path,
    source_path: Path,
    selected: Mapping[str, Any],
    mime_type: str,
    mo: Any,
) -> Any:
    label = str(selected.get("label") or "Plot artifact")
    return render_notebook_zoomable_image(
        mo=mo,
        image_bytes=path.read_bytes(),
        mime_type=mime_type,
        alt_text=label,
        caption=label,
        artifact_key=str(source_path),
    )


def _image_mime_type(*, path: Path, media_type: str) -> str:
    if media_type.startswith("image/"):
        return media_type
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/png"


__all__ = ["render_notebook_reader_evidence_artifact_visual"]

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

from dnadesign.opal.api.reader_evidence import optional_reader_evidence_artifact_adapter

from .reader_evidence_media import select_reader_media_artifact
from .reader_evidence_preview import reader_pdf_preview_path
from .zoomable_visual import render_notebook_zoomable_image


def render_notebook_reader_evidence_artifact_visual(
    surface: Mapping[str, Any],
    *,
    selected_plot_type_label: str | None,
    selected_artifact_label: str | None,
    mo: Any,
) -> Any:
    """Render one completed Reader plot artifact."""

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
    return _render_static_reader_artifact(selected, mo=mo)


def _render_static_reader_artifact(selected: Mapping[str, Any], *, mo: Any) -> Any:
    path = Path(str(selected.get("path") or ""))
    media_type = str(selected.get("media_type") or "")
    semantic_kind = str(selected.get("semantic_kind") or "").strip()
    adapter = optional_reader_evidence_artifact_adapter(semantic_kind)
    if adapter is not None:
        try:
            path = adapter.verify_artifact(selected)
        except Exception as exc:
            return mo.md(f"{adapter.verification_label} verification failed: `{exc}`")
    if not path.exists():
        return mo.md(f"Plot artifact missing: `{path}`")
    if media_type == "application/pdf" or path.suffix.lower() == ".pdf":
        try:
            preview_path = reader_pdf_preview_path(path)
        except RuntimeError as exc:
            return mo.md(f"PDF plot artifact could not be rendered as an image: `{path}`\n\n{exc}")
        visual = _render_zoomable_artifact(
            path=preview_path,
            source_path=path,
            selected=selected,
            mime_type="image/png",
            mo=mo,
        )
    elif media_type.startswith("image/") or path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        visual = _render_zoomable_artifact(
            path=path,
            source_path=path,
            selected=selected,
            mime_type=_image_mime_type(path=path, media_type=media_type),
            mo=mo,
        )
    else:
        return mo.md(f"Plot artifact: `{path}`")
    if adapter is None or adapter.render_details is None:
        return visual
    details = adapter.render_details(selected, mo=mo)
    return mo.vstack([visual, details], gap=0.25)


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

"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/structure_views/html.py

HTML rendering facade for browser structure views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.thread.structure_views.backends.py3dmol import (
    py3dmol_available,
    render_py3dmol_structure_view,
)
from dnadesign.thread.structure_views.models import StructureViewSpec


def structure_view_backend_available(backend: str = "py3dmol") -> bool:
    """Return whether a browser structure-view backend is importable."""

    if backend != "py3dmol":
        return False
    return py3dmol_available()


def render_structure_view_html(spec: StructureViewSpec, *, backend: str = "py3dmol") -> str:
    """Render an interactive HTML structure viewer with the selected backend."""

    if backend != "py3dmol":
        raise ValueError(f"Unsupported structure-view backend: {backend}")
    return render_py3dmol_structure_view(spec)

"""
Notebook scaffold rendering helpers for latentdna.
"""

from __future__ import annotations

from textwrap import dedent

from .scaffold_pages import render_page_cells
from .scaffold_selectors import render_bootstrap_cell, render_selector_cells, render_theme_cell


def _marimo_version() -> str:
    import marimo as _marimo

    version = getattr(_marimo, "__version__", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError("marimo import succeeded but __version__ is not available")
    return version


def _render_header() -> str:
    return dedent(
        """\
        import marimo

        __generated_with = "__GENERATED_WITH__"

        app = marimo.App(width="full")
        """
    )


def _render_footer() -> str:
    return dedent(
        """\
        if __name__ == "__main__":
            app.run()
        """
    )


def _render_template() -> str:
    return "\n\n".join(
        [
            _render_header(),
            render_bootstrap_cell(),
            render_theme_cell(),
            *render_selector_cells(),
            *render_page_cells(),
            _render_footer(),
        ]
    )


def render_workspace_notebook(
    *,
    workspace_id: str,
    notebook_id: str,
    title: str,
    description: str | None,
    default_deliverable: str,
    default_surface: str,
) -> str:
    description_text = description or "Read-only workspace notebook for persisted workspace artifacts."
    template = _render_template()
    return (
        template.replace("__GENERATED_WITH__", _marimo_version())
        .replace("__TITLE__", repr(title))
        .replace("__DESCRIPTION__", repr(description_text))
        .replace("__WORKSPACE_ID__", repr(workspace_id))
        .replace("__NOTEBOOK_ID__", repr(notebook_id))
        .replace("__DEFAULT_DELIVERABLE__", repr(default_deliverable))
        .replace("__DEFAULT_SURFACE__", repr(default_surface))
    )

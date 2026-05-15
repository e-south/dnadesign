"""
Page-panel cell templates for generated latentdna marimo notebooks.
"""

from __future__ import annotations

from textwrap import dedent


def render_scope_note_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(runtime):
            _identity = runtime.identity
            _support = runtime.support

            _plot_scope_text = _identity.title
            plot_scope_note = _support.mo.md(f"# {_plot_scope_text}")
            geometry_scope_note = _support.mo.md(
                (
                    "This surface is a projection browser for persisted geometry and metadata overlays. "
                    "Point positions are fixed by the saved coordinates, so hue changes only recolor the same geometry."
                )
            )
            return (geometry_scope_note, plot_scope_note)
        """
    )


def render_browser_surface_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(runtime):
            _support = runtime.support
            _plot_review = runtime.plot_review
            surface_options = {
                "Persisted plots": "plots",
                "Projection browser": "geometry_browser",
            }
            default_surface = (
                _plot_review.default_surface
                if _plot_review.default_surface in set(surface_options.values())
                else "plots"
            )
            surface_selector = _support.mo.ui.dropdown(
                options=surface_options,
                value=(
                    _support.option_key_for_value(surface_options, default_surface)
                    or next(iter(surface_options))
                ),
                label="Artifact group",
            )
            return (surface_selector,)


        @app.cell
        def _(geometry_panel, plot_review_panel, runtime, surface_selector):
            _support = runtime.support
            selected_surface = str(surface_selector.value)
            selected_panel = geometry_panel if selected_surface == "geometry_browser" else plot_review_panel
            browser_surface = _support.mo.vstack(
                [
                    _support.mo.hstack([surface_selector], justify="start", align="end", wrap=True, gap=0.28),
                    selected_panel,
                ],
                gap=0.35,
            )
            return (browser_surface,)
        """
    )


def render_page_display_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(browser_surface):
            browser_surface
            return
        """
    )

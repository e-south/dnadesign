"""
Selector and bootstrap cell templates for generated latentdna marimo notebooks.
"""

from __future__ import annotations

from textwrap import dedent


def render_bootstrap_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _():
            from pathlib import Path

            from dnadesign.latentdna.src.notebooks.browser_runtime import (
                build_workspace_browser_runtime,
                load_workspace_notebook_controls,
            )

            TITLE = __TITLE__
            DESCRIPTION = __DESCRIPTION__
            WORKSPACE_ID = __WORKSPACE_ID__
            NOTEBOOK_ID = __NOTEBOOK_ID__
            DEFAULT_DELIVERABLE = __DEFAULT_DELIVERABLE__
            NOTEBOOK_DIR = Path(__file__).resolve().parent
            CONTROL_PATH = NOTEBOOK_DIR / "controls.json"
            _controls = load_workspace_notebook_controls(CONTROL_PATH)
            _runtime_paths = _controls["runtime_paths"]
            WORKSPACE_DIR = (NOTEBOOK_DIR / str(_runtime_paths["workspace_relative_path"])).resolve()
            OUTPUT_ROOT = (NOTEBOOK_DIR / str(_runtime_paths["output_relative_path"])).resolve()
            CATALOG_PATH = (NOTEBOOK_DIR / str(_runtime_paths["catalog_relative_path"])).resolve()
            HEALTH_PATH = (NOTEBOOK_DIR / str(_runtime_paths["health_relative_path"])).resolve()
            runtime = build_workspace_browser_runtime(
                title=TITLE,
                description=DESCRIPTION,
                workspace_id=WORKSPACE_ID,
                notebook_id=NOTEBOOK_ID,
                default_deliverable=DEFAULT_DELIVERABLE,
                workspace_dir=WORKSPACE_DIR,
                output_root=OUTPUT_ROOT,
                catalog_path=CATALOG_PATH,
                health_path=HEALTH_PATH,
                controls=_controls,
            )
            return (runtime,)
        """
    )


def render_theme_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(runtime):
            notebook_theme = runtime.support.notebook_theme()
            notebook_theme
            return (notebook_theme,)
        """
    )


def render_selector_cells() -> tuple[str, ...]:
    return (
        dedent(
            """\
            @app.cell
            def _(runtime):
                _geometry = runtime.geometry
                _support = runtime.support

                _model_options = {value.upper(): value for value in _geometry.model_values}
                model_selector = _support.mo.ui.dropdown(
                    options=_model_options,
                    value=(
                        _support.option_key_for_value(_model_options, _geometry.model_default)
                        or next(iter(_model_options))
                    ),
                    label="Model",
                )
                layout_selector = _support.mo.ui.dropdown(
                    options=_geometry.layout_options,
                    value=(
                        _support.option_key_for_value(_geometry.layout_options, _geometry.layout_default)
                        or next(iter(_geometry.layout_options))
                    ),
                    label="Layout",
                )
                return (layout_selector, model_selector)
            """
        ),
        dedent(
            """\
            @app.cell
            def _(model_selector, runtime):
                _geometry = runtime.geometry
                _support = runtime.support

                selected_model = str(model_selector.value)
                family_values = _support.unique_in_order(
                    row.get("family")
                    for row in _geometry.geometry_rows
                    if str(row.get("model")) == selected_model
                ) or ["intermediate_embedding"]
                family_default = (
                    str(_geometry.geometry_control.get("default_family"))
                    if str(_geometry.geometry_control.get("default_family")) in family_values
                    else family_values[0]
                )
                _family_options = {
                    {
                        "intermediate_embedding": "Intermediate block mean",
                        "pooled_logits": "Pooled logits",
                    }.get(value, value.replace("_", " ")): value
                    for value in family_values
                }
                family_selector = _support.mo.ui.dropdown(
                    options=_family_options,
                    value=(
                        _support.option_key_for_value(_family_options, family_default)
                        or next(iter(_family_options))
                    ),
                    label="Family",
                )
                return (family_selector, selected_model)
            """
        ),
        dedent(
            """\
            @app.cell
            def _(family_selector, runtime, selected_model):
                _geometry = runtime.geometry
                _support = runtime.support

                selected_family = str(family_selector.value)
                context_values = _support.unique_in_order(
                    row.get("context")
                    for row in _geometry.geometry_rows
                    if str(row.get("model")) == selected_model and str(row.get("family")) == selected_family
                ) or ["anchor_60bp"]
                context_default = (
                    str(_geometry.geometry_control.get("default_context"))
                    if str(_geometry.geometry_control.get("default_context")) in context_values
                    else context_values[0]
                )
                _context_options = {
                    {
                        "anchor_60bp": "60 bp anchor",
                        "full_context_1kb": "1 kb construct context",
                    }.get(value, value.replace("_", " ")): value
                    for value in context_values
                }
                context_selector = _support.mo.ui.dropdown(
                    options=_context_options,
                    value=(
                        _support.option_key_for_value(_context_options, context_default)
                        or next(iter(_context_options))
                    ),
                    label="Context",
                )
                return (context_selector, selected_family)
            """
        ),
        dedent(
            """\
            @app.cell
            def _(context_selector, layout_selector, runtime, selected_family, selected_model):
                _geometry = runtime.geometry
                _support = runtime.support

                _selected_context = str(context_selector.value)
                compatible_geometries = [
                    row
                    for row in _geometry.geometry_rows
                    if str(row.get("model")) == selected_model
                    and str(row.get("family")) == selected_family
                    and str(row.get("context")) == _selected_context
                ]
                geometry_options = {
                    str(row.get("label") or row["view_id"]): str(row["view_id"])
                    for row in compatible_geometries
                } or {
                    str(row.get("label") or row["view_id"]): str(row["view_id"])
                    for row in _geometry.geometry_rows
                }
                geometry_default = next(iter(geometry_options.values()), "")
                geometry_selector = _support.mo.ui.dropdown(
                    options=geometry_options or {"No geometry": ""},
                    value=(
                        _support.option_key_for_value(geometry_options, geometry_default)
                        if geometry_options
                        else "No geometry"
                    ),
                    label="Geometry",
                )
                return (geometry_selector,)
            """
        ),
        dedent(
            """\
            @app.cell
            def _(runtime):
                _geometry = runtime.geometry
                _support = runtime.support

                compare_options = {
                    str(row.get("label") or row["view_id"]): str(row["view_id"]) for row in _geometry.geometry_rows
                } or {"No geometries": ""}
                compare_left_selector = _support.mo.ui.dropdown(
                    options=compare_options,
                    value=(
                        _support.option_key_for_value(compare_options, _geometry.compare_left_default)
                        or next(iter(compare_options))
                    ),
                    label="Left geometry",
                )
                compare_right_selector = _support.mo.ui.dropdown(
                    options=compare_options,
                    value=(
                        _support.option_key_for_value(compare_options, _geometry.compare_right_default)
                        or next(iter(compare_options))
                    ),
                    label="Right geometry",
                )
                return (compare_left_selector, compare_right_selector)
            """
        ),
    )

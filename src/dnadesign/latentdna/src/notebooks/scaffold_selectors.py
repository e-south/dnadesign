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

                projected_geometry_rows = [
                    row for row in _geometry.geometry_rows if row.get("projection_ids")
                ]
                _model_values = _support.unique_in_order(
                    row.get("model") for row in projected_geometry_rows
                ) or _geometry.model_values
                def _model_label(value):
                    _value = str(value)
                    if _value.startswith("evo2_"):
                        return "Evo 2 " + _value.removeprefix("evo2_").upper()
                    return _value.upper()

                _model_options = _support.labeled_options((_model_label(value), value) for value in _model_values)
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
                    label="Candidate set / mode",
                )
                return (layout_selector, model_selector, projected_geometry_rows)
            """
        ),
        dedent(
            """\
            @app.cell
            def _(model_selector, projected_geometry_rows, runtime):
                _geometry = runtime.geometry
                _support = runtime.support

                selected_model = str(model_selector.value)
                _selector_rows = projected_geometry_rows or _geometry.geometry_rows
                family_values = _support.unique_in_order(
                    row.get("family")
                    for row in _selector_rows
                    if str(row.get("model")) == selected_model
                ) or ["intermediate_embedding"]
                family_default = (
                    str(_geometry.geometry_control.get("default_family"))
                    if str(_geometry.geometry_control.get("default_family")) in family_values
                    else family_values[0]
                )
                _family_options = _support.labeled_options(
                    (
                        {
                            "intermediate_embedding": "Intermediate block mean",
                            "output_layer_mean": "Output-layer mean",
                        }.get(value, value.replace("_", " ")),
                        value,
                    )
                    for value in family_values
                )
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
            def _(family_selector, projected_geometry_rows, runtime, selected_model):
                _geometry = runtime.geometry
                _support = runtime.support

                selected_family = str(family_selector.value)
                _selector_rows = projected_geometry_rows or _geometry.geometry_rows
                context_values = _support.unique_in_order(
                    row.get("context")
                    for row in _selector_rows
                    if str(row.get("model")) == selected_model and str(row.get("family")) == selected_family
                ) or ["merged_anchor_insert_seq_mean"]
                context_default = (
                    str(_geometry.geometry_control.get("default_context"))
                    if str(_geometry.geometry_control.get("default_context")) in context_values
                    else context_values[0]
                )
                _context_options = _support.labeled_options(
                    (
                        {
                            "merged_anchor_insert_seq_mean": "Mixed-length anchor-source insert",
                            "full_context_1kb": "1 kb construct context",
                        }.get(value, value.replace("_", " ")),
                        value,
                    )
                    for value in context_values
                )
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
            def _(context_selector, layout_selector, projected_geometry_rows, runtime, selected_family, selected_model):
                _geometry = runtime.geometry
                _support = runtime.support

                _selected_context = str(context_selector.value)
                _selector_rows = projected_geometry_rows or _geometry.geometry_rows
                compatible_geometries = [
                    row
                    for row in _selector_rows
                    if str(row.get("model")) == selected_model
                    and str(row.get("family")) == selected_family
                    and str(row.get("context")) == _selected_context
                ]
                geometry_options = _support.labeled_options(
                    (
                        str(row.get("label") or row["view_id"]),
                        str(row["view_id"]),
                    )
                    for row in compatible_geometries
                )
                geometry_default = next(iter(geometry_options.values()), "")
                _empty_geometry_options = {"No compatible geometry for this selection": ""}
                geometry_selector = _support.mo.ui.dropdown(
                    options=geometry_options or _empty_geometry_options,
                    value=(
                        _support.option_key_for_value(geometry_options, geometry_default)
                        if geometry_options
                        else next(iter(_empty_geometry_options))
                    ),
                    label="Geometry",
                    searchable=True,
                    full_width=True,
                )
                return (geometry_selector,)
            """
        ),
        dedent(
            """\
            @app.cell
            def _(geometry_selector, runtime):
                _geometry = runtime.geometry
                _support = runtime.support

                _selected_geometry = _geometry.geometry_rows_by_id.get(str(geometry_selector.value))
                projection_ids = [
                    str(projection_id)
                    for projection_id in (_selected_geometry or {}).get("projection_ids", [])
                    if str(projection_id).strip()
                ]
                projection_options = _support.labeled_options(
                    (projection_id.replace("_", " "), projection_id)
                    for projection_id in projection_ids
                )
                empty_projection_options = {"Default projection": ""}
                projection_selector = _support.mo.ui.dropdown(
                    options=projection_options or empty_projection_options,
                    value=next(iter(projection_options or empty_projection_options)),
                    label="Projection",
                    searchable=True,
                    full_width=True,
                )
                return (projection_selector,)
            """
        ),
        dedent(
            """\
            @app.cell
            def _(runtime):
                _geometry = runtime.geometry
                _support = runtime.support

                default_hue = (
                    _geometry.selected_hue_default
                    if _geometry.selected_hue_default in _geometry.preferred_hues
                    else ""
                )
                get_requested_hue, set_requested_hue = _support.mo.state(default_hue)
                return (get_requested_hue, set_requested_hue)
            """
        ),
        dedent(
            """\
            @app.cell
            def _(runtime):
                _geometry = runtime.geometry
                _support = runtime.support

                reference_values = set((_geometry.reference_annotation_options or {"Off": ""}).values())
                default_reference = (
                    _geometry.reference_annotation_default
                    if _geometry.reference_annotation_default in reference_values
                    else ""
                )
                get_requested_reference, set_requested_reference = _support.mo.state(default_reference)
                return (get_requested_reference, set_requested_reference)
            """
        ),
    )

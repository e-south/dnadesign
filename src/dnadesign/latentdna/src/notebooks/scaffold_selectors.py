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
            controls = load_workspace_notebook_controls(CONTROL_PATH)
            runtime_paths = controls["runtime_paths"]
            WORKSPACE_DIR = (NOTEBOOK_DIR / str(runtime_paths["workspace_relative_path"])).resolve()
            OUTPUT_ROOT = (NOTEBOOK_DIR / str(runtime_paths["output_relative_path"])).resolve()
            CATALOG_PATH = (NOTEBOOK_DIR / str(runtime_paths["catalog_relative_path"])).resolve()
            HEALTH_PATH = (NOTEBOOK_DIR / str(runtime_paths["health_relative_path"])).resolve()
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
                controls=controls,
            )
            return (runtime, )
        """
    )


def render_selector_cells() -> tuple[str, ...]:
    return (
        dedent(
            """\
            @app.cell
            def _(runtime):
                _catalog = runtime.catalog
                _geometry = runtime.geometry
                _support = runtime.support

                section_selector = _support.mo.ui.dropdown(
                    options={
                        section_name: section_name
                        for section_name in _catalog.section_names or ["Unsectioned"]
                    },
                    value=_catalog.default_section,
                    label="Section",
                )
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
                        _support.option_key_for_value(
                            _geometry.layout_options,
                            _geometry.layout_default,
                        )
                        or next(iter(_geometry.layout_options))
                    ),
                    label="Layout",
                )
                _hue_options = {
                    "(none)": "",
                    **{
                        _support.display_hue_label(column): column
                        for column in _geometry.global_hue_columns
                    },
                }
                hue_selector = _support.mo.ui.dropdown(
                    options=_hue_options,
                    value=(
                        _support.option_key_for_value(_hue_options, _geometry.selected_hue_default)
                        or next(iter(_hue_options))
                    ),
                    label="Hue",
                )
                compare_options = {
                    str(row["label"]): str(row["view_id"]) for row in _geometry.geometry_rows
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
                return (
                    compare_left_selector,
                    compare_right_selector,
                    hue_selector,
                    layout_selector,
                    model_selector,
                    section_selector,
                )
            """
        ),
        dedent(
            """\
            @app.cell
            def _(runtime, section_selector):
                _catalog = runtime.catalog
                _identity = runtime.identity
                _support = runtime.support

                selected_section = str(section_selector.value)
                deliverables_in_section = [
                    row
                    for row in _catalog.deliverables
                    if str(row.get("section") or "Unsectioned") == selected_section
                ]
                deliverable_options = {
                    str(row.get("title") or row["deliverable_id"]): str(row["deliverable_id"])
                    for row in deliverables_in_section
                }
                default_deliverable_label = (
                    _support.option_key_for_value(deliverable_options, _identity.default_deliverable)
                    or next(iter(deliverable_options), "")
                )
                deliverable_selector = _support.mo.ui.dropdown(
                    options=deliverable_options or {"": "No deliverables"},
                    value=default_deliverable_label,
                    label="Deliverable",
                )
                return (deliverable_selector, selected_section)
            """
        ),
        dedent(
            """\
            @app.cell
            def _(deliverable_selector, runtime):
                _catalog = runtime.catalog
                _support = runtime.support

                selected_deliverable_id = str(deliverable_selector.value)
                selected_deliverable = next(
                    (
                        row
                        for row in _catalog.deliverables
                        if str(row.get("deliverable_id")) == selected_deliverable_id
                    ),
                    None,
                )
                selected_plot_rows = [
                    row
                    for row in _catalog.plots
                    if str(row.get("deliverable_id")) == selected_deliverable_id
                ]
                plot_selector = None
                if selected_plot_rows:
                    plot_options = {str(row["plot_id"]): str(row["plot_id"]) for row in selected_plot_rows}
                    plot_selector = _support.mo.ui.dropdown(
                        options=plot_options,
                        value=next(iter(plot_options)),
                        label="Plot",
                    )
                return (plot_selector, selected_deliverable, selected_deliverable_id, selected_plot_rows)
            """
        ),
        dedent(
            """\
            @app.cell
            def _(plot_selector, runtime, selected_plot_rows):
                _identity = runtime.identity
                _support = runtime.support

                selected_plot_id = str(plot_selector.value) if plot_selector is not None else ""
                selected_plot = next(
                    (
                        row
                        for row in selected_plot_rows
                        if str(row.get("plot_id")) == selected_plot_id
                    ),
                    selected_plot_rows[0] if selected_plot_rows else None,
                )
                plot_manifest = {}
                plot_files = []
                plot_render_path = None
                if selected_plot is not None:
                    plot_dir = _identity.output_root / "plots" / str(selected_plot["plot_id"])
                    plot_manifest = _support.load_json(plot_dir / "manifest.json")
                    plot_files = [
                        _identity.output_root / str(path_text)
                        for path_text in selected_plot.get("output_paths", [])
                    ]
                    for suffix in (".svg", ".png", ".jpg", ".jpeg", ".webp", ".pdf"):
                        candidate = next((path for path in plot_files if path.suffix.lower() == suffix), None)
                        if candidate is not None:
                            plot_render_path = candidate
                            break
                return (plot_files, plot_manifest, plot_render_path, selected_plot, selected_plot_id)
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
                ) or ["intermediate"]
                family_default = (
                    str(_geometry.geometry_control.get("default_family"))
                    if str(_geometry.geometry_control.get("default_family")) in family_values
                    else family_values[0]
                )
                _family_options = {value.replace("_", " "): value for value in family_values}
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
                ) or ["60bp"]
                context_default = (
                    str(_geometry.geometry_control.get("default_context"))
                    if str(_geometry.geometry_control.get("default_context")) in context_values
                    else context_values[0]
                )
                _context_options = {
                    {
                        "60bp": "60 bp anchor-only",
                        "1kb_anchor": "1 kb anchor-aligned context",
                        "1kb_seq": "1 kb expanded-context",
                        "1kb_drag": "1 kb context shift",
                    }.get(value, value): value
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
            def _(context_selector, runtime, selected_family, selected_model):
                _geometry = runtime.geometry
                _support = runtime.support

                selected_context = str(context_selector.value)
                compatible_geometries = [
                    row
                    for row in _geometry.geometry_rows
                    if str(row.get("model")) == selected_model and str(row.get("family")) == selected_family
                ]
                matching_geometries = [
                    row for row in compatible_geometries if str(row.get("context")) == selected_context
                ] or compatible_geometries or _geometry.geometry_rows
                _geometry_options = {
                    str(row["label"]): str(row["view_id"])
                    for row in matching_geometries
                } or {"No geometries": ""}
                geometry_selector = _support.mo.ui.dropdown(
                    options=_geometry_options,
                    value=(
                        _support.option_key_for_value(
                            _geometry_options,
                            str(matching_geometries[0]["view_id"]) if matching_geometries else "",
                        )
                        or next(iter(_geometry_options))
                    ),
                    label="Geometry",
                )
                return (geometry_selector, selected_context)
            """
        ),
    )

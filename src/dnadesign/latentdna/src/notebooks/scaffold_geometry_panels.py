"""Projection-browser geometry scaffold cells for the LatentDNA notebook."""

from __future__ import annotations

from textwrap import dedent


def render_geometry_resolution_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(
            context_selector,
            geometry_selector,
            layout_selector,
            projection_selector,
            runtime,
            selected_family,
            selected_model,
        ):
            _geometry = runtime.geometry

            _selected_context = str(context_selector.value)
            _selected_view_id = str(geometry_selector.value)
            _selected_layout_id = str(layout_selector.value)
            _selected_projection_id = str(projection_selector.value)
            selected_layout = next(
                (
                    row
                    for row in _geometry.layout_presets
                    if str(row.get("id")) == _selected_layout_id
                ),
                None,
            )
            selected_geometry = _geometry.geometry_rows_by_id.get(_selected_view_id)
            panel_specs = []
            if selected_layout is None or str(selected_layout.get("mode")) == "single_view":
                if selected_geometry is not None:
                    _projection_ids = [str(item) for item in selected_geometry.get("projection_ids", [])]
                    _projection_id = (
                        _selected_projection_id
                        if _selected_projection_id in _projection_ids
                        else next(iter(_projection_ids), "")
                    )
                    panel_specs = [
                        {
                            "view_id": _selected_view_id,
                            "projection_id": _projection_id,
                            "title": str(selected_geometry.get("label") or _selected_view_id),
                        }
                    ]
            elif str(selected_layout.get("mode")) == "model_pair":
                pair_view_ids = [
                    str(row["view_id"])
                    for row in _geometry.geometry_rows
                    if str(row.get("family")) == selected_family and str(row.get("context")) == _selected_context
                ]
                pair_view_ids = [
                    view_id for view_id in selected_layout.get("view_order", []) if view_id in pair_view_ids
                ] or sorted(pair_view_ids)
                for view_id in pair_view_ids:
                    geometry_row = _geometry.geometry_rows_by_id.get(view_id)
                    if geometry_row is None:
                        continue
                    _projection_id = next(iter(geometry_row.get("projection_ids", [])), "")
                    panel_specs.append(
                        {
                            "view_id": view_id,
                            "projection_id": _projection_id,
                            "title": str(geometry_row.get("label") or view_id),
                        }
                    )
            else:
                for index, view_id in enumerate(selected_layout.get("view_ids", [])):
                    geometry_row = _geometry.geometry_rows_by_id.get(str(view_id))
                    if geometry_row is None:
                        continue
                    _projection_id = next(iter(geometry_row.get("projection_ids", [])), "")
                    _panel_title = (
                        selected_layout.get("panel_titles", [])[index]
                        if index < len(selected_layout.get("panel_titles", []))
                        else str(geometry_row.get("label") or view_id)
                    )
                    panel_specs.append(
                        {
                            "view_id": str(view_id),
                            "projection_id": _projection_id,
                            "title": str(_panel_title),
                        }
                    )
            return (panel_specs, selected_geometry, selected_layout)
        """
    )


def render_geometry_frames_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(panel_specs, runtime, surface_selector):
            _geometry = runtime.geometry
            _renderers = runtime.renderers
            _support = runtime.support

            projection_frames = []
            if str(surface_selector.value) == "geometry_browser":
                for spec in panel_specs:
                    _view_id = str(spec.get("view_id") or "")
                    _projection_id = str(spec.get("projection_id") or "")
                    if not _projection_id:
                        projection_frames.append(_support.pd.DataFrame())
                        continue
                    try:
                        frame = _renderers.load_projection_frame(
                            _view_id or None,
                            _projection_id,
                            _geometry.joinable_tables,
                            required_columns=[
                                *_geometry.preferred_hues,
                                *_geometry.reference_required_columns,
                            ],
                            strict_required_columns=False,
                        )
                    except ValueError as exc:
                        frame = _support.pd.DataFrame()
                        frame.attrs["load_error"] = str(exc)
                    projection_frames.append(frame)
            available_hues = _support.available_hues_for_frames(
                projection_frames,
                preferred_hues=_geometry.preferred_hues,
                hue_kinds=_geometry.hue_kinds,
            )
            return (available_hues, projection_frames)
        """
    )


def render_geometry_hue_selector_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(
            available_hues,
            get_requested_hue,
            get_requested_reference,
            get_requested_reference_annotation_mode,
            runtime,
            set_requested_hue,
            set_requested_reference,
            set_requested_reference_annotation_mode,
        ):
            _geometry = runtime.geometry
            _support = runtime.support

            active_hues = [column for column in _geometry.preferred_hues if column in available_hues]
            _hue_options = {
                "(none)": "",
                **{
                    _support.display_hue_label(column): column
                    for column in active_hues
                },
            }
            _requested_hue = str(get_requested_hue() or "")
            _selected_hue = _requested_hue if _requested_hue in active_hues else ""
            hue_selector = _support.mo.ui.dropdown(
                options=_hue_options,
                value=(
                    _support.option_key_for_value(_hue_options, _selected_hue)
                    or next(iter(_hue_options))
                ),
                label="Population hue",
                on_change=set_requested_hue,
            )
            _reference_options = _geometry.reference_annotation_options or {"Off": ""}
            _reference_values = set(_reference_options.values())
            _requested_reference = str(get_requested_reference() or "")
            _selected_reference = (
                _requested_reference
                if _requested_reference in _reference_values
                else _geometry.reference_annotation_default
                if _geometry.reference_annotation_default in _reference_values
                else ""
            )
            _has_reference_overlay_options = any(str(value).strip() for value in _reference_options.values())
            geometry_reference_selector = (
                _support.mo.ui.dropdown(
                    options=_reference_options,
                    value=(
                        _support.option_key_for_value(
                            _reference_options,
                            _selected_reference,
                        )
                        or next(iter(_reference_options))
                    ),
                    label="Reference set",
                    on_change=set_requested_reference,
                )
                if _has_reference_overlay_options
                else None
            )
            _annotation_mode_options = _support.reference_annotation_mode_options()
            _annotation_mode_values = set(_annotation_mode_options.values())
            _requested_annotation_mode = str(get_requested_reference_annotation_mode() or "auto")
            _selected_annotation_mode = (
                _requested_annotation_mode if _requested_annotation_mode in _annotation_mode_values else "auto"
            )
            geometry_reference_annotation_selector = (
                _support.mo.ui.dropdown(
                    options=_annotation_mode_options,
                    value=(
                        _support.option_key_for_value(
                            _annotation_mode_options,
                            _selected_annotation_mode,
                        )
                        or next(iter(_annotation_mode_options))
                    ),
                    label="Label text",
                    on_change=set_requested_reference_annotation_mode,
                )
                if _has_reference_overlay_options
                else None
            )
            return (
                geometry_reference_annotation_selector,
                geometry_reference_selector,
                hue_selector,
            )


        @app.cell
        def _(
            geometry_reference_selector,
            get_requested_reference_hue,
            projection_frames,
            runtime,
            set_requested_reference_hue,
        ):
            _geometry = runtime.geometry
            _support = runtime.support

            geometry_reference_hue_selector = None
            if geometry_reference_selector is not None:
                _selected_reference = str(geometry_reference_selector.value or "")
                _reference_hue_options = (
                    _geometry.reference_hue_options_by_reference_set.get(_selected_reference)
                    or {"Single-color markers": ""}
                )
                _reference_annotation = _support.resolve_reference_annotation(
                    _selected_reference,
                    projection_frames,
                    workspace_dir=runtime.identity.workspace_dir,
                    fallback_labels=[],
                    label_limit=0,
                )
                _reference_labels = [
                    str(value)
                    for value in _reference_annotation.get("labels", [])
                    if str(value).strip()
                ]
                _reference_match_column = str(
                    _reference_annotation.get("match_column") or "usr_label__primary"
                )
                _reference_hue_columns = [
                    column for column in _reference_hue_options.values() if str(column).strip()
                ]
                _available_reference_hues = set(
                    _support.available_reference_hues_for_frames(
                        projection_frames,
                        preferred_hues=_reference_hue_columns,
                        hue_kinds=_geometry.reference_hue_kinds,
                        reference_labels=_reference_labels,
                        reference_match_column=_reference_match_column,
                        axis_styles=_geometry.axis_styles,
                    )
                )
                _reference_hue_options = {
                    label: column
                    for label, column in _reference_hue_options.items()
                    if not str(column).strip() or column in _available_reference_hues
                } or {"Single-color markers": ""}
                _reference_hue_values = set(_reference_hue_options.values())
                _requested_reference_hue = str(get_requested_reference_hue() or "")
                _selected_reference_hue = (
                    _requested_reference_hue if _requested_reference_hue in _reference_hue_values else ""
                )
                if len(_reference_hue_options) > 1:
                    geometry_reference_hue_selector = _support.mo.ui.dropdown(
                        options=_reference_hue_options,
                        value=(
                            _support.option_key_for_value(
                                _reference_hue_options,
                                _selected_reference_hue,
                            )
                            or next(iter(_reference_hue_options))
                        ),
                        label="Reference color",
                        on_change=set_requested_reference_hue,
                    )
            return (geometry_reference_hue_selector,)
        """
    )


def render_geometry_panel_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(
            available_hues,
            context_selector,
            family_selector,
            geometry_reference_annotation_selector,
            geometry_reference_hue_selector,
            geometry_reference_selector,
            geometry_scope_note,
            geometry_selector,
            hue_selector,
            layout_selector,
            model_selector,
            panel_specs,
            projection_selector,
            projection_frames,
            runtime,
            selected_geometry,
            selected_layout,
            surface_selector,
        ):
            _geometry = runtime.geometry
            _identity = runtime.identity
            _renderers = runtime.renderers
            _support = runtime.support

            if str(surface_selector.value) != "geometry_browser":
                geometry_panel = _support.mo.md("")
            else:
                requested_hue = str(hue_selector.value)
                effective_hue = requested_hue if requested_hue in available_hues else ""
                reference_set_id = (
                    str(geometry_reference_selector.value)
                    if geometry_reference_selector is not None
                    else ""
                )
                reference_label_limit = _support.reference_label_limit_for_annotation_mode(
                    str(geometry_reference_annotation_selector.value)
                    if geometry_reference_annotation_selector is not None
                    else "auto"
                )
                geometry_plot = _renderers.render_projection_grid(
                    panel_specs,
                    frames=projection_frames,
                    hue_column=effective_hue or None,
                    hue_kinds=_geometry.hue_kinds,
                    joinable_tables=_geometry.joinable_tables,
                    reference_labels=_geometry.reference_labels,
                    reference_set_id=reference_set_id,
                    reference_label_limit=reference_label_limit,
                    reference_hue_column=(
                        str(geometry_reference_hue_selector.value)
                        if geometry_reference_hue_selector is not None
                        else ""
                    )
                    or None,
                    reference_hue_kind=_geometry.reference_hue_kinds.get(
                        str(geometry_reference_hue_selector.value)
                        if geometry_reference_hue_selector is not None
                        else "",
                    ),
                )
                if selected_layout is None or str(selected_layout.get("mode")) == "single_view":
                    _control_widgets = [layout_selector, model_selector, family_selector, context_selector]
                    _control_widgets.extend([geometry_selector, projection_selector])
                else:
                    _control_widgets = [layout_selector]
                _control_widgets.append(hue_selector)
                if geometry_reference_selector is not None:
                    _control_widgets.append(geometry_reference_selector)
                if geometry_reference_annotation_selector is not None:
                    _control_widgets.append(geometry_reference_annotation_selector)
                if geometry_reference_hue_selector is not None:
                    _control_widgets.append(geometry_reference_hue_selector)
                _layout_label = str(selected_layout.get("label")) if selected_layout is not None else "Single view"
                _accordion_sections = {
                    "Selection": _support.mo.md(
                        "\\n".join(
                            [
                                f"- **Layout:** {_layout_label}",
                                f"- **Panels:** {len(panel_specs)}",
                                f"- **Hue:** {_support.display_hue_label(effective_hue) if effective_hue else 'None'}",
                                *(
                                    [f"- **Geometry:** {str(selected_geometry.get('label') or '')}"]
                                    if selected_geometry is not None
                                    else []
                                ),
                                *(
                                    [f"- **Rows:** {int(selected_geometry.get('rows')):,}"]
                                    if selected_geometry is not None and selected_geometry.get("rows") is not None
                                    else []
                                ),
                                *(
                                    [f"- **Dimensions:** {int(selected_geometry.get('dims')):,}"]
                                    if selected_geometry is not None and selected_geometry.get("dims") is not None
                                    else []
                                ),
                                *(
                                    [f"- **Role:** `{str(selected_geometry.get('role') or 'primary')}`"]
                                    if (
                                        selected_geometry is not None
                                        and str(selected_geometry.get("role") or "").strip()
                                    )
                                    else []
                                ),
                            ]
                        )
                    )
                }
                _accordion_sections["Reading the projection"] = _support.mo.md(
                    "\\n".join(
                        [
                            "- Coordinates come from persisted projection artifacts; changing hue or reference labels "
                            "does not refit UMAP or PCA.",
                            "- Distances are only comparable within a panel unless panels explicitly share one "
                            "projection artifact.",
                            "- Reference markers are matched rows in the selected view; a missing label means no row "
                            "matched the active view and reference-set selector.",
                        ]
                    )
                )
                _population_lines = []
                for _panel_spec in panel_specs:
                    _projection_id = str(_panel_spec.get("projection_id") or "")
                    if not _projection_id:
                        continue
                    _projection_manifest_path = _identity.output_root / "projections" / _projection_id / "manifest.json"
                    _projection_manifest = _support.load_json(_projection_manifest_path)
                    _manifest_stats = _projection_manifest.get("stats")
                    _stats = _manifest_stats if isinstance(_manifest_stats, dict) else {}
                    _projected_rows = _stats.get("projected_rows", _stats.get("rows"))
                    _population_rows = _stats.get("population_rows", _projected_rows)
                    _is_full_population = bool(_stats.get("is_full_population", _projected_rows == _population_rows))
                    if _projected_rows is None:
                        continue
                    _population_note = (
                        f"{int(_projected_rows):,} rows, full population"
                        if _is_full_population
                        else f"{int(_projected_rows):,} projected rows from {int(_population_rows):,}"
                    )
                    _population_lines.append(
                        f"- **{str(_panel_spec.get('title') or _projection_id)}:** {_population_note}"
                    )
                if _population_lines:
                    _accordion_sections["Projection population"] = _support.mo.md("\\n".join(_population_lines))
                geometry_panel = _support.mo.vstack(
                    [
                        geometry_scope_note,
                        _support.mo.hstack(_control_widgets, justify="start", align="end", wrap=True, gap=0.28),
                        geometry_plot,
                        _support.mo.accordion(_accordion_sections),
                    ],
                    gap=0.35,
                )
            return (geometry_panel,)
        """
    )

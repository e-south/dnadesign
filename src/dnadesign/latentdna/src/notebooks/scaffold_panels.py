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
            _support = runtime.support

            _plot_titles = []
            for _plot_section in runtime.plot_review.sections:
                for _plot_card in _plot_section.get("cards", []):
                    _plot_title = str(_plot_card.get("title") or _plot_card.get("plot_id") or "").strip()
                    if _plot_title and _plot_title not in _plot_titles:
                        _plot_titles.append(_plot_title)
            _plot_scope_text = (
                "Review the current artifact set: " + ", ".join(_plot_titles) + "."
                if _plot_titles
                else "Review the current artifact set."
            )
            plot_scope_note = _support.mo.md(_plot_scope_text)
            geometry_scope_note = _support.mo.md(
                (
                    "This surface is a projection browser for persisted geometry and metadata overlays. "
                    "Point positions are fixed by the saved coordinates, so hue changes only recolor the same geometry."
                )
            )
            return (geometry_scope_note, plot_scope_note)
        """
    )


def render_plot_review_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(runtime):
            _identity = runtime.identity
            _plot_review = runtime.plot_review
            _support = runtime.support

            plot_review_cards = []
            plot_selector = None
            if _plot_review.sections:
                plot_option_pairs = []
                for _plot_section in _plot_review.sections:
                    for _plot_card in _plot_section.get("cards", []):
                        _plot_label = str(_plot_card["title"])
                        plot_option_pairs.append((_plot_label, str(_plot_card["plot_id"])))
                        plot_review_cards.append(dict(_plot_card))
                plot_options = _support.labeled_options(plot_option_pairs)

                default_plot_id = next(
                    (
                        str(card["plot_id"])
                        for card in plot_review_cards
                        if str(card.get("deliverable_id") or "") == str(_identity.default_deliverable or "")
                    ),
                    next(
                        (
                            plot_id
                            for plot_id in _plot_review.ordered_plot_ids
                            if any(str(card["plot_id"]) == str(plot_id) for card in plot_review_cards)
                        ),
                        plot_review_cards[0]["plot_id"],
                    ),
                )
                plot_selector = _support.mo.ui.dropdown(
                    options=plot_options,
                    value=(
                        _support.option_key_for_value(plot_options, default_plot_id)
                        or next(iter(plot_options))
                    ),
                    label="Plot",
                    full_width=True,
                )
            return (plot_review_cards, plot_selector)


        @app.cell
        def _(plot_review_cards, plot_selector):
            selected_plot_card = None
            if plot_review_cards:
                _active_plot_id = (
                    str(plot_selector.value)
                    if plot_selector is not None
                    else str(plot_review_cards[0]["plot_id"])
                )
                selected_plot_card = next(
                    (card for card in plot_review_cards if str(card["plot_id"]) == _active_plot_id),
                    plot_review_cards[0],
                )
            return (selected_plot_card,)


        @app.cell
        def _(runtime, selected_plot_card):
            _renderers = runtime.renderers
            _support = runtime.support

            active_plot_frames = []
            available_plot_hues = []
            plot_hue_selector = None
            plot_reference_selector = None
            if selected_plot_card is not None and bool(selected_plot_card.get("live_render")):
                _plot_spec = dict(selected_plot_card.get("plot_spec") or {})
                active_plot_frames = _renderers.load_plot_review_frames(
                    _plot_spec,
                    joinable_tables=runtime.geometry.joinable_tables,
                    reference_required_columns=runtime.geometry.reference_required_columns,
                )
                _configured_hue_kinds = {
                    str(_option.get("column")): str(_option.get("type"))
                    for _option in _plot_spec.get("hue_options", [])
                    if isinstance(_option, dict) and _option.get("column") and _option.get("type")
                }
                _preferred_hues = [
                    str(_option.get("column"))
                    for _option in _plot_spec.get("hue_options", [])
                    if isinstance(_option, dict) and _option.get("column")
                ]
                available_plot_hues = _support.available_hues_for_frames(
                    active_plot_frames,
                    preferred_hues=_preferred_hues,
                    hue_kinds=_configured_hue_kinds,
                )
                _hue_options = {
                    str(_option.get("label")): str(_option.get("column"))
                    for _option in _plot_spec.get("hue_options", [])
                    if isinstance(_option, dict) and str(_option.get("column")) in available_plot_hues
                }
                if _hue_options:
                    _default_hue = str(_plot_spec.get("default_hue") or "")
                    plot_hue_selector = _support.mo.ui.dropdown(
                        options=_hue_options,
                        value=(
                            _support.option_key_for_value(_hue_options, _default_hue)
                            or next(iter(_hue_options))
                        ),
                        label="Hue",
                    )
                _reference_enabled_kinds = {
                    "projection_grid",
                    "projection_scatter",
                    "xy_scatter_grid",
                    "paired_xy_scatter_grid",
                }
                if str(_plot_spec.get("kind") or "") in _reference_enabled_kinds:
                    _reference_options = runtime.geometry.reference_annotation_options or {"Off": ""}
                    plot_reference_selector = _support.mo.ui.dropdown(
                        options=_reference_options,
                        value=(
                            _support.option_key_for_value(
                                _reference_options,
                                runtime.geometry.reference_annotation_default,
                            )
                            or next(iter(_reference_options))
                        ),
                        label="Reference labels",
                    )
            return (active_plot_frames, available_plot_hues, plot_hue_selector, plot_reference_selector)


        @app.cell
        def _(
            active_plot_frames,
            available_plot_hues,
            plot_hue_selector,
            plot_reference_selector,
            plot_review_cards,
            plot_scope_note,
            plot_selector,
            runtime,
            selected_plot_card,
        ):
            _renderers = runtime.renderers
            _support = runtime.support

            if not plot_review_cards:
                plot_review_panel = _support.mo.vstack(
                    [
                        plot_scope_note,
                        _support.mo.callout(
                            "No sanctioned plot review inventory is configured for this notebook.",
                            kind="warn",
                        ),
                    ],
                    gap=0.3,
                )
            else:
                _active_card = selected_plot_card or plot_review_cards[0]
                _heading = _support.mo.md(f"## {_active_card['title']}")
                if bool(_active_card.get("live_render")):
                    _requested_hue = str(plot_hue_selector.value) if plot_hue_selector is not None else ""
                    _effective_hue = _requested_hue if _requested_hue in available_plot_hues else None
                    _selected_reference_set = (
                        str(plot_reference_selector.value)
                        if plot_reference_selector is not None
                        else runtime.geometry.reference_annotation_default
                    )
                    _plot_spec = {
                        **dict(_active_card.get("plot_spec") or {}),
                        "alt_text": str(_active_card.get("alt_text") or _active_card.get("title") or ""),
                    }
                    _plot_surface = _renderers.render_plot_review_surface(
                        _plot_spec,
                        frames=active_plot_frames,
                        hue_column=_effective_hue,
                        reference_labels=runtime.geometry.reference_labels,
                        reference_set_id=_selected_reference_set,
                        joinable_tables=runtime.geometry.joinable_tables,
                    )
                else:
                    _plot_surface = (
                        _renderers.render_plot_asset(
                            _active_card["render_path"],
                            alt_text=str(_active_card.get("alt_text") or _active_card.get("title") or ""),
                        )
                        if _active_card.get("render_path") is not None
                        else _support.mo.callout(
                            f"No persisted plot asset is available for `{_active_card['plot_id']}`.",
                            kind="warn",
                        )
                    )

                _section_blocks = [
                    plot_scope_note,
                    _heading,
                ]
                _selectors = [
                    widget
                    for widget in [plot_selector, plot_hue_selector, plot_reference_selector]
                    if widget is not None
                ]
                if _selectors:
                    _section_blocks.insert(
                        1,
                        _support.mo.hstack(_selectors, justify="start", align="end", wrap=True, gap=0.28),
                    )
                _status = str(_active_card.get("status") or "missing")
                if (
                    _active_card.get("render_path") is None
                    and not bool(_active_card.get("live_render"))
                    and _status == "ok"
                ):
                    _status = "missing"
                _stale = bool(_active_card.get("stale"))
                if _status != "ok" or _stale:
                    _status_message = (
                        "This plot artifact is stale relative to the current workspace state."
                        if _stale
                        else f"This plot artifact status is `{_status}`."
                    )
                    _section_blocks.append(
                        _support.mo.callout(
                            _status_message + " Rebuild it or inspect the deliverable status before using it.",
                            kind="warn",
                        )
                    )
                if _active_card.get("render_mode_note"):
                    _section_blocks.append(
                        _support.mo.callout(
                            str(_active_card["render_mode_note"]),
                            kind="info",
                        )
                    )
                if _active_card.get("artifact_warning"):
                    _section_blocks.append(_support.mo.callout(str(_active_card["artifact_warning"]), kind="warn"))
                if _active_card.get("study_doc_warning"):
                    _section_blocks.append(_support.mo.callout(str(_active_card["study_doc_warning"]), kind="warn"))
                _accordion_sections = {}
                _at_a_glance_lines = []
                if str(_active_card.get("question") or "").strip():
                    _at_a_glance_lines.append(f"- **Question:** {str(_active_card['question'])}")
                if str(_active_card.get("scope") or "").strip():
                    _at_a_glance_lines.append(f"- **Scope:** {str(_active_card['scope'])}")
                if str(_active_card.get("decision_role") or "").strip():
                    _at_a_glance_lines.append(f"- **Role:** `{str(_active_card['decision_role'])}`")
                if str(_active_card.get("encoding") or "").strip():
                    _at_a_glance_lines.append(f"- **Encoding:** {str(_active_card['encoding'])}")
                if _at_a_glance_lines:
                    _accordion_sections["At a glance"] = _support.mo.md("\\n".join(_at_a_glance_lines))
                if str(_active_card.get("study_doc_md") or "").strip():
                    _accordion_sections["Study notes"] = _support.mo.md(str(_active_card["study_doc_md"]))
                _guardrails = [
                    str(item).strip()
                    for item in (_active_card.get("guardrails") or [])
                    if str(item).strip()
                ]
                if _guardrails:
                    _accordion_sections["Guardrails"] = _support.mo.md(
                        "\\n".join(f"- {item}" for item in _guardrails)
                    )
                for _section_title, _field_name in [
                    ("Caption", "caption_md"),
                    ("Preprocessing", "preprocessing_md"),
                    ("Math", "math_md"),
                    ("Why this helps choose X", "rationale_md"),
                    ("Limits", "limitations_md"),
                    ("Failure modes", "failure_modes_md"),
                    ("Plot details", "plot_details_md"),
                ]:
                    if str(_active_card.get(_field_name) or "").strip():
                        if _field_name == "math_md":
                            _accordion_sections[_section_title] = _support.render_math_markdown(
                                str(_active_card[_field_name])
                            )
                        else:
                            _accordion_sections[_section_title] = _support.mo.md(str(_active_card[_field_name]))
                _section_blocks.append(_plot_surface)
                if _accordion_sections:
                    _section_blocks.append(
                        _support.mo.accordion(_accordion_sections, lazy=True)
                    )

                plot_review_panel = _support.mo.vstack(_section_blocks, gap=0.4)
            return (plot_review_panel,)
        """
    )


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
        def _(panel_specs, runtime):
            _geometry = runtime.geometry
            _renderers = runtime.renderers
            _support = runtime.support

            projection_frames = []
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
        def _(available_hues, get_requested_hue, runtime, set_requested_hue):
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
                label="Hue",
                on_change=set_requested_hue,
            )
            _reference_options = _geometry.reference_annotation_options or {"Off": ""}
            geometry_reference_selector = _support.mo.ui.dropdown(
                options=_reference_options,
                value=(
                    _support.option_key_for_value(
                        _reference_options,
                        _geometry.reference_annotation_default,
                    )
                    or next(iter(_reference_options))
                ),
                label="Reference labels",
            )
            return (geometry_reference_selector, hue_selector)
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
        ):
            _geometry = runtime.geometry
            _identity = runtime.identity
            _renderers = runtime.renderers
            _support = runtime.support

            requested_hue = str(hue_selector.value)
            effective_hue = requested_hue if requested_hue in available_hues else ""
            geometry_plot = _renderers.render_projection_grid(
                panel_specs,
                frames=projection_frames,
                hue_column=effective_hue or None,
                hue_kinds=_geometry.hue_kinds,
                joinable_tables=_geometry.joinable_tables,
                reference_labels=_geometry.reference_labels,
                reference_set_id=str(geometry_reference_selector.value),
            )
            _control_widgets = [layout_selector, model_selector, family_selector, context_selector]
            if selected_layout is None or str(selected_layout.get("mode")) == "single_view":
                _control_widgets.extend([geometry_selector, projection_selector])
            else:
                _control_widgets.append(projection_selector)
            _control_widgets.extend([hue_selector, geometry_reference_selector])
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
                                if selected_geometry is not None and str(selected_geometry.get("role") or "").strip()
                                else []
                            ),
                        ]
                    )
                )
            }
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
                _population_lines.append(f"- **{str(_panel_spec.get('title') or _projection_id)}:** {_population_note}")
            if _population_lines:
                _accordion_sections["Projection population"] = _support.mo.md("\\n".join(_population_lines))
            geometry_panel = _support.mo.vstack(
                [
                    geometry_scope_note,
                    _support.mo.hstack(_control_widgets, justify="start", align="end", wrap=True, gap=0.28),
                    geometry_plot,
                    _support.mo.accordion(_accordion_sections, lazy=True),
                ],
                gap=0.35,
            )
            return (geometry_panel,)
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

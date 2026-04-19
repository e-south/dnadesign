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

            plot_scope_note = _support.mo.md(
                (
                    "This notebook is the LatentDNA pre-assay review surface for the current "
                    "`infer_batch_preparation` study snapshot. DenseGen remains the source of cohort semantics "
                    "and provenance. Use these plots to review representation health, design structure, "
                    "Sigma-35 organization, and context robustness. Do not use this notebook as the "
                    "study-status record."
                )
            )
            geometry_scope_note = _support.mo.md(
                (
                    "This tab is a projection browser for persisted geometry and metadata overlays. "
                    "Point positions are fixed by the saved coordinates, so hue changes only recolor the same geometry."
                )
            )
            comparison_scope_note = _support.mo.md(
                (
                    "This tab is a sampled diagnostic built from persisted artifacts. "
                    "Use it to check agreement between views, not as a hidden total score or final authority."
                )
            )
            return (comparison_scope_note, geometry_scope_note, plot_scope_note)
        """
    )


def render_plot_review_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(runtime):
            _plot_review = runtime.plot_review
            _support = runtime.support

            plot_review_cards = []
            plot_selector = None
            if _plot_review.sections:
                plot_options = {}
                for _plot_section in _plot_review.sections:
                    for _plot_card in _plot_section.get("cards", []):
                        _plot_label = str(_plot_card["title"])
                        plot_options[_plot_label] = str(_plot_card["plot_id"])
                        plot_review_cards.append(dict(_plot_card))

                default_plot_id = next(
                    (
                        plot_id
                        for plot_id in _plot_review.ordered_plot_ids
                        if any(str(card["plot_id"]) == str(plot_id) for card in plot_review_cards)
                    ),
                    plot_review_cards[0]["plot_id"],
                )
                plot_selector = _support.mo.ui.dropdown(
                    options=plot_options,
                    value=(
                        _support.option_key_for_value(plot_options, default_plot_id)
                        or next(iter(plot_options))
                    ),
                    label="Plot",
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
            if selected_plot_card is not None and bool(selected_plot_card.get("live_render")):
                _plot_spec = dict(selected_plot_card.get("plot_spec") or {})
                active_plot_frames = _renderers.load_plot_review_frames(
                    _plot_spec,
                    joinable_tables=runtime.geometry.joinable_tables,
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
            return (active_plot_frames, available_plot_hues, plot_hue_selector)


        @app.cell
        def _(
            active_plot_frames,
            available_plot_hues,
            plot_hue_selector,
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
                    _plot_surface = _renderers.render_plot_review_surface(
                        dict(_active_card.get("plot_spec") or {}),
                        frames=active_plot_frames,
                        hue_column=_effective_hue,
                        reference_labels=runtime.geometry.reference_labels,
                        joinable_tables=runtime.geometry.joinable_tables,
                    )
                else:
                    _plot_surface = (
                        _renderers.render_plot_asset(_active_card["render_path"])
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
                _selectors = [widget for widget in [plot_selector, plot_hue_selector] if widget is not None]
                if _selectors:
                    _section_blocks.insert(
                        1,
                        _support.mo.hstack(_selectors, justify="start", align="end", wrap=True, gap=0.28),
                    )
                if str(_active_card.get("status") or "missing") != "ok":
                    _section_blocks.append(
                        _support.mo.callout(
                            "This plot artifact is not current. Rebuild it or inspect the deliverable status "
                            "before using it.",
                            kind="warn",
                        )
                    )
                if _active_card.get("study_doc_warning"):
                    _section_blocks.append(_support.mo.callout(str(_active_card["study_doc_warning"]), kind="warn"))
                if str(_active_card.get("plot_details_md") or "").strip():
                    _section_blocks.append(
                        _support.mo.accordion(
                            {"Plot details": _support.mo.md(str(_active_card["plot_details_md"]))}
                        )
                    )
                _section_blocks.append(_plot_surface)
                if str(_active_card.get("caption_md") or "").strip():
                    _section_blocks.append(_support.mo.md(str(_active_card["caption_md"])))

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
            runtime,
            selected_family,
            selected_model,
        ):
            _geometry = runtime.geometry

            _selected_context = str(context_selector.value)
            _selected_view_id = str(geometry_selector.value)
            _selected_layout_id = str(layout_selector.value)
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
                    _projection_id = next(iter(selected_geometry.get("projection_ids", [])), "")
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
            _identity = runtime.identity
            _renderers = runtime.renderers
            _support = runtime.support

            projection_frames = []
            for spec in panel_specs:
                _projection_id = str(spec.get("projection_id") or "")
                if not _projection_id:
                    projection_frames.append(_support.pd.DataFrame())
                    continue
                frame = _support.load_table(_identity.output_root / "projections" / _projection_id / "coords.parquet")
                if not frame.empty:
                    frame = _renderers.enrich_projection_frame(frame, _geometry.joinable_tables)
                projection_frames.append(frame)
            available_hues = _support.available_hues_for_frames(
                projection_frames,
                preferred_hues=_geometry.preferred_hues,
                hue_kinds=_geometry.hue_kinds,
            )
            _hue_options = {
                "(none)": "",
                **{_support.display_hue_label(column): column for column in available_hues},
            }
            default_hue = _geometry.selected_hue_default if _geometry.selected_hue_default in available_hues else ""
            hue_selector = _support.mo.ui.dropdown(
                options=_hue_options,
                value=(
                    _support.option_key_for_value(_hue_options, default_hue)
                    or next(iter(_hue_options))
                ),
                label="Hue",
            )
            return (available_hues, hue_selector, projection_frames)
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
            geometry_scope_note,
            geometry_selector,
            hue_selector,
            layout_selector,
            model_selector,
            panel_specs,
            projection_frames,
            runtime,
            selected_geometry,
            selected_layout,
        ):
            _geometry = runtime.geometry
            _renderers = runtime.renderers
            _support = runtime.support

            requested_hue = str(hue_selector.value)
            effective_hue = requested_hue if requested_hue in available_hues else ""
            hue_notice = None
            if requested_hue and not effective_hue:
                hue_notice = _support.mo.callout(
                    "Hue reset because it is not available for the active layout.",
                    kind="info",
                )
            geometry_plot = _renderers.render_projection_grid(
                panel_specs,
                frames=projection_frames,
                hue_column=effective_hue or None,
                hue_kinds=_geometry.hue_kinds,
                joinable_tables=_geometry.joinable_tables,
                reference_labels=_geometry.reference_labels,
            )
            _control_widgets = [layout_selector, model_selector, family_selector, context_selector]
            if selected_layout is None or str(selected_layout.get("mode")) == "single_view":
                _control_widgets.append(geometry_selector)
            _control_widgets.append(hue_selector)
            _layout_label = (
                str(selected_layout.get("label"))
                if selected_layout is not None
                else "Single view"
            )
            _geometry_label = str(selected_geometry.get("label")) if selected_geometry is not None else ""
            _selection_summary = (
                f"Layout: **{_layout_label}**. "
                f"Panels: **{len(panel_specs)}**. "
                f"Hue: **{_support.display_hue_label(effective_hue) if effective_hue else 'None'}**."
            )
            if _geometry_label:
                _selection_summary += f" Geometry: **{_geometry_label}**."
            geometry_panel = _support.mo.vstack(
                [
                    geometry_scope_note,
                    _support.mo.hstack(_control_widgets, justify="start", align="end", wrap=True, gap=0.28),
                    _support.mo.md(_selection_summary),
                    *([hue_notice] if hue_notice is not None else []),
                    geometry_plot,
                ],
                gap=0.35,
            )
            return (geometry_panel,)
        """
    )


def render_compare_panel_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(compare_left_selector, compare_right_selector, comparison_scope_note, runtime):
            _geometry = runtime.geometry
            _renderers = runtime.renderers
            _support = runtime.support

            def _format_metric_value(value):
                if value is None:
                    return "n/a"
                if isinstance(value, float):
                    if abs(value) >= 1e-3:
                        return f"{value:.4f}"
                    return f"{value:.3e}"
                return value

            selected_compare_left = str(compare_left_selector.value)
            selected_compare_right = str(compare_right_selector.value)
            compare_payload = _renderers.compare_pair_payload(
                selected_compare_left,
                selected_compare_right,
                geometry_rows_by_id=_geometry.geometry_rows_by_id,
                comparison_bases=_geometry.comparison_bases,
                compare_metrics=_geometry.compare_metrics,
            )
            distance_correlation_plot = _renderers.render_distance_correlation(
                compare_payload,
                title="Pairwise distance Spearman",
            )
            rowwise_cosine_plot = _renderers.render_rowwise_distribution(
                compare_payload,
                value_key="rowwise_cosine",
                title="Row-wise cosine similarity",
                xlabel="Row-wise cosine similarity",
            )
            rowwise_diff_plot = _renderers.render_rowwise_distribution(
                compare_payload,
                value_key="rowwise_diff_norm",
                title="Row-wise L2 difference",
                xlabel="Row-wise L2 norm",
            )
            metrics_payload = compare_payload.get("metrics")
            metrics_rows = [
                {"Metric": "Basis", "Value": compare_payload.get("basis_display", "Unavailable")},
                {"Metric": "Status", "Value": compare_payload.get("status", "missing")},
                {"Metric": "Rows compared", "Value": compare_payload.get("rows", "n/a")},
                {
                    "Metric": "Sampling strategy",
                    "Value": compare_payload.get("sample_strategy_display", "Unavailable"),
                },
                {
                    "Metric": "Distance Spearman",
                    "Value": (
                        _format_metric_value(metrics_payload.get("distance_spearman"))
                        if isinstance(metrics_payload, dict)
                        else "n/a"
                    ),
                },
                {
                    "Metric": "Linear CKA",
                    "Value": (
                        _format_metric_value(metrics_payload.get("linear_cka"))
                        if isinstance(metrics_payload, dict)
                        else "n/a"
                    ),
                },
                {
                    "Metric": "Neighbor-set Jaccard",
                    "Value": (
                        _format_metric_value(metrics_payload.get("neighbor_set_jaccard"))
                        if isinstance(metrics_payload, dict)
                        else "n/a"
                    ),
                },
                {
                    "Metric": "Median row-wise cosine",
                    "Value": (
                        _format_metric_value(metrics_payload.get("median_rowwise_cosine"))
                        if isinstance(metrics_payload, dict)
                        else "n/a"
                    ),
                },
                {
                    "Metric": "Median row-wise L2",
                    "Value": (
                        _format_metric_value(metrics_payload.get("median_rowwise_diff_norm"))
                        if isinstance(metrics_payload, dict)
                        else "n/a"
                    ),
                },
            ]
            compare_panel = _support.mo.vstack(
                [
                    comparison_scope_note,
                    _support.mo.hstack(
                        [compare_left_selector, compare_right_selector],
                        justify="start",
                        align="end",
                        wrap=True,
                        gap=0.28,
                    ),
                    distance_correlation_plot,
                    _support.mo.hstack(
                        [rowwise_cosine_plot, rowwise_diff_plot],
                        gap=0.35,
                        wrap=True,
                        align="start",
                        justify="center",
                    ),
                    _support.key_value_table([(row["Metric"], row["Value"]) for row in metrics_rows]),
                ],
                gap=0.35,
            )
            return (compare_panel,)
        """
    )


def render_page_tabs_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(runtime):
            _support = runtime.support
            _plot_review = runtime.plot_review
            default_tab = {
                "plots": "Plots",
                "geometry_audit": "Geometry audit",
                "comparison_audit": "Comparison audit",
            }.get(_plot_review.default_surface, "Plots")
            get_active_top_tab, set_active_top_tab = _support.mo.state(default_tab)
            return (default_tab, get_active_top_tab, set_active_top_tab)


        @app.cell
        def _(compare_panel, geometry_panel, get_active_top_tab, plot_review_panel, runtime, set_active_top_tab):
            _support = runtime.support
            active_top_tab = get_active_top_tab() or "Plots"
            page_tabs = _support.mo.ui.tabs(
                {
                    "Plots": plot_review_panel,
                    "Geometry audit": geometry_panel,
                    "Comparison audit": compare_panel,
                },
                value=active_top_tab,
                lazy=True,
                on_change=set_active_top_tab,
            )
            return (page_tabs,)
        """
    )


def render_page_display_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(page_tabs):
            page_tabs
            return
        """
    )

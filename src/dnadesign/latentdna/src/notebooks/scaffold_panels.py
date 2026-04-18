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

            plot_scope_note = _support.mo.callout(
                (
                    "This notebook is for reviewing generated representation-comparison plots in scientific order. "
                    "Geometry and Comparison are secondary audit views. LatentDNA is downstream analysis, not the "
                    "study-status authority. Appendix plots are orientation and proxy surfaces, not primary "
                    "decision evidence."
                ),
                kind="info",
            )
            geometry_scope_note = _support.mo.callout(
                (
                    "Projection browser for geometry and metadata overlays. Point positions come from persisted "
                    "coordinates and do not change when hue changes. Use this to inspect geometry, not to pick a "
                    "winner by UMAP appearance."
                ),
                kind="info",
            )
            comparison_scope_note = _support.mo.callout(
                (
                    "Sampled live diagnostic computed from persisted artifacts. Useful for agreement and audit "
                    "checks, but not full-population proof and not a hidden total score."
                ),
                kind="info",
            )
            return (comparison_scope_note, geometry_scope_note, plot_scope_note)
        """
    )


def render_plot_review_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(plot_scope_note, runtime):
            _plot_review = runtime.plot_review
            _renderers = runtime.renderers
            _support = runtime.support

            if not _plot_review.sections:
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
                jump_labels = []
                for section in _plot_review.sections:
                    for card in section.get("cards", []):
                        jump_labels.append(f"`{card['plot_id']}`")
                jump_list = _support.mo.md("Jump list: " + " · ".join(jump_labels))

                section_blocks = [plot_scope_note, jump_list]
                for section in _plot_review.sections:
                    section_blocks.append(_support.mo.md(f"## {section['title']}"))
                    if str(section.get("summary") or "").strip():
                        section_blocks.append(_support.mo.md(str(section["summary"])))
                    for card in section.get("cards", []):
                        badge_class = "appendix" if str(card.get("visibility_tier")) == "appendix" else "primary"
                        badge = _support.mo.Html(
                            "<span class='latentdna-badge "
                            f"latentdna-badge--{badge_class}'>{card['badge']}</span>"
                        )
                        heading = _support.mo.hstack(
                            [_support.mo.md(f"### {card['title']}"), badge],
                            justify="space-between",
                            align="center",
                            gap=0.25,
                        )
                        card_blocks = [
                            heading,
                            (
                                _renderers.render_plot_asset(card["render_path"])
                                if card.get("render_path") is not None
                                else _support.mo.callout(
                                    f"No persisted plot asset is available for `{card['plot_id']}`.",
                                    kind="warn",
                                )
                            ),
                        ]
                        if str(card.get("caption_md") or "").strip():
                            card_blocks.append(_support.mo.md(str(card["caption_md"])))
                        if card.get("study_doc_warning"):
                            card_blocks.append(_support.mo.callout(str(card["study_doc_warning"]), kind="warn"))
                        if str(card.get("study_doc_md") or "").strip():
                            card_blocks.append(_support.mo.md(str(card["study_doc_md"])))
                        if str(card.get("guardrail_text") or "").strip():
                            card_blocks.append(_support.mo.callout(str(card["guardrail_text"]), kind="info"))
                        section_blocks.append(_support.mo.vstack(card_blocks, gap=0.22))

                plot_review_panel = _support.mo.vstack(section_blocks, gap=0.4)
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
                    title = (
                        selected_layout.get("panel_titles", [])[index]
                        if index < len(selected_layout.get("panel_titles", []))
                        else str(geometry_row.get("label") or view_id)
                    )
                    panel_specs.append(
                        {
                            "view_id": str(view_id),
                            "projection_id": _projection_id,
                            "title": str(title),
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
            geometry_status = _support.table_from_records(
                [
                    {
                        "Field": "Layout",
                        "Value": (
                            selected_layout.get("label")
                            if selected_layout is not None
                            else "single view"
                        ),
                    },
                    {"Field": "Panels", "Value": len(panel_specs)},
                    {
                        "Field": "Geometry",
                        "Value": (
                            selected_geometry.get("label")
                            if selected_geometry is not None
                            else "layout-selected"
                        ),
                    },
                    {"Field": "Hue", "Value": effective_hue or "none"},
                    {
                        "Field": "Available hues",
                        "Value": ", ".join(available_hues) if available_hues else "none",
                    },
                ],
                columns=["Field", "Value"],
            )
            geometry_panel = _support.mo.vstack(
                [
                    geometry_scope_note,
                    _support.mo.hstack(_control_widgets, justify="start", align="end", wrap=True, gap=0.28),
                    *([hue_notice] if hue_notice is not None else []),
                    geometry_plot,
                    geometry_status,
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
                title="Pairwise distance correlation",
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
                {"Metric": "Basis", "Value": compare_payload.get("basis", "unavailable")},
                {"Metric": "Status", "Value": compare_payload.get("status", "missing")},
                {"Metric": "Rows", "Value": compare_payload.get("rows", "n/a")},
                {"Metric": "Sampling strategy", "Value": compare_payload.get("sample_strategy", "n/a")},
                {
                    "Metric": "Distance Spearman",
                    "Value": (
                        metrics_payload.get("distance_spearman")
                        if isinstance(metrics_payload, dict)
                        else None
                    ),
                },
                {
                    "Metric": "Linear CKA",
                    "Value": (
                        metrics_payload.get("linear_cka")
                        if isinstance(metrics_payload, dict)
                        else None
                    ),
                },
                {
                    "Metric": "Neighbor-set Jaccard",
                    "Value": (
                        metrics_payload.get("neighbor_set_jaccard")
                        if isinstance(metrics_payload, dict)
                        else None
                    ),
                },
                {
                    "Metric": "Median row-wise cosine",
                    "Value": (
                        metrics_payload.get("median_rowwise_cosine")
                        if isinstance(metrics_payload, dict)
                        else None
                    ),
                },
                {
                    "Metric": "Median row-wise L2",
                    "Value": (
                        metrics_payload.get("median_rowwise_diff_norm")
                        if isinstance(metrics_payload, dict)
                        else None
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
                    _support.table_from_records(metrics_rows, columns=["Metric", "Value"]),
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
        def _(compare_panel, geometry_panel, plot_review_panel, runtime):
            _support = runtime.support
            _plot_review = runtime.plot_review
            default_tab = {
                "plots": "Plots",
                "geometry_audit": "Geometry audit",
                "comparison_audit": "Comparison audit",
            }.get(_plot_review.default_surface, "Plots")

            page_tabs = _support.mo.ui.tabs(
                {
                    "Plots": plot_review_panel,
                    "Geometry audit": geometry_panel,
                    "Comparison audit": compare_panel,
                },
                value=default_tab,
                lazy=True,
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

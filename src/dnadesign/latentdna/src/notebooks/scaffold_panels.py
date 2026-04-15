"""
Page-panel cell templates for generated latentdna marimo notebooks.
"""

from __future__ import annotations

from textwrap import dedent


def render_context_audit_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(runtime):
            _catalog = runtime.catalog
            _support = runtime.support

            context_audit = _catalog.controls.get("context_audit", {})
            context_audit_kind = (
                "info"
                if str(context_audit.get("decision")) == "whole_sequence_primary"
                else "warn"
            )
            context_audit_md = _support.mo.callout(
                "\\n".join(
                    [
                        f"Context audit status: `{context_audit.get('status', 'missing')}`",
                        f"Decision: `{context_audit.get('decision', 'not_evaluated')}`",
                        (
                            "Median construct_shift20_norm: "
                            f"`{context_audit.get('metrics', {}).get('construct_shift20_norm_median')}`"
                            if isinstance(context_audit.get("metrics"), dict)
                            else "Median construct_shift20_norm: unavailable"
                        ),
                        (
                            "Median construct_self_cosine20: "
                            f"`{context_audit.get('metrics', {}).get('construct_self_cosine20_median')}`"
                            if isinstance(context_audit.get("metrics"), dict)
                            else "Median construct_self_cosine20: unavailable"
                        ),
                        (
                            "Median 20B log likelihood / token (60 bp anchor-only): "
                            f"`{context_audit.get('metrics', {}).get('anchor20_log_likelihood_per_token_median')}`"
                            if isinstance(context_audit.get("metrics"), dict)
                            and context_audit.get("metrics", {}).get("anchor20_log_likelihood_per_token_median")
                            is not None
                            else "Median 20B log likelihood / token (60 bp anchor-only): unavailable"
                        ),
                        (
                            "Median 20B log likelihood / token (1 kb expanded-context): "
                            f"`{context_audit.get('metrics', {}).get("
                            "'expanded_context20_log_likelihood_per_token_median')}`"
                            if isinstance(context_audit.get("metrics"), dict)
                            and context_audit.get("metrics", {}).get(
                                "expanded_context20_log_likelihood_per_token_median"
                            )
                            is not None
                            else "Median 20B log likelihood / token (1 kb expanded-context): unavailable"
                        ),
                        (
                            "Mean kNN overlap: "
                            f"`{context_audit.get('metrics', {}).get('mean_knn_overlap')}`"
                            if isinstance(context_audit.get("metrics"), dict)
                            else "Mean kNN overlap: unavailable"
                        ),
                        (
                            "Mean landmark Jaccard: "
                            f"`{context_audit.get('metrics', {}).get('mean_jaccard_overlap')}`"
                            if isinstance(context_audit.get("metrics"), dict)
                            else "Mean landmark Jaccard: unavailable"
                        ),
                    ]
                ),
                kind=context_audit_kind,
            )
            return (context_audit_md,)
        """
    )


def render_overview_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(context_audit_md, runtime):
            _catalog = runtime.catalog
            _identity = runtime.identity
            _support = runtime.support

            context_block = _support.mo.md(
                "\\n".join(
                    [
                        f"# {_identity.title}",
                        "",
                        _identity.description or "",
                        "",
                        f"- Workspace: `{_identity.workspace_id}`",
                        f"- Notebook: `{_identity.notebook_id}`",
                        f"- Datasets: {', '.join(f'`{item}`' for item in _identity.source_labels) or 'none'}",
                        f"- Row count: {_identity.row_count_text}",
                        f"- Dimensionality: {_identity.dimensionality_text}",
                        (
                            f"- Key vector columns: "
                            f"{', '.join(f'`{item}`' for item in _identity.vector_columns) or 'none'}"
                        ),
                        f"- Deliverables present: `{len(_catalog.deliverables)}`",
                        (
                            f"- Visual families present: "
                            f"{', '.join(f'`{item}`' for item in _identity.visual_families) or 'none'}"
                        ),
                    ]
                )
            )
            overview_panel = _support.mo.vstack([context_block, context_audit_md])
            return (overview_panel,)
        """
    )


def render_geometry_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(
            context_audit_md,
            context_selector,
            family_selector,
            geometry_selector,
            hue_selector,
            layout_selector,
            model_selector,
            runtime,
            selected_context,
            selected_family,
        ):
            _geometry = runtime.geometry
            _identity = runtime.identity
            _renderers = runtime.renderers
            _support = runtime.support

            selected_view_id = str(geometry_selector.value)
            selected_geometry = next(
                (
                    row
                    for row in _geometry.geometry_rows
                    if str(row.get("view_id")) == selected_view_id
                ),
                None,
            )
            selected_layout_id = str(layout_selector.value)
            selected_layout = next(
                (
                    row
                    for row in _geometry.layout_presets
                    if str(row.get("id")) == selected_layout_id
                ),
                None,
            )
            projection_ids = (
                list(selected_geometry.get("projection_ids", []))
                if selected_geometry is not None
                else []
            )
            selected_projection_id = projection_ids[0] if projection_ids else ""
            projection_frame = (
                _support.load_table(
                    _identity.output_root / "projections" / selected_projection_id / "coords.parquet"
                )
                if selected_projection_id
                else _support.pd.DataFrame()
            )
            projection_frame = _renderers.enrich_projection_frame(
                projection_frame,
                _geometry.joinable_tables,
            )
            hue_columns = _support.candidate_hue_columns(
                projection_frame,
                _geometry.preferred_hues,
                _geometry.joinable_artifact_suffixes,
            )
            selected_hue = str(hue_selector.value)

            panel_specs: list[dict[str, object]] = []
            if selected_layout is None or str(selected_layout.get("mode")) == "single_view":
                if selected_geometry is not None:
                    panel_specs = [
                        {
                            "view_id": selected_view_id,
                            "projection_id": selected_projection_id,
                            "title": str(selected_geometry.get("label") or selected_view_id),
                        }
                    ]
            elif str(selected_layout.get("mode")) == "model_pair":
                pair_view_ids = [
                    str(row["view_id"])
                    for row in _geometry.geometry_rows
                    if str(row.get("family")) == selected_family and str(row.get("context")) == selected_context
                ]
                pair_view_ids = [
                    _view_id
                    for _view_id in selected_layout.get("view_order", [])
                    if _view_id in pair_view_ids
                ] or sorted(pair_view_ids)
                for _view_id in pair_view_ids:
                    geometry_row = _geometry.geometry_rows_by_id.get(_view_id)
                    if geometry_row is None:
                        continue
                    projection_id = next(iter(geometry_row.get("projection_ids", [])), "")
                    panel_specs.append(
                        {
                            "view_id": _view_id,
                            "projection_id": projection_id,
                            "title": str(geometry_row.get("label") or _view_id),
                        }
                    )
            else:
                for index, _view_id in enumerate(selected_layout.get("view_ids", [])):
                    geometry_row = _geometry.geometry_rows_by_id.get(str(_view_id))
                    if geometry_row is None:
                        continue
                    projection_id = next(iter(geometry_row.get("projection_ids", [])), "")
                    title = (
                        selected_layout.get("panel_titles", [])[index]
                        if index < len(selected_layout.get("panel_titles", []))
                        else str(geometry_row.get("label") or _view_id)
                    )
                    panel_specs.append(
                        {
                            "view_id": str(_view_id),
                            "projection_id": projection_id,
                            "title": str(title),
                        }
                    )

            geometry_plot = _renderers.render_projection_grid(
                panel_specs,
                hue_column=selected_hue or None,
                joinable_tables=_geometry.joinable_tables,
                reference_labels=_geometry.reference_labels,
            )

            geometry_controls = _support.mo.vstack(
                [
                    _support.mo.md("## Geometry Views"),
                    model_selector,
                    family_selector,
                    context_selector,
                    layout_selector,
                    geometry_selector,
                    (
                        _support.mo.md(f"Projection: `{selected_projection_id}`")
                        if selected_projection_id
                        else _support.mo.callout(
                            "No projection has been materialized yet for the selected geometry.",
                            kind="warn",
                        )
                    ),
                    hue_selector,
                ]
            )
            geometry_status = _support.mo.md(
                "\\n".join(
                    [
                        (
                            f"- Layout: `{selected_layout.get('label')}`"
                            if selected_layout is not None
                            else "- Layout: single view"
                        ),
                        (
                            f"- View id: `{selected_geometry.get('view_id')}`"
                            if selected_geometry is not None
                            else "- View id: unavailable"
                        ),
                        (
                            f"- Projection ids: {', '.join(f'`{item}`' for item in projection_ids)}"
                            if projection_ids
                            else "- Projection ids: none yet"
                        ),
                        (
                            f"- Materialized view: `{selected_geometry.get('materialized')}`"
                            if selected_geometry is not None
                            else "- Materialized view: unavailable"
                        ),
                        (
                            f"- Hue columns available: {', '.join(f'`{item}`' for item in hue_columns) or 'none'}"
                            if selected_geometry is not None
                            else "- Hue columns available: none"
                        ),
                        (
                            f"- Panel count: `{len(panel_specs)}`"
                            if panel_specs
                            else "- Panel count: `0`"
                        ),
                    ]
                )
            )
            geometry_panel = _support.mo.vstack(
                [
                    geometry_controls,
                    context_audit_md,
                    geometry_status,
                    geometry_plot,
                ]
            )
            return (geometry_panel,)
        """
    )


def render_compare_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(compare_left_selector, compare_right_selector, runtime):
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
            compare_summary_table = _support.mo.ui.table(
                _support.pd.DataFrame(
                    [
                        {
                            "basis": compare_payload.get("basis"),
                            "rows": compare_payload.get("rows"),
                            "left_dims": compare_payload.get("left_dims"),
                            "right_dims": compare_payload.get("right_dims"),
                            "distance_spearman": (
                                compare_payload.get("metrics", {}).get("distance_spearman")
                                if isinstance(compare_payload.get("metrics"), dict)
                                else None
                            ),
                            "linear_cka": (
                                compare_payload.get("metrics", {}).get("linear_cka")
                                if isinstance(compare_payload.get("metrics"), dict)
                                else None
                            ),
                            "mean_knn_overlap": (
                                compare_payload.get("metrics", {}).get("mean_knn_overlap")
                                if isinstance(compare_payload.get("metrics"), dict)
                                else None
                            ),
                            "coordinate_r2_diagnostic": (
                                compare_payload.get("metrics", {}).get("coordinate_r2_diagnostic")
                                if isinstance(compare_payload.get("metrics"), dict)
                                else None
                            ),
                            "median_rowwise_cosine": (
                                compare_payload.get("metrics", {}).get("median_rowwise_cosine")
                                if isinstance(compare_payload.get("metrics"), dict)
                                else None
                            ),
                            "median_rowwise_diff_norm": (
                                compare_payload.get("metrics", {}).get("median_rowwise_diff_norm")
                                if isinstance(compare_payload.get("metrics"), dict)
                                else None
                            ),
                        }
                    ]
                ),
                page_size=1,
                show_download=False,
            )
            distance_correlation_plot = _renderers.render_distance_correlation(
                compare_payload,
                title=f"{selected_compare_left} vs {selected_compare_right}: pairwise distance correlation",
            )
            rowwise_cosine_plot = _renderers.render_rowwise_distribution(
                compare_payload,
                value_key="rowwise_cosine",
                title=f"{selected_compare_left} vs {selected_compare_right}: row-wise cosine",
                xlabel="Row-wise cosine similarity",
            )
            rowwise_diff_plot = _renderers.render_rowwise_distribution(
                compare_payload,
                value_key="rowwise_diff_norm",
                title=f"{selected_compare_left} vs {selected_compare_right}: row-wise L2 difference",
                xlabel="Row-wise L2 norm",
            )
            compare_status = _support.mo.md(
                "\\n".join(
                    [
                        f"- Left geometry: `{selected_compare_left}`",
                        f"- Right geometry: `{selected_compare_right}`",
                        f"- Basis: `{compare_payload.get('basis', 'unavailable')}`",
                        f"- Status: `{compare_payload.get('status', 'missing')}`",
                        (
                            f"- Error: {compare_payload.get('error')}"
                            if compare_payload.get("error")
                            else "- Error: none"
                        ),
                    ]
                )
            )
            compare_panel = _support.mo.vstack(
                [
                    _support.mo.md("## Compare Views"),
                    _support.mo.vstack([compare_left_selector, compare_right_selector]),
                    compare_status,
                    compare_summary_table,
                    distance_correlation_plot,
                    rowwise_cosine_plot,
                    rowwise_diff_plot,
                ]
            )
            return (compare_panel,)
        """
    )


def render_deliverable_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(
            deliverable_selector,
            plot_files,
            plot_manifest,
            plot_render_path,
            plot_selector,
            runtime,
            section_selector,
            selected_deliverable,
            selected_plot_rows,
        ):
            _catalog = runtime.catalog
            _identity = runtime.identity
            _renderers = runtime.renderers
            _support = runtime.support

            navigation = _support.mo.vstack(
                [
                    _support.mo.md("## Navigation"),
                    section_selector,
                    deliverable_selector,
                    (
                        plot_selector
                        if plot_selector is not None
                        else _support.mo.callout("No rendered plots are available yet.", kind="warn")
                    ),
                ]
            )
            main_plot = (
                _renderers.render_plot_asset(plot_render_path)
                if plot_render_path is not None and plot_render_path.is_file()
                else _support.mo.callout(
                    "No persisted plot asset is available for the current selection.",
                    kind="warn",
                )
            )

            docs_blocks = []
            if selected_deliverable is not None:
                for docs_ref in selected_deliverable.get("docs_refs", []):
                    if not isinstance(docs_ref, dict):
                        continue
                    content = _support.read_text(docs_ref.get("path"))
                    if content is None:
                        continue
                    docs_blocks.append(_support.mo.md(content))

            plot_file_lines = [
                f"- `{path.relative_to(_identity.workspace_dir).as_posix()}`"
                for path in plot_files
                if path.is_file()
            ] or ["- No persisted files indexed for the selected plot."]
            exports_for_deliverable = [
                row
                for row in _catalog.exports
                if selected_deliverable is not None
                and any(
                    str(output.get("name") or "").startswith(f"export:{row.get('export_id')}")
                    for output in selected_deliverable.get("outputs", [])
                    if isinstance(output, dict)
                )
            ]

            disclosure_tabs = {
                "What this shows": _support.mo.md(
                    "\\n".join(
                        [
                            (
                                f"### {selected_deliverable.get('title')}"
                                if selected_deliverable is not None
                                else "### No deliverable selected"
                            ),
                            "",
                            (
                                f"- Section: `{selected_deliverable.get('section')}`"
                                if selected_deliverable is not None
                                else ""
                            ),
                            (
                                f"- Question: {selected_deliverable.get('question')}"
                                if selected_deliverable is not None
                                else ""
                            ),
                            (
                                f"- Summary: {selected_deliverable.get('summary')}"
                                if selected_deliverable is not None
                                else ""
                            ),
                        ]
                    ).strip()
                ),
                "How to read it": _support.mo.vstack(
                    docs_blocks
                    or [
                        _support.mo.callout(
                            "No study docs are linked for the selected deliverable.",
                            kind="info",
                        )
                    ]
                ),
                "Provenance": _support.mo.md(
                    f"```json\\n{_support.json.dumps(plot_manifest.get('inputs', []), indent=2, sort_keys=True)}\\n```"
                ),
                "Inputs and Upstream Artifacts": _support.mo.md(
                    "\\n".join(
                        [
                            f"- `{entry.get('kind')}:{entry.get('id')}`"
                            for entry in plot_manifest.get("inputs", [])
                            if isinstance(entry, dict)
                        ]
                    )
                    or "- No upstream inputs recorded."
                ),
                "Metadata Tables": _support.mo.ui.table(
                    (
                        _support.pd.DataFrame(selected_plot_rows)
                        if selected_plot_rows
                        else _support.pd.DataFrame(columns=["plot_id", "status"])
                    ),
                    page_size=min(max(len(selected_plot_rows), 1), 10),
                    show_download=False,
                ),
                "Export Files": _support.mo.vstack(
                    [
                        _support.mo.md("\\n".join(plot_file_lines)),
                        _support.mo.ui.table(
                            _support.pd.DataFrame(exports_for_deliverable),
                            page_size=min(max(len(exports_for_deliverable), 1), 10),
                            show_download=False,
                        ),
                    ]
                ),
                "Notebook Control Plane": _support.mo.md(
                    f"```json\\n{_support.json.dumps(_catalog.controls, indent=2, sort_keys=True)}\\n```"
                ),
                "Manifest and QA Details": _support.mo.vstack(
                    [
                        _support.mo.md(
                            "```json\\n"
                            f"{_support.json.dumps(
                                selected_deliverable.get('acceptance_checks', [])
                                if selected_deliverable is not None
                                else [],
                                indent=2,
                                sort_keys=True,
                            )}"
                            "\\n```"
                        ),
                        _support.mo.md(
                            f"```json\\n{_support.json.dumps(plot_manifest, indent=2, sort_keys=True)}\\n```"
                        ),
                        _support.mo.md(
                            f"```json\\n{_support.json.dumps(_catalog.health, indent=2, sort_keys=True)}\\n```"
                        ),
                    ]
                ),
            }

            deliverable_panel = _support.mo.vstack(
                [
                    navigation,
                    _support.mo.md("## Main Plot"),
                    main_plot,
                    _support.mo.md("## Details"),
                    _support.mo.ui.tabs(disclosure_tabs),
                ]
            )
            return (deliverable_panel,)
        """
    )


def render_inventory_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(runtime):
            _catalog = runtime.catalog
            _support = runtime.support

            inventory = _support.mo.ui.tabs(
                {
                    "Plots": _support.mo.ui.table(
                        _support.pd.DataFrame(_catalog.plots),
                        page_size=min(max(len(_catalog.plots), 1), 10),
                        show_download=False,
                    ),
                    "Exports": _support.mo.ui.table(
                        _support.pd.DataFrame(_catalog.exports),
                        page_size=min(max(len(_catalog.exports), 1), 10),
                        show_download=False,
                    ),
                    "Notebooks": _support.mo.ui.table(
                        _support.pd.DataFrame(_catalog.notebooks),
                        page_size=min(max(len(_catalog.notebooks), 1), 10),
                        show_download=False,
                    ),
                    "Runs": _support.mo.ui.table(
                        _support.pd.DataFrame(_catalog.runs),
                        page_size=min(max(len(_catalog.runs), 1), 10),
                        show_download=False,
                    ),
                }
            )
            return (inventory,)
        """
    )


def render_page_tabs_cell() -> str:
    return dedent(
        """\
        @app.cell
        def _(compare_panel, deliverable_panel, geometry_panel, inventory, overview_panel, runtime):
            _support = runtime.support

            page_tabs = _support.mo.ui.tabs(
                {
                    "Overview": overview_panel,
                    "Geometry": geometry_panel,
                    "Compare": compare_panel,
                    "Deliverables": deliverable_panel,
                    "Catalog": inventory,
                }
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

"""Plot-review scaffold cells for the LatentDNA browser notebook."""

from __future__ import annotations

from textwrap import dedent


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
                _plot_id_options = _support.labeled_options(plot_option_pairs)
                plot_options = {label: label for label in _plot_id_options}

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
                        _support.option_key_for_value(_plot_id_options, default_plot_id)
                        or next(iter(plot_options))
                    ),
                    label="Plot",
                    full_width=True,
                )
            return (plot_review_cards, plot_selector)


        @app.cell
        def _(plot_review_cards, plot_selector, runtime):
            _support = runtime.support
            selected_plot_card = None
            if plot_review_cards:
                _active_plot_value = (
                    str(plot_selector.value)
                    if plot_selector is not None
                    else str(plot_review_cards[0]["plot_id"])
                )
                selected_plot_card = _support.resolve_labeled_option_card(
                    plot_review_cards,
                    _active_plot_value,
                )
            return (selected_plot_card,)


        @app.cell
        def _(runtime):
            _geometry = runtime.geometry
            _support = runtime.support

            _reference_values = set((_geometry.reference_annotation_options or {"Off": ""}).values())
            _default_reference = (
                _geometry.reference_annotation_default
                if _geometry.reference_annotation_default in _reference_values
                else ""
            )
            get_requested_plot_reference, set_requested_plot_reference = _support.mo.state(_default_reference)
            return (get_requested_plot_reference, set_requested_plot_reference)


        @app.cell
        def _(runtime):
            _geometry = runtime.geometry
            _support = runtime.support

            _default_reference_hue = ""
            get_requested_plot_reference_hue, set_requested_plot_reference_hue = _support.mo.state(
                _default_reference_hue
            )
            return (get_requested_plot_reference_hue, set_requested_plot_reference_hue)


        @app.cell
        def _(runtime):
            _support = runtime.support

            _annotation_modes = _support.reference_annotation_mode_options()
            _default_annotation_mode = "auto"
            get_requested_plot_reference_annotation_mode, set_requested_plot_reference_annotation_mode = (
                _support.mo.state(_default_annotation_mode)
            )
            return (get_requested_plot_reference_annotation_mode, set_requested_plot_reference_annotation_mode)


        @app.cell
        def _(runtime):
            _support = runtime.support

            get_requested_plot_hue, set_requested_plot_hue = _support.mo.state("")
            return (get_requested_plot_hue, set_requested_plot_hue)


        @app.cell
        def _(
            get_requested_plot_hue,
            get_requested_plot_reference,
            get_requested_plot_reference_annotation_mode,
            runtime,
            selected_plot_card,
            set_requested_plot_hue,
            set_requested_plot_reference,
            set_requested_plot_reference_annotation_mode,
            surface_selector,
        ):
            _renderers = runtime.renderers
            _support = runtime.support

            active_plot_frames = []
            available_plot_hues = []
            plot_filter_selector = None
            plot_hue_selector = None
            plot_reference_annotation_selector = None
            plot_reference_selector = None
            if (
                str(surface_selector.value) == "plots"
                and selected_plot_card is not None
                and bool(selected_plot_card.get("live_render"))
            ):
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
                    _requested_hue = str(get_requested_plot_hue() or "")
                    _selected_hue = _requested_hue if _requested_hue in set(_hue_options.values()) else _default_hue
                    plot_hue_selector = _support.mo.ui.dropdown(
                        options=_hue_options,
                        value=(
                            _support.option_key_for_value(_hue_options, _selected_hue)
                            or next(iter(_hue_options))
                        ),
                            label="Population hue",
                        on_change=set_requested_plot_hue,
                    )
                _filter_options = [
                    _option
                    for _option in _plot_spec.get("filter_options", [])
                    if isinstance(_option, dict) and _option.get("column")
                ]
                if _filter_options:
                    _filter_option = dict(_filter_options[0])
                    _filter_column = str(_filter_option.get("column"))
                    _configured_values = [
                        dict(_value)
                        for _value in _filter_option.get("values", [])
                        if isinstance(_value, dict) and str(_value.get("value") or "").strip()
                    ]
                    if _configured_values:
                        _value_options = {
                            str(_value.get("label") or _value.get("value")): str(_value.get("value"))
                            for _value in _configured_values
                        }
                    else:
                        _observed_values = _support.unique_in_order(
                            str(_value)
                            for _frame in active_plot_frames
                            if _filter_column in _frame.columns
                            for _value in _frame[_filter_column].dropna().astype(str).tolist()
                            if str(_value).strip()
                        )
                        _value_options = {
                            _support.display_hue_label(_value): _value
                            for _value in _observed_values
                        }
                    if bool(_filter_option.get("include_all", True)):
                        _value_options = {"All": "", **_value_options}
                    if _value_options:
                        plot_filter_selector = _support.mo.ui.dropdown(
                            options=_value_options,
                            value=next(iter(_value_options)),
                            label=str(_filter_option.get("label") or "Subset"),
                        )
                _reference_enabled_kinds = {
                    "projection_grid",
                    "projection_scatter",
                    "xy_scatter_grid",
                    "paired_xy_scatter_grid",
                }
                if str(_plot_spec.get("kind") or "") in _reference_enabled_kinds:
                    _reference_options = runtime.geometry.reference_annotation_options or {"Off": ""}
                    _reference_values = set(_reference_options.values())
                    _requested_reference = str(get_requested_plot_reference() or "")
                    _selected_reference = (
                        _requested_reference
                        if _requested_reference in _reference_values
                        else runtime.geometry.reference_annotation_default
                        if runtime.geometry.reference_annotation_default in _reference_values
                        else ""
                    )
                    plot_reference_selector = _support.mo.ui.dropdown(
                        options=_reference_options,
                        value=(
                            _support.option_key_for_value(
                                _reference_options,
                                _selected_reference,
                            )
                            or next(iter(_reference_options))
                        ),
                        label="Reference set",
                        on_change=set_requested_plot_reference,
                    )
                    _annotation_mode_options = _support.reference_annotation_mode_options()
                    _annotation_mode_values = set(_annotation_mode_options.values())
                    _requested_annotation_mode = str(get_requested_plot_reference_annotation_mode() or "auto")
                    _selected_annotation_mode = (
                        _requested_annotation_mode if _requested_annotation_mode in _annotation_mode_values else "auto"
                    )
                    plot_reference_annotation_selector = _support.mo.ui.dropdown(
                        options=_annotation_mode_options,
                        value=(
                            _support.option_key_for_value(
                                _annotation_mode_options,
                                _selected_annotation_mode,
                            )
                            or next(iter(_annotation_mode_options))
                        ),
                        label="Label text",
                        on_change=set_requested_plot_reference_annotation_mode,
                    )
            return (
                active_plot_frames,
                available_plot_hues,
                plot_filter_selector,
                plot_hue_selector,
                plot_reference_annotation_selector,
                plot_reference_selector,
            )


        @app.cell
        def _(
            active_plot_frames,
            get_requested_plot_reference_hue,
            plot_reference_selector,
            runtime,
            selected_plot_card,
            set_requested_plot_reference_hue,
            surface_selector,
        ):
            _support = runtime.support

            plot_reference_hue_selector = None
            if (
                str(surface_selector.value) == "plots"
                and selected_plot_card is not None
                and bool(selected_plot_card.get("live_render"))
                and plot_reference_selector is not None
            ):
                _plot_spec = dict(selected_plot_card.get("plot_spec") or {})
                _reference_enabled_kinds = {
                    "projection_grid",
                    "projection_scatter",
                    "xy_scatter_grid",
                    "paired_xy_scatter_grid",
                }
                if str(_plot_spec.get("kind") or "") in _reference_enabled_kinds:
                    _selected_reference = str(plot_reference_selector.value or "")
                    _reference_annotation = _support.resolve_reference_annotation(
                        _selected_reference,
                        active_plot_frames,
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
                    _reference_hue_options = (
                        runtime.geometry.reference_hue_options_by_reference_set.get(_selected_reference)
                        or {"Single-color markers": ""}
                    )
                    _reference_hue_columns = [
                        column for column in _reference_hue_options.values() if str(column).strip()
                    ]
                    _reference_x_column = str(_plot_spec.get("x_column") or "x")
                    _reference_y_column = str(_plot_spec.get("y_column") or "y")
                    _available_reference_hues = set(
                        _support.available_reference_hues_for_frames(
                            active_plot_frames,
                            preferred_hues=_reference_hue_columns,
                            hue_kinds=runtime.geometry.reference_hue_kinds,
                            reference_labels=_reference_labels,
                            reference_match_column=_reference_match_column,
                            axis_styles=runtime.geometry.axis_styles,
                            x_column=_reference_x_column,
                            y_column=_reference_y_column,
                        )
                    )
                    _reference_hue_options = {
                        label: column
                        for label, column in _reference_hue_options.items()
                        if not str(column).strip() or column in _available_reference_hues
                    } or {"Single-color markers": ""}
                    _reference_hue_values = set(_reference_hue_options.values())
                    _requested_reference_hue = str(get_requested_plot_reference_hue() or "")
                    _selected_reference_hue = (
                        _requested_reference_hue if _requested_reference_hue in _reference_hue_values else ""
                    )
                    if len(_reference_hue_options) > 1:
                        plot_reference_hue_selector = _support.mo.ui.dropdown(
                            options=_reference_hue_options,
                            value=(
                                _support.option_key_for_value(
                                    _reference_hue_options,
                                    _selected_reference_hue,
                                )
                                or next(iter(_reference_hue_options))
                            ),
                            label="Reference color",
                            on_change=set_requested_plot_reference_hue,
                        )
            return (plot_reference_hue_selector,)


        @app.cell
        def _(
            available_plot_hues,
            get_requested_plot_hue,
            get_requested_plot_reference,
            get_requested_plot_reference_annotation_mode,
            get_requested_plot_reference_hue,
            plot_filter_selector,
            plot_hue_selector,
            plot_reference_annotation_selector,
            plot_reference_hue_selector,
            plot_reference_selector,
            runtime,
            selected_plot_card,
        ):
            plot_effective_hue = None
            plot_filter = None
            plot_reference_label_limit = None
            plot_selected_reference_set = runtime.geometry.reference_annotation_default
            plot_selected_reference_hue_column = ""
            _support = runtime.support
            if selected_plot_card is not None and bool(selected_plot_card.get("live_render")):
                _active_plot_spec = dict(selected_plot_card.get("plot_spec") or {})
                _default_hue = str(_active_plot_spec.get("default_hue") or "")
                _plot_hue_value = plot_hue_selector.value if plot_hue_selector is not None else None
                _requested_hue = (
                    str(_plot_hue_value)
                    if _plot_hue_value is not None
                    else str(get_requested_plot_hue() or _default_hue)
                    if plot_hue_selector is not None
                    else ""
                )
                plot_effective_hue = _requested_hue if _requested_hue in available_plot_hues else None
                _plot_reference_value = plot_reference_selector.value if plot_reference_selector is not None else None
                _selected_reference = (
                    str(_plot_reference_value)
                    if _plot_reference_value is not None
                    else str(get_requested_plot_reference() or "")
                    if plot_reference_selector is not None
                    else runtime.geometry.reference_annotation_default
                )
                plot_selected_reference_set = _selected_reference
                _plot_reference_hue_value = (
                    plot_reference_hue_selector.value if plot_reference_hue_selector is not None else None
                )
                plot_selected_reference_hue_column = (
                    str(_plot_reference_hue_value)
                    if _plot_reference_hue_value is not None
                    else str(get_requested_plot_reference_hue() or "")
                    if plot_reference_hue_selector is not None
                    else ""
                )
                _plot_reference_annotation_value = (
                    plot_reference_annotation_selector.value
                    if plot_reference_annotation_selector is not None
                    else None
                )
                _selected_reference_annotation_mode = (
                    str(_plot_reference_annotation_value)
                    if _plot_reference_annotation_value is not None
                    else str(get_requested_plot_reference_annotation_mode() or "auto")
                    if plot_reference_annotation_selector is not None
                    else "auto"
                )
                plot_reference_label_limit = _support.reference_label_limit_for_annotation_mode(
                    _selected_reference_annotation_mode
                )
                _filter_options = [
                    _option
                    for _option in _active_plot_spec.get("filter_options", [])
                    if isinstance(_option, dict) and _option.get("column")
                ]
                if plot_filter_selector is not None and _filter_options:
                    _filter_value = str(plot_filter_selector.value)
                    if _filter_value:
                        plot_filter = {
                            "column": str(dict(_filter_options[0]).get("column")),
                            "value": _filter_value,
                        }
            return (
                plot_effective_hue,
                plot_filter,
                plot_reference_label_limit,
                plot_selected_reference_hue_column,
                plot_selected_reference_set,
            )


        @app.cell
        def _(
            active_plot_frames,
            plot_filter_selector,
            plot_filter,
            plot_effective_hue,
            plot_hue_selector,
            plot_reference_annotation_selector,
            plot_reference_hue_selector,
            plot_reference_selector,
            plot_reference_label_limit,
            plot_selected_reference_hue_column,
            plot_selected_reference_set,
            plot_review_cards,
            plot_scope_note,
            plot_selector,
            runtime,
            selected_plot_card,
            surface_selector,
        ):
            _renderers = runtime.renderers
            _support = runtime.support

            if str(surface_selector.value) != "plots":
                plot_review_panel = _support.mo.md("")
            elif not plot_review_cards:
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
                    _plot_spec = {
                        **dict(_active_card.get("plot_spec") or {}),
                        "alt_text": str(_active_card.get("alt_text") or _active_card.get("title") or ""),
                    }
                    _plot_surface = _renderers.render_plot_review_surface(
                        _plot_spec,
                        frames=active_plot_frames,
                        hue_column=plot_effective_hue,
                        filter_spec=plot_filter,
                        reference_labels=runtime.geometry.reference_labels,
                        reference_set_id=plot_selected_reference_set,
                        reference_label_limit=plot_reference_label_limit,
                        reference_hue_column=plot_selected_reference_hue_column or None,
                        reference_hue_kind=runtime.geometry.reference_hue_kinds.get(
                            plot_selected_reference_hue_column or "",
                        ),
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
                    for widget in [
                        plot_selector,
                        plot_filter_selector,
                        plot_hue_selector,
                        plot_reference_selector,
                        plot_reference_annotation_selector,
                        plot_reference_hue_selector,
                    ]
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
                    ("Rationale", "rationale_md"),
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
                    _section_blocks.append(_support.mo.accordion(_accordion_sections))

                plot_review_panel = _support.mo.vstack(_section_blocks, gap=0.4)
            return (plot_review_panel,)
        """
    )

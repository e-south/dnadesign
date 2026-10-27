"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli/notebook_cells_template_gallery.py

Gallery/export marimo notebook cell template segment for DenseGen notebook scaffolding.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

NOTEBOOK_TEMPLATE_CELLS_GALLERY = r"""
@app.cell
def _(Path, config_path, load_current_inventory_strict, plot_inventory_path, resolve_plot_record):
    plot_entries = []
    plot_inventory_load_error = None
    plot_root = plot_inventory_path.parent
    image_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}
    video_suffixes = {".mp4", ".webm", ".ogg"}
    supported_suffixes = image_suffixes | video_suffixes | {".pdf"}
    seen_paths = set()

    try:
        _payload = load_current_inventory_strict(plot_root, config_path=config_path)
    except Exception as exc:
        plot_inventory_load_error = (
            "Plot gallery requires a current `outputs/plots/current_inventory.json` "
            "with the full notebook-visible surface. "
            + str(exc)
            + ". Regenerate plot and notebook artifacts with `uv run dense plot`."
        )
        payload_entries = []
    else:
        payload_entries = list(_payload.get("plots", []))

    for _entry in payload_entries:
        _rel_path = str(_entry.get("path") or "").strip()
        if not _rel_path:
            continue
        _candidate = (plot_root / _rel_path).resolve()
        if not _candidate.exists():
            continue
        _suffix = str(_candidate.suffix).lower()
        if _suffix not in supported_suffixes:
            continue
        _key = str(_candidate)
        if _key in seen_paths:
            continue
        seen_paths.add(_key)
        plot_entries.append(
            resolve_plot_record(
                plot_root=plot_root,
                plot_path=_candidate,
                manifest_entry=_entry,
                source_rank=0,
            )
        )

    def _stem_priority(entry: dict[str, object]) -> tuple[int, int, int, str]:
        _stem = str(getattr(entry["path"], "stem", "")).strip().lower()
        _tail = _stem.rsplit("_", 1)[-1]
        _has_numeric_tail = int("_" in _stem and _tail.isdigit())
        _has_digest_like_token = int("__" in _stem)
        return (_has_numeric_tail, _has_digest_like_token, len(_stem), _stem)

    def _suffix_priority(entry: dict[str, object]) -> tuple[int, str]:
        _suffix = str(getattr(entry["path"], "suffix", "")).lower()
        if _suffix in image_suffixes:
            return (0, _suffix)
        if _suffix == ".pdf":
            return (1, _suffix)
        if _suffix in video_suffixes:
            return (2, _suffix)
        return (3, _suffix)

    def _entry_priority(entry: dict[str, object]) -> tuple[int, tuple[int, str], tuple[int, int, int, str]]:
        _source_rank = int(entry.get("_source_rank", 1))
        return (_source_rank, _suffix_priority(entry), _stem_priority(entry))

    preferred_entries: dict[tuple[str, str, str], dict[str, object]] = {}
    for _entry in plot_entries:
        _key = (
            str(_entry.get("visual_plot_type") or ""),
            str(_entry.get("plan_name") or ""),
            str(_entry.get("input_name") or ""),
        )
        _current = preferred_entries.get(_key)
        if _current is None or _entry_priority(_entry) < _entry_priority(_current):
            preferred_entries[_key] = _entry

    plot_entries = sorted(
        preferred_entries.values(),
        key=lambda entry: (
            str(entry["plan_name"]),
            str(entry.get("visual_plot_type") or ""),
            str(entry["path"]),
        ),
    )
    return image_suffixes, plot_entries, plot_inventory_load_error, video_suffixes


@app.cell
def _(mo, plot_entries, plot_inventory_load_error, require):
    require(plot_inventory_load_error is not None, plot_inventory_load_error or "Plot inventory is invalid.")
    plot_gallery_notice = ""
    if not plot_entries:
        plot_gallery_notice = (
            "No `outputs/plots/current_inventory.json` plots found yet. "
            "Run `uv run dense plot` to generate plot artifacts for this run."
        )
    return plot_gallery_notice


@app.cell
def _(
    HIDDEN_VISUAL_PLOT_TYPES,
    base_plot_id,
    describe_visual_plot_type,
    mo,
    notebook_visible_plot_ids,
    pd,
    plot_entries,
    resolve_plot_availability,
):
    entries_for_gallery = [
        _entry
        for _entry in plot_entries
        if str(_entry.get("visual_plot_type") or "").strip() not in HIDDEN_VISUAL_PLOT_TYPES
    ]

    known_plot_ids = notebook_visible_plot_ids()
    generated_plot_ids = sorted(
        {
            str(_entry.get("visual_plot_type") or "").strip()
            for _entry in entries_for_gallery
            if str(_entry.get("visual_plot_type") or "").strip()
        }
    )

    plot_id_label_to_id = {}
    _generated_counts = {}
    _generated_base_counts = {}
    for _entry in entries_for_gallery:
        _plot_id = str(_entry.get("visual_plot_type") or "").strip()
        if not _plot_id:
            continue
        _generated_counts[_plot_id] = int(_generated_counts.get(_plot_id, 0)) + 1
        _generated_base = base_plot_id(_plot_id)
        if _generated_base:
            _generated_base_counts[_generated_base] = int(_generated_base_counts.get(_generated_base, 0)) + 1

    for _plot_id in known_plot_ids:
        _count = int(_generated_counts.get(_plot_id, _generated_base_counts.get(_plot_id, 0)))
        if _count <= 0:
            continue
        _status = resolve_plot_availability(
            _plot_id,
            generated_plot_ids=generated_plot_ids,
        )
        _label = f"{describe_visual_plot_type(_plot_id)} [{_status}]"
        plot_id_label_to_id[_label] = _plot_id

    plot_id_options = list(plot_id_label_to_id.keys())
    if not plot_id_options:
        plot_id_options = ["(no plot types)"]
    if "(no plot types)" in plot_id_options:
        plot_id_label_to_id["(no plot types)"] = "(no plot types)"

    plot_availability_rows = []
    for _plot_id in known_plot_ids:
        _count = int(_generated_counts.get(_plot_id, _generated_base_counts.get(_plot_id, 0)))
        plot_availability_rows.append(
            {
                "Plot type": _plot_id,
                "Status": resolve_plot_availability(
                    _plot_id,
                    generated_plot_ids=generated_plot_ids,
                ),
                "Generated files": _count,
            }
        )
    plot_availability_table = pd.DataFrame(
        plot_availability_rows,
        columns=["Plot type", "Status", "Generated files"],
    )

    plot_id_filter = mo.ui.dropdown(options=plot_id_options, value=plot_id_options[0], label="Plot type")
    return (
        base_plot_id,
        entries_for_gallery,
        generated_plot_ids,
        plot_availability_table,
        plot_id_label_to_id,
        plot_id_filter,
    )


@app.cell
def _(
    base_plot_id,
    entries_for_gallery,
    generated_plot_ids,
    plot_availability_table,
    plot_id_label_to_id,
    compact_plan_label,
    plot_id_filter,
    plot_missing_hint,
    plot_required_artifacts,
    resolve_plot_availability,
):
    selected_plot_label = str(plot_id_filter.value or "")
    selected_plot_id = str(plot_id_label_to_id.get(selected_plot_label, selected_plot_label))

    def _entry_matches_selected_plot_id(_entry: dict[str, object]) -> bool:
        _visual_plot_type = str(_entry.get("visual_plot_type") or "").strip()
        if not _visual_plot_type:
            return False
        if _visual_plot_type == selected_plot_id:
            return True
        return base_plot_id(_visual_plot_type) == selected_plot_id

    _filtered_entries = [
        _entry
        for _entry in entries_for_gallery
        if _entry_matches_selected_plot_id(_entry)
    ]

    label_to_entry = {}
    plot_filter_message = ""
    plot_options = []
    if selected_plot_id == "(no plot types)":
        plot_filter_message = "No plot types are available. Run `uv run dense plot` to generate plot artifacts."
        plot_options = ["(no plots for current filters)"]
    elif not _filtered_entries:
        _generated_plot_ids = set(generated_plot_ids)
        _generated_base_plot_ids = {
            base_plot_id(_plot_id)
            for _plot_id in _generated_plot_ids
            if str(_plot_id).strip()
        }
        _generated_base_plot_ids = {
            _plot_id for _plot_id in _generated_base_plot_ids if str(_plot_id).strip()
        }
        if (
            selected_plot_id
            and selected_plot_id not in _generated_plot_ids
            and selected_plot_id not in _generated_base_plot_ids
        ):
            _base_id = base_plot_id(selected_plot_id)
            _generation_hint = _base_id if _base_id else selected_plot_id
            _availability = resolve_plot_availability(
                selected_plot_id,
                generated_plot_ids=_generated_plot_ids,
            )
            _required_artifacts = [
                str(_item).strip()
                for _item in plot_required_artifacts(selected_plot_id)
                if str(_item).strip()
            ]
            _missing_hint = str(plot_missing_hint(selected_plot_id) or "").strip()
            plot_filter_message = (
                "No generated plots for plot type `"
                + selected_plot_id
                + "`. Run `uv run dense plot --only "
                + _generation_hint
                + "` to generate it."
            )
            if _availability != "generated":
                plot_filter_message += " Availability: `" + _availability + "`."
            if _required_artifacts:
                plot_filter_message += " Required artifacts: " + ", ".join(
                    f"`{_item}`" for _item in _required_artifacts
                ) + "."
            if _missing_hint:
                plot_filter_message += " " + _missing_hint
        else:
            plot_filter_message = (
                "No plots found for plot type `"
                + selected_plot_id
                + "`. Select another plot type."
            )
        plot_options = ["(no plots for current filters)"]
    else:
        for _entry_index, _entry in enumerate(_filtered_entries):
            _plan = str(_entry["plan_name"])
            compact_plan_name = compact_plan_label(_plan)
            _label = str(_entry.get("title") or _entry["plot_name"])
            _option_label = f"{_entry_index + 1}. [{compact_plan_name}] {_label}"
            plot_options.append(_option_label)
            label_to_entry[_option_label] = _entry

    plot_selector = mo.ui.dropdown(options=plot_options, value=plot_options[0], label="Artifact")
    return (
        label_to_entry,
        plot_filter_message,
        plot_availability_table,
        plot_id_filter,
        plot_selector,
    )


@app.cell
def _(label_to_entry, plot_filter_message, plot_selector):
    _selected_plot_option = str(plot_selector.value or "")
    active_plot_error = str(plot_filter_message or "").strip()
    active_plot_entry = None
    if not active_plot_error and _selected_plot_option not in label_to_entry:
        active_plot_error = "Selected plot is not available for the current plot filter."
    if not active_plot_error and _selected_plot_option in label_to_entry:
        active_plot_entry = label_to_entry[_selected_plot_option]
    return active_plot_entry, active_plot_error


@app.cell
def _(
    active_plot_entry,
    active_plot_error,
    image_suffixes,
    label_to_entry,
    mo,
    plot_availability_table,
    plot_id_label_to_id,
    plot_id_filter,
    plot_gallery_notice,
    plot_missing_hint,
    resolve_plot_display_media,
    plot_required_artifacts,
    plot_selector,
    video_suffixes,
):
    _selected_plot_type_label = str(plot_id_filter.value or "")
    _selected_plot_type = str(plot_id_label_to_id.get(_selected_plot_type_label, _selected_plot_type_label))
    _matching_plots = int(len(label_to_entry))
    _filters_summary = mo.md(
        " | ".join(
            [
                f"Plot type: `{_selected_plot_type_label}`",
                f"Matching plots: `{_matching_plots}`",
            ]
        )
    )
    gallery_metadata = mo.accordion(
        {
            "Plot availability": mo.ui.table(plot_availability_table),
        },
        multiple=True,
    )
    _controls = mo.hstack(
        [plot_id_filter, plot_selector],
        justify="start",
        align="center",
        wrap=True,
        gap=0.3,
        widths=[1.8, 8.2],
    )
    _content = [mo.md("### Plot gallery"), _filters_summary, gallery_metadata]
    if str(plot_gallery_notice).strip():
        _content.append(mo.md(str(plot_gallery_notice)))
    _content.append(_controls)
    if active_plot_entry is None:
        _content.append(mo.md(str(active_plot_error or "No plot selected.")))
    else:
        _plan_name = str(active_plot_entry["plan_name"])
        _plot_id = str(active_plot_entry["plot_id"])
        _plot_name = str(active_plot_entry["plot_name"])
        _variant = str(active_plot_entry["variant"]).strip()
        _title = str(active_plot_entry.get("title") or _plot_name).strip() or _plot_name
        _caption = str(active_plot_entry.get("caption") or active_plot_entry.get("description") or "").strip()
        _alt_text = str(active_plot_entry.get("alt_text") or _caption or _title).strip() or _title
        _plot_path = active_plot_entry["path"]
        _plot_suffix = str(getattr(_plot_path, "suffix", "")).lower()
        _display_kind, _display_payload = resolve_plot_display_media(_plot_path)
        _variant_text = _variant if _variant else "none"
        _required_artifacts = [
            str(_item).strip() for _item in plot_required_artifacts(_plot_id) if str(_item).strip()
        ]
        _required_artifacts_text = ", ".join(f"`{_item}`" for _item in _required_artifacts) or "none"
        _missing_hint = str(plot_missing_hint(_plot_id) or "").strip()
        _preview_mode = {
            "image_path": "source image",
            "image_bytes": "inline PNG preview from source PDF",
            "video_path": "source video",
            "pdf_path": "embedded source PDF",
        }.get(str(_display_kind), "unavailable")
        _content.append(
            mo.accordion(
                {
                    "Selected plot metadata": mo.md(
                        "\n".join(
                            [
                                f"- Plan scope: `{_plan_name}`",
                                f"- Plot id: `{_plot_id or 'n/a'}`",
                                f"- Plot title: `{_title}`",
                                f"- Variant: `{_variant_text}`",
                                "- Availability: `generated`",
                                f"- Preview mode: `{_preview_mode}`",
                                f"- Required artifacts: {_required_artifacts_text}",
                                *([f"- Supporting caption: {_caption}"] if _caption else []),
                                *([f"- Alt text: {_alt_text}"] if _alt_text else []),
                                f"- File: `{str(_plot_path)}`",
                                *([f"- Contract hint: {_missing_hint}"] if _missing_hint else []),
                            ]
                        )
                    )
                },
                multiple=True,
            )
        )
        _content.append(mo.md(f"#### {_title}"))
        if _display_kind == "video_path":
            _content.append(
                mo.video(
                    _plot_path.read_bytes(),
                    controls=True,
                    width="100%",
                    rounded=True,
                )
            )
        elif _display_kind in {"image_path", "image_bytes"}:
            _content.append(
                mo.image(
                    _display_payload,
                    alt=_alt_text,
                    caption=_caption or None,
                    rounded=True,
                    style={
                        "border-radius": "14px",
                        "width": "100%",
                        "max-width": "860px",
                        "max-height": "560px",
                        "height": "auto",
                        "object-fit": "contain",
                        "margin": "0 auto",
                        "display": "block",
                    },
                )
            )
        elif _display_kind == "pdf_path" or _plot_suffix == ".pdf":
            _content.append(mo.pdf(_plot_path))
            if _caption:
                _content.append(
                    mo.md(
                        "<div style='max-width:860px;margin:0 auto;color:#4b5563;font-size:0.96rem;'>"
                        + _caption
                        + "</div>"
                    )
                )
        else:
            _content.append(mo.md("Preview unavailable for this plot type."))
    mo.vstack(_content)
    return


@app.cell
def _(Path, shutil, subprocess, tempfile):
    def _convert_plot_artifact(source_path: Path, destination_path: Path, fmt: str) -> None:
        source_path = Path(source_path).expanduser().resolve()
        destination_path = Path(destination_path).expanduser().resolve()
        source_suffix = str(source_path.suffix).lower()
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "artifact":
            shutil.copy2(source_path, destination_path)
            return

        if source_suffix == f".{fmt}":
            shutil.copy2(source_path, destination_path)
            return

        if fmt == "png":
            if source_suffix in {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}:
                shutil.copy2(source_path, destination_path)
                return
            if source_suffix == ".pdf":
                ghostscript = shutil.which("gs")
                pdftoppm = shutil.which("pdftoppm")
                magick = shutil.which("magick")
                convert = shutil.which("convert")
                sips = shutil.which("sips")
                output_root = destination_path.with_suffix("")
                commands = []
                if ghostscript:
                    commands.append(
                        [
                            ghostscript,
                            "-dSAFER",
                            "-dBATCH",
                            "-dNOPAUSE",
                            "-sDEVICE=pngalpha",
                            "-r450",
                            "-dFirstPage=1",
                            "-dLastPage=1",
                            f"-sOutputFile={destination_path}",
                            str(source_path),
                        ]
                    )
                if pdftoppm:
                    commands.append(
                        [
                            pdftoppm,
                            "-png",
                            "-singlefile",
                            "-r",
                            "450",
                            str(source_path),
                            str(output_root),
                        ]
                    )
                if magick:
                    commands.append(
                        [magick, "-density", "450", f"{source_path}[0]", "-quality", "100", str(destination_path)]
                    )
                if convert:
                    commands.append(
                        [convert, "-density", "450", f"{source_path}[0]", "-quality", "100", str(destination_path)]
                    )
                if sips:
                    commands.append([sips, "-s", "format", "png", str(source_path), "--out", str(destination_path)])
                for command in commands:
                    try:
                        subprocess.run(command, check=True, capture_output=True)
                    except Exception:
                        continue
                    generated_png = output_root.with_suffix(".png")
                    if generated_png.exists() and generated_png != destination_path:
                        shutil.move(str(generated_png), str(destination_path))
                    if destination_path.exists() and destination_path.stat().st_size > 0:
                        return
                raise RuntimeError(
                    "PNG export from PDF requires `gs`, `pdftoppm`, `magick`, `convert`, or `sips`."
                )
            raise RuntimeError(f"Cannot export `{source_path.name}` to PNG.")

        if fmt == "svg":
            if source_suffix == ".svg":
                shutil.copy2(source_path, destination_path)
                return
            if source_suffix == ".pdf":
                pdftocairo = shutil.which("pdftocairo")
                if not pdftocairo:
                    raise RuntimeError("SVG export from PDF requires `pdftocairo` in PATH.")
                output_root = destination_path.with_suffix("")
                subprocess.run(
                    [pdftocairo, "-svg", str(source_path), str(output_root)],
                    check=True,
                    capture_output=True,
                )
                generated_svg = output_root.with_suffix(".svg")
                if generated_svg.exists() and generated_svg != destination_path:
                    shutil.move(str(generated_svg), str(destination_path))
                if destination_path.exists() and destination_path.stat().st_size > 0:
                    return
                raise RuntimeError(f"Failed to export SVG for `{source_path.name}`.")
            raise RuntimeError(f"Cannot export `{source_path.name}` to SVG.")

        if fmt == "pdf":
            if source_suffix == ".pdf":
                shutil.copy2(source_path, destination_path)
                return
            magick = shutil.which("magick")
            convert = shutil.which("convert")
            commands = []
            if magick:
                commands.append([magick, str(source_path), str(destination_path)])
            if convert:
                commands.append([convert, str(source_path), str(destination_path)])
            for command in commands:
                try:
                    subprocess.run(command, check=True, capture_output=True)
                except Exception:
                    continue
                if destination_path.exists() and destination_path.stat().st_size > 0:
                    return
            raise RuntimeError(
                f"Cannot export `{source_path.name}` to PDF. Install `magick` or `convert` to enable this conversion."
            )

        raise RuntimeError(f"Unsupported plot export format `{fmt}`.")

    def export_plot_artifact(source_path: Path, destination_path: Path, fmt: str) -> None:
        _convert_plot_artifact(source_path, destination_path, fmt)

    def resolve_plot_display_media(source_path: Path) -> tuple[str, object]:
        source_path = Path(source_path).expanduser().resolve()
        source_suffix = str(source_path.suffix).lower()
        if source_suffix in {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}:
            return ("image_path", source_path)
        if source_suffix in {".mp4", ".webm", ".ogg"}:
            return ("video_path", source_path)
        if source_suffix == ".pdf":
            sibling_png = source_path.with_suffix(".png")
            if sibling_png.exists() and sibling_png.stat().st_size > 0:
                return ("image_path", sibling_png)
            try:
                with tempfile.TemporaryDirectory(prefix="densegen_plot_preview_") as tmpdir:
                    preview_path = Path(tmpdir) / f"{source_path.stem}.png"
                    _convert_plot_artifact(source_path, preview_path, "png")
                    if preview_path.exists() and preview_path.stat().st_size > 0:
                        return ("image_bytes", preview_path.read_bytes())
            except Exception:
                return ("pdf_path", source_path)
            return ("pdf_path", source_path)
        return ("unknown", source_path)

    return export_plot_artifact, resolve_plot_display_media


@app.cell
def _(mo, run_root, to_repo_relative_path):
    plot_export_target = mo.ui.dropdown(
        options=["selected", "filtered", "all"],
        value="selected",
        label="",
    )
    plot_export_format = mo.ui.dropdown(options=["artifact", "png", "pdf", "svg"], value="png", label="")
    default_plot_export_dir = run_root / "outputs" / "notebooks" / "plots_export"
    default_plot_export_dir_text = to_repo_relative_path(default_plot_export_dir)
    plot_export_path = mo.ui.text(
        value=str(default_plot_export_dir_text),
        label="Plot export directory",
        full_width=True,
    )
    plot_export_button = mo.ui.run_button(label="Export", kind="neutral")
    plot_export_details = mo.accordion(
        {
            "Export behavior": mo.md(
                        "\n".join(
                            [
                                (
                                    "Export selected, filtered, or all plots into one output format. "
                                    "selected = currently visible plot, filtered = every plot "
                                    "matching the current plot-type filter, all = all plots in this run."
                                ),
                                "- Format `artifact`: copy each plot in its generated source format.",
                                "- Target `selected`: export the plot currently shown in Plot gallery.",
                                "- Target `filtered`: export every plot matching the current plot-type filter.",
                                "- Target `all`: export every plot listed for this run.",
                                "- Path behavior: relative export paths resolve from the repository root.",
                            ]
                )
            )
        },
        multiple=True,
    )
    mo.vstack(
        [
            mo.md("### Plot export"),
            mo.hstack(
                [
                    plot_export_target,
                    plot_export_format,
                    plot_export_path,
                    plot_export_button,
                ],
                justify="start",
                align="end",
                gap=0.2,
                widths=[1.0, 1.0, 8.0, 0.9],
                wrap=False,
            ),
            plot_export_details,
        ],
        align="stretch",
    )
    return plot_export_button, plot_export_format, plot_export_path, plot_export_target


@app.cell
def _(mo):
    get_plot_export_handled_click, set_plot_export_handled_click = mo.state(0)
    get_plot_export_status, set_plot_export_status = mo.state("")
    return (
        get_plot_export_handled_click,
        get_plot_export_status,
        set_plot_export_handled_click,
        set_plot_export_status,
    )


@app.cell
def _(
    Path,
    active_plot_entry,
    consume_click,
    export_plot_artifact,
    get_plot_export_handled_click,
    get_plot_export_status,
    label_to_entry,
    mo,
    plot_entries,
    plot_export_button,
    plot_export_format,
    plot_export_path,
    plot_export_target,
    require,
    repo_root,
    set_plot_export_handled_click,
    set_plot_export_status,
):
    _plot_click_count = int(plot_export_button.value or 0)
    _plot_status_text = str(get_plot_export_status() or "")
    _should_export, _handled_click = consume_click(
        _plot_click_count,
        int(get_plot_export_handled_click() or 0),
    )

    def _slug(value: str) -> str:
        text = str(value or "").strip().replace("__", "_")
        keep = []
        for ch in text:
            if ch.isalnum() or ch in {"-", "_"}:
                keep.append(ch)
            else:
                keep.append("-")
        slug = "".join(keep).strip("-_")
        return slug or "plot"

    if _should_export:
        try:
            _selected_target = str(plot_export_target.value or "selected").strip()
            require(
                _selected_target not in {"selected", "filtered", "all"},
                f"Plot export set must be selected|filtered|all, got `{_selected_target}`.",
            )
            _selected_format = str(plot_export_format.value or "").strip()
            require(
                _selected_format not in {"artifact", "pdf", "png", "svg"},
                f"Plot export format must be artifact|pdf|png|svg, got `{_selected_format}`.",
            )
            _raw_export_dir = str(plot_export_path.value or "").strip()
            require(not _raw_export_dir, "Plot export directory cannot be empty.")
            _export_dir = Path(_raw_export_dir).expanduser()
            if not _export_dir.is_absolute():
                _export_dir = repo_root / _export_dir
            _export_dir.mkdir(parents=True, exist_ok=True)

            if _selected_target == "selected":
                require(active_plot_entry is None, "No selected plot is available to export.")
                _entries = [active_plot_entry]
            elif _selected_target == "filtered":
                _entries = list(label_to_entry.values())
                require(not _entries, "No filtered plots are available to export.")
            else:
                _entries = list(plot_entries)
                require(not _entries, "No plots are available to export.")

            _exported_n = 0
            for _idx, _entry in enumerate(_entries):
                _source_path = Path(_entry["path"]).expanduser().resolve()
                _plan_name = _slug(str(_entry.get("plan_name") or "run"))
                _plot_name = _slug(str(_entry.get("plot_id") or _entry.get("plot_name") or _source_path.stem))
                _variant = _slug(str(_entry.get("variant") or "default"))
                _destination_suffix = _source_path.suffix if _selected_format == "artifact" else f".{_selected_format}"
                _destination_path = (
                    _export_dir / f"{_idx + 1:03d}__{_plan_name}__{_plot_name}__{_variant}{_destination_suffix}"
                )
                export_plot_artifact(_source_path, _destination_path, _selected_format)
                _exported_n += 1
            _plot_status_text = "Saved `" + str(_exported_n) + "` plot(s) to `" + str(_export_dir) + "`."
        except Exception as exc:
            _plot_status_text = "Plot export failed: " + str(exc)
        set_plot_export_handled_click(_handled_click)
        set_plot_export_status(_plot_status_text)
    mo.md(_plot_status_text)
    return
"""

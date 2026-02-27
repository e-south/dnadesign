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
def _(Path, json, plot_manifest_path):
    plot_entries = []
    plot_manifest_load_error = None
    plot_root = plot_manifest_path.parent
    image_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}
    supported_suffixes = image_suffixes | {".pdf"}
    seen_paths = set()

    def _infer_plot_id_from_path(relative_parts: tuple[str, ...], stem: str) -> str:
        if not relative_parts:
            return ""
        _head = str(relative_parts[0]).strip().lower()
        _stem = str(stem).strip().lower()
        if _head == "stage_a":
            return "stage_a_summary"
        if _head == "stage_b":
            if "usage" in _stem:
                return "tfbs_usage"
            return "placement_map"
        if _head == "run_health":
            return "run_health"
        return ""

    def _visual_plot_type(plot_id: str, *, plot_name: str, variant: str, stem: str) -> str:
        _base = str(plot_id or "").strip()
        _variant = str(variant or "").strip()
        _plot_name = str(plot_name or "").strip()
        _stem = str(stem or "").strip()
        if _base and _variant and _variant != _base:
            return f"{_base}/{_variant}"
        if _base:
            return _base
        if _variant:
            return _variant
        if _plot_name:
            return _plot_name
        return _stem

    if plot_manifest_path.exists():
        try:
            _payload = json.loads(plot_manifest_path.read_text())
        except Exception as exc:
            plot_manifest_load_error = f"Failed to parse `plot_manifest.json`: {exc}"
        else:
            for _entry in _payload.get("plots", []):
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
                _rel_parts = tuple(str(_part) for _part in Path(_rel_path).parts)
                _plot_id = str(_entry.get("plot_id") or _entry.get("name") or "").strip()
                if not _plot_id:
                    _plot_id = _infer_plot_id_from_path(_rel_parts, _candidate.stem)
                _plan_name = str(_entry.get("plan_name") or "").strip()
                if not _plan_name:
                    if len(_rel_parts) >= 2 and _rel_parts[0] == "stage_b":
                        _plan_name = str(_rel_parts[1]).strip() or "unscoped"
                    elif len(_rel_parts) >= 1 and _rel_parts[0] == "stage_a":
                        _plan_name = "stage_a"
                    else:
                        _plan_name = "unscoped"
                _input_name = str(_entry.get("input_name") or "").strip()
                if not _input_name and len(_rel_parts) >= 4 and _rel_parts[0] == "stage_b":
                    _input_name = str(_rel_parts[2]).strip()
                if not _input_name and len(_rel_parts) >= 2 and _rel_parts[0] == "stage_a":
                    _stem = str(_candidate.stem)
                    if _stem == "background_logo":
                        _input_name = "background"
                    elif _stem.endswith("__background_logo"):
                        _input_name = _stem[: -len("__background_logo")].strip()
                _plot_name = str(_entry.get("name") or _candidate.stem)
                _variant = str(_entry.get("variant") or _candidate.stem or "")
                plot_entries.append(
                    {
                        "path": _candidate,
                        "plot_id": _plot_id,
                        "visual_plot_type": _visual_plot_type(
                            _plot_id,
                            plot_name=_plot_name,
                            variant=_variant,
                            stem=_candidate.stem,
                        ),
                        "plan_name": _plan_name,
                        "input_name": _input_name,
                        "plot_name": _plot_name,
                        "variant": _variant,
                        "description": str(_entry.get("description") or ""),
                        "_source_rank": 0,
                    }
                )

    for _candidate in sorted(plot_root.rglob("*")):
        if not _candidate.is_file():
            continue
        _resolved = _candidate.resolve()
        _suffix = str(_resolved.suffix).lower()
        if _suffix not in supported_suffixes:
            continue
        _key = str(_resolved)
        if _key in seen_paths:
            continue
        seen_paths.add(_key)
        _relative_parts = tuple(str(_part) for _part in _resolved.relative_to(plot_root.resolve()).parts)
        if any(str(_part).startswith(".") for _part in _relative_parts):
            continue
        _plan_name = "unscoped"
        _input_name = ""
        if len(_relative_parts) >= 2 and _relative_parts[0] == "stage_b":
            _plan_name = str(_relative_parts[1])
            if len(_relative_parts) >= 3:
                _input_name = str(_relative_parts[2]).strip()
        elif len(_relative_parts) >= 1 and _relative_parts[0] == "stage_a":
            _plan_name = "stage_a"
            _stem = str(_resolved.stem)
            if _stem == "background_logo":
                _input_name = "background"
            elif _stem.endswith("__background_logo"):
                _input_name = _stem[: -len("__background_logo")].strip()
        _inferred_plot_id = _infer_plot_id_from_path(_relative_parts, _resolved.stem)
        _plot_name = str(_resolved.stem)
        _variant = str(_resolved.stem)
        plot_entries.append(
            {
                "path": _resolved,
                "plot_id": _inferred_plot_id,
                "visual_plot_type": _visual_plot_type(
                    _inferred_plot_id,
                    plot_name=_plot_name,
                    variant=_variant,
                    stem=_resolved.stem,
                ),
                "plan_name": _plan_name,
                "input_name": _input_name,
                "plot_name": _plot_name,
                "variant": _variant,
                "description": "",
                "_source_rank": 1,
            }
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
        return (2, _suffix)

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
    return plot_entries, plot_manifest_load_error


@app.cell
def _(PLOT_SPECS, mo, plot_entries, plot_manifest_load_error, require):
    require(plot_manifest_load_error is not None, plot_manifest_load_error or "Plot manifest is invalid.")
    plot_gallery_notice = ""
    if not plot_entries:
        plot_gallery_notice = (
            "No `outputs/plots/plot_manifest.json` plots found yet. "
            "Run `uv run dense plot` to generate plot artifacts for this run."
        )
    available_plot_names = sorted(list(PLOT_SPECS.keys()))
    generated_plot_names = sorted(
        {
            str(entry.get("plot_id") or "").strip()
            for entry in plot_entries
            if str(entry.get("plot_id") or "").strip()
        }
    )
    missing_plot_names = [name for name in available_plot_names if name not in generated_plot_names]
    if missing_plot_names:
        _joined = ",".join(missing_plot_names)
        mo.md(
            "Available but not generated: "
            + ", ".join(f"`{name}`" for name in missing_plot_names)
            + f". Run `uv run dense plot --only {_joined}` to generate them."
        )

    def compact_plan_label(plan_name: str) -> str:
        _plan_text = str(plan_name or "").strip()
        if not _plan_text:
            return "run-level"
        if _plan_text == "unscoped":
            return "run-level"
        if _plan_text == "stage_a":
            return "stage-a"
        _parts = [part for part in _plan_text.split("__") if part]
        _base_label = str(_parts[0] if _parts else _plan_text).strip().replace("__", "-")
        _variant_tokens = []
        for _token in _parts[1:]:
            _token = str(_token).strip()
            if not _token:
                continue
            if "=" in _token:
                _key, _value = _token.split("=", 1)
            elif "_" in _token:
                _key, _value = _token.split("_", 1)
            else:
                _key, _value = _token, ""
            _key = str(_key).strip()
            _value = str(_value).strip()
            if _key and _value:
                _variant_tokens.append(f"{_key}={_value}")
        if not _variant_tokens:
            return _base_label
        _variant_label = " ".join(_variant_tokens)
        return f"{_base_label} [{_variant_label}]"

    _plan_names = sorted({str(entry["plan_name"]) for entry in plot_entries})
    all_scope_label = "all scopes (all plots)"
    plan_label_to_name = {all_scope_label: "all"}
    for _plan_name in _plan_names:
        _label_candidate = compact_plan_label(_plan_name)
        if _label_candidate in plan_label_to_name and plan_label_to_name[_label_candidate] != _plan_name:
            _label_candidate = f"{_label_candidate} [{_plan_name}]"
        plan_label_to_name[_label_candidate] = _plan_name

    plan_options = list(plan_label_to_name.keys())
    plot_scope_filter = mo.ui.dropdown(options=plan_options, value=plan_options[0], label="")
    return all_scope_label, compact_plan_label, plan_label_to_name, plot_gallery_notice, plot_scope_filter


@app.cell
def _(PLOT_SPECS, all_scope_label, mo, pd, plan_label_to_name, plot_entries, plot_scope_filter):
    selected_scope_label = str(plot_scope_filter.value or all_scope_label)
    selected_plot_scope = str(plan_label_to_name.get(selected_scope_label, "all"))
    hidden_plot_types = {"run_health/summary_table"}
    entries_for_scope = list(plot_entries)
    if selected_plot_scope != "all":
        entries_for_scope = [_entry for _entry in plot_entries if str(_entry["plan_name"]) == selected_plot_scope]
    entries_for_scope = [
        _entry
        for _entry in entries_for_scope
        if str(_entry.get("visual_plot_type") or "").strip() not in hidden_plot_types
    ]

    known_plot_ids = sorted([str(_name) for _name in PLOT_SPECS.keys()])

    def base_plot_id(plot_type: str) -> str:
        _token = str(plot_type or "").strip()
        if "/" in _token:
            return str(_token.split("/", 1)[0]).strip()
        return _token

    def _ordered_unique(values: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for value in values:
            token = str(value).strip()
            if not token or token in seen:
                continue
            ordered.append(token)
            seen.add(token)
        return ordered

    plot_ids_by_scope = {}
    generated_plot_ids_by_scope: dict[str, list[str]] = {}
    _plot_ids_all_generated_raw = []
    for entry in plot_entries:
        _plot_id = str(entry.get("visual_plot_type") or "").strip()
        if not _plot_id:
            continue
        if _plot_id in hidden_plot_types:
            continue
        _plot_ids_all_generated_raw.append(_plot_id)
    _plot_ids_all_generated = sorted(set(_plot_ids_all_generated_raw))
    generated_plot_ids_by_scope["all"] = _plot_ids_all_generated
    plot_ids_by_scope["all"] = _ordered_unique(_plot_ids_all_generated)
    for _plan_name in sorted({str(entry["plan_name"]) for entry in plot_entries}):
        _plot_ids_scope_generated_raw = []
        for entry in plot_entries:
            if str(entry["plan_name"]) != _plan_name:
                continue
            _plot_id = str(entry.get("visual_plot_type") or "").strip()
            if not _plot_id:
                continue
            if _plot_id in hidden_plot_types:
                continue
            _plot_ids_scope_generated_raw.append(_plot_id)
        _plot_ids_scope_generated = sorted(set(_plot_ids_scope_generated_raw))
        generated_plot_ids_by_scope[_plan_name] = _plot_ids_scope_generated
        plot_ids_by_scope[_plan_name] = _ordered_unique(_plot_ids_scope_generated)

    def _format_plot_id_list(values: list[str]) -> str:
        if not values:
            return "`(none)`"
        return ", ".join(f"`{plot_id}`" for plot_id in values)

    _scope_available_ids = list(plot_ids_by_scope.get(selected_plot_scope, []))

    plot_id_label_to_id = {}
    _generated_set = set(generated_plot_ids_by_scope.get(selected_plot_scope, []))
    _generated_set.update(
        base_plot_id(_plot_id) for _plot_id in list(_generated_set) if base_plot_id(_plot_id)
    )
    for _plot_id in _scope_available_ids:
        _status = "generated" if _plot_id in _generated_set else "available"
        _label = f"{_plot_id} [{_status}]"
        plot_id_label_to_id[_label] = _plot_id

    plot_id_options = list(plot_id_label_to_id.keys())
    if selected_plot_scope == "all":
        if not plot_id_options:
            plot_id_options = ["(no plot types)"]
    elif not plot_id_options:
        plot_id_options = ["(no plot types)"]
    if "(no plot types)" in plot_id_options:
        plot_id_label_to_id["(no plot types)"] = "(no plot types)"

    _generated_counts = {}
    for _entry in entries_for_scope:
        _plot_id = str(_entry.get("visual_plot_type") or "").strip()
        if not _plot_id:
            continue
        _generated_counts[_plot_id] = int(_generated_counts.get(_plot_id, 0)) + 1
    plot_availability_rows = []
    for _plot_id in _scope_available_ids:
        _count = int(_generated_counts.get(_plot_id, 0))
        plot_availability_rows.append(
            {
                "Plot type": _plot_id,
                "Status": "generated" if _count > 0 else "available",
                "Generated files": _count,
            }
        )
    plot_availability_table = pd.DataFrame(
        plot_availability_rows,
        columns=["Plot type", "Status", "Generated files"],
    )

    plot_id_filter = mo.ui.dropdown(options=plot_id_options, value=plot_id_options[0], label="")
    return (
        base_plot_id,
        entries_for_scope,
        generated_plot_ids_by_scope,
        plot_availability_table,
        plot_id_label_to_id,
        plot_id_filter,
        plot_scope_filter,
        selected_plot_scope,
        selected_scope_label,
    )


@app.cell
def _(
    base_plot_id,
    entries_for_scope,
    generated_plot_ids_by_scope,
    plot_availability_table,
    plot_id_label_to_id,
    compact_plan_label,
    pd,
    plot_id_filter,
    plot_scope_filter,
    selected_plot_scope,
    selected_scope_label,
):
    selected_plot_label = str(plot_id_filter.value or "")
    selected_plot_id = str(plot_id_label_to_id.get(selected_plot_label, selected_plot_label))

    def _entry_matches_selected_plot_id(_entry: dict[str, object]) -> bool:
        _visual_plot_type = str(_entry.get("visual_plot_type") or "").strip()
        if not _visual_plot_type:
            return False
        return _visual_plot_type == selected_plot_id

    _filtered_entries = [
        _entry
        for _entry in entries_for_scope
        if _entry_matches_selected_plot_id(_entry)
    ]

    label_to_entry = {}
    plot_filter_message = ""

    plot_options = []
    if selected_plot_id == "(no plot types)":
        plot_filter_message = (
            "No plot types are available for scope `"
            + selected_scope_label
            + "`. Select another scope."
        )
        plot_options = ["(no plots for current filters)"]
    elif not _filtered_entries:
        _generated_plot_ids = set(generated_plot_ids_by_scope.get(selected_plot_scope, []))
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
            plot_filter_message = (
                "No generated plots for scope `"
                + selected_scope_label
                + "` and plot type `"
                + selected_plot_id
                + "`. Run `uv run dense plot --only "
                + _generation_hint
                + "` to generate it."
            )
        else:
            plot_filter_message = (
                "No plots found for scope `"
                + selected_scope_label
                + "` and plot type `"
                + selected_plot_id
                + "`. Select another scope or plot type."
            )
        plot_options = ["(no plots for current filters)"]
    else:
        for _entry_index, _entry in enumerate(_filtered_entries):
            _plan = str(_entry["plan_name"])
            compact_plan_name = compact_plan_label(_plan)
            _variant = str(_entry["variant"]).strip()
            _label = str(_entry["plot_name"])
            if _variant:
                _label = f"{_label} ({_variant})"
            _option_label = f"{_entry_index + 1}. [{compact_plan_name}] {_label}"
            plot_options.append(_option_label)
            label_to_entry[_option_label] = _entry

    plot_selector = mo.ui.dropdown(options=plot_options, value=plot_options[0], label="")
    return (
        label_to_entry,
        plot_filter_message,
        plot_availability_table,
        plot_id_filter,
        plot_scope_filter,
        plot_selector,
    )


@app.cell
def _(label_to_entry, plot_filter_message, plot_selector):
    _selected_plot_option = str(plot_selector.value or "")
    active_plot_error = str(plot_filter_message or "").strip()
    active_plot_entry = None
    if not active_plot_error and _selected_plot_option not in label_to_entry:
        active_plot_error = "Selected plot is not available for the current plan filter."
    if not active_plot_error and _selected_plot_option in label_to_entry:
        active_plot_entry = label_to_entry[_selected_plot_option]
    return active_plot_entry, active_plot_error


@app.cell
def _(Path, hashlib, plot_manifest_path, shutil, subprocess):
    preview_dir = plot_manifest_path.parent / ".preview_png"
    _image_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}

    def resolve_plot_preview_image(plot_path: Path) -> Path:
        source_path = Path(plot_path).expanduser().resolve()
        suffix = str(source_path.suffix).lower()
        if suffix in _image_suffixes:
            return source_path
        if suffix != ".pdf":
            raise RuntimeError(f"Unsupported plot format: `{source_path.suffix}`.")

        preview_dir.mkdir(parents=True, exist_ok=True)

        ghostscript = shutil.which("gs")
        pdftoppm = shutil.which("pdftoppm")
        magick = shutil.which("magick")
        convert = shutil.which("convert")
        sips = shutil.which("sips")
        command_signature = "|".join(
            label
            for label, enabled in (
                ("gs", bool(ghostscript)),
                ("pdftoppm", bool(pdftoppm)),
                ("magick", bool(magick)),
                ("convert", bool(convert)),
                ("sips", bool(sips)),
            )
            if enabled
        )
        preview_version = "preview-v2-gs450"
        digest = hashlib.sha1(f"{source_path}|{command_signature}|{preview_version}".encode("utf-8")).hexdigest()[:12]
        preview_path = preview_dir / f"{source_path.stem}__{digest}.png"
        if preview_path.exists() and preview_path.stat().st_mtime >= source_path.stat().st_mtime:
            return preview_path

        for stale_preview in preview_dir.glob(f"{source_path.stem}__*.png"):
            if stale_preview == preview_path:
                continue
            stale_preview.unlink(missing_ok=True)

        command_groups = []
        if ghostscript:
            command_groups.append(
                [
                    ghostscript,
                    "-dSAFER",
                    "-dBATCH",
                    "-dNOPAUSE",
                    "-sDEVICE=pngalpha",
                    "-r450",
                    "-dFirstPage=1",
                    "-dLastPage=1",
                    f"-sOutputFile={preview_path}",
                    str(source_path),
                ]
            )
        if pdftoppm:
            command_groups.append(
                [
                    pdftoppm,
                    "-png",
                    "-singlefile",
                    "-r",
                    "450",
                    str(source_path),
                    str(preview_path.with_suffix("")),
                ]
            )
        if magick:
            command_groups.append(
                [magick, "-density", "450", f"{source_path}[0]", "-quality", "100", str(preview_path)]
            )
        if convert:
            command_groups.append(
                [convert, "-density", "450", f"{source_path}[0]", "-quality", "100", str(preview_path)]
            )
        if sips:
            command_groups.append([sips, "-s", "format", "png", str(source_path), "--out", str(preview_path)])

        if preview_path.exists():
            preview_path.unlink()
        for command in command_groups:
            try:
                subprocess.run(command, check=True, capture_output=True)
            except Exception:
                continue
            if preview_path.exists() and preview_path.stat().st_size > 0:
                return preview_path
        raise RuntimeError(
            "Unable to render PDF plot preview. Install `gs`, `pdftoppm`, "
            "`magick`, `convert`, or `sips` for PNG previews."
        )

    return resolve_plot_preview_image


@app.cell
def _(
    active_plot_entry,
    active_plot_error,
    label_to_entry,
    mo,
    plot_availability_table,
    plot_id_label_to_id,
    plot_id_filter,
    plot_gallery_notice,
    plot_scope_filter,
    plot_selector,
    resolve_plot_preview_image,
):
    _selected_scope_label = str(plot_scope_filter.value or "")
    _selected_plot_type_label = str(plot_id_filter.value or "")
    _selected_plot_type = str(plot_id_label_to_id.get(_selected_plot_type_label, _selected_plot_type_label))
    _matching_plots = int(len(label_to_entry))
    _filters_summary = mo.md(
        " | ".join(
            [
                f"Scope: `{_selected_scope_label}`",
                f"Plot type: `{_selected_plot_type}`",
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
        [plot_scope_filter, plot_id_filter, plot_selector],
        justify="start",
        align="center",
        wrap=True,
        gap=0.3,
        widths=[1.4, 1.6, 7.0],
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
        _plot_path = active_plot_entry["path"]
        _variant_text = _variant if _variant else "none"
        _content.append(
            mo.accordion(
                {
                    "Selected plot metadata": mo.md(
                        "\n".join(
                            [
                                f"- Plan scope: `{_plan_name}`",
                                f"- Plot id: `{_plot_id or 'n/a'}`",
                                f"- Plot name: `{_plot_name}`",
                                f"- Variant: `{_variant_text}`",
                                f"- File: `{str(_plot_path)}`",
                            ]
                        )
                    )
                },
                multiple=True,
            )
        )
        try:
            _preview_path = resolve_plot_preview_image(_plot_path)
            _preview_error = ""
        except Exception as exc:
            _preview_path = None
            _preview_error = str(exc)
        if _preview_path is not None:
            _content.append(
                mo.image(
                    str(_preview_path),
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
        else:
            if str(getattr(_plot_path, "suffix", "")).lower() == ".pdf":
                _content.append(mo.pdf(str(_plot_path)))
                _content.append(
                    mo.md(
                        "PNG preview unavailable: "
                        + _preview_error
                        + ". Showing PDF directly for this plot."
                    )
                )
            else:
                _content.append(
                    mo.md(
                        "Preview unavailable: "
                        + _preview_error
                        + ". Install `gs`, `pdftoppm`, `magick`, `convert`, or `sips` for PNG plot previews."
                    )
                )
    mo.vstack(_content)
    return


@app.cell
def _(mo, run_root, to_repo_relative_path):
    plot_export_target = mo.ui.dropdown(
        options=["selected", "filtered", "all"],
        value="selected",
        label="",
    )
    plot_export_format = mo.ui.dropdown(options=["pdf", "png", "svg"], value="png", label="")
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
                        "Export selected, filtered, or all plots into one format. selected = currently visible plot, "
                        "filtered = every plot matching current gallery filters, all = all plots in this run.",
                        "- Target `selected`: export the plot currently shown in Plot gallery.",
                        "- Target `filtered`: export every plot matching current scope and plot-type filters.",
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
def _(
    Path,
    active_plot_entry,
    label_to_entry,
    mo,
    plot_entries,
    plot_export_button,
    plot_export_format,
    plot_export_path,
    plot_export_target,
    require,
    repo_root,
    resolve_plot_preview_image,
    shutil,
    subprocess,
):
    _plot_click_count = int(plot_export_button.value or 0)
    _plot_status_text = ""
    if _plot_click_count > 0:
        _selected_target = str(plot_export_target.value or "selected").strip()
        require(
            _selected_target not in {"selected", "filtered", "all"},
            f"Plot export set must be selected|filtered|all, got `{_selected_target}`.",
        )
        _selected_format = str(plot_export_format.value or "").strip()
        require(
            _selected_format not in {"pdf", "png", "svg"},
            f"Plot export format must be pdf|png|svg, got `{_selected_format}`.",
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

    def _export_plot(source_path: Path, destination_path: Path, fmt: str) -> None:
        source_suffix = str(source_path.suffix).lower()
        if source_suffix == f".{fmt}":
            shutil.copy2(source_path, destination_path)
            return
        if fmt == "png":
            png_source = resolve_plot_preview_image(source_path)
            shutil.copy2(png_source, destination_path)
            return
        if fmt == "svg":
            if source_suffix == ".pdf":
                pdftocairo = shutil.which("pdftocairo")
                if not pdftocairo:
                    raise RuntimeError(
                        "SVG export from PDF requires `pdftocairo` to be installed and available in PATH."
                    )
                output_root = destination_path.with_suffix("")
                subprocess.run(
                    [pdftocairo, "-svg", str(source_path), str(output_root)],
                    check=True,
                    capture_output=True,
                )
                generated_svg = output_root.with_suffix(".svg")
                if not generated_svg.exists() or generated_svg.stat().st_size <= 0:
                    raise RuntimeError(f"Failed to export SVG for `{source_path.name}`.")
                if generated_svg != destination_path:
                    shutil.move(str(generated_svg), str(destination_path))
                return
            raise RuntimeError(
                f"Cannot export `{source_path.name}` to SVG. Only PDF sources can be exported to SVG."
            )
        if fmt == "pdf":
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

    if _plot_click_count > 0:
        _exported_n = 0
        for _idx, _entry in enumerate(_entries):
            _source_path = Path(_entry["path"]).expanduser().resolve()
            _plan_name = _slug(str(_entry.get("plan_name") or "run"))
            _plot_name = _slug(str(_entry.get("plot_id") or _entry.get("plot_name") or _source_path.stem))
            _variant = _slug(str(_entry.get("variant") or "default"))
            _destination_path = (
                _export_dir / f"{_idx + 1:03d}__{_plan_name}__{_plot_name}__{_variant}.{_selected_format}"
            )
            _export_plot(_source_path, _destination_path, _selected_format)
            _exported_n += 1
        _plot_status_text = "Saved `" + str(_exported_n) + "` plot(s) to `" + str(_export_dir) + "`."
    mo.md(_plot_status_text)
    return"""

"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/public/api.py

Baserender vNext public API for job execution and record rendering helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

from ..config import (
    AdapterCfg,
    RenderJobV4,
    load_render_job_from_mapping,
    render_contract_descriptor,
    render_contract_kinds,
    resolve_style,
    validate_render_contract_renderer,
)
from ..config import (
    validate_render_job as _validate_render_job,
)
from ..core import Record, SchemaError, ensure
from ..execution import run_render_job as _run_render_job
from ..integrations import (
    adapter_contract,
    adapter_grid_record_limit,
    build_adapter,
    finalize_adapter,
    normalize_adapter_config,
    required_source_columns,
)
from ..integrations.styles import integration_style_overrides
from ..io import iter_parquet_rows
from ..runtime import initialize_runtime
from . import catalog as _catalog
from .sequence_panel import (
    BASERENDER_SEQUENCE_PANEL_CONTRACT_ID,
    BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION,
    SequencePanelConfig,
    SequencePanelDiagnostics,
    SequencePanelImage,
    sequence_panel_config_for_adapter,
)
from .sequence_panel_layout import normalize_panel_image, sequence_center_y_px

get_adapter_descriptor = _catalog.get_adapter_descriptor
get_render_contract_descriptor = _catalog.get_render_contract_descriptor
get_renderer_descriptor = _catalog.get_renderer_descriptor
get_style_profile_descriptor = _catalog.get_style_profile_descriptor
get_transform_descriptor = _catalog.get_transform_descriptor
list_adapters = _catalog.list_adapters
list_render_contracts = _catalog.list_render_contracts
list_renderers = _catalog.list_renderers
list_style_profiles = _catalog.list_style_profiles
list_transforms = _catalog.list_transforms


def _legend_tags(record: Record) -> tuple[str, ...]:
    tags: set[str] = set()
    for feature in record.features:
        for tag in feature.tags:
            text = str(tag).strip()
            if text:
                tags.add(text)
    return tuple(sorted(tags))


def render_sequence_panel_image(
    row: Mapping[str, object],
    *,
    config: SequencePanelConfig | None = None,
    adapter_kind: str | None = None,
    style_profile: str | None = None,
    adapter_columns: Mapping[str, object] | None = None,
    adapter_policies: Mapping[str, object] | None = None,
    style_overrides: Mapping[str, object] | None = None,
    target_width_px: int = 2200,
    target_height_px: int = 430,
    vertical_anchor: str = "center",
    canvas_top_pad_px: int = 0,
    title: str | None = None,
) -> SequencePanelImage:
    """Render one adapter row into a fixed-size sequence panel.

    ``title`` is caller-owned display text. Any adapter-provided record label is
    preserved as a second header line; BaseRender does not interpret either value.
    """
    import matplotlib.pyplot as plt

    if config is None:
        ensure(adapter_kind is not None, "adapter_kind is required when config is not provided", SchemaError)
        ensure(style_profile is not None, "style_profile is required when config is not provided", SchemaError)
        config = sequence_panel_config_for_adapter(
            adapter_kind=str(adapter_kind),
            style_profile=str(style_profile),
            adapter_columns=adapter_columns,
            adapter_policies=adapter_policies,
            style_overrides=style_overrides,
            target_width_px=target_width_px,
            target_height_px=target_height_px,
            vertical_anchor=vertical_anchor,
            canvas_top_pad_px=canvas_top_pad_px,
        )

    record = adapt_record(
        row,
        adapter_kind=config.adapter_kind,
        adapter_columns=config.adapter_columns,
        adapter_policies=config.adapter_policies,
        alphabet=config.alphabet,
    )
    record_label = record.display.overlay_text
    normalized_title: str | None = None
    if title is not None:
        ensure(isinstance(title, str) and title.strip() != "", "sequence panel title must be non-empty", SchemaError)
        normalized_title = title.strip()
        header_lines = [normalized_title]
        if record_label is not None and record_label.strip() and record_label.strip() != normalized_title:
            header_lines.append(record_label.strip())
        record = replace(
            record,
            display=replace(record.display, overlay_text="\n".join(header_lines)),
            meta={**record.meta, "overlay_text_role": "caller_header"},
        )
    fig = render_record_figure(
        record,
        renderer_name=config.renderer_name,
        style_preset=config.style_preset,
        style_overrides=config.style_overrides,
    )
    _force_opaque_white_figure(fig)
    image = _figure_rgba(fig)
    source_strand_center_y_px = sequence_center_y_px(fig)
    plt.close(fig)
    image, strand_center_y_px = normalize_panel_image(
        image,
        target_width_px=config.target_width_px,
        target_height_px=config.target_height_px,
        vertical_anchor=config.vertical_anchor,
        canvas_top_pad_px=config.canvas_top_pad_px,
        source_anchor_y_px=source_strand_center_y_px,
    )
    diagnostics = SequencePanelDiagnostics(
        contract_id=BASERENDER_SEQUENCE_PANEL_CONTRACT_ID,
        contract_version=BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION,
        style_profile=config.style_profile,
        style_preset=str(config.style_preset) if config.style_preset is not None else None,
        adapter_kind=config.adapter_kind,
        renderer_name=config.renderer_name,
        sequence_length_bp=len(str(record.sequence)),
        feature_count=len(record.features),
        strand_count=2 if bool((config.style_overrides or {}).get("show_reverse_complement", False)) else 1,
        legend_entries=_legend_tags(record),
        image_width_px=int(image.shape[1]),
        image_height_px=int(image.shape[0]),
        strand_center_y_px=float(strand_center_y_px),
        title=normalized_title,
        record_label=record_label,
    )
    return SequencePanelImage(image=image, diagnostics=diagnostics)


def _build_public_adapter(
    *,
    adapter_kind: str,
    adapter_columns: Mapping[str, object] | None,
    adapter_policies: Mapping[str, object] | None,
    alphabet: str,
):
    normalized_alphabet = str(alphabet).upper()
    kind, columns, policies = normalize_adapter_config(
        kind=adapter_kind,
        columns={} if adapter_columns is None else dict(adapter_columns),
        policies={} if adapter_policies is None else dict(adapter_policies),
        alphabet=normalized_alphabet,
    )
    cfg = AdapterCfg(
        kind=kind,
        columns=columns,
        policies=policies,
    )
    return cfg, build_adapter(cfg, alphabet=normalized_alphabet)


def _apply_public_adapter_row(
    *,
    adapter,
    row: Mapping[str, object],
    row_index: int,
) -> Record:
    ensure(isinstance(row, Mapping), "row must be a mapping", SchemaError)
    return adapter.apply(dict(row), row_index=row_index)


def _reject_partial_document_adapter(*, adapter_kind: str, surface: str) -> None:
    if adapter_contract(adapter_kind).validation_scope == "document":
        raise SchemaError(
            f"{surface} cannot use document-scoped adapter {adapter_kind!r}; "
            "use adapt_records with the complete document"
        )


def adapt_record(
    row: Mapping[str, object],
    *,
    adapter_kind: str,
    adapter_columns: Mapping[str, object] | None = None,
    adapter_policies: Mapping[str, object] | None = None,
    alphabet: str = "DNA",
    row_index: int = 0,
) -> Record:
    initialize_runtime()
    cfg, adapter = _build_public_adapter(
        adapter_kind=adapter_kind,
        adapter_columns=adapter_columns,
        adapter_policies=adapter_policies,
        alphabet=alphabet,
    )
    _reject_partial_document_adapter(adapter_kind=cfg.kind, surface="adapt_record")
    return _apply_public_adapter_row(adapter=adapter, row=row, row_index=row_index)


def adapt_records(
    rows: Iterable[Mapping[str, object]],
    *,
    adapter_kind: str,
    adapter_columns: Mapping[str, object] | None = None,
    adapter_policies: Mapping[str, object] | None = None,
    alphabet: str = "DNA",
) -> list[Record]:
    initialize_runtime()
    _cfg, adapter = _build_public_adapter(
        adapter_kind=adapter_kind,
        adapter_columns=adapter_columns,
        adapter_policies=adapter_policies,
        alphabet=alphabet,
    )
    records = [
        _apply_public_adapter_row(adapter=adapter, row=row, row_index=row_index) for row_index, row in enumerate(rows)
    ]
    finalize_adapter(adapter)
    return records


def load_record_from_parquet(
    dataset_path: str | Path,
    *,
    record_id: str,
    adapter_kind: str,
    adapter_columns: Mapping[str, object],
    adapter_policies: Mapping[str, object] | None = None,
    alphabet: str = "DNA",
    match_column: str | None = None,
) -> Record:
    initialize_runtime()

    cfg, adapter = _build_public_adapter(
        adapter_kind=adapter_kind,
        adapter_columns=adapter_columns,
        adapter_policies=adapter_policies,
        alphabet=alphabet,
    )
    _reject_partial_document_adapter(adapter_kind=cfg.kind, surface="load_record_from_parquet")
    source_columns = required_source_columns(cfg)

    if match_column is None:
        raw_match = cfg.columns.get("id")
        ensure(
            raw_match is not None,
            "adapter columns must include 'id' when match_column is not provided",
            SchemaError,
        )
        key_col = str(raw_match)
    else:
        key_col = str(match_column)
        ensure(key_col.strip() != "", "match_column must be non-empty", SchemaError)
        if key_col not in source_columns:
            source_columns = [*source_columns, key_col]

    target_id = str(record_id)
    for row_index, row in enumerate(iter_parquet_rows(dataset_path, columns=source_columns)):
        if str(row.get(key_col)) != target_id:
            continue
        return adapter.apply(row, row_index=row_index)

    raise SchemaError(f"Record '{target_id}' not found in dataset by column '{key_col}'")


def load_records_from_parquet(
    dataset_path: str | Path,
    *,
    record_ids: Iterable[str],
    adapter_kind: str,
    adapter_columns: Mapping[str, object],
    adapter_policies: Mapping[str, object] | None = None,
    alphabet: str = "DNA",
    match_column: str | None = None,
) -> list[Record]:
    initialize_runtime()
    ensure(not isinstance(record_ids, (str, bytes)), "record_ids must be an iterable of ids", SchemaError)
    requested_ids = [str(record_id) for record_id in record_ids]
    ensure(len(requested_ids) > 0, "record_ids must contain at least one id", SchemaError)
    ensure(
        all(record_id.strip() != "" for record_id in requested_ids),
        "record_ids cannot contain blank ids",
        SchemaError,
    )

    cfg, adapter = _build_public_adapter(
        adapter_kind=adapter_kind,
        adapter_columns=adapter_columns,
        adapter_policies=adapter_policies,
        alphabet=alphabet,
    )
    _reject_partial_document_adapter(adapter_kind=cfg.kind, surface="load_records_from_parquet")
    source_columns = required_source_columns(cfg)

    if match_column is None:
        raw_match = cfg.columns.get("id")
        ensure(
            raw_match is not None,
            "adapter columns must include 'id' when match_column is not provided",
            SchemaError,
        )
        key_col = str(raw_match)
    else:
        key_col = str(match_column)
        ensure(key_col.strip() != "", "match_column must be non-empty", SchemaError)
        if key_col not in source_columns:
            source_columns = [*source_columns, key_col]

    remaining: set[str] = set()
    for record_id in requested_ids:
        if record_id not in remaining:
            remaining.add(record_id)

    found: dict[str, Record] = {}
    for row_index, row in enumerate(iter_parquet_rows(dataset_path, columns=source_columns)):
        row_id = str(row.get(key_col))
        if row_id not in remaining:
            continue
        if row_id not in found:
            found[row_id] = adapter.apply(row, row_index=row_index)
        remaining.discard(row_id)
        if not remaining:
            break

    if remaining:
        raise SchemaError(f"Records not found in dataset by column '{key_col}': {sorted(remaining)}")

    return [found[record_id] for record_id in requested_ids]


def render_record_figure(
    record: Record,
    *,
    renderer_name: str = "sequence_rows",
    style_preset: str | Path | None = None,
    style_overrides: Mapping[str, object] | None = None,
    renderer_options: Mapping[str, object] | None = None,
):
    initialize_runtime()
    style = resolve_style(
        preset=style_preset,
        overrides={} if style_overrides is None else dict(style_overrides),
    )
    from ..render import Palette

    palette = Palette(style.palette)
    from ..render import render_record

    if renderer_options:
        return render_record(
            record,
            renderer_name=renderer_name,
            style=style,
            palette=palette,
            renderer_options=renderer_options,
        )
    return render_record(record, renderer_name=renderer_name, style=style, palette=palette)


def _force_opaque_white_figure(fig) -> None:
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    for axis in fig.axes:
        axis.set_facecolor("white")
        axis.patch.set_alpha(1.0)


def _figure_rgba(fig):
    import numpy as np

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return data.reshape((height, width, 4)).copy()


def render_record_grid_figure(
    records: Iterable[Record],
    *,
    renderer_name: str = "sequence_rows",
    style_preset: str | Path | None = None,
    style_overrides: Mapping[str, object] | None = None,
    ncols: int = 3,
    renderer_options: Mapping[str, object] | None = None,
):
    import matplotlib.pyplot as plt

    initialize_runtime()
    records_list = list(records)
    ensure(len(records_list) > 0, "render_record_grid_figure requires at least one record", SchemaError)
    ensure(isinstance(ncols, int) and ncols >= 1, "ncols must be >= 1", SchemaError)
    style = resolve_style(
        preset=style_preset,
        overrides={} if style_overrides is None else dict(style_overrides),
    )
    from ..render import Palette, validate_records_for_rendering

    validation_kwargs = {
        "renderer_name": renderer_name,
        "style": style,
        "palette": Palette(style.palette),
    }
    if renderer_options:
        validation_kwargs["renderer_options"] = renderer_options
    records_list = list(validate_records_for_rendering(records_list, **validation_kwargs))
    limits = tuple(
        limit
        for limit in (
            get_renderer_descriptor(renderer_name).max_grid_records,
            adapter_grid_record_limit(records_list),
        )
        if limit is not None
    )
    grid_limit = min(limits) if limits else None
    if grid_limit is not None and len(records_list) > grid_limit:
        raise SchemaError(
            f"renderer {renderer_name!r} supports at most {grid_limit} record per grid; render records individually"
        )

    panel_images: list[object] = []
    for record in records_list:
        panel_kwargs = {
            "renderer_name": renderer_name,
            "style_preset": style_preset,
            "style_overrides": style_overrides,
        }
        if renderer_options:
            panel_kwargs["renderer_options"] = renderer_options
        panel = render_record_figure(record, **panel_kwargs)
        panel_images.append(_figure_rgba(panel))
        plt.close(panel)

    max_h = max(image.shape[0] for image in panel_images)
    max_w = max(image.shape[1] for image in panel_images)
    cols = min(ncols, len(panel_images))
    rows = int(math.ceil(len(panel_images) / cols))
    dpi = 120
    fig_w = (cols * max_w) / dpi
    fig_h = (rows * max_h) / dpi
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)
    flat_axes = list(axes.flat)

    for idx, image in enumerate(panel_images):
        ax = flat_axes[idx]
        ax.imshow(image)
        ax.set_axis_off()

    for ax in flat_axes[len(panel_images) :]:
        ax.set_axis_off()

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.02, hspace=0.02)
    return fig


def render_parquet_record_figure(
    dataset_path: str | Path,
    *,
    record_id: str,
    adapter_kind: str,
    adapter_columns: Mapping[str, object],
    adapter_policies: Mapping[str, object] | None = None,
    alphabet: str = "DNA",
    match_column: str | None = None,
    renderer_name: str = "sequence_rows",
    style_preset: str | Path | None = None,
    style_overrides: Mapping[str, object] | None = None,
):
    record = load_record_from_parquet(
        dataset_path,
        record_id=record_id,
        adapter_kind=adapter_kind,
        adapter_columns=adapter_columns,
        adapter_policies=adapter_policies,
        alphabet=alphabet,
        match_column=match_column,
    )
    return render_record_figure(
        record,
        renderer_name=renderer_name,
        style_preset=style_preset,
        style_overrides=style_overrides,
    )


def validate_render_job(
    job_or_path: str,
    *,
    caller_root: str | Path | None = None,
) -> RenderJobV4:
    return _validate_render_job(job_or_path, caller_root=caller_root)


def run_render_job(job_or_path: RenderJobV4 | str, *, caller_root: str | Path | None = None):
    return _run_render_job(job_or_path, caller_root=caller_root)


def _check_job_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    try:
        return render_contract_descriptor(kind).kind
    except SchemaError as exc:
        allowed = ", ".join(render_contract_kinds(include_aliases=True))
        raise SchemaError(f"kind must be one of: {allowed}") from exc


def _validate_requested_job_kind(job: RenderJobV4, kind: str | None) -> None:
    contract_kind = _check_job_kind(kind)
    if contract_kind is not None:
        validate_render_contract_renderer(contract_kind, job.render.renderer, field="kind")


def validate_job(
    path_or_dict: str | Path | Mapping[str, object],
    *,
    kind: str | None = None,
    caller_root: str | Path | None = None,
) -> RenderJobV4:
    if isinstance(path_or_dict, Mapping):
        job = load_render_job_from_mapping(path_or_dict, caller_root=caller_root)
    else:
        job = validate_render_job(path_or_dict, caller_root=caller_root)
    _validate_requested_job_kind(job, kind)
    return job


def run_job(
    path_or_dict: RenderJobV4 | str | Path | Mapping[str, object],
    *,
    kind: str | None = None,
    strict: bool | None = None,
    caller_root: str | Path | None = None,
):
    if isinstance(path_or_dict, RenderJobV4):
        job = path_or_dict
    elif isinstance(path_or_dict, Mapping):
        job = load_render_job_from_mapping(path_or_dict, caller_root=caller_root)
    else:
        job = validate_render_job(path_or_dict, caller_root=caller_root)
    _validate_requested_job_kind(job, kind)

    if strict is not None:
        job = replace(job, run=replace(job.run, strict=bool(strict)))
    return run_render_job(job, caller_root=caller_root)


def render(
    record_or_records: Record | Iterable[Record],
    *,
    renderer: str = "sequence_rows",
    style: Mapping[str, object] | None = None,
    grid: Mapping[str, object] | None = None,
    options: Mapping[str, object] | None = None,
):
    if style is None:
        style_preset = None
        style_overrides: Mapping[str, object] | None = None
    else:
        style_preset_raw = style.get("preset") if isinstance(style, Mapping) else None
        style_preset = None if style_preset_raw is None else str(style_preset_raw)
        if isinstance(style, Mapping) and "overrides" in style:
            overrides_raw = style.get("overrides") or {}
            if not isinstance(overrides_raw, Mapping):
                raise SchemaError("style.overrides must be a mapping")
            style_overrides = dict(overrides_raw)
        elif isinstance(style, Mapping):
            style_overrides = {k: v for k, v in style.items() if k not in {"preset", "overrides"}}
        else:
            raise SchemaError("style must be a mapping")

    if isinstance(record_or_records, Record):
        figure_kwargs = {
            "renderer_name": renderer,
            "style_preset": style_preset,
            "style_overrides": style_overrides,
        }
        if options:
            figure_kwargs["renderer_options"] = options
        return render_record_figure(record_or_records, **figure_kwargs)

    records_list = list(record_or_records)
    ncols = len(records_list)
    if grid is not None:
        if not isinstance(grid, Mapping):
            raise SchemaError("grid must be a mapping")
        unknown = sorted(str(k) for k in grid.keys() if str(k) != "ncols")
        if unknown:
            raise SchemaError(f"grid contains unknown keys: {unknown}; allowed keys: ['ncols']")
        if "ncols" in grid:
            ncols = int(grid["ncols"])
    if ncols < 1:
        raise SchemaError("grid.ncols must be >= 1")
    grid_kwargs = {
        "renderer_name": renderer,
        "style_preset": style_preset,
        "style_overrides": style_overrides,
        "ncols": ncols,
    }
    if options:
        grid_kwargs["renderer_options"] = options
    return render_record_grid_figure(records_list, **grid_kwargs)


def style_profile_overrides(profile_name: str) -> Mapping[str, object]:
    """Return a defensive copy of a registered style profile."""
    return integration_style_overrides(profile_name)

"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/public/api.py

Baserender vNext public API for job execution and record rendering helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..adapters import build_adapter, list_adapter_descriptors, required_source_columns
from ..adapters import get_adapter_descriptor as _get_adapter_descriptor
from ..config import (
    AdapterCfg,
    RenderJobV3,
    load_sequence_rows_job_from_mapping,
    render_contract_descriptor,
    render_contract_descriptors,
    render_contract_kinds,
    resolve_style,
    validate_render_contract_renderer,
)
from ..config import (
    validate_render_job as _validate_render_job,
)
from ..config import (
    validate_sequence_rows_job as _validate_sequence_rows_job,
)
from ..config.adapter_contracts import normalize_adapter_config
from ..core import Record, SchemaError, ensure
from ..execution import run_sequence_rows_job as _run_sequence_rows_job
from ..io import iter_parquet_rows
from ..render.renderer import get_renderer_descriptor as _get_renderer_descriptor
from ..render.renderer import renderer_descriptors
from ..runtime import initialize_runtime
from ..styles.curated import cruncher_showcase_style_overrides as _cruncher_showcase_style_overrides
from .sequence_panel import (
    BASERENDER_SEQUENCE_PANEL_CONTRACT_ID,
    BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION,
    DEFAULT_SEQUENCE_PANEL_PROFILE,
    SequencePanelConfig,
    SequencePanelDiagnostics,
    SequencePanelImage,
    sequence_panel_config_for_adapter,
)


def _normalize_panel_image(
    image: Any,
    *,
    target_width_px: int,
    target_height_px: int,
    vertical_anchor: str,
    canvas_top_pad_px: int,
) -> Any:
    import numpy as np
    from PIL import Image

    if int(target_width_px) <= 0 or int(target_height_px) <= 0:
        raise SchemaError("sequence panel target dimensions must be positive")

    rgba = np.asarray(image)
    ensure(rgba.ndim == 3 and rgba.shape[2] in {3, 4}, "sequence panel image must be RGB/RGBA", SchemaError)
    if rgba.shape[2] == 3:
        alpha = np.full(rgba.shape[:2], 255, dtype=np.uint8)
        rgba = np.dstack([rgba[:, :, :3], alpha])

    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    content_mask = ((rgb < 245).any(axis=2)) & (alpha > 0)
    if content_mask.any():
        ys, xs = np.where(content_mask)
        pad = 8
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(rgba.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(rgba.shape[1], int(xs.max()) + pad + 1)
        rgba = rgba[y0:y1, x0:x1, :]

    source = Image.fromarray(rgba.astype(np.uint8, copy=False))
    scale = min(int(target_width_px) / max(source.width, 1), int(target_height_px) / max(source.height, 1))
    resized = source.resize(
        (max(1, int(source.width * scale)), max(1, int(source.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", (int(target_width_px), int(target_height_px)), (255, 255, 255, 255))
    x = (canvas.width - resized.width) // 2
    anchor = str(vertical_anchor).strip().lower()
    if anchor == "top":
        y = min(max(0, int(canvas_top_pad_px)), max(0, canvas.height - resized.height))
    elif anchor == "bottom":
        y = max(0, canvas.height - resized.height)
    elif anchor == "center":
        y = (canvas.height - resized.height) // 2
    else:
        raise SchemaError("sequence panel vertical_anchor must be 'top', 'center', or 'bottom'")
    canvas.alpha_composite(resized, dest=(x, y))
    return np.asarray(canvas)


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
    style_profile: str = DEFAULT_SEQUENCE_PANEL_PROFILE,
    adapter_columns: Mapping[str, object] | None = None,
    adapter_policies: Mapping[str, object] | None = None,
    style_overrides: Mapping[str, object] | None = None,
    target_width_px: int = 2200,
    target_height_px: int = 310,
    vertical_anchor: str = "center",
    canvas_top_pad_px: int = 0,
) -> SequencePanelImage:
    import matplotlib.pyplot as plt

    if config is None:
        ensure(adapter_kind is not None, "adapter_kind is required when config is not provided", SchemaError)
        config = sequence_panel_config_for_adapter(
            adapter_kind=str(adapter_kind),
            style_profile=style_profile,
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
    fig = render_record_figure(
        record,
        renderer_name=config.renderer_name,
        style_preset=config.style_preset,
        style_overrides=config.style_overrides,
    )
    image = _figure_rgba(fig)
    plt.close(fig)
    image = _normalize_panel_image(
        image,
        target_width_px=config.target_width_px,
        target_height_px=config.target_height_px,
        vertical_anchor=config.vertical_anchor,
        canvas_top_pad_px=config.canvas_top_pad_px,
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
    _cfg, adapter = _build_public_adapter(
        adapter_kind=adapter_kind,
        adapter_columns=adapter_columns,
        adapter_policies=adapter_policies,
        alphabet=alphabet,
    )
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
    return [
        _apply_public_adapter_row(adapter=adapter, row=row, row_index=row_index) for row_index, row in enumerate(rows)
    ]


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
):
    initialize_runtime()
    style = resolve_style(
        preset=style_preset,
        overrides={} if style_overrides is None else dict(style_overrides),
    )
    from ..render import Palette

    palette = Palette(style.palette)
    from ..render import render_record

    return render_record(record, renderer_name=renderer_name, style=style, palette=palette)


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
):
    import matplotlib.pyplot as plt

    initialize_runtime()
    records_list = list(records)
    ensure(len(records_list) > 0, "render_record_grid_figure requires at least one record", SchemaError)
    ensure(isinstance(ncols, int) and ncols >= 1, "ncols must be >= 1", SchemaError)

    panel_images: list[object] = []
    for record in records_list:
        panel = render_record_figure(
            record,
            renderer_name=renderer_name,
            style_preset=style_preset,
            style_overrides=style_overrides,
        )
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


def validate_sequence_rows_job(
    job_or_path: str,
    *,
    caller_root: str | Path | None = None,
) -> RenderJobV3:
    return _validate_sequence_rows_job(job_or_path, caller_root=caller_root)


def run_sequence_rows_job(job_or_path: RenderJobV3 | str, *, caller_root: str | Path | None = None):
    return _run_sequence_rows_job(job_or_path, caller_root=caller_root)


def validate_render_job(
    job_or_path: str,
    *,
    caller_root: str | Path | None = None,
) -> RenderJobV3:
    return _validate_render_job(job_or_path, caller_root=caller_root)


def run_render_job(job_or_path: RenderJobV3 | str, *, caller_root: str | Path | None = None):
    return run_sequence_rows_job(job_or_path, caller_root=caller_root)


def validate_cruncher_showcase_job(
    job_or_path: str,
    *,
    caller_root: str | Path | None = None,
) -> RenderJobV3:
    # Backward-compatible alias; BaseRenderJobV3 / RenderJobV3 is canonical.
    return validate_sequence_rows_job(job_or_path, caller_root=caller_root)


def run_cruncher_showcase_job(job_or_path: RenderJobV3 | str, *, caller_root: str | Path | None = None):
    # Backward-compatible alias; BaseRenderJobV3 / RenderJobV3 is canonical.
    return run_sequence_rows_job(job_or_path, caller_root=caller_root)


def _check_job_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    try:
        return render_contract_descriptor(kind).kind
    except SchemaError as exc:
        allowed = ", ".join(render_contract_kinds(include_aliases=True))
        raise SchemaError(f"kind must be one of: {allowed}") from exc


def _validate_requested_job_kind(job: RenderJobV3, kind: str | None) -> None:
    contract_kind = _check_job_kind(kind)
    if contract_kind is not None:
        validate_render_contract_renderer(contract_kind, job.render.renderer, field="kind")


def validate_job(
    path_or_dict: str | Path | Mapping[str, object],
    *,
    kind: str | None = None,
    caller_root: str | Path | None = None,
) -> RenderJobV3:
    if isinstance(path_or_dict, Mapping):
        job = load_sequence_rows_job_from_mapping(path_or_dict, caller_root=caller_root)
    else:
        job = validate_render_job(path_or_dict, caller_root=caller_root)
    _validate_requested_job_kind(job, kind)
    return job


def run_job(
    path_or_dict: RenderJobV3 | str | Path | Mapping[str, object],
    *,
    kind: str | None = None,
    strict: bool | None = None,
    caller_root: str | Path | None = None,
):
    if isinstance(path_or_dict, RenderJobV3):
        job = path_or_dict
    elif isinstance(path_or_dict, Mapping):
        job = load_sequence_rows_job_from_mapping(path_or_dict, caller_root=caller_root)
    else:
        job = validate_render_job(path_or_dict, caller_root=caller_root)
    _validate_requested_job_kind(job, kind)

    if strict is not None:
        job = replace(job, run=replace(job.run, strict=bool(strict)))
    return run_sequence_rows_job(job, caller_root=caller_root)


def list_adapters() -> tuple[str, ...]:
    return tuple(descriptor.kind for descriptor in list_adapter_descriptors())


def get_adapter_descriptor(kind: str):
    return _get_adapter_descriptor(kind)


def list_renderers() -> tuple[str, ...]:
    return tuple(descriptor.name for descriptor in renderer_descriptors())


def get_renderer_descriptor(name: str):
    return _get_renderer_descriptor(name)


def list_render_contracts() -> tuple[str, ...]:
    return tuple(descriptor.kind for descriptor in render_contract_descriptors())


def get_render_contract_descriptor(kind: str):
    return render_contract_descriptor(kind)


def render(
    record_or_records: Record | Iterable[Record],
    *,
    renderer: str = "sequence_rows",
    style: Mapping[str, object] | None = None,
    grid: Mapping[str, object] | None = None,
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
        return render_record_figure(
            record_or_records,
            renderer_name=renderer,
            style_preset=style_preset,
            style_overrides=style_overrides,
        )

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
    return render_record_grid_figure(
        records_list,
        renderer_name=renderer,
        style_preset=style_preset,
        style_overrides=style_overrides,
        ncols=ncols,
    )


def cruncher_showcase_style_overrides() -> Mapping[str, object]:
    return _cruncher_showcase_style_overrides()

"""
Shared browser runtime support helpers for generated latentdna marimo notebooks.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import marimo as mo
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..annotation_layout import choose_annotation_placement
from ..contracts.notebook import WorkspaceNotebookControls
from ..labels import humanize_column_name
from ..metadata_axes import (
    AxisStyle,
    axis_color_map,
    axis_display_label,
    axis_display_text,
    axis_style_map_from_payload,
    legend_categories,
    normalize_axis_category,
    ordered_categories_for_axis,
)
from ..metadata_join_keys import candidate_join_key_pairs_for_columns
from ..reference_sets import reference_set_required_columns, resolve_reference_set_ids_from_columns
from ..visual_style import (
    ANNOTATION_LABEL_BOX_ALPHA,
    GRID_COLOR,
    NOTEBOOK_FONT_STACK,
    PANEL_BACKGROUND_COLOR,
    PLOT_FONT_FAMILY,
    PLOT_LABEL_FONT_SIZE,
    PLOT_LEGEND_FONT_SIZE,
    PLOT_TICK_FONT_SIZE,
    PLOT_TITLE_FONT_SIZE,
    PUBLICATION_PALETTE,
    SPINE_COLOR,
    TEXT_COLOR,
    contrast_safe_scatter_color_map,
    humanize_display_text,
    reference_annotation_label,
)
from ..visual_style import scatter_style as shared_scatter_style
from ..workspaces.loader import load_workspace_config
from .raster_scatter import BACKGROUND_RASTER_MIN_ALPHA

_RUNTIME_TABLE_ARTIFACT_KINDS = {
    "alignments": "alignment_set",
    "clusters": "cluster_set",
    "distances": "distance_set",
    "projections": "projection",
    "scalars": "scalar_table",
    "views": "view",
}


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifact_manifest(
    artifact_dir: Path,
    *,
    artifact_kind: str,
    artifact_id: str,
    allow_missing_status: bool = False,
    allowed_statuses: set[str] | None = None,
) -> dict[str, object]:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{artifact_kind} manifest is missing for `{artifact_id}`")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"{artifact_kind} manifest is invalid for `{artifact_id}`")
    manifest_artifact_id = str(manifest.get("artifact_id") or "").strip()
    manifest_kind = str(manifest.get("artifact_kind") or "").strip()
    if not manifest_artifact_id:
        raise ValueError(f"{artifact_kind} manifest is invalid for `{artifact_id}`: artifact_id=missing")
    if not manifest_kind:
        raise ValueError(f"{artifact_kind} manifest is invalid for `{artifact_id}`: artifact_kind=missing")
    if manifest_kind and manifest_kind != artifact_kind:
        raise ValueError(
            f"{artifact_kind} manifest has unexpected artifact kind for `{artifact_id}`: `{manifest_kind}`"
        )
    if manifest_artifact_id != artifact_id:
        raise ValueError(f"{artifact_kind} manifest is stale for `{artifact_id}`: artifact_id=`{manifest_artifact_id}`")
    status = str(manifest.get("status") or "").strip().lower()
    valid_statuses = {str(item).strip().lower() for item in (allowed_statuses or {"ok"}) if str(item).strip()}
    if bool(manifest.get("stale")) or (status and status not in valid_statuses):
        stale_reason = "stale" if bool(manifest.get("stale")) else f"status={status}"
        raise ValueError(f"{artifact_kind} artifact is not fresh for `{artifact_id}`: {stale_reason}")
    if not status and not allow_missing_status:
        raise ValueError(f"{artifact_kind} artifact is not fresh for `{artifact_id}`: status=missing")
    return manifest


def _table_artifact_metadata(path: Path) -> tuple[Path, str, str] | None:
    artifact_dir = path.parent
    root_name = artifact_dir.parent.name
    artifact_kind = _RUNTIME_TABLE_ARTIFACT_KINDS.get(root_name)
    if artifact_kind is None:
        return None
    return artifact_dir, artifact_kind, artifact_dir.name


def _manifest_attention_warning(manifest: dict[str, object], *, artifact_kind: str, artifact_id: str) -> str | None:
    status = str(manifest.get("status") or "").strip().lower()
    if status != "attention":
        return None
    warnings = [str(item).strip() for item in manifest.get("warnings", []) if str(item).strip()]
    if warnings:
        return f"{artifact_kind} `{artifact_id}` is rendered from an attention-state artifact: {warnings[0]}"
    return f"{artifact_kind} `{artifact_id}` is rendered from an attention-state artifact."


def _load_table_uncached(
    path: Path,
    *,
    require_fresh_manifest: bool = False,
    allowed_statuses: set[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    manifest: dict[str, object] | None = None
    artifact_kind = ""
    artifact_id = ""
    if require_fresh_manifest:
        metadata = _table_artifact_metadata(path)
        if metadata is not None:
            artifact_dir, artifact_kind, artifact_id = metadata
            manifest = load_artifact_manifest(
                artifact_dir,
                artifact_kind=artifact_kind,
                artifact_id=artifact_id,
                allow_missing_status=artifact_kind != "view",
                allowed_statuses=allowed_statuses,
            )
    frame = pd.read_parquet(path, columns=columns)
    if manifest is not None:
        status = str(manifest.get("status") or "").strip().lower()
        if status and status != "ok":
            frame.attrs["artifact_status"] = status
            warning = _manifest_attention_warning(manifest, artifact_kind=artifact_kind, artifact_id=artifact_id)
            if warning:
                frame.attrs["artifact_warning"] = warning
    return frame


def _cache_token(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return 0, 0
    return int(stat.st_mtime_ns), int(stat.st_size)


def _manifest_cache_identity(path: Path) -> tuple[str, int, int]:
    metadata = _table_artifact_metadata(path)
    if metadata is None:
        return "", 0, 0
    artifact_dir, _, _ = metadata
    manifest_path = artifact_dir / "manifest.json"
    mtime_ns, size = _cache_token(manifest_path)
    return str(manifest_path), mtime_ns, size


@lru_cache(maxsize=64)
def _cached_parquet_table(
    path_text: str,
    path_mtime_ns: int,
    path_size: int,
    manifest_path_text: str,
    manifest_mtime_ns: int,
    manifest_size: int,
    require_fresh_manifest: bool,
    allowed_statuses_key: tuple[str, ...],
    columns_key: tuple[str, ...],
) -> pd.DataFrame:
    del path_mtime_ns, path_size, manifest_path_text, manifest_mtime_ns, manifest_size
    return _load_table_uncached(
        Path(path_text),
        require_fresh_manifest=require_fresh_manifest,
        allowed_statuses=set(allowed_statuses_key) if allowed_statuses_key else None,
        columns=list(columns_key) if columns_key else None,
    )


def load_table(
    path: Path,
    *,
    require_fresh_manifest: bool = False,
    allowed_statuses: set[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    path_mtime_ns, path_size = _cache_token(path)
    manifest_path_text, manifest_mtime_ns, manifest_size = _manifest_cache_identity(path)
    cached = _cached_parquet_table(
        str(path),
        path_mtime_ns,
        path_size,
        manifest_path_text,
        manifest_mtime_ns,
        manifest_size,
        require_fresh_manifest,
        tuple(sorted(str(status) for status in (allowed_statuses or []))),
        tuple(columns or ()),
    )
    frame = cached.copy()
    frame.attrs = dict(cached.attrs)
    return frame


def read_text(path_text: str | None) -> str | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def load_workspace_notebook_controls(control_path: Path) -> dict[str, object]:
    return WorkspaceNotebookControls.model_validate(load_json(control_path)).model_dump(mode="json")


def notebook_theme():
    return mo.Html(
        f"""
        <style>
          .latentdna-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 5.5rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 650;
            letter-spacing: 0.02em;
            font-family: {NOTEBOOK_FONT_STACK};
            color: {TEXT_COLOR};
            background: rgba(226, 232, 240, 0.82);
          }}

          .latentdna-badge--primary {{
            background: rgba(59, 130, 246, 0.12);
          }}

          .latentdna-badge--appendix {{
            background: rgba(236, 201, 75, 0.22);
          }}
        </style>
        """
    )


def unique_in_order(values):
    seen = set()
    ordered = []
    for value in values:
        key = str(value or "Unsectioned").strip() or "Unsectioned"
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def option_key_for_value(options: dict[str, object], target_value: object) -> str | None:
    target_text = str(target_value)
    for key, value in options.items():
        if str(value) == target_text:
            return key
    return None


def labeled_options(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    normalized: list[tuple[str, object]] = []
    counts: dict[str, int] = {}
    for label, value in pairs:
        value_text = str(value).strip()
        base_label = str(label).strip() or value_text
        normalized.append((base_label, value))
        counts[base_label] = counts.get(base_label, 0) + 1

    options: dict[str, object] = {}
    for base_label, value in normalized:
        value_text = str(value).strip()
        if not value_text:
            continue
        label = base_label if counts[base_label] == 1 else f"{base_label} [{value_text}]"
        if label in options:
            if str(options[label]).strip() == value_text:
                continue
            suffix = 2
            while f"{label} #{suffix}" in options:
                suffix += 1
            label = f"{label} #{suffix}"
        options[label] = value
    return options


def resolve_labeled_option_card(
    cards: Iterable[dict[str, object]],
    selected_value: object,
    *,
    id_column: str = "plot_id",
    title_column: str = "title",
) -> dict[str, object] | None:
    """Resolve a Marimo dropdown selection against stable IDs and display labels.

    Marimo dropdowns are configured from ``label -> value`` dictionaries, but the
    browser-side select element exposes display labels. Accepting both the
    durable ID and the rendered label makes generated notebooks robust to manual
    use and browser-level review automation.
    """
    ordered_cards = [dict(card) for card in cards]
    if not ordered_cards:
        return None
    selected_text = str(selected_value or "").strip()
    if not selected_text:
        return ordered_cards[0]
    for card in ordered_cards:
        card_id = str(card.get(id_column) or "").strip()
        card_title = str(card.get(title_column) or "").strip()
        accepted = {value for value in (card_id, card_title) if value}
        if card_id and card_title:
            accepted.add(f"{card_title} [{card_id}]")
        if selected_text in accepted:
            return card
    return ordered_cards[0]


def normalize_label(value) -> str:
    return str(value or "").strip().lower()


def display_reference_label(value) -> str:
    return reference_annotation_label(value)


def _missing_reference_display_value(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool):
        return missing
    return False


def _scalar_missing_mask(values: pd.Series) -> pd.Series | None:
    """Use vectorized missing checks when a hue column is scalar-valued."""
    if pd.api.types.is_numeric_dtype(values.dtype):
        return values.replace([np.inf, -np.inf], np.nan).isna()
    if (
        pd.api.types.is_bool_dtype(values.dtype)
        or isinstance(values.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(values.dtype)
    ):
        return values.isna()
    if pd.api.types.is_object_dtype(values.dtype):
        sample = values.dropna().head(128).tolist()
        if any(_listlike_hue_values(value) is not None for value in sample):
            return None
        return values.isna()
    return None


@lru_cache(maxsize=16)
def _load_workspace_reference_set(workspace_dir_text: str, reference_set_id: str):
    context = load_workspace_config(Path(workspace_dir_text))
    return context.config.reference_sets.get(reference_set_id)


def _reference_annotation_coverage_warnings(
    frames: list[pd.DataFrame],
    *,
    reference_set_id: str,
    reference_labels: list[str],
    reference_match_column: str,
) -> list[str]:
    targets = {normalize_label(label) for label in reference_labels if str(label).strip()}
    if not targets:
        return []

    panel_count = 0
    missing_column_panels: list[str] = []
    empty_match_panels: list[str] = []
    for index, frame in enumerate(frames):
        if frame.empty:
            continue
        panel_count += 1
        panel_label = str(
            frame.attrs.get("view_id")
            or frame.attrs.get("projection_id")
            or frame.attrs.get("scalar_id")
            or f"panel {index + 1}"
        )
        if reference_match_column not in frame.columns:
            missing_column_panels.append(panel_label)
            continue
        match_values = frame[reference_match_column].astype("string").str.strip().str.lower()
        if not bool(match_values.isin(targets).any()):
            empty_match_panels.append(panel_label)

    warnings: list[str] = []
    if missing_column_panels:
        warnings.append(
            f"reference set `{reference_set_id}` cannot be checked in "
            f"{len(missing_column_panels)} panel(s) missing `{reference_match_column}`: "
            + ", ".join(missing_column_panels[:4])
            + (" ..." if len(missing_column_panels) > 4 else "")
        )
    if empty_match_panels:
        warnings.append(
            f"reference set `{reference_set_id}` has no matched overlay rows in "
            f"{len(empty_match_panels)} of {panel_count} non-empty panel(s): "
            + ", ".join(empty_match_panels[:4])
            + (" ..." if len(empty_match_panels) > 4 else "")
        )
    return warnings


def resolve_reference_annotation(
    reference_set_id: str | None,
    frames: list[pd.DataFrame],
    *,
    workspace_dir: Path,
    fallback_labels: list[str] | None = None,
    label_limit: int | None = None,
) -> dict[str, object]:
    if reference_set_id is None:
        return {
            "reference_set_id": "",
            "match_column": "usr_label__primary",
            "labels": list(fallback_labels or []),
            "display_labels": {},
            "label_limit": label_limit,
            "warnings": [],
        }
    selected_reference_set_id = str(reference_set_id or "").strip()
    if not selected_reference_set_id:
        return {
            "reference_set_id": "",
            "match_column": "usr_label__primary",
            "labels": [],
            "display_labels": {},
            "label_limit": 0,
            "warnings": [],
        }

    try:
        reference_set = _load_workspace_reference_set(str(workspace_dir), selected_reference_set_id)
    except Exception as exc:
        return {
            "reference_set_id": selected_reference_set_id,
            "match_column": "usr_label__primary",
            "labels": [],
            "display_labels": {},
            "label_limit": 0,
            "warnings": [f"reference set `{selected_reference_set_id}` could not be loaded: {exc}"],
        }
    if reference_set is None:
        return {
            "reference_set_id": selected_reference_set_id,
            "match_column": "usr_label__primary",
            "labels": [],
            "display_labels": {},
            "label_limit": 0,
            "warnings": [f"reference set `{selected_reference_set_id}` is not configured"],
        }

    required_columns = reference_set_required_columns(reference_set)
    nonempty_frames = [frame for frame in frames if not frame.empty]
    column_values: dict[str, np.ndarray] = {}
    for column in required_columns:
        parts: list[pd.Series] = []
        column_available = False
        for frame in nonempty_frames:
            if column in frame.columns:
                parts.append(frame[column].reset_index(drop=True))
                column_available = True
            else:
                parts.append(pd.Series([None] * len(frame), dtype=object))
        if column_available and parts:
            column_values[column] = pd.concat(parts, ignore_index=True).to_numpy(dtype=object)

    resolution = resolve_reference_set_ids_from_columns(reference_set, column_values)
    match_column = str(getattr(reference_set, "match_column", "usr_label__primary"))
    label_column = getattr(reference_set, "label_column", None)
    configured_display_labels = {
        str(key): str(value)
        for key, value in dict(getattr(reference_set, "display_labels", {}) or {}).items()
        if str(key).strip() and str(value).strip()
    }
    display_labels: dict[str, str] = {}
    for row in resolution.selected_rows:
        match_value = str(row.get(match_column) or "").strip()
        if not match_value:
            continue
        display_value = None
        if label_column:
            candidate_display = row.get(str(label_column))
            if not _missing_reference_display_value(candidate_display) and str(candidate_display).strip():
                display_value = str(candidate_display).strip()
        if display_value is None:
            display_value = configured_display_labels.get(match_value, match_value)
        display_labels[match_value] = display_value

    configured_label_limit = getattr(reference_set, "label_limit", None)
    if configured_label_limit is not None:
        default_label_limit = int(configured_label_limit)
    elif len(resolution.matched_ids) > 5:
        default_label_limit = 0
    else:
        default_label_limit = 5
    warnings = []
    if resolution.missing_columns:
        warnings.append(
            f"reference set `{selected_reference_set_id}` is missing columns: " + ", ".join(resolution.missing_columns)
        )
    if resolution.expected_ids and set(resolution.matched_ids) != set(resolution.expected_ids):
        missing_ids = [value for value in resolution.expected_ids if value not in set(resolution.matched_ids)]
        if missing_ids:
            warnings.append(
                f"reference set `{selected_reference_set_id}` has {len(missing_ids)} unmatched reference rows"
            )
    warnings.extend(
        _reference_annotation_coverage_warnings(
            frames,
            reference_set_id=selected_reference_set_id,
            reference_labels=resolution.matched_ids,
            reference_match_column=match_column,
        )
    )
    return {
        "reference_set_id": selected_reference_set_id,
        "match_column": match_column,
        "labels": resolution.matched_ids,
        "display_labels": display_labels,
        "label_limit": default_label_limit if label_limit is None else label_limit,
        "warnings": warnings,
    }


def _axis_style(axis_styles: dict[str, object] | None, column: str | None):
    if column is None:
        return None
    return axis_style_map_from_payload(axis_styles).get(str(column))


def display_hue_label(column: str, *, axis_styles: dict[str, object] | None = None) -> str:
    style = _axis_style(axis_styles, column)
    if style is not None:
        return axis_display_label(style, column)
    if column == "gc_fraction":
        return "GC fraction"
    if column == "log_likelihood_per_token_7b":
        return "7B log likelihood / token"
    if column == "log_likelihood_per_token_20b":
        return "20B log likelihood / token"
    if column.startswith("log_likelihood_per_token_"):
        return humanize_display_text(column)
    if column == "promoter_standard__strength_value_numeric":
        return "Reference strength"
    if column == "sfxi_ref__metric_value":
        return "SFXI selected metric"
    if column == "sfxi_ref__sfxi":
        return "SFXI score"
    if column == "sfxi_ref__logic_fidelity":
        return "SFXI logic fidelity"
    if column == "sfxi_ref__effect_scaled":
        return "SFXI effect scaled"
    if column == "sfxi_ref__effect_raw":
        return "SFXI raw effect"
    if column == "tf_bin":
        return "Native TF bin"
    if column.startswith("infer__evo2_") and "__log_likelihood__mean_per_token" in column:
        model = "7B" if "__7b__" in column else "20B"
        scope = "1 kb construct context" if "__template_1kb_" in column else "anchor-source insert"
        return f"{model} log likelihood / token ({scope})"
    if column.startswith("cluster_label__"):
        return humanize_display_text(column.replace("cluster_label__", ""))
    return humanize_column_name(column)


def hue_option_scale(
    hue_options: object,
    column: str | None,
    *,
    panel_count: int = 1,
) -> str:
    """Return the configured continuous-hue scale contract for a selected column."""

    column_name = str(column or "").strip()
    if not column_name:
        return "global"
    if isinstance(hue_options, list):
        for option in hue_options:
            option_column = ""
            option_scale = ""
            if isinstance(option, dict):
                option_column = str(option.get("column") or "").strip()
                option_scale = str(option.get("scale") or "").strip()
            else:
                option_column = str(getattr(option, "column", "") or "").strip()
                option_scale = str(getattr(option, "scale", "") or "").strip()
            if option_column == column_name and option_scale in {"global", "panel"}:
                return option_scale
    if column_name == "gc_fraction" and panel_count > 1:
        return "panel"
    return "global"


def continuous_hue_colorbar_label(
    column: str,
    *,
    axis_styles: dict[str, object] | None = None,
    panel_scaled: bool = False,
) -> str:
    label = display_hue_label(column, axis_styles=axis_styles)
    if panel_scaled:
        return f"{label} (panel scale)"
    return label


def display_hue_value(column: str | None, value: object, *, axis_styles: dict[str, object] | None = None) -> str:
    style = _axis_style(axis_styles, column)
    if style is not None:
        return axis_display_text(style, value)
    return humanize_display_text(value)


def normalize_categorical_hue_value(
    column: str | None,
    value: object,
    *,
    axis_styles: dict[str, object] | None = None,
    row: dict[str, object] | None = None,
) -> str:
    style = _axis_style(axis_styles, column)
    return _normalize_categorical_hue_value_with_style(style, value, row=row)


def _normalize_categorical_hue_value_with_style(
    style: AxisStyle | None,
    value: object,
    *,
    row: dict[str, object] | None = None,
) -> str:
    if _is_missing_hue_value(value):
        if style is not None and style.noncanonical_bucket is not None:
            return style.noncanonical_bucket
        return "NA"
    listlike_text = _format_listlike_hue_value(value)
    if listlike_text is not None:
        return listlike_text
    if style is not None:
        return normalize_axis_category(style, value, row=row)
    text = str(value)
    return text if text.strip() else "NA"


def _normalize_categorical_values_without_row_context(
    values: pd.Series,
    *,
    style: AxisStyle | None,
) -> pd.Series:
    try:
        unique_values = pd.unique(values)
        normalized_by_value = {
            value: _normalize_categorical_hue_value_with_style(style, value) for value in unique_values
        }
        normalized = values.map(normalized_by_value)
        missing = normalized.isna()
        if bool(missing.any()):
            normalized.loc[missing] = values.loc[missing].map(
                lambda value: _normalize_categorical_hue_value_with_style(style, value)
            )
        return normalized.astype(str)
    except TypeError:
        return values.map(lambda value: _normalize_categorical_hue_value_with_style(style, value)).astype(str)


def _selector_match_mask(frame: pd.DataFrame, selector: dict[str, object]) -> pd.Series | None:
    column = str(selector.get("column") or "").strip()
    if not column or column not in frame.columns:
        return None
    value_text = frame[column].astype(str)
    if selector.get("equals") is not None:
        return value_text == str(selector.get("equals"))
    in_values = selector.get("in_values") or []
    if in_values:
        return value_text.isin({str(item) for item in in_values})
    if bool(selector.get("non_null", False)):
        return frame[column].notna() & frame[column].astype(str).str.strip().ne("")
    return pd.Series(False, index=frame.index)


def _canonical_scope_mask(frame: pd.DataFrame, *, style: AxisStyle) -> pd.Series:
    selector_masks = [
        mask
        for selector in style.canonical_row_selectors
        if (mask := _selector_match_mask(frame, selector)) is not None
    ]
    if not selector_masks:
        return pd.Series(True, index=frame.index)
    stacked = pd.concat(selector_masks, axis=1)
    if style.canonical_row_match == "all":
        return stacked.all(axis=1)
    return stacked.any(axis=1)


def normalize_categorical_hue_series(
    column: str | None,
    values: pd.Series,
    *,
    axis_styles: dict[str, object] | None = None,
    frame: pd.DataFrame | None = None,
) -> pd.Series:
    style = _axis_style(axis_styles, column)
    normalized = _normalize_categorical_values_without_row_context(values, style=style)
    if frame is None or style is None or style.noncanonical_bucket is None or not style.canonical_row_selectors:
        return normalized
    scoped_frame = frame.reindex(values.index)
    scope_mask = _canonical_scope_mask(scoped_frame, style=style)
    if bool(scope_mask.all()):
        return normalized
    missing_mask = _scalar_missing_mask(values)
    if missing_mask is None:
        missing_mask = values.map(_is_missing_hue_value)
    normalized.loc[(~scope_mask) & (~missing_mask)] = style.noncanonical_bucket
    return normalized


def resolve_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> tuple[str, str] | None:
    candidates = candidate_join_keys(left, right)
    if not candidates:
        return None
    return candidates[0]


def candidate_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[tuple[str, str]]:
    return candidate_join_key_pairs_for_columns(left.columns, right.columns)


def shared_join_key(left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    resolved = resolve_join_keys(left, right)
    if resolved is None:
        return None
    left_key, right_key = resolved
    if left_key == right_key:
        return left_key
    return None


def geometry_map(geometry_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["view_id"]): row for row in geometry_rows}


def _existing_parquet_columns(path: Path, requested_columns: list[str] | None) -> list[str] | None:
    if requested_columns is None:
        return None
    existing = set(pq.read_schema(path).names)
    return [column for column in dict.fromkeys(requested_columns) if column in existing]


def load_view_rows(view_id: str, *, output_root: Path, columns: list[str] | None = None) -> pd.DataFrame:
    view_dir = output_root / "views" / view_id
    load_view_manifest(view_id, output_root=output_root)
    rows_path = view_dir / "rows.parquet"
    if not rows_path.is_file():
        raise ValueError(f"view rows artifact is missing for `{view_id}`")
    return load_table(rows_path, columns=_existing_parquet_columns(rows_path, columns))


def load_view_matrix(view_id: str, *, output_root: Path) -> np.ndarray:
    view_dir = output_root / "views" / view_id
    load_view_manifest(view_id, output_root=output_root)
    matrix_path = view_dir / "matrix.npy"
    if not matrix_path.is_file():
        raise ValueError(f"view matrix artifact is missing for `{view_id}`")
    return np.load(matrix_path, mmap_mode="r")


def load_view_manifest(view_id: str, *, output_root: Path) -> dict[str, object]:
    return load_artifact_manifest(output_root / "views" / view_id, artifact_kind="view", artifact_id=view_id)


def include_hue_column(column: str, artifact_suffixes: set[str] | None = None) -> bool:
    blocked = {"x", "y", "left_count", "right_count", "left_indices", "right_indices"}
    if column in blocked:
        return False
    if column.startswith(("left_count__", "right_count__", "left_indices__", "right_indices__")):
        return False
    if column.startswith("cluster_label__"):
        return True
    if artifact_suffixes:
        suffix = column.rsplit("__", 1)[-1]
        if suffix in artifact_suffixes:
            return False
    return True


def candidate_hue_columns(
    frame: pd.DataFrame, preferred: list[str], artifact_suffixes: set[str] | None = None
) -> list[str]:
    if frame.empty:
        return []
    return [column for column in preferred if column in frame.columns and include_hue_column(column, artifact_suffixes)]


def normalize_hue_kind(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"categorical", "binary", "continuous", "ordinal"}:
        return text
    return None


def _listlike_hue_values(value: object) -> list[object] | None:
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=lambda item: str(item))
    return None


def _is_missing_hue_value(value: object) -> bool:
    listlike = _listlike_hue_values(value)
    if listlike is not None:
        return not listlike or all(_is_missing_hue_value(item) for item in listlike)
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _format_hue_token(value: object) -> str:
    text = str(value or "").strip()
    sigma_match = re.fullmatch(r"sigma(\d+)", text, flags=re.IGNORECASE)
    if sigma_match:
        return f"Sigma{sigma_match.group(1)}"
    return humanize_display_text(value)


def _format_listlike_hue_value(value: object) -> str | None:
    listlike = _listlike_hue_values(value)
    if listlike is None:
        return None
    parts = [_format_hue_token(item) for item in listlike if not _is_missing_hue_value(item)]
    compact_parts = [part for part in parts if part]
    return " + ".join(compact_parts) if compact_parts else "NA"


def _finite_non_null_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=object)
    series = frame[column]
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.replace([np.inf, -np.inf], np.nan).dropna()
    missing_mask = _scalar_missing_mask(series)
    if missing_mask is not None:
        return series.loc[~missing_mask]
    mask = [not _is_missing_hue_value(value) for value in series.tolist()]
    if not any(mask):
        return pd.Series(dtype=series.dtype)
    return series.loc[mask]


def finite_non_null_hue_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return _finite_non_null_series(frame, column)


def available_hues_for_frames(
    frames: list[pd.DataFrame],
    *,
    preferred_hues: list[str],
    hue_kinds: dict[str, str],
) -> list[str]:
    if not frames:
        return []
    nonempty_frames = [frame for frame in frames if not frame.empty]
    if not nonempty_frames:
        return []
    available: list[str] = []
    for hue in preferred_hues:
        kind = normalize_hue_kind(hue_kinds.get(hue))
        if kind is None:
            continue
        continuous_values: list[pd.Series] = []
        supported = True
        for frame in nonempty_frames:
            series = _finite_non_null_series(frame, hue)
            if series.empty:
                supported = False
                break
            if kind == "continuous":
                numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if numeric.empty:
                    supported = False
                    break
                continuous_values.append(numeric)
        if not supported:
            continue
        if kind == "continuous":
            combined = pd.concat(continuous_values, ignore_index=True) if continuous_values else pd.Series(dtype=float)
            if combined.nunique() < 2:
                continue
        available.append(hue)
    return available


def available_reference_hues_for_frames(
    frames: list[pd.DataFrame],
    *,
    preferred_hues: list[str],
    hue_kinds: dict[str, str],
    reference_labels: list[str],
    reference_match_column: str = "usr_label__primary",
    axis_styles: dict[str, object] | None = None,
    x_column: str = "x",
    y_column: str = "y",
) -> list[str]:
    available: list[str] = []
    for hue in preferred_hues:
        hue_column = str(hue or "").strip()
        if not hue_column:
            continue
        hue_kind = normalize_hue_kind(hue_kinds.get(hue_column)) or "continuous"
        if (
            reference_hue_render_params(
                frames,
                reference_labels=reference_labels,
                reference_match_column=reference_match_column,
                reference_hue_column=hue_column,
                reference_hue_kind=hue_kind,
                axis_styles=axis_styles,
                x_column=x_column,
                y_column=y_column,
            )
            is not None
        ):
            available.append(hue_column)
    return available


def classify_hue_series(series: pd.Series, *, configured_kind: object = None) -> str:
    explicit_kind = normalize_hue_kind(configured_kind)
    if explicit_kind is not None:
        return explicit_kind
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if (
        pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "continuous"
    return "categorical"


def continuous_hue_render_params(column: str | None, values: pd.Series) -> dict[str, object]:
    from matplotlib import colors as mcolors

    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {"cmap": "viridis", "norm": None, "vmin": None, "vmax": None}

    column_name = str(column or "").strip()
    if "margin" in column_name:
        robust_limit = float(np.nanquantile(np.abs(numeric.to_numpy(dtype=float)), 0.99))
        if not np.isfinite(robust_limit) or robust_limit <= 0.0:
            robust_limit = float(np.nanmax(np.abs(numeric.to_numpy(dtype=float))))
        robust_limit = max(robust_limit, 1e-6)
        return {
            "cmap": "coolwarm",
            "norm": mcolors.TwoSlopeNorm(vmin=-robust_limit, vcenter=0.0, vmax=robust_limit),
            "vmin": None,
            "vmax": None,
        }

    lower = float(np.nanquantile(numeric.to_numpy(dtype=float), 0.01))
    upper = float(np.nanquantile(numeric.to_numpy(dtype=float), 0.99))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower = float(numeric.min())
        upper = float(numeric.max())
    return {
        "cmap": "viridis",
        "norm": None,
        "vmin": lower,
        "vmax": upper,
    }


def _reference_categorical_hue_series(values: pd.Series) -> pd.Series:
    labels: list[str] = []
    for value in values.tolist():
        if _is_missing_hue_value(value):
            labels.append("")
            continue
        labels.append(_format_listlike_hue_value(value) or _format_hue_token(value))
    return pd.Series(labels, index=values.index, dtype="string")


def selected_reference_rows(
    frame: pd.DataFrame,
    *,
    reference_labels: list[str],
    reference_match_column: str = "usr_label__primary",
    x_column: str = "x",
    y_column: str = "y",
) -> pd.DataFrame:
    match_column = str(reference_match_column or "usr_label__primary")
    if frame.empty or match_column not in frame.columns:
        return pd.DataFrame()
    if x_column not in frame.columns or y_column not in frame.columns:
        return pd.DataFrame()
    targets = {normalize_label(label) for label in reference_labels}
    if not targets:
        return pd.DataFrame()
    match_values = frame[match_column].astype("string").str.strip().str.lower()
    selected = frame[match_values.isin(targets)].copy()
    if selected.empty:
        return selected
    finite_xy = (
        pd.to_numeric(selected[x_column], errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
        & pd.to_numeric(selected[y_column], errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
    )
    return selected.loc[finite_xy].copy()


def reference_hue_render_params(
    frames: list[pd.DataFrame],
    *,
    reference_labels: list[str],
    reference_match_column: str = "usr_label__primary",
    reference_hue_column: str | None = None,
    reference_hue_kind: str | None = None,
    axis_styles: dict[str, object] | None = None,
    x_column: str = "x",
    y_column: str = "y",
) -> dict[str, object] | None:
    hue_column = str(reference_hue_column or "").strip()
    if not hue_column:
        return None
    hue_kind = normalize_hue_kind(reference_hue_kind) or "continuous"
    series_parts: list[pd.Series] = []
    for frame in frames:
        if hue_column not in frame.columns:
            continue
        selected = selected_reference_rows(
            frame,
            reference_labels=reference_labels,
            reference_match_column=reference_match_column,
            x_column=x_column,
            y_column=y_column,
        )
        if selected.empty or hue_column not in selected.columns:
            continue
        if hue_kind == "continuous":
            numeric = pd.to_numeric(selected[hue_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not numeric.empty:
                series_parts.append(numeric)
        else:
            categorical = _reference_categorical_hue_series(selected[hue_column])
            categorical = categorical.loc[categorical.astype(str).str.strip() != ""]
            if not categorical.empty:
                series_parts.append(categorical)
    if not series_parts:
        return None
    combined = pd.concat(series_parts, ignore_index=True)
    if hue_kind != "continuous":
        categories = category_values_for_legend(
            [str(value) for value in combined.tolist() if str(value).strip()],
            column=hue_column,
            axis_styles=axis_styles,
        )
        if not categories:
            return None
        return {
            "kind": hue_kind,
            "categories": categories,
            "color_map": category_color_map(categories, column=hue_column, axis_styles=axis_styles),
        }
    if combined.nunique() < 2:
        return None
    return {"kind": "continuous", **continuous_hue_render_params(hue_column, combined)}


def reference_annotation_mode_options() -> dict[str, str]:
    return {
        "Auto labels": "auto",
        "Markers only": "markers_only",
        "Show text labels": "show_labels",
    }


def reference_label_limit_for_annotation_mode(mode: str | None) -> int | None:
    normalized = str(mode or "auto").strip()
    if normalized == "markers_only":
        return 0
    if normalized == "show_labels":
        return -1
    return None


def reference_hue_coverage_warnings(
    frames: list[pd.DataFrame],
    *,
    reference_labels: list[str],
    reference_match_column: str = "usr_label__primary",
    reference_hue_column: str | None = None,
    reference_hue_kind: str | None = None,
    x_column: str = "x",
    y_column: str = "y",
) -> list[str]:
    hue_column = str(reference_hue_column or "").strip()
    if not hue_column:
        return []
    hue_kind = normalize_hue_kind(reference_hue_kind) or "continuous"
    missing_column_panels: list[str] = []
    empty_hue_panels: list[str] = []
    for index, frame in enumerate(frames):
        selected = selected_reference_rows(
            frame,
            reference_labels=reference_labels,
            reference_match_column=reference_match_column,
            x_column=x_column,
            y_column=y_column,
        )
        if selected.empty:
            continue
        panel_label = str(
            frame.attrs.get("view_id")
            or frame.attrs.get("projection_id")
            or frame.attrs.get("scalar_id")
            or f"panel {index + 1}"
        )
        if hue_column not in selected.columns:
            missing_column_panels.append(panel_label)
            continue
        if hue_kind == "continuous":
            values = pd.to_numeric(selected[hue_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            has_values = bool(values.notna().any())
        else:
            values = _reference_categorical_hue_series(selected[hue_column])
            has_values = bool((values.astype(str).str.strip() != "").any())
        if not has_values:
            empty_hue_panels.append(panel_label)

    warnings: list[str] = []
    if missing_column_panels:
        warnings.append(
            f"reference hue `{hue_column}` is missing for selected reference rows in "
            f"{len(missing_column_panels)} panel(s): "
            + ", ".join(missing_column_panels[:4])
            + (" ..." if len(missing_column_panels) > 4 else "")
        )
    if empty_hue_panels:
        warnings.append(
            f"reference hue `{hue_column}` has no values for selected reference rows in "
            f"{len(empty_hue_panels)} panel(s): "
            + ", ".join(empty_hue_panels[:4])
            + (" ..." if len(empty_hue_panels) > 4 else "")
        )
    return warnings


def scatter_style(row_count: int) -> tuple[float, float]:
    style = shared_scatter_style(row_count)
    return style.point_size, style.alpha


MISSING_HUE_BACKGROUND_COLOR = "#CBD5E1"


def draw_missing_hue_background(
    ax,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    point_style,
    rasterize_panel: bool,
    raster_renderer,
) -> None:
    """Draw rows that lack the selected continuous hue without dropping the cloud."""
    if len(x_values) == 0:
        return
    alpha = max(float(point_style.alpha) * 0.64, 0.10)
    if rasterize_panel:
        raster_renderer(
            ax,
            x_values=x_values,
            y_values=y_values,
            color=MISSING_HUE_BACKGROUND_COLOR,
            point_alpha=alpha,
            min_point_alpha=BACKGROUND_RASTER_MIN_ALPHA,
        )
        return
    ax.scatter(
        x_values,
        y_values,
        c=MISSING_HUE_BACKGROUND_COLOR,
        s=point_style.point_size,
        alpha=alpha,
        linewidths=0.0,
        edgecolors="none",
        rasterized=point_style.rasterized,
        zorder=0.8,
    )


def category_color_map(
    categories: list[str],
    *,
    column: str | None = None,
    axis_styles: dict[str, object] | None = None,
) -> dict[str, str]:
    style = _axis_style(axis_styles, column)
    if style is not None:
        ordered = legend_categories(style, categories)
        return contrast_safe_scatter_color_map(axis_color_map(style, ordered, fallback_palette=PUBLICATION_PALETTE))
    return contrast_safe_scatter_color_map(
        axis_color_map(None, ordered_categories_for_axis(None, categories), fallback_palette=PUBLICATION_PALETTE)
    )


def category_values_for_legend(
    categories: list[str],
    *,
    column: str | None = None,
    axis_styles: dict[str, object] | None = None,
) -> list[str]:
    style = _axis_style(axis_styles, column)
    if style is not None:
        return legend_categories(style, categories)
    return ordered_categories_for_axis(None, categories)


def table_from_records(
    records: pd.DataFrame | list[dict[str, object]],
    *,
    columns: list[str] | None = None,
    page_size: int | None = None,
):
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=columns)
    if columns is not None:
        frame = frame.reindex(columns=columns)
    if page_size is None:
        return mo.ui.table(frame)
    return mo.ui.table(frame, page_size=page_size)


def key_value_table(
    rows: list[tuple[str, object]],
    *,
    field_name: str = "Field",
    value_name: str = "Value",
):
    normalized_rows = [{field_name: str(field), value_name: value} for field, value in rows]
    return table_from_records(
        normalized_rows,
        columns=[field_name, value_name],
        page_size=min(max(len(normalized_rows), 1), 12),
    )


def style_notebook_axes(ax, *, grid: bool = True, square: bool = False) -> None:
    ax.set_facecolor(PANEL_BACKGROUND_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(colors=TEXT_COLOR, labelsize=PLOT_TICK_FONT_SIZE, length=4.5, width=0.8, direction="out")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_fontsize(PLOT_LABEL_FONT_SIZE)
    ax.yaxis.label.set_fontsize(PLOT_LABEL_FONT_SIZE)
    ax.title.set_color(TEXT_COLOR)
    ax.title.set_fontsize(PLOT_TITLE_FONT_SIZE)
    ax.title.set_fontweight("semibold")
    ax.title.set_fontfamily(PLOT_FONT_FAMILY)
    ax.margins(x=0.04, y=0.05)
    if square:
        ax.set_box_aspect(1)
    if grid:
        ax.grid(True, color=GRID_COLOR, linewidth=0.75, alpha=0.58)
        ax.set_axisbelow(True)


def style_notebook_legend(legend) -> None:
    if legend is None:
        return
    title = legend.get_title()
    if title is not None:
        title.set_visible(False)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
        text.set_fontsize(PLOT_LEGEND_FONT_SIZE)
        text.set_fontfamily(PLOT_FONT_FAMILY)


def draw_reference_labels(
    ax,
    frame: pd.DataFrame,
    *,
    reference_labels: list[str],
    reference_match_column: str = "usr_label__primary",
    reference_display_labels: dict[str, str] | None = None,
    reference_label_limit: int | None = None,
    x_column: str = "x",
    y_column: str = "y",
    right_padding_px: float = 0.0,
    left_padding_px: float = 0.0,
    reference_hue_column: str | None = None,
    reference_hue_params: dict[str, object] | None = None,
) -> None:
    match_column = str(reference_match_column or "usr_label__primary")
    selected = selected_reference_rows(
        frame,
        reference_labels=reference_labels,
        reference_match_column=match_column,
        x_column=x_column,
        y_column=y_column,
    )
    if selected.empty:
        return
    placed_boxes: list[tuple[float, float, float, float]] = []
    axes_box = ax.get_window_extent()
    display_x_mid = float((axes_box.x0 + axes_box.x1) / 2.0)
    display_y_mid = float((axes_box.y0 + axes_box.y1) / 2.0)
    hue_column = str(reference_hue_column or "").strip()
    can_color_by_reference_hue = (
        bool(hue_column)
        and reference_hue_params is not None
        and hue_column in selected.columns
        and (reference_hue_params.get("cmap") is not None or reference_hue_params.get("color_map") is not None)
    )
    if can_color_by_reference_hue:
        reference_hue_kind = normalize_hue_kind(reference_hue_params.get("kind")) or "continuous"
        if reference_hue_kind == "continuous":
            hue_values = pd.to_numeric(selected[hue_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite_hue = hue_values.notna()
            if (~finite_hue).any():
                ax.scatter(
                    selected.loc[~finite_hue, x_column].to_numpy(dtype=float),
                    selected.loc[~finite_hue, y_column].to_numpy(dtype=float),
                    c="#111111",
                    s=125,
                    marker="*",
                    linewidths=0.8,
                    edgecolors="white",
                    zorder=5,
                )
            if finite_hue.any():
                ax.scatter(
                    selected.loc[finite_hue, x_column].to_numpy(dtype=float),
                    selected.loc[finite_hue, y_column].to_numpy(dtype=float),
                    c=hue_values.loc[finite_hue].to_numpy(dtype=float),
                    cmap=str(reference_hue_params["cmap"]),
                    norm=reference_hue_params.get("norm"),
                    vmin=None if reference_hue_params.get("norm") is not None else reference_hue_params.get("vmin"),
                    vmax=None if reference_hue_params.get("norm") is not None else reference_hue_params.get("vmax"),
                    s=132,
                    marker="*",
                    linewidths=0.85,
                    edgecolors="white",
                    zorder=5,
                )
        else:
            hue_values = _reference_categorical_hue_series(selected[hue_column])
            finite_hue = hue_values.astype(str).str.strip() != ""
            color_map = {
                str(key): str(value) for key, value in dict(reference_hue_params.get("color_map") or {}).items()
            }
            if (~finite_hue).any():
                ax.scatter(
                    selected.loc[~finite_hue, x_column].to_numpy(dtype=float),
                    selected.loc[~finite_hue, y_column].to_numpy(dtype=float),
                    c="#111111",
                    s=125,
                    marker="*",
                    linewidths=0.8,
                    edgecolors="white",
                    zorder=5,
                )
            if finite_hue.any():
                colors = [color_map.get(str(value), "#111111") for value in hue_values.loc[finite_hue].tolist()]
                ax.scatter(
                    selected.loc[finite_hue, x_column].to_numpy(dtype=float),
                    selected.loc[finite_hue, y_column].to_numpy(dtype=float),
                    c=colors,
                    s=132,
                    marker="*",
                    linewidths=0.85,
                    edgecolors="white",
                    zorder=5,
                )
    else:
        ax.scatter(
            selected[x_column].to_numpy(dtype=float),
            selected[y_column].to_numpy(dtype=float),
            c="#111111",
            s=125,
            marker="*",
            linewidths=0.8,
            edgecolors="white",
            zorder=5,
        )
    label_rows = selected.sort_values(match_column)
    if reference_label_limit is not None and reference_label_limit >= 0:
        label_rows = label_rows.head(reference_label_limit)
    display_labels = reference_display_labels or {}
    label_font_size = 7.4 if len(label_rows) > 5 else PLOT_TICK_FONT_SIZE
    label_font_weight = "medium" if len(label_rows) > 5 else "semibold"
    label_box_padding = 0.12 if len(label_rows) > 5 else 0.18
    for row in label_rows.to_dict(orient="records"):
        point_x = float(row[x_column])
        point_y = float(row[y_column])
        match_value = str(row.get(match_column) or "")
        label = display_reference_label(display_labels.get(match_value, match_value))
        display_x, display_y = ax.transData.transform((point_x, point_y))
        placement = choose_annotation_placement(
            display_x=display_x,
            display_y=display_y,
            label_text=label,
            axes_box=axes_box,
            placed_boxes=placed_boxes,
            x_mid=display_x_mid,
            y_mid=display_y_mid,
            font_size=label_font_size,
            left_padding_px=left_padding_px,
            right_padding_px=right_padding_px,
        )
        placed_boxes.append(placement.box)
        annotation = ax.annotate(
            label,
            xy=(point_x, point_y),
            xytext=(placement.offset_x, placement.offset_y),
            textcoords="offset pixels",
            fontsize=label_font_size,
            fontweight=label_font_weight,
            ha=placement.ha,
            va=placement.va,
            color=TEXT_COLOR,
            bbox={
                "boxstyle": f"round,pad={label_box_padding}",
                "fc": "white",
                "ec": "none",
                "alpha": ANNOTATION_LABEL_BOX_ALPHA,
            },
            arrowprops={"arrowstyle": "-", "color": SPINE_COLOR, "linewidth": 0.9},
            zorder=6,
        )
        annotation.set_clip_on(True)
        if annotation.arrow_patch is not None:
            annotation.arrow_patch.set_clip_on(True)

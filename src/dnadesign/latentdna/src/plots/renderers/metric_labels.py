"""Tick-label contracts for metric-panel plot renderers."""

from __future__ import annotations

import re
from typing import Any

from ...labels import humanize_candidate
from ...visual_style import PLOT_TICK_FONT_SIZE, humanize_display_text, wrap_plot_title


def wrapped_metric_tick_label(value: object, *, width: int = 16, max_lines: int | None = None) -> str:
    return wrap_plot_title(humanize_display_text(str(value)), width=width, max_lines=max_lines)


def _candidate_row_label(
    row: dict[str, object],
    *,
    fallback_column: str,
    include_family: bool = True,
) -> str:
    candidate_fields = {
        key: str(row.get(key) or "").strip()
        for key in ("candidate_model", "candidate_scope", "candidate_family")
        if str(row.get(key) or "").strip()
    }
    if not include_family:
        candidate_fields.pop("candidate_family", None)
    if candidate_fields:
        return humanize_candidate(candidate_fields)
    return humanize_display_text(str(row.get(fallback_column) or ""))


def _short_candidate_model(value: object) -> str:
    text = humanize_display_text(value)
    normalized = text.casefold()
    if "20b" in normalized:
        return "20B"
    if "7b" in normalized:
        return "7B"
    return text


def _short_candidate_scope(value: object) -> str:
    text = humanize_display_text(value)
    normalized = text.casefold()
    if normalized in {
        "60 bp anchor",
        "anchor-source insert",
        "merged anchor insert seq mean",
        "mixed-length anchor-source insert",
        "anchor-source insert mean",
    }:
        return "anchor insert"
    if normalized == "1 kb construct context":
        return "1 kb ctx"
    if normalized in {"1 kb context anchor mean", "full context anchor mean"}:
        return "1 kb anchor mean"
    if normalized in {"context anchor mean bidir concat", "context anchor mean bidirectional concat"}:
        return "bidir anchor mean"
    if normalized == "reverse complement context 1 kb":
        return "RC 1 kb ctx"
    if normalized == "reverse complement context anchor mean":
        return "RC 1 kb anchor mean"
    if normalized == "reference core60":
        return "ref core60"
    if normalized == "reference context forward 1 kb":
        return "ref forward 1 kb"
    if normalized == "reference context forward anchor mean":
        return "ref forward anchor mean"
    if normalized == "reference context reverse complement 1 kb":
        return "ref RC 1 kb"
    if normalized == "reference context reverse complement anchor mean":
        return "ref RC anchor mean"
    if normalized == "anchor + anchor-mean concat":
        return "anchor + anchor-mean"
    if normalized == "anchor + 1 kb context concat":
        return "anchor + 1 kb ctx"
    if normalized == "native source record":
        return "native 81 bp"
    if normalized in {"core60 tss upstream", "core60 tss-upstream"}:
        return "core60 TSS"
    return text


def _compact_candidate_scope(value: object) -> str:
    text = humanize_display_text(value)
    normalized = text.casefold()
    if normalized in {
        "60 bp anchor",
        "anchor-source insert",
        "merged anchor insert seq mean",
        "mixed-length anchor-source insert",
        "anchor-source insert mean",
    }:
        return "anchor insert"
    if normalized == "1 kb construct context":
        return "1kb ctx"
    if normalized in {"1 kb context anchor mean", "full context anchor mean"}:
        return "1kb anchor mean"
    if normalized in {"context anchor mean bidir concat", "context anchor mean bidirectional concat"}:
        return "bidir anchor"
    if normalized == "reverse complement context 1 kb":
        return "RC 1kb ctx"
    if normalized == "reverse complement context anchor mean":
        return "RC 1kb anchor"
    if normalized == "reference core60":
        return "ref core60"
    if normalized == "reference context forward 1 kb":
        return "ref fwd 1kb"
    if normalized == "reference context forward anchor mean":
        return "ref fwd anchor"
    if normalized == "reference context reverse complement 1 kb":
        return "ref RC 1kb"
    if normalized == "reference context reverse complement anchor mean":
        return "ref RC anchor"
    if normalized == "anchor + anchor-mean concat":
        return "anchor+anchor-mean"
    if normalized == "anchor + 1 kb context concat":
        return "anchor+1kb ctx"
    if normalized == "native source record":
        return "native81"
    if normalized in {"core60 tss upstream", "core60 tss-upstream"}:
        return "core60 TSS"
    return text


def _short_candidate_family(value: object) -> str:
    text = humanize_display_text(value)
    normalized = text.casefold()
    if normalized == "intermediate block mean":
        return "Block"
    if normalized == "output-layer mean":
        return "Output"
    return text


def _normalized_context_label(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    for prefix in (
        "7b intermediate: anchor vs ",
        "20b intermediate: anchor vs ",
        "7b output: anchor vs ",
        "20b output: anchor vs ",
        "anchor vs ",
    ):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def compact_context_comparison_label(value: object) -> str | None:
    """Return compact labels for anchor-vs-context comparison ticks."""

    normalized = _normalized_context_label(value)
    if normalized in {"context anchor mean", "1 kb anchor mean", "1kb anchor mean"}:
        return "Fwd\nanchor"
    if normalized in {
        "full 1 kb context",
        "1 kb sequence mean",
        "1kb sequence mean",
        "context full sequence mean",
    }:
        return "Fwd\nseq"
    if normalized in {
        "rc context anchor mean",
        "rc 1 kb anchor mean",
        "reverse complement anchor mean",
        "reverse complement context anchor mean",
    }:
        return "RC\nanchor"
    if normalized in {
        "rc full 1 kb context",
        "rc 1 kb sequence mean",
        "reverse complement full sequence mean",
        "reverse complement context sequence mean",
    }:
        return "RC\nseq"
    return None


def candidate_tick_label(
    row: dict[str, object],
    *,
    fallback_column: str,
    plot_id: str | None = None,
    include_family: bool = True,
    include_scope: bool = True,
    multiline: bool = True,
    force_fallback: bool = False,
) -> str:
    fallback_value = row.get(fallback_column) or row.get("candidate_label") or ""
    if plot_id in {"context_pair_summary", "context_robustness_summary"}:
        for context_value in (
            row.get("candidate_label"),
            fallback_value,
            row.get("comparison_label"),
            row.get("label"),
        ):
            compact_context = compact_context_comparison_label(context_value)
            if compact_context is not None:
                return compact_context

    model = str(row.get("candidate_model") or "").strip()
    scope = str(row.get("candidate_scope") or "").strip()
    family = str(row.get("candidate_family") or "").strip()
    if model and not force_fallback:
        short_model = _short_candidate_model(model)
        parts = [short_model]
        if include_scope and scope:
            scope_label = _compact_candidate_scope(scope) if not multiline else _short_candidate_scope(scope)
            parts[-1] = f"{short_model} {scope_label}"
        if include_family and family:
            short_family = _short_candidate_family(family)
            parts.append(short_family)
        separator = "\n" if multiline else " "
        return separator.join(part for part in parts if part)

    if force_fallback:
        base_label = humanize_display_text(str(fallback_value))
    else:
        base_label = _candidate_row_label(
            row,
            fallback_column=fallback_column,
            include_family=include_family,
        )
    if multiline:
        return wrapped_metric_tick_label(base_label, width=12, max_lines=4)
    return base_label


def _compact_multiline_labels(labels: list[str]) -> bool:
    if not labels:
        return False
    return all("\n" in str(label) and max(len(line) for line in str(label).splitlines()) <= 12 for label in labels)


def metric_tick_labels_need_rotation(labels: list[str], *, grouped_family_bars: bool, plot_id: str | None) -> bool:
    """Return whether metric-panel x tick labels need angled placement."""

    if grouped_family_bars:
        return True
    if _compact_multiline_labels(labels):
        return False
    if plot_id == "context_pair_summary":
        return True
    if not labels:
        return False
    max_line_length = max(len(line) for label in labels for line in str(label).splitlines())
    return len(labels) <= 6 and max_line_length > 14


def style_metric_tick_labels(
    ax: Any,
    *,
    label_count: int,
    axis: str = "x",
    rotation: float = 0.0,
    ha: str | None = None,
    va: str | None = None,
) -> None:
    if label_count >= 8:
        font_size = PLOT_TICK_FONT_SIZE - 3.1
    elif label_count >= 6:
        font_size = PLOT_TICK_FONT_SIZE - 1.4
    else:
        font_size = PLOT_TICK_FONT_SIZE - 0.6
    if axis == "x" and rotation:
        font_size -= 0.8
    tick_labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    default_ha = "right" if axis == "y" or rotation else "center"
    default_va = "center" if axis == "y" else "top"
    for label in tick_labels:
        label.set_fontsize(font_size)
        label.set_linespacing(0.95)
        label.set_multialignment("right" if rotation else "center")
        label.set_rotation(rotation)
        label.set_rotation_mode("anchor")
        label.set_ha(ha or default_ha)
        label.set_va(va or default_va)

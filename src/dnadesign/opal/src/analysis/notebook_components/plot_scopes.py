"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/plot_scopes.py

Notebook component builders for plot scopes OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ._support import compact_path, display_name, first_media_output, mapping, sequence
from .plot_text import plot_alt_text, rounds_text


def build_notebook_plot_scope_options(choice: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return manifest-backed scope choices for a selected visual plot."""

    options = sequence(mapping(choice).get("scope_options")) or [choice]
    control_label = plot_scope_control_label(options)
    rows: list[dict[str, Any]] = []
    for option in options:
        if not isinstance(option, Mapping):
            continue
        rows.append(
            {
                "label": str(option.get("scope_label") or plot_scope_label(option)),
                "control_label": control_label,
                "rounds": option.get("rounds"),
                "run_id": option.get("run_id"),
                "path": option.get("path"),
                "path_label": option.get("path_label"),
                "freshness": option.get("freshness"),
            }
        )
    return rows


def select_notebook_plot_scope(choice: Mapping[str, Any], scope_label: str | None) -> dict[str, Any]:
    """Select one concrete manifest-backed plot scope from a plot-level choice."""

    options = sequence(mapping(choice).get("scope_options")) or [choice]
    if not options:
        raise ValueError("Plot choice has no manifest-backed scope options.")
    if scope_label in (None, ""):
        return dict(options[0])
    for option in options:
        if not isinstance(option, Mapping):
            continue
        if str(option.get("scope_label") or plot_scope_label(option)) == str(scope_label):
            return dict(option)
    labels = [
        str(option.get("scope_label") or plot_scope_label(option)) for option in options if isinstance(option, Mapping)
    ]
    raise ValueError(f"Plot scope selection not found: {scope_label}. Available: {labels}")


def plot_choice_from_manifest(
    *,
    entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    workdir: str,
    label: str,
    capability: Mapping[str, Any],
) -> dict[str, Any] | None:
    media_output = first_media_output(manifest)
    if media_output is None:
        return None
    path = str(media_output.get("path"))
    title = str(
        entry.get("title")
        or manifest.get("title")
        or mapping(manifest.get("params")).get("title")
        or display_name(entry.get("name") or manifest.get("name"))
    )
    warnings = sequence(manifest.get("warnings"))
    tidy_csv = manifest.get("tidy_csv")
    inputs = [
        item
        for item in sequence(manifest.get("inputs"))
        if isinstance(item, Mapping) and (item.get("path") or item.get("role"))
    ]
    summary = str(
        manifest.get("caption")
        or manifest.get("review_purpose")
        or mapping(manifest.get("metadata")).get("summary")
        or ""
    )
    freshness = mapping(manifest.get("freshness"))
    freshness_status = str(freshness.get("status") or manifest.get("stale_state") or "unknown")
    return {
        "label": str(label),
        "scope_label": plot_scope_label(manifest),
        "title": title,
        "name": str(entry.get("name") or manifest.get("name") or ""),
        "kind": entry.get("kind") or manifest.get("kind") or "unknown",
        "path": path,
        "workdir": workdir,
        "path_label": compact_path(path, base=workdir),
        "filename": Path(path).name,
        "tidy_label": compact_path(tidy_csv, base=workdir) if tidy_csv else "none",
        "source_labels": [
            f"{item.get('role') or 'input'}={compact_path(item.get('path'), base=workdir)}" for item in inputs[:5]
        ],
        "freshness": freshness_status,
        "status": manifest.get("status"),
        "run_id": manifest.get("run_id"),
        "rounds": manifest.get("rounds"),
        "capability": dict(capability),
        "warning_count": len(warnings),
        "caption": summary,
        "alt_text": plot_alt_text(
            title=title,
            kind=entry.get("kind") or manifest.get("kind"),
            summary=summary,
            params=manifest.get("params"),
            metadata=manifest.get("metadata"),
            rounds=manifest.get("rounds"),
            run_id=manifest.get("run_id"),
            freshness=freshness.get("status") or manifest.get("stale_state"),
            warning_count=len(warnings),
        ),
        "entry": dict(entry),
        "manifest": dict(manifest),
    }


def sort_plot_scope_manifests(manifests: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted([manifest for manifest in manifests if isinstance(manifest, Mapping)], key=plot_scope_sort_key)


def plot_scope_sort_key(manifest: Mapping[str, Any]) -> tuple[int, int, str]:
    rounds = manifest.get("rounds")
    if rounds == "all":
        return (0, -1, "")
    if rounds == "latest":
        return (1, -1, "")
    items = sequence(rounds)
    if len(items) == 1:
        try:
            return (2, -int(items[0]), "")
        except Exception:
            return (2, 0, str(items[0]))
    if items:
        try:
            return (3, min(int(item) for item in items), ",".join(str(item) for item in items))
        except Exception:
            return (3, 0, ",".join(str(item) for item in items))
    return (4, 0, str(rounds or ""))


def dedupe_scope_labels(choices: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for choice in choices:
        option = dict(choice)
        label = str(option.get("scope_label") or plot_scope_label(option))
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label} ({option.get('filename') or option.get('path_label')})"
        option["scope_label"] = label
        options.append(option)
    return options


def plot_scope_label(value: Mapping[str, Any]) -> str:
    return rounds_text(value.get("rounds"))


def plot_scope_control_label(values: Iterable[Mapping[str, Any]]) -> str:
    options = [value for value in values if isinstance(value, Mapping)]
    if not options:
        return "Plot scope"
    if all(_is_single_round_scope(option.get("rounds")) for option in options):
        has_run_scope = any(option.get("run_id") not in (None, "") for option in options)
        return "Round/run" if has_run_scope else "Round"
    return "Plot scope"


def _is_single_round_scope(value: Any) -> bool:
    items = sequence(value)
    if len(items) != 1:
        return False
    try:
        int(items[0])
    except Exception:
        return False
    return True

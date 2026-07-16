"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/plots.py

Notebook component builders for plots OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ...registries.plots import describe_plot_kind
from ._support import (
    compact_path,
    display_name,
    first_media_output,
    mapping,
    plot_entries_from_manifests,
    sequence,
)
from .layered_scatter import (
    build_notebook_layered_scatter_contract,
    render_notebook_layered_scatter_image,
)
from .plot_scopes import (
    dedupe_scope_labels,
    plot_choice_from_manifest,
    sort_plot_scope_manifests,
)
from .plot_text import rounds_text


def build_notebook_visual_surface_model(
    view_model: Mapping[str, Any],
    *,
    selection_view_id: str | None = None,
    plot_entries: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build manifest-authoritative visual choices for OPAL marimo templates."""

    campaign = mapping(view_model.get("campaign"))
    workdir = campaign.get("workdir") or ""
    plots_dir = str(Path(str(workdir)) / "outputs" / "plots") if workdir else "outputs/plots"
    manifest_rows = [
        manifest
        for manifest in sequence(view_model.get("plot_manifests"))
        if isinstance(manifest, Mapping) and manifest.get("status") == "written"
    ]
    if selection_view_id is not None:
        missing_view_identity = [row.get("manifest_path") for row in manifest_rows if not row.get("selection_view_id")]
        if missing_view_identity:
            raise ValueError(
                "Written plot manifests require selection_view_id: "
                + ", ".join(str(path) for path in missing_view_identity)
            )
        manifest_rows = [row for row in manifest_rows if str(row.get("selection_view_id")) == str(selection_view_id)]
    active_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for row in manifest_rows:
        active_by_name.setdefault(str(row.get("name")), []).append(row)
    if plot_entries is None:
        configured_entries = list(sequence(view_model.get("configured_plots"))) or plot_entries_from_manifests(
            manifest_rows
        )
    else:
        configured_entries = list(plot_entries)

    choices: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    missing_outputs: list[str] = []
    labels_seen: dict[str, int] = {}
    for entry in configured_entries:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "")
        if not name:
            continue
        manifests = sort_plot_scope_manifests(active_by_name.get(name) or [])
        manifest = manifests[0] if manifests else None
        kind = str(entry.get("kind") or mapping(manifest).get("kind") or "unknown")
        metadata = _plot_kind_metadata(kind)
        capability = mapping(metadata.get("capability"))
        if manifest is None:
            missing_outputs.append(name)
            inventory.append(
                _plot_inventory_entry(
                    entry=entry,
                    manifest=None,
                    status="configured_missing_output",
                    workdir=workdir,
                    capability=capability,
                )
            )
            continue
        media_output = first_media_output(manifest)
        scope_choices = [
            scope_choice
            for scope_choice in (
                plot_choice_from_manifest(
                    entry=entry,
                    manifest=scope_manifest,
                    workdir=workdir,
                    label=display_name(name),
                    capability=capability,
                )
                for scope_manifest in manifests
            )
            if scope_choice is not None
        ]
        path = str(scope_choices[0].get("path") or "") if scope_choices else ""
        freshness = mapping(manifest.get("freshness"))
        freshness_status = str(freshness.get("status") or manifest.get("stale_state") or "unknown")
        if media_output is None or not scope_choices:
            missing_outputs.append(name)
            inventory.append(
                _plot_inventory_entry(
                    entry=entry,
                    manifest=manifest,
                    status="generated_missing_media",
                    workdir=workdir,
                    capability=capability,
                )
            )
            continue
        params = mapping(manifest.get("params"))
        title = str(entry.get("title") or manifest.get("title") or params.get("title") or display_name(name))
        label = str(params.get("surface_label") or params.get("display_label") or title)
        labels_seen[label] = labels_seen.get(label, 0) + 1
        if labels_seen[label] > 1:
            label = f"{label} ({Path(path).name})"
        for scope_choice in scope_choices:
            scope_choice["label"] = label
            scope_choice["title"] = title
        choice = dict(scope_choices[0])
        choice["scope_options"] = dedupe_scope_labels(scope_choices)
        choice["scope_count"] = len(scope_choices)
        choices.append(choice)
        inventory.append(
            _plot_inventory_entry(
                entry=entry,
                manifest=manifest,
                status=_generated_inventory_status(freshness_status),
                workdir=workdir,
                capability=capability,
                media_path=path,
            )
        )
    stale_artifacts = list(sequence(view_model.get("stale_artifacts")))
    for artifact in stale_artifacts:
        if not isinstance(artifact, Mapping):
            continue
        artifact_path = str(artifact.get("path") or "")
        if not artifact_path:
            continue
        inventory.append(
            {
                "name": Path(artifact_path).stem,
                "kind": "unmanifested",
                "status": "stale_unmanifested",
                "rounds": "not recorded",
                "freshness": "stale",
                "path": artifact_path,
                "path_label": compact_path(artifact_path, base=workdir),
                "objective_family": "unknown",
                "data_layer": "generated_artifact",
                "round_scope": "not recorded",
                "label_requirement": "not recorded",
                "requires_model_artifact": False,
                "tidy_available": artifact_path.endswith(".csv"),
            }
        )
    return {
        "plots_dir": plots_dir,
        "choices": choices,
        "missing_outputs": missing_outputs,
        "stale_artifacts": stale_artifacts,
        "inventory": inventory,
        "inventory_status_counts": _inventory_status_counts(inventory),
    }


def build_notebook_plot_inventory_rows(visual_surface_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build rows that distinguish configured, generated, and stale plot surfaces."""

    rows: list[dict[str, Any]] = []
    for item in sequence(mapping(visual_surface_model).get("inventory")):
        if not isinstance(item, Mapping):
            continue
        rows.append(
            {
                "plot": item.get("name") or "not recorded",
                "kind": item.get("kind") or "not recorded",
                "status": item.get("status") or "unknown",
                "rounds": rounds_text(item.get("rounds")),
                "objective": item.get("objective_family") or "unknown",
                "data": item.get("data_layer") or "unspecified",
                "round behavior": item.get("round_scope") or "unspecified",
                "labels": item.get("label_requirement") or "none",
                "model artifact": bool(item.get("requires_model_artifact")),
                "tidy": bool(item.get("tidy_available")),
                "path": item.get("path_label") or "not generated",
            }
        )
    return rows


def render_notebook_plot_choice_image(
    plot_choice: Mapping[str, Any],
    *,
    mo: Any,
    view_state: Mapping[str, Any] | None = None,
) -> Any:
    """Render one plot-choice media artifact at notebook-column width."""

    if build_notebook_layered_scatter_contract(plot_choice) is not None:
        return render_notebook_layered_scatter_image(plot_choice, state=view_state, mo=mo)

    path = Path(str(plot_choice.get("path") or ""))
    path_label = str(plot_choice.get("path_label") or plot_choice.get("path") or "not generated")
    if not path.exists():
        return mo.md(f"Plot media missing: `{path_label}`")
    return mo.image(
        path.read_bytes(),
        alt=str(plot_choice.get("alt_text") or plot_choice.get("title") or plot_choice.get("label")),
        caption=str(plot_choice.get("caption") or "") or None,
        rounded=True,
        style={
            "width": "auto",
            "max-height": "min(62vh, 720px)",
            "max-width": "100%",
            "height": "auto",
            "object-fit": "contain",
            "margin": "0 auto",
            "display": "block",
            "background": "white",
        },
    )


def _plot_kind_metadata(kind: str) -> Mapping[str, Any]:
    return describe_plot_kind(kind)


def _plot_inventory_entry(
    *,
    entry: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    status: str,
    workdir: str,
    capability: Mapping[str, Any],
    media_path: str | None = None,
) -> dict[str, Any]:
    manifest_map = mapping(manifest)
    path = media_path or ""
    return {
        "name": str(entry.get("name") or manifest_map.get("name") or "unknown"),
        "kind": str(entry.get("kind") or manifest_map.get("kind") or "unknown"),
        "status": status,
        "rounds": manifest_map.get("rounds") or entry.get("round_selector") or "not generated",
        "freshness": mapping(manifest_map.get("freshness")).get("status") or "not generated",
        "path": path,
        "path_label": compact_path(path, base=workdir) if path else "not generated",
        "objective_family": capability.get("objective_family") or "unknown",
        "data_layer": capability.get("data_layer") or "unspecified",
        "round_scope": capability.get("round_scope") or "unspecified",
        "label_requirement": capability.get("label_requirement") or "none",
        "requires_model_artifact": bool(capability.get("requires_model_artifact")),
        "tidy_available": bool(capability.get("tidy_available")),
    }


def _generated_inventory_status(freshness_status: str) -> str:
    if freshness_status == "fresh":
        return "generated_current"
    if freshness_status == "stale":
        return "generated_stale"
    if freshness_status == "missing_outputs":
        return "generated_missing_media"
    return "generated_unknown"


def _inventory_status_counts(inventory: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in inventory:
        status = str(mapping(item).get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts

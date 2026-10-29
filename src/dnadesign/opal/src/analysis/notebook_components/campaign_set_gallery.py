"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/campaign_set_gallery.py

Notebook component builders for campaign set gallery OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Iterable, Mapping

from ...plots._mpl_utils import pretty_label
from ._support import display_name, mapping
from .campaign_set_relationships import campaign_pair_contexts, metadata_fields, relationship_pair_membership
from .campaign_set_sources import campaign_plot_manifest, manifest_media_path, manifest_tidy_csv_path


def build_notebook_campaign_set_plot_gallery_items(
    campaigns: Iterable[Mapping[str, Any]],
    *,
    plot_name: str,
    plot_kind: str,
    group_key: str,
    relationship: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return source plot media entries for a relationship-scoped campaign set."""

    items: list[dict[str, Any]] = []
    pair_membership = relationship_pair_membership(relationship)
    for campaign_model in campaigns:
        campaign = mapping(campaign_model.get("campaign"))
        slug = str(campaign.get("slug") or "unknown")
        pair_contexts = campaign_pair_contexts(campaign_model, pair_membership) if pair_membership else [None]
        if not pair_contexts:
            continue
        metadata = mapping(campaign.get("metadata"))
        manifest = campaign_plot_manifest(campaign_model, name=plot_name, kind=plot_kind)
        if manifest is None:
            continue
        media_path = manifest_media_path(manifest)
        if media_path is None or not media_path.exists():
            continue
        tidy_path = manifest_tidy_csv_path(manifest)
        group_value = str(metadata.get(str(group_key), "not recorded"))
        for pair_context in pair_contexts:
            items.append(
                {
                    **metadata_fields(metadata),
                    **(pair_context or {}),
                    "campaign": slug,
                    "campaign_label": display_name(slug),
                    "group_key": group_key,
                    "group": group_value,
                    "media_path": str(media_path),
                    "tidy_csv": str(tidy_path) if tidy_path is not None else "",
                }
            )
    return sorted(
        items,
        key=lambda row: (
            str(row.get("pair_key") or ""),
            _role_sort_key(str(row.get("comparison_role") or "")),
            str(row.get("campaign") or ""),
        ),
    )


def render_notebook_campaign_set_plot_gallery_image(
    items: Iterable[Mapping[str, Any]],
    *,
    title: str,
    group_key: str,
    dpi: int = 180,
) -> dict[str, Any] | None:
    """Render a compact source-plot gallery for a campaign set."""

    data = [dict(item) for item in items if item.get("media_path") and Path(str(item.get("media_path"))).exists()]
    if not data:
        return None

    import matplotlib.pyplot as plt

    from ...plots._mpl_utils import apply_plot_style, pretty_title

    apply_plot_style()
    columns = min(2, len(data))
    rows = (len(data) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6.4 * columns, 4.8 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, item in zip(axes.ravel(), data, strict=False):
        image = plt.imread(str(item["media_path"]))
        axis.imshow(image)
        axis.set_title(
            f"{pretty_label(str(item.get('comparison_role') or item.get('group') or 'campaign'))}: "
            f"{display_name(item.get('campaign'))}",
            fontsize=9,
        )
    fig.suptitle(pretty_title(title), fontsize=12)
    fig.tight_layout(pad=0.45)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=int(dpi), facecolor="white")
    plt.close(fig)
    roles = ", ".join(pretty_label(str(item.get("comparison_role") or "")) for item in data)
    return {
        "image_bytes": buffer.getvalue(),
        "caption": (
            f"Campaign-set source-plot gallery grouped by `{group_key}`. Panels show manifest-backed source "
            f"plots for the matched campaigns; no aggregate interval is computed."
        ),
        "alt_text": f"Campaign-set gallery for {title}. Roles shown: {roles}.",
        "group_count": len({str(item.get("group") or "") for item in data}),
        "row_count": len(data),
        "interval": {
            "kind": "none",
            "unit": "source plots",
            "rounds_with_interval": 0,
            "min_unit_count": 0,
            "max_unit_count": 0,
            "is_confidence_interval": False,
        },
    }


def _role_sort_key(role: str) -> int:
    normalized = role.lower()
    if normalized == "positive":
        return 0
    if normalized == "null":
        return 1
    return 2

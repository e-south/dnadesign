from __future__ import annotations

from typing import Any, Iterable, Mapping

from ._support import display_name, mapping, sequence

CAMPAIGN_PLOT_SURFACE_KIND = "campaign_plot"
CAMPAIGN_SET_COMPARISON_SURFACE_KIND = "campaign_set_metric_comparison"


def build_notebook_campaign_set_visual_choices(
    plot_choices: Iterable[Mapping[str, Any]],
    campaigns: Iterable[Mapping[str, Any]],
    collection: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return one selectable visual surface per rendered campaign-set visual."""

    campaign_list = [campaign for campaign in campaigns if isinstance(campaign, Mapping)]
    choices: list[dict[str, Any]] = []
    labels_seen: set[str] = set()
    for raw_choice in sequence(plot_choices):
        if not isinstance(raw_choice, Mapping):
            continue
        choice = dict(raw_choice)
        label = _unique_label(_choice_label(choice), labels_seen)
        choice["label"] = label
        choice.setdefault("surface_kind", CAMPAIGN_PLOT_SURFACE_KIND)
        choices.append(choice)

    comparison_lenses = [
        lens for lens in sequence(mapping(collection).get("comparison_lenses")) if isinstance(lens, Mapping)
    ]
    if len(campaign_list) <= 1 or not comparison_lenses:
        return choices

    for choice in list(choices):
        if choice.get("surface_kind") != CAMPAIGN_PLOT_SURFACE_KIND:
            continue
        if str(choice.get("kind") or "") != "metric_over_rounds":
            continue
        source_plot_name = str(choice.get("name") or "")
        if not source_plot_name:
            continue
        title = str(choice.get("title") or choice.get("label") or display_name(source_plot_name))
        for lens in comparison_lenses:
            group_key = str(lens.get("group_key") or "").strip()
            if not group_key:
                continue
            label = str(lens.get("label") or "Campaign-set comparison")
            choices.append(
                {
                    "label": _unique_label(f"{title} - {label}", labels_seen),
                    "title": title,
                    "kind": CAMPAIGN_SET_COMPARISON_SURFACE_KIND,
                    "surface_kind": CAMPAIGN_SET_COMPARISON_SURFACE_KIND,
                    "source_plot_name": source_plot_name,
                    "source_plot_label": choice["label"],
                    "source_kind": choice.get("kind"),
                    "relationship_kind": lens.get("kind"),
                    "role_dimension": lens.get("role_dimension") or group_key,
                    "left_role": lens.get("left_role"),
                    "right_role": lens.get("right_role"),
                    "match_on": list(sequence(lens.get("match_on"))),
                    "replicate_on": list(sequence(lens.get("replicate_on"))),
                    "pair_count": lens.get("pair_count"),
                    "pairs": list(sequence(lens.get("pairs"))),
                    "comparison_group_options": [group_key],
                }
            )
    return choices


def _choice_label(choice: Mapping[str, Any]) -> str:
    return str(choice.get("label") or choice.get("title") or display_name(choice.get("name") or "visual"))


def _unique_label(label: str, seen: set[str]) -> str:
    base = str(label or "Visual").strip() or "Visual"
    candidate = base
    index = 2
    while candidate in seen:
        candidate = f"{base} ({index})"
        index += 1
    seen.add(candidate)
    return candidate

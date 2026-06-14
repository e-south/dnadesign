from __future__ import annotations

from typing import Any, Iterable, Mapping

from ...plots._mpl_utils import pretty_label
from ._support import mapping, sequence

CAMPAIGN_SET_BASERENDER_SURFACE_KIND = "campaign_set_baserender"


def build_notebook_collection_baserender_role_choices(
    campaigns: Iterable[Mapping[str, Any]],
    collection: Mapping[str, Any] | None,
    selected_collection_set_choice: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    """Return positive/null campaign choices for a selected collection comparison set."""

    slug_to_campaign = {
        str(mapping(campaign.get("campaign")).get("slug") or ""): campaign for campaign in sequence(campaigns)
    }
    selected_match = mapping((selected_collection_set_choice or {}).get("match"))
    if not selected_match:
        selected_match = mapping((selected_collection_set_choice or {}).get("comparison_set_match"))
    if not collection or not selected_match:
        return []

    for lens in sequence(collection.get("comparison_lenses")):
        if mapping(lens).get("kind") != "control_pair":
            continue
        for pair in sequence(mapping(lens).get("pairs")):
            pair_match = mapping(pair.get("match"))
            if not _match_overlapping_dimensions(pair_match, selected_match):
                continue
            return [
                choice
                for choice in (
                    _role_choice(
                        role=mapping(lens).get("left_role"),
                        campaign_slug=pair.get("left"),
                        slug_to_campaign=slug_to_campaign,
                        control_role=selected_match.get("control_role"),
                    ),
                    _role_choice(
                        role=mapping(lens).get("right_role"),
                        campaign_slug=pair.get("right"),
                        slug_to_campaign=slug_to_campaign,
                        control_role=selected_match.get("control_role"),
                    ),
                )
                if choice is not None
            ]
    return []


def _match_overlapping_dimensions(pair_match: Mapping[str, Any], selected_match: Mapping[str, Any]) -> bool:
    compared = 0
    for selected_key, selected_value in selected_match.items():
        if selected_key == "review_surface":
            continue
        pair_key = _pair_match_key(str(selected_key), pair_match)
        if pair_key is None:
            continue
        compared += 1
        if str(pair_match[pair_key]) != str(selected_value):
            return False
    return compared > 0


def _pair_match_key(selected_key: str, pair_match: Mapping[str, Any]) -> str | None:
    if selected_key in pair_match:
        return selected_key
    aliases = {
        "label_name": "target",
        "target_label": "target",
        "target": "label_name",
    }
    alias = aliases.get(selected_key)
    if alias in pair_match:
        return alias
    return None


def _role_choice(
    *,
    role: Any,
    campaign_slug: Any,
    slug_to_campaign: Mapping[str, Mapping[str, Any]],
    control_role: Any = None,
) -> dict[str, str] | None:
    slug = str(campaign_slug or "").strip()
    if not slug or slug not in slug_to_campaign:
        return None
    role_token = str(role or "").strip() or "campaign"
    return {
        "label": _role_label(role_token, slug=slug, control_role=control_role),
        "role": role_token,
        "campaign_slug": slug,
    }


def _role_label(role: str, *, slug: str, control_role: Any = None) -> str:
    token = str(role).strip().lower().replace("-", "_")
    slug_token = str(slug).lower().replace("-", "_")
    if token == "positive":
        return "DenseGen label"
    if token in {"null", "matched_null"} or "matched_null" in slug_token:
        control = str(control_role or "").strip()
        if control == "count_fixed_shuffled_slot_negative_control":
            return "Count-fixed shuffled-slot control"
        if control == "count_preserving_slot_confound_control":
            return "Count-preserving slot diagnostic control"
        return "Scrambled control"
    return pretty_label(role)

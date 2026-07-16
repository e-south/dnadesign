"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/campaign_set_baserender.py

Notebook component builders for campaign set BaseRender OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ...plots._mpl_utils import pretty_label
from ._support import mapping, sequence
from .baserender_selection_scope import build_notebook_baserender_role_control

CAMPAIGN_SET_BASERENDER_SURFACE_KIND = "campaign_set_baserender"


def build_notebook_collection_baserender_role_control(
    campaigns: Iterable[Mapping[str, Any]],
    collection: Mapping[str, Any] | None,
    selected_collection_set_choice: Mapping[str, Any] | None,
    *,
    active_view_mode: str,
    mo: Any,
) -> tuple[list[dict[str, str]], Any | None]:
    """Build role choices and their campaign-set label-source control."""

    choices = (
        build_notebook_collection_baserender_role_choices(campaigns, collection, selected_collection_set_choice)
        if active_view_mode == "Campaign set"
        else []
    )
    control = build_notebook_baserender_role_control(
        active_view_mode=active_view_mode,
        role_choices=choices,
        mo=mo,
    )
    return choices, control


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
        return _role_choices_from_campaign_metadata(
            campaigns=sequence(campaigns),
            selected_match=selected_match,
        )

    choices: list[dict[str, str]] = []
    for lens in sequence(collection.get("comparison_lenses")):
        if mapping(lens).get("kind") != "control_pair":
            continue
        for pair in sequence(mapping(lens).get("pairs")):
            pair_match = mapping(pair.get("match"))
            if not _match_overlapping_dimensions(pair_match, selected_match):
                continue
            choices.extend(
                choice
                for choice in (
                    _role_choice(
                        role=mapping(lens).get("left_role"),
                        role_label=mapping(lens).get("left_role_label"),
                        campaign_slug=pair.get("left"),
                        slug_to_campaign=slug_to_campaign,
                        control_role=selected_match.get("control_role"),
                    ),
                    _role_choice(
                        role=mapping(lens).get("right_role"),
                        role_label=mapping(lens).get("right_role_label"),
                        campaign_slug=pair.get("right"),
                        slug_to_campaign=slug_to_campaign,
                        control_role=selected_match.get("control_role"),
                    ),
                )
                if choice is not None
            )
    if choices:
        return _disambiguate_duplicate_choice_labels(choices)
    return _role_choices_from_campaign_metadata(
        campaigns=sequence(campaigns),
        selected_match=selected_match,
    )


def _role_choices_from_campaign_metadata(
    *,
    campaigns: list[Mapping[str, Any]],
    selected_match: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Infer target/control BaseRender campaign choices from campaign metadata."""

    if not selected_match:
        return []
    selected_label = _first_present(selected_match, ("label_name", "target_label", "target"))
    if not selected_label:
        return []
    control_role = str(selected_match.get("control_role") or "").strip()
    rows: list[dict[str, Any]] = []
    for campaign in campaigns:
        campaign_map = mapping(campaign.get("campaign"))
        metadata = mapping(campaign_map.get("metadata"))
        if not _campaign_matches_selected_label(metadata, selected_label):
            continue
        if not _campaign_matches_control_surface(metadata, control_role=control_role):
            continue
        role = _campaign_role(metadata)
        if role not in {"positive", "null", "matched_null"}:
            continue
        slug = str(campaign_map.get("slug") or "").strip()
        if not slug:
            continue
        rows.append(
            {
                "label": _role_label(role, slug=slug, control_role=control_role),
                "role": role,
                "campaign_slug": slug,
                "_seed": _campaign_seed(metadata),
                "_role_order": "0" if role == "positive" else "1",
            }
        )
    rows.sort(key=lambda row: (_seed_sort_key(row["_seed"]), row["_role_order"], row["label"], row["campaign_slug"]))
    return _disambiguate_duplicate_choice_labels(
        [{"label": row["label"], "role": row["role"], "campaign_slug": row["campaign_slug"]} for row in rows],
        seeds=[str(row["_seed"]) for row in rows],
    )


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
    role_label: Any = None,
    campaign_slug: Any,
    slug_to_campaign: Mapping[str, Mapping[str, Any]],
    control_role: Any = None,
) -> dict[str, str] | None:
    slug = str(campaign_slug or "").strip()
    if not slug or slug not in slug_to_campaign:
        return None
    role_token = str(role or "").strip() or "campaign"
    return {
        "label": _role_label(role_token, slug=slug, control_role=control_role, role_label=role_label),
        "role": role_token,
        "campaign_slug": slug,
    }


def _role_label(role: str, *, slug: str, control_role: Any = None, role_label: Any = None) -> str:
    explicit = str(role_label or "").strip()
    if explicit:
        return _normalise_role_label(explicit)
    token = str(role).strip().lower().replace("-", "_")
    slug_token = str(slug).lower().replace("-", "_")
    if token == "positive":
        control = str(control_role or "").strip()
        if control in {
            "matched_label_permutation_negative_control",
            "count_fixed_shuffled_slot_negative_control",
            "count_preserving_slot_confound_control",
        }:
            return "Sequence-matched metadata"
        return "Positive label source"
    if token in {"null", "matched_null"} or "matched_null" in slug_token:
        control = str(control_role or "").strip()
        if control == "matched_label_permutation_negative_control":
            return "Row-shuffled control"
        if control == "count_fixed_shuffled_slot_negative_control":
            return "Slot-shuffled control"
        if control == "count_preserving_slot_confound_control":
            return "Count-preserving diagnostic"
        return pretty_label(control) if control else "Control label source"
    return pretty_label(role)


def _normalise_role_label(label: str) -> str:
    text = str(label or "").strip()
    normalized = text.lower().replace("-", " ")
    replacements = {
        "dense array metadata": "Sequence-matched metadata",
        "sequence matched metadata": "Sequence-matched metadata",
        "row shuffled metadata control": "Row-shuffled control",
        "row shuffled control": "Row-shuffled control",
        "count fixed slot shuffle control": "Slot-shuffled control",
        "count fixed slot shuffled control": "Slot-shuffled control",
        "slot shuffled control": "Slot-shuffled control",
        "count preserving slot diagnostic": "Count-preserving diagnostic",
    }
    return replacements.get(normalized, text)


def _disambiguate_duplicate_choice_labels(
    choices: list[dict[str, str]],
    *,
    seeds: list[str] | None = None,
) -> list[dict[str, str]]:
    """Make dropdown labels unique without changing campaign identity."""

    counts: dict[str, int] = {}
    for choice in choices:
        counts[choice["label"]] = counts.get(choice["label"], 0) + 1
    out: list[dict[str, str]] = []
    for index, choice in enumerate(choices):
        row = dict(choice)
        if counts.get(row["label"], 0) > 1:
            seed = str((seeds or [])[index] if seeds and index < len(seeds) else "").strip()
            suffix = f"seed {seed}" if seed else row["campaign_slug"]
            row["label"] = f"{row['label']} ({suffix})"
        out.append(row)
    return out


def _campaign_matches_selected_label(metadata: Mapping[str, Any], selected_label: str) -> bool:
    selected = str(selected_label or "").strip()
    if not selected:
        return False
    return any(str(metadata.get(key) or "").strip() == selected for key in ("label_name", "target", "target_label"))


def _campaign_matches_control_surface(metadata: Mapping[str, Any], *, control_role: str) -> bool:
    role = str(control_role or "").strip()
    scope = str(metadata.get("candidate_scope_policy_id") or "").strip()
    null_version = str(metadata.get("null_version") or "").strip()
    if role == "count_fixed_shuffled_slot_negative_control":
        return scope == "tfbs_slot_position_target_count_eq_1_v1"
    if role == "count_preserving_slot_confound_control":
        return not scope and (
            not null_version or null_version == "densegen_tfbs_learnability_slot_geometry_count_matched_null_v1"
        )
    return True


def _campaign_role(metadata: Mapping[str, Any]) -> str:
    oracle_role = str(metadata.get("oracle_role") or "").strip()
    if oracle_role in {"positive", "matched_null", "null"}:
        return oracle_role
    label_oracle_kind = str(metadata.get("label_oracle_kind") or "").strip()
    if label_oracle_kind == "positive":
        return "positive"
    if label_oracle_kind in {"null", "matched_null"}:
        return "null"
    return label_oracle_kind


def _campaign_seed(metadata: Mapping[str, Any]) -> str:
    return _first_present(metadata, ("replicate_seed", "seed")) or ""


def _first_present(mapping_value: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(mapping_value.get(key) or "").strip()
        if value:
            return value
    return ""


def _seed_sort_key(value: object) -> tuple[int, str]:
    text = str(value or "").strip()
    try:
        return int(text), text
    except ValueError:
        return 10_000_000, text

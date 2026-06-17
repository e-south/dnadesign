"""Target-profile contracts for DenseGen TFBS learnability probe campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

from .schema import (
    TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES,
    TFBS_LEARNABILITY_CANONICAL_COUNT_FRACTION_TARGET_SET,
    TFBS_LEARNABILITY_SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_TARGET_SET,
    TFBS_LEARNABILITY_SLOT_POSITION_COUNT_FIXED_SENTINEL_TARGET_SET,
    TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET,
    TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET,
)

CANONICAL_COUNT_FRACTION_PROFILE_ID = "tfbs_count_fraction_probe_v1"
SLOT_POSITION_SENTINEL_PROFILE_ID = "tfbs_slot_position_sentinel_probe_v1"
SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID = "tfbs_slot_position_count_fixed_sentinel_probe_v1"
SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID = "tfbs_slot_position_count_fixed_baer_middle_probe_v1"
SLOT_POSITION_PROFILE_ID = "tfbs_slot_position_probe_v1"
CUSTOM_TFBS_TARGET_PROFILE_ID = "custom_tfbs_learnability_label_set"
DEFAULT_TFBS_TARGET_PROFILE_ID = CANONICAL_COUNT_FRACTION_PROFILE_ID


@dataclass(frozen=True)
class TfbsProbeTargetProfile:
    """Manifest contract for a concrete TFBS probe campaign label set."""

    profile_id: str
    profile_role: str
    label_names: tuple[str, ...]
    label_family_ids: tuple[str, ...]
    canonical: bool
    interpretation_boundary: str

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["label_names"] = list(self.label_names)
        payload["label_family_ids"] = list(self.label_family_ids)
        return payload


CANONICAL_COUNT_FRACTION_PROFILE = TfbsProbeTargetProfile(
    profile_id=CANONICAL_COUNT_FRACTION_PROFILE_ID,
    profile_role="canonical_stage_b_probe",
    label_names=TFBS_LEARNABILITY_CANONICAL_COUNT_FRACTION_TARGET_SET,
    label_family_ids=("tf_family_count_fraction",),
    canonical=True,
    interpretation_boundary=(
        "Canonical TFBS count-fraction synthetic-oracle learnability probe. "
        "This profile supports claims about OPAL enrichment of literal DenseGen "
        "count_fraction construction labels against matched nulls; it does not "
        "claim measured stress growth, TF binding, or slot-position geometry."
    ),
)

SLOT_POSITION_PROFILE = TfbsProbeTargetProfile(
    profile_id=SLOT_POSITION_PROFILE_ID,
    profile_role="boundary_stage_b_probe",
    label_names=TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET,
    label_family_ids=("tf_slot_family_presence",),
    canonical=False,
    interpretation_boundary=(
        "Count-matched slot-position synthetic-oracle boundary probe. This profile tests whether "
        "promoter embeddings support enrichment of literal DenseGen slot_family_presence construction labels "
        "after preserving target-family counts; interpret outcomes as slot-position evidence or boundary "
        "diagnostics, not measured TF binding, stress growth, or mechanism."
    ),
)

SLOT_POSITION_SENTINEL_PROFILE = TfbsProbeTargetProfile(
    profile_id=SLOT_POSITION_SENTINEL_PROFILE_ID,
    profile_role="boundary_stage_b_sentinel_probe",
    label_names=TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET,
    label_family_ids=("tf_slot_family_presence",),
    canonical=False,
    interpretation_boundary=(
        "Slim count-matched slot-position sentinel probe. This profile tests two predeclared slot-family "
        "presence objectives, LexA in the leftmost slot and CpxR/BaeR in the rightmost slot, as a boundary "
        "screen for positional learnability. It is intentionally narrower than a full slot-by-family resolution "
        "map and does not support measured TF binding, stress growth, or mechanism claims."
    ),
)

SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE = TfbsProbeTargetProfile(
    profile_id=SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID,
    profile_role="boundary_stage_b_count_fixed_sentinel_probe",
    label_names=TFBS_LEARNABILITY_SLOT_POSITION_COUNT_FIXED_SENTINEL_TARGET_SET,
    label_family_ids=("tf_slot_family_presence",),
    canonical=False,
    interpretation_boundary=(
        "Count-fixed slot-position sentinel probe. This profile restricts each label's candidate universe to rows "
        "with exactly one target-family motif, then compares DenseGen slot labels against a count-fixed shuffled-slot "
        "negative control. It supports only synthetic construction-label learnability claims about coarse motif "
        "placement, not measured TF binding, stress growth, mechanism, or biology."
    ),
)

SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE = TfbsProbeTargetProfile(
    profile_id=SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID,
    profile_role="boundary_stage_b_count_fixed_minimal_placement_probe",
    label_names=TFBS_LEARNABILITY_SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_TARGET_SET,
    label_family_ids=("tf_slot_family_presence",),
    canonical=False,
    interpretation_boundary=(
        "Count-fixed BaeR middle-slot placement probe. This profile restricts the candidate universe to rows "
        "with exactly one BaeR motif, then asks whether active selection can enrich BaeR in the middle TFBS slot "
        "against a count-fixed shuffled-slot negative control. It adds a minimal middle-slot placement check "
        "without expanding into a full regulator-by-slot map, and supports only synthetic construction-label "
        "learnability claims."
    ),
)

_NAMED_TARGET_PROFILES = MappingProxyType(
    {
        CANONICAL_COUNT_FRACTION_PROFILE.profile_id: CANONICAL_COUNT_FRACTION_PROFILE,
        SLOT_POSITION_SENTINEL_PROFILE.profile_id: SLOT_POSITION_SENTINEL_PROFILE,
        SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE.profile_id: SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE,
        SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE.profile_id: SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE,
        SLOT_POSITION_PROFILE.profile_id: SLOT_POSITION_PROFILE,
    }
)

_COUNT_FIXED_SLOT_POSITION_PROFILE_IDS = frozenset(
    {
        SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID,
        SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID,
    }
)


def canonical_count_fraction_label_names() -> tuple[str, ...]:
    """Return the current canonical TFBS probe labels."""

    return CANONICAL_COUNT_FRACTION_PROFILE.label_names


def slot_position_label_names() -> tuple[str, ...]:
    """Return the first-class TFBS slot-position boundary labels."""

    return SLOT_POSITION_PROFILE.label_names


def slot_position_sentinel_label_names() -> tuple[str, ...]:
    """Return the slim TFBS slot-position sentinel boundary labels."""

    return SLOT_POSITION_SENTINEL_PROFILE.label_names


def slot_position_count_fixed_sentinel_label_names() -> tuple[str, ...]:
    """Return the count-fixed TFBS slot-position sentinel boundary labels."""

    return SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE.label_names


def slot_position_count_fixed_baer_middle_label_names() -> tuple[str, ...]:
    """Return the count-fixed BaeR middle-slot placement label."""

    return SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE.label_names


def is_count_fixed_slot_position_profile_id(profile_id: str) -> bool:
    """Return whether a named profile requires count-fixed slot-position controls."""

    return str(profile_id or "").strip() in _COUNT_FIXED_SLOT_POSITION_PROFILE_IDS


def tfbs_target_profile_ids() -> tuple[str, ...]:
    """Return stable named TFBS target-profile ids for CLIs and manifests."""

    return tuple(_NAMED_TARGET_PROFILES)


def tfbs_target_profile_for_profile_id(profile_id: str) -> TfbsProbeTargetProfile:
    """Return a named target profile by stable id, or fail fast."""

    token = str(profile_id or "").strip()
    try:
        return _NAMED_TARGET_PROFILES[token]
    except KeyError as exc:
        raise ValueError(f"unsupported TFBS target profile id: {profile_id!r}") from exc


def tfbs_label_names_for_profile_id(profile_id: str) -> tuple[str, ...]:
    """Return the label set owned by a named TFBS target profile."""

    return tfbs_target_profile_for_profile_id(profile_id).label_names


def resolve_tfbs_target_label_names(
    *,
    target_profile_id: str | None,
    label_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve CLI target labels from either a named profile or explicit label list."""

    explicit_labels = tuple(str(label).strip() for label in label_names if str(label).strip())
    if target_profile_id and explicit_labels:
        raise ValueError("--target-profile cannot be combined with explicit --label-name values")
    if target_profile_id:
        return tfbs_label_names_for_profile_id(target_profile_id)
    if explicit_labels:
        return _validate_label_names(explicit_labels)
    return canonical_count_fraction_label_names()


def resolve_tfbs_target_profile(
    *,
    target_profile_id: str | None,
    label_names: tuple[str, ...],
) -> TfbsProbeTargetProfile:
    """Return the manifest profile for an explicit profile id or label tuple."""

    explicit_labels = tuple(str(label).strip() for label in label_names if str(label).strip())
    if target_profile_id:
        if str(target_profile_id) == CUSTOM_TFBS_TARGET_PROFILE_ID:
            return tfbs_target_profile_for_labels(explicit_labels)
        profile = tfbs_target_profile_for_profile_id(target_profile_id)
        if explicit_labels and _validate_label_names(explicit_labels) != profile.label_names:
            raise ValueError(
                "target profile label mismatch: "
                f"profile_id={target_profile_id!r} labels={list(profile.label_names)} "
                f"requested={list(explicit_labels)}"
            )
        return profile
    if explicit_labels:
        return tfbs_target_profile_for_labels(explicit_labels)
    return CANONICAL_COUNT_FRACTION_PROFILE


def tfbs_target_profile_for_labels(label_names: tuple[str, ...]) -> TfbsProbeTargetProfile:
    """Return the known target profile, or an explicit custom profile."""

    labels = _validate_label_names(label_names)
    if labels == CANONICAL_COUNT_FRACTION_PROFILE.label_names:
        return CANONICAL_COUNT_FRACTION_PROFILE
    if labels == SLOT_POSITION_SENTINEL_PROFILE.label_names:
        raise ValueError(
            "ambiguous TFBS slot-position sentinel labels: "
            f"{list(labels)} match both {SLOT_POSITION_SENTINEL_PROFILE_ID!r} and "
            f"{SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID!r}; pass target_profile_id explicitly"
        )
    if labels == SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE.label_names:
        return SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE
    if labels == SLOT_POSITION_PROFILE.label_names:
        return SLOT_POSITION_PROFILE
    return TfbsProbeTargetProfile(
        profile_id=CUSTOM_TFBS_TARGET_PROFILE_ID,
        profile_role="custom_operator_selected_probe",
        label_names=labels,
        label_family_ids=tuple(dict.fromkeys(_label_family_id(label) for label in labels)),
        canonical=False,
        interpretation_boundary=(
            "Custom TFBS synthetic-oracle learnability label set. Interpret this as an "
            "operator-selected probe surface; do not treat it as the canonical count-fraction claim "
            "without a separate review gate."
        ),
    )


def _validate_label_names(label_names: tuple[str, ...]) -> tuple[str, ...]:
    labels = tuple(dict.fromkeys(str(label).strip() for label in label_names if str(label).strip()))
    if not labels:
        raise ValueError("TFBS target profile requires at least one label")
    unknown = sorted(set(labels) - set(TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES))
    if unknown:
        raise ValueError(f"unsupported TFBS learnability target label(s): {unknown}")
    return labels


def _label_family_id(label_name: str) -> str:
    if label_name.endswith("_present"):
        return "tf_family_presence"
    if label_name.endswith("_count_fraction"):
        return "tf_family_count_fraction"
    if "_in_slot" in label_name:
        return "tf_slot_family_presence"
    raise ValueError(f"cannot infer TFBS label family for target profile label: {label_name}")

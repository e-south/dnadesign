"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/realization/normalize_anchor.py

Normalize-anchor realization contracts for Construct analysis windows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ..annotations.features import AnnotationFeature, load_annotation_features
from ..annotations.focal import FocalSelection, resolve_focal_selection
from ..annotations.retention import FeatureRetentionSummary, classify_feature_retention
from ..contracts.config import JobConfig, NormalizeTemplateConfig
from ..contracts.errors import ValidationError
from .placement import template_match_offsets
from .sequences import ensure_dna_text


class NormalizeAnchorTemplate(Protocol):
    id: str
    kind: str
    sequence: str
    source: str
    dataset: str | None
    field: str | None
    record_id: str | None
    circular: bool


NormalizeTemplateLoader = Callable[[NormalizeTemplateConfig], NormalizeAnchorTemplate]


@dataclass(frozen=True)
class NormalizeAnchorRealization:
    source_sequence: str
    analysis_sequence: str
    source_start_0: int
    source_end_0: int
    template: NormalizeAnchorTemplate | None
    template_sha256: str | None
    added_left_bp: int
    added_right_bp: int
    derived_start_offset_0: int
    focal_selection: FocalSelection
    retention: FeatureRetentionSummary


def require_normalize_target_length_match(*, cfg: JobConfig) -> None:
    if cfg.job.normalize_anchor is None:
        raise ValidationError("job.normalize_anchor is required when job.mode='normalize_anchor'.")
    if cfg.job.normalize_anchor.over_length_policy.target_length != cfg.job.normalize_anchor.target_length:
        raise ValidationError(
            "normalize_anchor.over_length_policy.target_length must match normalize_anchor.target_length."
        )
    policy = cfg.job.normalize_anchor.under_length_policy
    if policy is not None and policy.target_length != cfg.job.normalize_anchor.target_length:
        raise ValidationError(
            "normalize_anchor.under_length_policy.target_length must match normalize_anchor.target_length."
        )


def realize_normalize_anchor(
    *,
    row: dict[str, object],
    cfg: JobConfig,
    load_template: NormalizeTemplateLoader,
) -> NormalizeAnchorRealization:
    if cfg.job.normalize_anchor is None:
        raise ValidationError("job.normalize_anchor is required when job.mode='normalize_anchor'.")
    normalize_cfg = cfg.job.normalize_anchor
    input_field = cfg.job.input.field
    if input_field is None:
        raise ValidationError("job.input.field is required when job.mode='normalize_anchor'.")
    source_value = row.get(input_field)
    if source_value is None:
        raise ValidationError(f"Input row '{row.get('id')}' is missing field '{input_field}'.")
    sequence = ensure_dna_text(str(source_value), label=f"input field '{input_field}'")
    features = load_annotation_features(row)
    try:
        focal_selection = resolve_focal_selection(
            sequence_length=len(sequence),
            features=features,
            selector_chain=normalize_cfg.focal_selector,
            allow_low_confidence=bool(normalize_cfg.fallback_policy.allow_low_confidence),
        )
    except ValueError as exc:
        raise ValidationError(
            f"normalize_anchor could not resolve a focal point for row '{row.get('id')}': {exc}"
        ) from exc

    source_start_0 = 0
    source_end_0 = len(sequence)
    template: NormalizeAnchorTemplate | None = None
    template_sha256: str | None = None
    added_left_bp = 0
    added_right_bp = 0
    derived_start_offset_0 = 0
    analysis_sequence = sequence

    if len(sequence) > normalize_cfg.target_length:
        source_start_0, source_end_0 = best_trim_window(
            sequence=sequence,
            features=features,
            focal_selection=focal_selection,
            target_length=normalize_cfg.target_length,
            required_roles=list(normalize_cfg.feature_retention_policy.fail_if_loses_roles),
            window_anchor=normalize_cfg.over_length_policy.window_anchor,
        )
        if normalize_cfg.over_length_policy.require_focal_inside and not (
            source_start_0 <= float(focal_selection.focal_point_0) < source_end_0
        ):
            raise ValidationError(f"normalize_anchor trim window excludes the focal point for row '{row.get('id')}'.")
        analysis_sequence = sequence[source_start_0:source_end_0]
    elif len(sequence) < normalize_cfg.target_length:
        if normalize_cfg.over_length_policy.window_anchor != "retention_optimized":
            raise ValidationError(
                "normalize_anchor under-length template expansion currently supports only "
                "retention_optimized trim semantics."
            )
        policy = normalize_cfg.under_length_policy
        if policy is None:
            raise ValidationError(
                f"normalize_anchor requires under_length_policy for short input row '{row.get('id')}'."
            )
        template = load_template(policy.template)
        template_sha256 = hashlib.sha256(template.sequence.encode("utf-8")).hexdigest()
        analysis_sequence, embedded_anchor_start, added_left_bp, added_right_bp = expand_short_sequence_from_template(
            sequence=sequence,
            template=template,
            target_length=normalize_cfg.target_length,
            focal_selection=focal_selection,
            placement_ref=policy.placement_ref,
        )
        derived_start_offset_0 = embedded_anchor_start

    if len(analysis_sequence) != normalize_cfg.target_length:
        raise ValidationError(
            f"normalize_anchor produced {len(analysis_sequence)} bp, expected {normalize_cfg.target_length}."
        )

    retention = classify_feature_retention(
        features=features,
        source_start_0=source_start_0,
        source_end_0=source_end_0,
        derived_start_offset_0=derived_start_offset_0,
    )
    lost_roles = {
        str(entry.get("role_hint") or "") for entry in retention.lost if str(entry.get("role_hint") or "").strip()
    }
    required_lost_roles = sorted(
        set(normalize_cfg.feature_retention_policy.fail_if_loses_roles).intersection(lost_roles)
    )
    if required_lost_roles:
        joined = ", ".join(required_lost_roles)
        raise ValidationError(f"normalize_anchor would lose required roles for row '{row.get('id')}': {joined}.")

    return NormalizeAnchorRealization(
        source_sequence=sequence,
        analysis_sequence=analysis_sequence,
        source_start_0=source_start_0,
        source_end_0=source_end_0,
        template=template,
        template_sha256=template_sha256,
        added_left_bp=added_left_bp,
        added_right_bp=added_right_bp,
        derived_start_offset_0=derived_start_offset_0,
        focal_selection=focal_selection,
        retention=retention,
    )


def best_trim_window(
    *,
    sequence: str,
    features: list[AnnotationFeature],
    focal_selection: FocalSelection,
    target_length: int,
    required_roles: list[str],
    window_anchor: str = "retention_optimized",
) -> tuple[int, int]:
    if len(sequence) < target_length:
        raise ValidationError("trim window requested for sequence shorter than target length")
    focal_point = float(focal_selection.focal_point_0)
    if window_anchor == "upstream_of_focal":
        end = int(focal_point)
        start = end - target_length
        if start < 0 or end > len(sequence):
            raise ValidationError(
                "normalize_anchor upstream_of_focal trim requires the focal offset to have "
                f"{target_length} upstream bases inside the input sequence."
            )
        return int(start), int(end)
    if window_anchor != "retention_optimized":
        raise ValidationError(f"Unsupported normalize_anchor trim window_anchor: {window_anchor!r}.")
    min_start = max(0, int(focal_point) - target_length + 1)
    max_start = min(len(sequence) - target_length, int(focal_point))
    if min_start > max_start:
        start = max(0, min(len(sequence) - target_length, round(focal_point) - target_length // 2))
        return int(start), int(start + target_length)

    best_key: tuple[int, int, int, int] | None = None
    best_window: tuple[int, int] | None = None
    for start in range(min_start, max_start + 1):
        end = start + target_length
        retained_roles = 0
        retained_features = 0
        clipped_features = 0
        for feature in features:
            intervals = list(feature.intervals_0)
            if not intervals:
                continue
            fully_retained = all(interval.start_0 >= start and interval.end_0 <= end for interval in intervals)
            intersects = any(interval.end_0 > start and interval.start_0 < end for interval in intervals)
            if fully_retained:
                retained_features += 1
                if feature.role_hint in required_roles:
                    retained_roles += 1
            elif intersects:
                clipped_features += 1
        key = (retained_roles, retained_features, -clipped_features, -start)
        if best_key is None or key > best_key:
            best_key = key
            best_window = (start, end)
    if best_window is None:
        raise ValidationError("normalize_anchor could not resolve a retention-optimized trim window.")
    return best_window


def expand_short_sequence_from_template(
    *,
    sequence: str,
    template: NormalizeAnchorTemplate,
    target_length: int,
    focal_selection: FocalSelection,
    placement_ref: str,
) -> tuple[str, int, int, int]:
    replacement_interval = parse_replacement_placement_ref(placement_ref)
    effective_template_sequence = template.sequence
    if replacement_interval is None:
        anchor_start = resolve_under_length_anchor_start(
            sequence=sequence,
            template=template,
            placement_ref=placement_ref,
        )
    else:
        replacement_start, replacement_end = replacement_interval
        _validate_replacement_interval(
            template=template,
            placement_ref=placement_ref,
            replacement_start=replacement_start,
            replacement_end=replacement_end,
        )
        anchor_start = replacement_start
        effective_template_sequence = (
            template.sequence[:replacement_start] + sequence + template.sequence[replacement_end:]
        )
    absolute_focal = anchor_start + focal_selection.focal_point_0
    window_start = int(round(absolute_focal - (target_length / 2.0)))
    if not template.circular and window_start < 0:
        window_start = 0
    window_end = window_start + target_length
    if not template.circular and window_end > len(effective_template_sequence):
        max_window_start = len(effective_template_sequence) - target_length
        if max_window_start < 0:
            raise ValidationError(
                f"normalize_anchor template '{template.id}' cannot provide {target_length} bp around the focal point."
            )
        window_start = max_window_start
        window_end = window_start + target_length
    if template.circular:
        expanded = "".join(
            effective_template_sequence[(window_start + idx) % len(effective_template_sequence)]
            for idx in range(target_length)
        )
    else:
        expanded = effective_template_sequence[window_start:window_end]
    if len(expanded) != target_length:
        raise ValidationError(
            f"normalize_anchor template expansion produced {len(expanded)} bp instead of {target_length}."
        )
    embedded_anchor_start = (
        (anchor_start - window_start) % len(effective_template_sequence)
        if template.circular
        else anchor_start - window_start
    )
    if embedded_anchor_start < 0 or embedded_anchor_start + len(sequence) > target_length:
        raise ValidationError(
            "normalize_anchor template expansion could not embed the anchor contiguously inside the derived window."
        )
    added_left = embedded_anchor_start
    added_right = target_length - embedded_anchor_start - len(sequence)
    return expanded, embedded_anchor_start, added_left, added_right


def parse_replacement_placement_ref(placement_ref: str) -> tuple[int, int] | None:
    text = str(placement_ref or "").strip()
    lowered = text.casefold()
    prefix = "replace:"
    if not lowered.startswith(prefix):
        return None
    body = text[len(prefix) :].strip()
    separator = ".." if ".." in body else "-"
    try:
        raw_start, raw_end = body.split(separator, maxsplit=1)
        start = int(raw_start.strip())
        end = int(raw_end.strip())
    except ValueError as exc:
        raise ValidationError(
            f"normalize_anchor under_length_policy.placement_ref '{placement_ref}' must use "
            "'replace:<start_0>-<end_0>'."
        ) from exc
    if end <= start:
        raise ValidationError(
            f"normalize_anchor under_length_policy.placement_ref '{placement_ref}' must use end > start."
        )
    return start, end


def resolve_under_length_anchor_start(
    *,
    sequence: str,
    template: NormalizeAnchorTemplate,
    placement_ref: str,
) -> int:
    explicit_start = parse_placement_ref_start(placement_ref)
    if explicit_start is not None:
        anchor_start = explicit_start % len(template.sequence) if template.circular else explicit_start
        if anchor_start < 0 or (not template.circular and anchor_start + len(sequence) > len(template.sequence)):
            raise ValidationError(
                f"normalize_anchor under_length_policy.placement_ref '{placement_ref}' is outside template "
                f"'{template.id}'."
            )
        observed = template_subsequence(
            template.sequence,
            start=anchor_start,
            length=len(sequence),
            circular=template.circular,
        )
        if observed.upper() != sequence.upper():
            raise ValidationError(
                f"normalize_anchor under_length_policy.placement_ref '{placement_ref}' points to template "
                f"sequence '{observed}', not the input anchor."
            )
        return anchor_start

    offsets = template_match_offsets(template.sequence, sequence, circular=template.circular)
    if len(offsets) != 1:
        raise ValidationError(
            "normalize_anchor under-length template expansion requires exactly one forward-strand match for the "
            f"input anchor in template '{template.id}', found {len(offsets)}. Set placement_ref to "
            "'offset:<start_0>' to disambiguate a configured placement."
        )
    return offsets[0]


def parse_placement_ref_start(placement_ref: str) -> int | None:
    text = str(placement_ref or "").strip()
    lowered = text.casefold()
    for prefix in ("offset:", "start:"):
        if lowered.startswith(prefix):
            try:
                return int(text[len(prefix) :].strip())
            except ValueError as exc:
                raise ValidationError(
                    f"normalize_anchor under_length_policy.placement_ref '{placement_ref}' must use an integer "
                    "offset after the prefix."
                ) from exc
    return int(text) if text.isdecimal() else None


def template_subsequence(template_seq: str, *, start: int, length: int, circular: bool) -> str:
    if circular:
        return "".join(template_seq[(start + idx) % len(template_seq)] for idx in range(length))
    return template_seq[start : start + length]


def _validate_replacement_interval(
    *,
    template: NormalizeAnchorTemplate,
    placement_ref: str,
    replacement_start: int,
    replacement_end: int,
) -> None:
    if replacement_start < 0 or replacement_end > len(template.sequence):
        raise ValidationError(
            f"normalize_anchor under_length_policy.placement_ref '{placement_ref}' is outside template '{template.id}'."
        )

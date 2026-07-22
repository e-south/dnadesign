"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_realization_normalize_anchor.py

Normalize-anchor realization contract tests for Construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dnadesign.construct.src.annotations.features import AnnotationFeature, AnnotationInterval
from dnadesign.construct.src.annotations.focal import FocalSelection
from dnadesign.construct.src.contracts.config import JobConfig
from dnadesign.construct.src.contracts.errors import ValidationError
from dnadesign.construct.src.realization.normalize_anchor import (
    best_trim_window,
    expand_short_sequence_from_template,
    realize_normalize_anchor,
    require_normalize_target_length_match,
)
from dnadesign.construct.src.sources.input_rows import normalize_input_scan_fields_for_schema


@dataclass(frozen=True)
class _Template:
    id: str
    sequence: str
    circular: bool = False
    kind: str = "literal"
    source: str = "fixture"
    dataset: str | None = None
    field: str | None = None
    record_id: str | None = None


def _feature(
    *,
    feature_id: str,
    start: int,
    end: int,
    role_hint: str | None = None,
) -> AnnotationFeature:
    return AnnotationFeature(
        feature_id=feature_id,
        feature_order=1,
        feature_type="misc_feature",
        label=feature_id,
        role_hint=role_hint,
        start_0=start,
        end_0=end,
        intervals_0=(AnnotationInterval(start_0=start, end_0=end, strand=1, partial=False),),
        confidence="high",
    )


def _seq_annot_feature_payload(
    *,
    feature_id: str,
    start: int,
    end: int,
    role_hint: str,
) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "feature_order": 1,
        "feature_type": "misc_feature",
        "label": feature_id,
        "role_hint": role_hint,
        "start_0": start,
        "end_0": end,
        "intervals_0": [{"start_0": start, "end_0": end, "strand": 1, "partial": False}],
        "confidence": "high",
    }


def _normalize_cfg(
    *,
    target_length: int = 6,
    input_field: str = "sequence",
    focal_offset_0: int | None = None,
    over_target_length: int | None = None,
    under_target_length: int | None = None,
    under_policy: bool = False,
    fail_if_loses_roles: list[str] | None = None,
) -> JobConfig:
    selector: dict[str, object]
    if focal_offset_0 is None:
        selector = {
            "kind": "sequence_midpoint",
            "allowed": True,
        }
    else:
        selector = {
            "kind": "sequence_offset",
            "offset_0": focal_offset_0,
        }
    normalize_anchor: dict[str, object] = {
        "product_kind": "analysis_window",
        "target_length": target_length,
        "focal_selector": {
            "kind": "chain",
            "selectors": [selector],
        },
        "over_length_policy": {
            "kind": "trim",
            "target_length": over_target_length if over_target_length is not None else target_length,
        },
        "fallback_policy": {
            "allow_low_confidence": True,
        },
        "feature_retention_policy": {
            "fail_if_loses_roles": list(fail_if_loses_roles or []),
        },
    }
    if under_policy:
        normalize_anchor["under_length_policy"] = {
            "kind": "expand_from_template",
            "target_length": under_target_length if under_target_length is not None else target_length,
            "template": {
                "source": {
                    "kind": "literal",
                    "sequence": "AAAACCCCGGGG",
                }
            },
            "placement_ref": "offset:4",
        }
    return JobConfig.model_validate(
        {
            "job": {
                "id": "normalize_anchor_fixture",
                "mode": "normalize_anchor",
                "input": {
                    "source": {
                        "kind": "usr",
                        "dataset": "input_refs",
                        "root": "/tmp/usr_root",
                    },
                    "field": input_field,
                },
                "normalize_anchor": normalize_anchor,
                "output": {
                    "target": {
                        "kind": "usr",
                        "dataset": "normalized_refs",
                        "root": "/tmp/usr_root",
                    }
                },
            }
        }
    )


def test_normalize_input_scan_fields_requires_explicit_input_field() -> None:
    with pytest.raises(ValidationError, match="job.input.field is required"):
        normalize_input_scan_fields_for_schema(input_field=None, available_fields=["id", "sequence"])


def test_normalize_input_scan_fields_adds_optional_annotation_and_label_fields() -> None:
    fields = normalize_input_scan_fields_for_schema(
        input_field="sequence",
        available_fields=[
            "id",
            "sequence",
            "usr_label__primary",
            "seq_annot__features",
            "seq_annot__record_name",
            "unrelated",
        ],
    )

    assert fields == ["id", "seq_annot__features", "seq_annot__record_name", "sequence", "usr_label__primary"]


def test_require_normalize_target_length_match_rejects_under_policy_mismatch() -> None:
    cfg = _normalize_cfg(target_length=6, under_policy=True, under_target_length=5)

    with pytest.raises(ValidationError, match="under_length_policy.target_length must match"):
        require_normalize_target_length_match(cfg=cfg)


def test_best_trim_window_prefers_retaining_required_roles_over_nearby_features() -> None:
    selection = FocalSelection(
        focal_point_0=9.0,
        focal_rule="fixture",
        focal_features=(),
        focal_confidence="high",
    )
    features = [
        _feature(feature_id="required", start=5, end=7, role_hint="must_keep"),
        _feature(feature_id="nearby", start=10, end=12, role_hint="optional"),
    ]

    assert best_trim_window(
        sequence="A" * 16,
        features=features,
        focal_selection=selection,
        target_length=6,
        required_roles=["must_keep"],
    ) == (4, 10)


def test_expand_short_sequence_requires_unique_template_match_without_offset() -> None:
    template = _Template(id="duplicate_template", sequence="AAACCCGGGCCCTTT")
    selection = FocalSelection(
        focal_point_0=1.5,
        focal_rule="fixture",
        focal_features=(),
        focal_confidence="high",
    )

    with pytest.raises(ValidationError, match="found 2"):
        expand_short_sequence_from_template(
            sequence="CCC",
            template=template,
            target_length=7,
            focal_selection=selection,
            placement_ref="template_fixture",
        )


def test_realize_normalize_anchor_rejects_short_input_without_under_length_policy() -> None:
    cfg = _normalize_cfg(target_length=4)

    with pytest.raises(ValidationError, match="requires under_length_policy"):
        realize_normalize_anchor(
            row={"id": "row_a", "sequence": "ACG"},
            cfg=cfg,
            load_template=lambda _template_cfg: pytest.fail("template loader should not be called"),
        )


def test_realize_normalize_anchor_rejects_lost_required_role() -> None:
    cfg = _normalize_cfg(target_length=4, focal_offset_0=8, fail_if_loses_roles=["must_keep"])

    with pytest.raises(ValidationError, match="would lose required roles"):
        realize_normalize_anchor(
            row={
                "id": "row_a",
                "sequence": "A" * 10,
                "seq_annot__features": [
                    _seq_annot_feature_payload(
                        feature_id="required_left_edge",
                        start=0,
                        end=2,
                        role_hint="must_keep",
                    )
                ],
            },
            cfg=cfg,
            load_template=lambda _template_cfg: pytest.fail("template loader should not be called"),
        )

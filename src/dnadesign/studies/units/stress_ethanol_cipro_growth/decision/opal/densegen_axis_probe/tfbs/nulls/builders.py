"""Public builders for DenseGen TFBS matched null label tables."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..oracle import validate_tfbs_label_algebra
from ..schema import (
    TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION,
    TFBS_LEARNABILITY_SLOT_GEOMETRY_NULL_VERSION,
)
from .contracts import (
    TFBS_ACTIVE_NUMERIC_COLUMNS,
    TFBS_CONTENT_BLOCK_COLUMNS,
    TFBS_PASSIVE_STRATUM_COLUMNS,
    TFBS_SLOT_COUNT_MATCH_COLUMNS,
    TFBS_SLOT_EVENT_COLUMNS,
    TFBS_SLOT_FAMILY_COLUMNS,
    TfbsNullBuild,
    TfbsNullConfig,
)
from .reports import _content_warnings, _null_viability_report, _slot_warnings, _with_null_metadata
from .strata import _permuted_donor_positions, _require_columns, _select_viable_stratum
from .validation import (
    _recompute_slot_event_columns,
    _validate_count_matching,
    _validate_label_distribution,
    _validate_slot_label_consistency,
)


def build_tfbs_family_content_matched_null(
    labels: pd.DataFrame,
    *,
    seed: int,
    label_name: str = "tf_family_content_block",
    stratum_candidates: Sequence[Sequence[str]] = (
        ("sigma35_variant", "spacer_length"),
        ("sigma35_variant",),
        (),
    ),
    config: TfbsNullConfig | None = None,
) -> TfbsNullBuild:
    """Permute the v1 TFBS content-label block within matched sigma-core strata."""

    cfg = config or TfbsNullConfig()
    frame = labels.reset_index(drop=True).copy()
    _require_columns(frame, ("id", "quality_flag", *TFBS_CONTENT_BLOCK_COLUMNS, *TFBS_PASSIVE_STRATUM_COLUMNS))
    selected = _select_viable_stratum(frame, stratum_candidates=stratum_candidates, config=cfg)
    donor_positions = _permuted_donor_positions(frame, selected.stratum_columns, seed=seed)
    out = frame.copy()
    donor = frame.iloc[donor_positions].reset_index(drop=True)
    for column in TFBS_CONTENT_BLOCK_COLUMNS:
        out[column] = donor[column].to_numpy()
    _validate_label_distribution(frame, out, columns=TFBS_ACTIVE_NUMERIC_COLUMNS)
    validate_tfbs_label_algebra(out)
    _validate_slot_label_consistency(out)
    report = _null_viability_report(
        before=frame,
        after=out,
        null_version=TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION,
        seed=seed,
        label_name=label_name,
        selected=selected,
        config=cfg,
        compare_columns=TFBS_CONTENT_BLOCK_COLUMNS,
        label_joint_columns=TFBS_ACTIVE_NUMERIC_COLUMNS,
        null_control_role="matched_label_permutation_negative_control",
        preserved_signal="sigma-core stratum and label marginal distributions",
        disrupted_signal="row association between sequence identity and TFBS content labels",
        negative_control_claim_status="VALID_AS_NEGATIVE_CONTROL",
        warnings=_content_warnings(selected),
    )
    return TfbsNullBuild(
        labels=_with_null_metadata(
            out,
            null_version=TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION,
            null_control_role="matched_label_permutation_negative_control",
            negative_control_claim_status="VALID_AS_NEGATIVE_CONTROL",
            seed=seed,
            positive_labels=frame,
            selected=selected,
        ),
        null_viability_report=report,
    )


def build_tfbs_slot_geometry_count_matched_null(
    labels: pd.DataFrame,
    *,
    label_name: str,
    seed: int,
    stratum_candidates: Sequence[Sequence[str]] = (
        ("sigma35_variant", "spacer_length", "lexA_count", "cpxR_count", "baeR_count"),
        ("sigma35_variant", "lexA_count", "cpxR_count", "baeR_count"),
        ("lexA_count", "cpxR_count", "baeR_count"),
    ),
    config: TfbsNullConfig | None = None,
) -> TfbsNullBuild:
    """Permute slot-family geometry while preserving row-level TF family counts."""

    if label_name not in TFBS_SLOT_EVENT_COLUMNS:
        raise ValueError(f"slot-geometry null label_name must be a v1 slot label, got {label_name!r}")
    cfg = config or TfbsNullConfig()
    frame = labels.reset_index(drop=True).copy()
    _require_columns(
        frame,
        (
            "id",
            "quality_flag",
            *TFBS_SLOT_FAMILY_COLUMNS,
            *TFBS_SLOT_EVENT_COLUMNS,
            *TFBS_SLOT_COUNT_MATCH_COLUMNS,
            *TFBS_PASSIVE_STRATUM_COLUMNS,
        ),
    )
    selected = _select_viable_stratum(frame, stratum_candidates=stratum_candidates, config=cfg)
    donor_positions = _permuted_donor_positions(frame, selected.stratum_columns, seed=seed)
    out = frame.copy()
    donor = frame.iloc[donor_positions].reset_index(drop=True)
    for column in TFBS_SLOT_FAMILY_COLUMNS:
        out[column] = donor[column].to_numpy()
    _recompute_slot_event_columns(out)
    _validate_count_matching(frame, out)
    _validate_label_distribution(frame, out, columns=TFBS_SLOT_EVENT_COLUMNS)
    validate_tfbs_label_algebra(out)
    _validate_slot_label_consistency(out)
    report = _null_viability_report(
        before=frame,
        after=out,
        null_version=TFBS_LEARNABILITY_SLOT_GEOMETRY_NULL_VERSION,
        seed=seed,
        label_name=label_name,
        selected=selected,
        config=cfg,
        compare_columns=(label_name,),
        label_joint_columns=(*TFBS_SLOT_FAMILY_COLUMNS, *TFBS_SLOT_EVENT_COLUMNS),
        null_control_role="count_preserving_slot_confound_control",
        preserved_signal="row-level TF family counts",
        disrupted_signal="slot-family assignment conditional on preserved counts",
        negative_control_claim_status="CONFOUND_CONTROL_ONLY",
        warnings=_slot_warnings(selected),
    )
    return TfbsNullBuild(
        labels=_with_null_metadata(
            out,
            null_version=TFBS_LEARNABILITY_SLOT_GEOMETRY_NULL_VERSION,
            null_control_role="count_preserving_slot_confound_control",
            negative_control_claim_status="CONFOUND_CONTROL_ONLY",
            seed=seed,
            positive_labels=frame,
            selected=selected,
        ),
        null_viability_report=report,
    )

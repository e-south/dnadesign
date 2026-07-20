"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_normalization.py

Reader-bootstrap normalization for the stress-study behavior shadow objective.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.contracts import StressTargetView
from .multistate_behavior_cohort import (
    VerifiedBehaviorCohortReceipt,
    behavior_normalization_source_rows_sha256,
    validated_behavior_evidence,
    verify_behavior_cohort_receipt,
)
from .multistate_behavior_protocol import MultistateBehaviorShadowProtocol

NORMALIZATION_SCHEMA_ID = "stress_ethanol_cipro_growth.multistate_response_behavior_normalization.v1"
_SOURCE_DIGEST_FIELDS = {
    "reader_bundle_manifest_sha256",
    "reader_request_sha256",
    "candidate_bindings_manifest_sha256",
    "observation_policy_sha256",
}


@dataclass(frozen=True)
class MultistateBehaviorNormalizationEvidence:
    """One soft-min scale plus the response and signal rows that establish it."""

    protocol: MultistateBehaviorShadowProtocol
    softmin_scale: float
    bootstrap_samples: int
    unit_count: int
    response_pair_count: int
    response_resolution_rows: pd.DataFrame
    signal_resolution_rows: pd.DataFrame
    source_rows_sha256: str
    verified_cohort_receipt: VerifiedBehaviorCohortReceipt | None = None

    @property
    def scale_basis(self) -> str:
        return self.protocol.normalization.scale_basis

    @property
    def event_time_role(self) -> str:
        return self.protocol.normalization.event_time_role

    @property
    def repeat_role(self) -> str:
        return self.protocol.normalization.repeat_role

    @property
    def censor_role(self) -> str:
        return self.protocol.normalization.censor_role

    @property
    def normalization(self) -> dict[str, float]:
        return {"softmin_scale": self.softmin_scale}


def derive_multistate_behavior_normalization(
    labels: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
    target_views: tuple[StressTargetView, ...],
    verified_cohort_receipt: VerifiedBehaviorCohortReceipt | None = None,
) -> MultistateBehaviorNormalizationEvidence:
    """Derive one soft-min scale from pooled response and signal resolution rows."""

    protocol.assert_target_views(target_views)
    label_rows, draw_rows = validated_behavior_evidence(labels, draws, protocol=protocol)
    pairs = _declared_pair_union(protocol, target_views=target_views)
    pair_views = _declaring_views(protocol, target_views=target_views)
    response_records: list[dict[str, object]] = []
    signal_records: list[dict[str, object]] = []
    draw_count = int(draw_rows.groupby("id", sort=False)["draw_index"].nunique().iloc[0])

    for label in label_rows.itertuples(index=False):
        unit_id = str(label.id)
        unit_draws = draw_rows.loc[draw_rows["id"].astype(str).eq(unit_id)]
        identity = {
            "id": unit_id,
            "candidate_id": str(label.candidate_id),
            "reader_experiment_id": str(label.reader_experiment_id),
        }
        for state_a, state_b in pairs:
            contrast = unit_draws[f"r{state_a}"].to_numpy(dtype=float) - unit_draws[f"r{state_b}"].to_numpy(dtype=float)
            response_records.append(
                {
                    **identity,
                    "state_a": state_a,
                    "state_b": state_b,
                    "declared_by_selection_views": ",".join(pair_views[(state_a, state_b)]),
                    "bootstrap_sd": float(np.std(contrast, ddof=1)),
                    "bootstrap_samples": draw_count,
                }
            )
        for state_id in protocol.state_ids:
            signal_records.append(
                {
                    **identity,
                    "state_id": state_id,
                    "bootstrap_sd": float(np.std(unit_draws[f"b{state_id}"].to_numpy(dtype=float), ddof=1)),
                    "bootstrap_samples": draw_count,
                }
            )

    response_rows = pd.DataFrame.from_records(response_records).sort_values(
        ["id", "state_a", "state_b"], kind="mergesort"
    )
    signal_rows = pd.DataFrame.from_records(signal_records).sort_values(["id", "state_id"], kind="mergesort")
    quantile = protocol.normalization.scale_quantile
    method = protocol.normalization.quantile_method
    pooled_sd = np.concatenate(
        [
            response_rows["bootstrap_sd"].to_numpy(dtype=float),
            signal_rows["bootstrap_sd"].to_numpy(dtype=float),
        ]
    )
    softmin_scale = float(np.quantile(pooled_sd, quantile, method=method))
    _assert_positive_scale(softmin_scale)
    source_rows_sha256 = behavior_normalization_source_rows_sha256(
        label_rows,
        draw_rows,
        protocol=protocol,
    )
    if verified_cohort_receipt is not None:
        verify_behavior_cohort_receipt(
            label_rows,
            protocol=protocol,
            receipt=verified_cohort_receipt,
        )
        if source_rows_sha256 != verified_cohort_receipt.source_rows_sha256:
            raise ValueError("behavior cohort receipt source-row digest disagrees with the exhaustive projection.")
    return MultistateBehaviorNormalizationEvidence(
        protocol=protocol,
        softmin_scale=softmin_scale,
        bootstrap_samples=draw_count,
        unit_count=len(label_rows),
        response_pair_count=len(pairs),
        response_resolution_rows=response_rows.reset_index(drop=True),
        signal_resolution_rows=signal_rows.reset_index(drop=True),
        source_rows_sha256=source_rows_sha256,
        verified_cohort_receipt=verified_cohort_receipt,
    )


def build_multistate_behavior_normalization_record(
    evidence: MultistateBehaviorNormalizationEvidence,
    *,
    source_artifact_digests: Mapping[str, str],
) -> dict[str, object]:
    """Build the digest-bearing record a shadow publication must persist."""

    observed = set(source_artifact_digests)
    if observed != _SOURCE_DIGEST_FIELDS:
        raise ValueError(
            "behavior normalization source digests must be exactly "
            f"{sorted(_SOURCE_DIGEST_FIELDS)}; missing={sorted(_SOURCE_DIGEST_FIELDS - observed)}, "
            f"extra={sorted(observed - _SOURCE_DIGEST_FIELDS)}."
        )
    digests = {key: _canonical_digest(value, field=key) for key, value in source_artifact_digests.items()}
    protocol = evidence.protocol
    receipt = evidence.verified_cohort_receipt
    if receipt is None:
        raise ValueError("behavior normalization publication requires an exhaustive verified cohort receipt.")
    expected_receipt_digests = {
        "reader_bundle_manifest_sha256": _canonical_digest(
            receipt.reader_bundle_manifest_sha256,
            field="receipt.reader_bundle_manifest_sha256",
        ),
        "candidate_bindings_manifest_sha256": _canonical_digest(
            receipt.candidate_bindings_manifest_sha256,
            field="receipt.candidate_bindings_manifest_sha256",
        ),
    }
    if any(digests[field] != value for field, value in expected_receipt_digests.items()):
        raise ValueError("behavior normalization source digests disagree with the exhaustive cohort receipt.")
    return {
        "schema_id": NORMALIZATION_SCHEMA_ID,
        "schema_version": "1",
        "study_id": protocol.study_id,
        "protocol_id": protocol.protocol_id,
        "status": protocol.status,
        "activation": {
            "campaign": protocol.campaign_activation,
            "synthesis": protocol.synthesis_authorization,
        },
        "objective": {
            "name": protocol.objective_name,
            "family_weighting": protocol.family_weighting,
        },
        "assay": {
            "state_ids": list(protocol.state_ids),
            "primary_reduction_id": protocol.primary_reduction_id,
            "fluorescence_reference": protocol.fluorescence_reference,
        },
        "target_views": [
            {"id": view.id, "target_mask": [int(value) for value in view.target_mask]} for view in protocol.target_views
        ],
        "normalization": {
            "softmin_scale": evidence.softmin_scale,
            "scale_quantile": protocol.normalization.scale_quantile,
            "quantile_method": protocol.normalization.quantile_method,
            "scale_basis": protocol.normalization.scale_basis,
            "pair_deduplication": protocol.normalization.pair_deduplication,
            "cohort_id": protocol.normalization.cohort_id,
            "unit": protocol.normalization.unit,
            "unit_count": evidence.unit_count,
            "candidate_count": receipt.candidate_count,
            "reader_experiment_count": receipt.reader_experiment_count,
            "excluded_nonexact_unit_count": receipt.excluded_nonexact_unit_count,
            "response_pair_count": evidence.response_pair_count,
            "bootstrap_samples": evidence.bootstrap_samples,
        },
        "evidence_roles": {
            "bootstrap": protocol.normalization.bootstrap_role,
            "event_time": evidence.event_time_role,
            "repeat": evidence.repeat_role,
            "censor": evidence.censor_role,
        },
        "source": {
            "protocol_sha256": f"sha256:{protocol.source_sha256}",
            "source_rows_sha256": f"sha256:{evidence.source_rows_sha256}",
            **digests,
        },
    }


def verify_multistate_behavior_normalization_source(
    labels: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    evidence: MultistateBehaviorNormalizationEvidence,
) -> None:
    """Require scoring evidence to reproduce the cohort that fixed the scale."""

    label_rows, draw_rows = validated_behavior_evidence(labels, draws, protocol=evidence.protocol)
    observed_digest = behavior_normalization_source_rows_sha256(
        label_rows,
        draw_rows,
        protocol=evidence.protocol,
    )
    if observed_digest != evidence.source_rows_sha256:
        raise ValueError(
            "behavior scoring evidence does not reproduce the normalization source rows: "
            f"expected=sha256:{evidence.source_rows_sha256}, observed=sha256:{observed_digest}."
        )
    if len(label_rows) != evidence.unit_count:
        raise ValueError(
            "behavior normalization cohort unit count drifted: "
            f"expected={evidence.unit_count}, observed={len(label_rows)}."
        )


def _declared_pair_union(
    protocol: MultistateBehaviorShadowProtocol,
    *,
    target_views: tuple[StressTargetView, ...],
) -> tuple[tuple[str, str], ...]:
    state_index = {state_id: index for index, state_id in enumerate(protocol.state_ids)}
    pairs: set[tuple[str, str]] = set()
    for view in target_views:
        on_states = [protocol.state_ids[index] for index, value in enumerate(view.target_mask) if value == 1.0]
        off_states = [protocol.state_ids[index] for index, value in enumerate(view.target_mask) if value == 0.0]
        for on_state in on_states:
            for off_state in off_states:
                left, right = sorted((on_state, off_state), key=state_index.__getitem__)
                pairs.add((left, right))
    return tuple(sorted(pairs, key=lambda pair: (state_index[pair[0]], state_index[pair[1]])))


def _declaring_views(
    protocol: MultistateBehaviorShadowProtocol,
    *,
    target_views: tuple[StressTargetView, ...],
) -> dict[tuple[str, str], tuple[str, ...]]:
    state_index = {state_id: index for index, state_id in enumerate(protocol.state_ids)}
    declared: dict[tuple[str, str], list[str]] = {}
    for view in target_views:
        on_states = [protocol.state_ids[index] for index, value in enumerate(view.target_mask) if value == 1.0]
        off_states = [protocol.state_ids[index] for index, value in enumerate(view.target_mask) if value == 0.0]
        for on_state in on_states:
            for off_state in off_states:
                pair = tuple(sorted((on_state, off_state), key=state_index.__getitem__))
                declared.setdefault(pair, []).append(view.id)  # type: ignore[arg-type]
    return {pair: tuple(sorted(set(view_ids))) for pair, view_ids in declared.items()}


def _assert_positive_scale(value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"behavior soft-min scale must be positive and finite; observed {value!r}.")


def _canonical_digest(value: str, *, field: str) -> str:
    text = str(value).removeprefix("sha256:")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest.")
    return f"sha256:{text}"


__all__ = [
    "NORMALIZATION_SCHEMA_ID",
    "MultistateBehaviorNormalizationEvidence",
    "build_multistate_behavior_normalization_record",
    "derive_multistate_behavior_normalization",
    "verify_multistate_behavior_normalization_source",
]

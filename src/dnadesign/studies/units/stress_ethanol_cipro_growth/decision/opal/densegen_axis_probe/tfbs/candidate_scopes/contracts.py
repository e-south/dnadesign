"""Stable candidate-scope ontology for DenseGen TFBS probe campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

COUNT_FIXED_SLOT_POSITION_SCOPE_POLICY_ID = "tfbs_slot_position_target_count_eq_1_v1"
COUNT_FIXED_SLOT_POSITION_SCOPE_VALUE = 1


@dataclass(frozen=True)
class TfbsCandidateScopePolicy:
    """Contract for a label-specific candidate universe restriction."""

    label_name: str
    policy_id: str
    target_family_count_column: str
    required_count_value: int
    claim_boundary: str

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_scope_policy_id"] = payload.pop("policy_id")
        return payload


@dataclass(frozen=True)
class TfbsCandidateScope:
    """Materialized candidate IDs for a label-specific Stage B campaign scope."""

    policy: TfbsCandidateScopePolicy
    ids: tuple[str, ...]
    row_count: int
    positive_label_marginal: dict[str, int]

    def to_manifest(self) -> dict[str, Any]:
        payload = self.policy.to_manifest()
        payload.update(
            {
                "row_count": int(self.row_count),
                "positive_label_marginal": dict(self.positive_label_marginal),
            }
        )
        return payload


_COUNT_FIXED_SLOT_POSITION_POLICIES = MappingProxyType(
    {
        "lexA_in_slot0": TfbsCandidateScopePolicy(
            label_name="lexA_in_slot0",
            policy_id=COUNT_FIXED_SLOT_POSITION_SCOPE_POLICY_ID,
            target_family_count_column="lexA_count",
            required_count_value=COUNT_FIXED_SLOT_POSITION_SCOPE_VALUE,
            claim_boundary=(
                "Candidate universe is restricted to rows with exactly one LexA motif. Enrichment can therefore "
                "not be explained by selecting rows with more LexA motifs."
            ),
        ),
        "cpxR_or_baeR_in_slot2": TfbsCandidateScopePolicy(
            label_name="cpxR_or_baeR_in_slot2",
            policy_id=COUNT_FIXED_SLOT_POSITION_SCOPE_POLICY_ID,
            target_family_count_column="cpxR_or_baeR_count",
            required_count_value=COUNT_FIXED_SLOT_POSITION_SCOPE_VALUE,
            claim_boundary=(
                "Candidate universe is restricted to rows with exactly one CpxR-or-BaeR motif. Enrichment can "
                "therefore not be explained by selecting rows with more CpxR/BaeR motifs."
            ),
        ),
    }
)


def count_fixed_slot_position_scope_policy(label_name: str) -> TfbsCandidateScopePolicy:
    """Return the declared count-fixed policy for a supported slot-position label."""

    label = str(label_name).strip()
    try:
        return _COUNT_FIXED_SLOT_POSITION_POLICIES[label]
    except KeyError as exc:
        raise ValueError(f"unsupported count-fixed slot-position label: {label_name!r}") from exc


def is_count_fixed_slot_position_label(label_name: str) -> bool:
    """Return whether a label owns a declared count-fixed slot-position scope."""

    return str(label_name).strip() in _COUNT_FIXED_SLOT_POSITION_POLICIES

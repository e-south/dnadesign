"""Contracts and label-block ontology for TFBS permutation nulls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..schema import TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES

TFBS_COUNT_COLUMNS = ("lexA_count", "cpxR_count", "baeR_count", "cpxR_or_baeR_count")
TFBS_PRESENCE_COLUMNS = ("lexA_present", "cpxR_present", "baeR_present", "cpxR_or_baeR_present")
TFBS_COUNT_FRACTION_COLUMNS = (
    "lexA_count_fraction",
    "cpxR_count_fraction",
    "baeR_count_fraction",
    "cpxR_or_baeR_count_fraction",
)
TFBS_SLOT_EVENT_COLUMNS = (
    "lexA_in_slot0",
    "lexA_in_slot1",
    "lexA_in_slot2",
    "baeR_in_slot1",
    "cpxR_or_baeR_in_slot0",
    "cpxR_or_baeR_in_slot1",
    "cpxR_or_baeR_in_slot2",
)
TFBS_SLOT_FAMILY_COLUMNS = ("slot0_family", "slot1_family", "slot2_family")
TFBS_PASSIVE_STRATUM_COLUMNS = ("sigma35_variant", "spacer_length")
TFBS_CONTENT_BLOCK_COLUMNS = (
    *TFBS_COUNT_COLUMNS,
    *TFBS_PRESENCE_COLUMNS,
    *TFBS_COUNT_FRACTION_COLUMNS,
    *TFBS_SLOT_EVENT_COLUMNS,
    *TFBS_SLOT_FAMILY_COLUMNS,
)
TFBS_SLOT_COUNT_MATCH_COLUMNS = ("lexA_count", "cpxR_count", "baeR_count")
TFBS_ACTIVE_NUMERIC_COLUMNS = tuple(TFBS_LEARNABILITY_ACTIVE_LABEL_NAMES)


@dataclass(frozen=True)
class TfbsNullConfig:
    """Viability thresholds for matched TFBS permutation nulls."""

    tiny_stratum_threshold: int = 3
    fail_if_fraction_rows_in_singleton_strata_gt: float = 0.01
    fail_if_fraction_rows_in_tiny_strata_gt: float = 0.05
    fail_on_weak_exchangeability: bool = True
    warn_if_unchanged_label_fraction_ge: float = 0.50
    fail_if_unchanged_label_fraction_ge: float = 0.75


@dataclass(frozen=True)
class TfbsNullBuild:
    """Null label table plus the manifest-ready viability report."""

    labels: pd.DataFrame
    null_viability_report: dict[str, Any]

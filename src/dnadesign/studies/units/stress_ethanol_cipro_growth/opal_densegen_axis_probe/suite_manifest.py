"""Manifest-backed suite planning for the DenseGen motif QA probe."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .constants import (
    ACTIVE_LABEL_FAMILY_ID,
    ACTIVE_LABEL_FAMILY_IDS,
    DEFAULT_INITIAL_LABELS,
    DEFAULT_SUITE_ID,
    DEFAULT_SUITE_SEEDS,
    DEFAULT_TOP_K,
    PASSIVE_LABEL_FAMILY_IDS,
    SPLITS,
)
from .label_families import label_family_records


@dataclass(frozen=True)
class ProbeSuiteSpec:
    suite_id: str
    selection_k: int
    initial_label_count: int
    seeds: tuple[int, ...]
    rounds: int
    splits: tuple[str, ...]
    active_label_family: str
    active_label_families: tuple[str, ...]
    passive_label_families: tuple[str, ...]
    active_tasks: tuple[str, ...]
    null_strategy: str
    interpretation_boundary: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("seeds", "splits", "active_label_families", "passive_label_families", "active_tasks"):
            payload[key] = list(payload[key])
        return payload


def default_probe_suite() -> ProbeSuiteSpec:
    return ProbeSuiteSpec(
        suite_id=DEFAULT_SUITE_ID,
        selection_k=DEFAULT_TOP_K,
        initial_label_count=DEFAULT_INITIAL_LABELS,
        seeds=DEFAULT_SUITE_SEEDS,
        rounds=12,
        splits=SPLITS,
        active_label_family=ACTIVE_LABEL_FAMILY_ID,
        active_label_families=ACTIVE_LABEL_FAMILY_IDS,
        passive_label_families=PASSIVE_LABEL_FAMILY_IDS,
        active_tasks=("cipro", "ethanol", "dual"),
        null_strategy="global_quality_ok_permutation",
        interpretation_boundary=(
            "QA for motif-composition recoverability in the configured OPAL X space; "
            "not a growth phenotype or biology claim."
        ),
    )


def suite_manifest_payload(suite: ProbeSuiteSpec | None = None) -> dict[str, Any]:
    spec = suite or default_probe_suite()
    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_motif_qa_suite.v1",
        **spec.to_dict(),
        "label_families": label_family_records(),
        "qa_metrics": {
            "primary": [
                "positive_lift_auc",
                "null_lift_auc",
                "paired_auc_delta",
                "final_positive_minus_null_lift",
            ],
            "diagnostic": [
                "null_round_spikes",
                "final_lift_delta",
                "seed_level_spread",
                "passive_label_family_readouts",
            ],
        },
        "notebook_boundary": (
            "Canonical OPAL notebooks consume campaign-generic plot manifests. "
            "Study aggregate probe figures stay in the study report unless promoted through the OPAL plot registry."
        ),
    }

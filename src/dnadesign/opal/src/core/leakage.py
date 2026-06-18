"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/core/leakage.py

Fail-fast leakage and contamination guards for generic OPAL campaign state.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config.types import LabelSourceUSRSidecar, RootConfig
from .utils import LeakageContractError

LEAKAGE_GUARD_SCHEMA_VERSION = "opal.leakage_guard.v1"


@dataclass(frozen=True)
class LeakageViolation:
    code: str
    scope: str
    message: str
    count: int = 0
    sample_ids: tuple[str, ...] = ()
    owner: str = "opal"
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sample_ids"] = list(self.sample_ids)
        return payload


@dataclass(frozen=True)
class LeakageGuardReport:
    scope: str
    checks: tuple[str, ...]
    violations: tuple[LeakageViolation, ...]
    owner: str = "opal"
    schema_version: str = LEAKAGE_GUARD_SCHEMA_VERSION

    @property
    def status(self) -> str:
        return "fail" if self.violations else "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "owner": self.owner,
            "scope": self.scope,
            "checks": list(self.checks),
            "status": self.status,
            "violations": [violation.to_dict() for violation in self.violations],
        }


def _sample(values: set[str], limit: int = 10) -> tuple[str, ...]:
    return tuple(sorted(values)[:limit])


def _value_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        if len(value) == 0:
            return False
        return any(_value_is_present(item) for item in list(value))
    try:
        missing = pd.isna(value)
    except Exception:
        return True
    if isinstance(missing, (bool, np.bool_)):
        return not bool(missing)
    return not bool(np.asarray(missing).all())


def build_shared_label_source_contamination_report(
    *,
    cfg: RootConfig,
    store: Any,
    df: pd.DataFrame,
) -> LeakageGuardReport:
    checks = (
        "usr_sidecar_no_current_y_column_values",
        "usr_sidecar_no_campaign_label_entries",
        "ledger_only_no_campaign_label_history_entries",
    )
    if not isinstance(cfg.labels.source, LabelSourceUSRSidecar):
        return LeakageGuardReport(scope="label_source", checks=checks, violations=())

    violations: list[LeakageViolation] = []
    y_col = str(cfg.data.y_column_name)
    if y_col in df.columns:
        y_ids = {
            str(row_id)
            for row_id, value in df[["id", y_col]].itertuples(index=False, name=None)
            if _value_is_present(value)
        }
        if y_ids:
            violations.append(
                LeakageViolation(
                    code="records_y_column_contamination",
                    scope="label_source",
                    message=(
                        f"Shared usr_sidecar campaign {cfg.campaign.slug!r} found non-empty current-Y values "
                        f"in records column {y_col!r}; shared labels must come from the configured sidecar."
                    ),
                    count=len(y_ids),
                    sample_ids=_sample(y_ids),
                )
            )

    hist_col = store.label_hist_col()
    if hist_col in df.columns:
        entry_ids: set[str] = set()
        label_entry_ids: set[str] = set()
        malformed_ids: set[str] = set()
        for row_id, cell in df[["id", hist_col]].itertuples(index=False, name=None):
            rid = str(row_id)
            try:
                entries = store._parse_hist_cell_strict(cell)
            except Exception:
                malformed_ids.add(rid)
                continue
            if entries:
                entry_ids.add(rid)
            if any(str(entry.get("kind", "")).strip() == "label" for entry in entries):
                label_entry_ids.add(rid)

        if malformed_ids:
            violations.append(
                LeakageViolation(
                    code="records_label_history_malformed",
                    scope="label_source",
                    message=(
                        f"Shared usr_sidecar campaign {cfg.campaign.slug!r} found malformed campaign-local "
                        f"label history in {hist_col!r}."
                    ),
                    count=len(malformed_ids),
                    sample_ids=_sample(malformed_ids),
                )
            )
        if label_entry_ids and str(cfg.writeback.prediction_records) != "ledger_only":
            violations.append(
                LeakageViolation(
                    code="records_label_history_label_contamination",
                    scope="label_source",
                    message=(
                        f"Shared usr_sidecar campaign {cfg.campaign.slug!r} found campaign-local observed labels "
                        f"in {hist_col!r}; observed labels must come from the configured sidecar."
                    ),
                    count=len(label_entry_ids),
                    sample_ids=_sample(label_entry_ids),
                )
            )
        if str(cfg.writeback.prediction_records) == "ledger_only" and entry_ids:
            violations.append(
                LeakageViolation(
                    code="records_label_history_contamination",
                    scope="label_source",
                    message=(
                        f"Shared usr_sidecar campaign {cfg.campaign.slug!r} is ledger_only but records column "
                        f"{hist_col!r} contains campaign-local label-history entries."
                    ),
                    count=len(entry_ids),
                    sample_ids=_sample(entry_ids),
                )
            )

    return LeakageGuardReport(scope="label_source", checks=checks, violations=tuple(violations))


def build_train_eval_leakage_report(
    *,
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    as_of_round: int,
    selection_excludes_labeled: bool,
) -> LeakageGuardReport:
    checks = ("train_eval_disjoint_when_excluding_labeled",)
    if not selection_excludes_labeled or train_df.empty or candidate_df.empty:
        return LeakageGuardReport(scope=f"round:{int(as_of_round)}", checks=checks, violations=())
    train_ids = set(train_df["id"].astype(str).tolist()) if "id" in train_df.columns else set()
    candidate_ids = set(candidate_df["id"].astype(str).tolist()) if "id" in candidate_df.columns else set()
    overlap = train_ids & candidate_ids
    violations: list[LeakageViolation] = []
    if overlap:
        violations.append(
            LeakageViolation(
                code="train_eval_overlap",
                scope=f"round:{int(as_of_round)}",
                message=(
                    "Training IDs remain in the candidate/eval pool even though selection is configured "
                    "to exclude already-labeled candidates."
                ),
                count=len(overlap),
                sample_ids=_sample(overlap),
            )
        )
    return LeakageGuardReport(scope=f"round:{int(as_of_round)}", checks=checks, violations=tuple(violations))


def build_prediction_identity_report(
    *,
    prediction_ids: Any,
    scope: str,
) -> LeakageGuardReport:
    checks = ("prediction_ids_unique",)
    ids = pd.Series([str(value) for value in list(prediction_ids)], dtype="object")
    duplicate_mask = ids.duplicated(keep=False)
    violations: list[LeakageViolation] = []
    if bool(duplicate_mask.any()):
        duplicate_ids = set(ids.loc[duplicate_mask].tolist())
        violations.append(
            LeakageViolation(
                code="duplicate_prediction_ids",
                scope=scope,
                message="Prediction evidence contains duplicate candidate IDs for the selected run/round scope.",
                count=len(duplicate_ids),
                sample_ids=_sample(duplicate_ids),
            )
        )
    return LeakageGuardReport(scope=scope, checks=checks, violations=tuple(violations))


def build_selected_ids_scope_report(
    *,
    selected_ids: Any,
    prediction_ids: Any,
    scope: str,
) -> LeakageGuardReport:
    checks = ("selected_ids_within_prediction_scope",)
    selected = {str(value) for value in list(selected_ids)}
    predicted = {str(value) for value in list(prediction_ids)}
    outside_scope = selected - predicted
    violations: list[LeakageViolation] = []
    if outside_scope:
        violations.append(
            LeakageViolation(
                code="selected_ids_outside_eval",
                scope=scope,
                message="Selection evidence references IDs absent from the run-scoped prediction/eval evidence.",
                count=len(outside_scope),
                sample_ids=_sample(outside_scope),
            )
        )
    return LeakageGuardReport(scope=scope, checks=checks, violations=tuple(violations))


def assert_no_leakage_violations(report: LeakageGuardReport) -> None:
    if not report.violations:
        return
    details = "; ".join(
        f"{violation.code}(count={violation.count}, sample_ids={list(violation.sample_ids)})"
        for violation in report.violations
    )
    raise LeakageContractError(f"LeakageContractError: {report.scope} failed OPAL leakage guard: {details}")

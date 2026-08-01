"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/evaluation/readiness.py

Bridge receipt validation and readiness-only meta-study decisions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from ..contracts._values import MetastudyContractError, canonical_digest
from ..contracts.decision import DECISION_CONTRACT_ID, MetastudyDecision
from ..contracts.materialization import EvidenceReadiness
from ..contracts.protocol import DEFAULT_PROTOCOL, MetastudyProtocol, protocol_digest

METASTUDY_ROUTE_ID = "rt_lnrna_reporter_response_metastudy"


def decision_from_readiness(
    readiness: EvidenceReadiness,
    *,
    protocol: MetastudyProtocol = DEFAULT_PROTOCOL,
) -> MetastudyDecision:
    """Create the evidence-free blocked decision required by an unready route."""

    if not isinstance(readiness, EvidenceReadiness) or not readiness.is_receipt_validated:
        raise MetastudyContractError("readiness must come from readiness_from_receipt")
    ready_kinetic_ids = set(readiness.ready_experiment_ids) & set(protocol.planned_kinetic_experiment_ids)
    if len(ready_kinetic_ids) >= protocol.minimum_kinetic_experiments:
        raise MetastudyContractError("ready evidence requires profile evaluation, not a readiness-only decision")
    blocker = f"reader_evidence_ready_{readiness.ready_experiment_count}_of_{readiness.selected_experiment_count}"
    return MetastudyDecision(
        contract_id=DECISION_CONTRACT_ID,
        protocol_id=protocol.protocol_id,
        condition_ontology_digest=protocol.condition_ontology_digest,
        status="blocked",
        selection_use="descriptive_comparison",
        evidence_grade="none",
        selected_reduction=None,
        blockers=(blocker, "minimum_7_of_8_kinetic_experiments_not_met"),
        limitations=(),
        policy_digest=protocol_digest(protocol),
        evidence_digest=canonical_digest(
            {
                "receipt_digest": readiness.receipt_digest,
                "selected_experiment_count": readiness.selected_experiment_count,
                "ready_experiment_count": readiness.ready_experiment_count,
                "ready_experiment_ids": readiness.ready_experiment_ids,
                "blocked_experiment_ids": readiness.blocked_experiment_ids,
            }
        ),
        readiness=readiness,
        evaluations=(),
        materialization_attempts=(),
    )


def readiness_from_receipt(payload: Mapping[str, object]) -> EvidenceReadiness:
    """Adapt one public read-only readiness receipt without importing its producer."""

    if not isinstance(payload, Mapping):
        raise MetastudyContractError("readiness receipt must be an object")
    expected_top_level = {
        "available_protocols",
        "contract_errors",
        "experiments",
        "ok",
        "reader_command",
        "route_id",
        "selected_blockers",
        "summary",
    }
    if set(payload) != expected_top_level:
        raise MetastudyContractError("readiness receipt top-level fields do not match the exact contract")
    if payload["route_id"] != METASTUDY_ROUTE_ID:
        raise MetastudyContractError(f"readiness receipt route_id must equal {METASTUDY_ROUTE_ID}")
    summary = payload["summary"]
    blockers = payload["selected_blockers"]
    experiments = payload["experiments"]
    contract_errors = payload["contract_errors"]
    if (
        not isinstance(summary, Mapping)
        or not isinstance(blockers, list)
        or not isinstance(experiments, list)
        or not isinstance(contract_errors, list)
    ):
        raise MetastudyContractError("readiness receipt requires summary and selected_blockers")
    expected_summary = {
        "contract_error_count",
        "experiment_count",
        "membership_count",
        "related_membership_count",
        "selected_blocker_count",
        "selected_membership_count",
        "selected_ready_count",
    }
    if set(summary) != expected_summary:
        raise MetastudyContractError("readiness receipt summary fields do not match the exact contract")
    for field in expected_summary:
        value = summary[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MetastudyContractError(f"readiness receipt summary.{field} must be a non-negative integer")
    if len(contract_errors) != summary["contract_error_count"]:
        raise MetastudyContractError("readiness contract_error_count does not match contract_errors")
    if contract_errors:
        raise MetastudyContractError("readiness receipt contains contract_errors")
    selected = summary.get("selected_membership_count")
    ready = summary.get("selected_ready_count")
    if isinstance(selected, bool) or not isinstance(selected, int):
        raise MetastudyContractError("selected_membership_count must be an integer")
    if isinstance(ready, bool) or not isinstance(ready, int):
        raise MetastudyContractError("selected_ready_count must be an integer")
    blocked_ids: list[str] = []
    for index, blocker in enumerate(blockers):
        if not isinstance(blocker, Mapping) or set(blocker) != {"experiment_id", "route_id"}:
            raise MetastudyContractError(f"selected_blockers[{index}] must be an object")
        if blocker["route_id"] != payload["route_id"]:
            raise MetastudyContractError(f"selected_blockers[{index}].route_id changed")
        experiment_id = blocker.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise MetastudyContractError(f"selected_blockers[{index}].experiment_id must be text")
        blocked_ids.append(experiment_id)
    if len(blocked_ids) != summary.get("selected_blocker_count"):
        raise MetastudyContractError("selected blocker count does not match blocker identities")
    ready_ids: list[str] = []
    selected_ids: list[str] = []
    related_ids: list[str] = []
    membership_count = 0
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, Mapping):
            raise MetastudyContractError(f"experiments[{index}] must be an object")
        experiment_id = experiment.get("experiment_id")
        memberships = experiment.get("memberships")
        if not isinstance(experiment_id, str) or not experiment_id.strip() or not isinstance(memberships, list):
            raise MetastudyContractError(f"experiments[{index}] identity or memberships are malformed")
        for membership_index, membership in enumerate(memberships):
            if not isinstance(membership, Mapping) or set(membership) != {
                "membership",
                "ready",
                "required_reader_state",
                "route_id",
            }:
                raise MetastudyContractError(f"experiments[{index}].memberships[{membership_index}] fields changed")
            membership_count += 1
            if membership["route_id"] != payload["route_id"]:
                continue
            if membership["required_reader_state"] != "records_ready" or not isinstance(membership["ready"], bool):
                raise MetastudyContractError("meta-study readiness membership semantics changed")
            if membership["membership"] == "selected":
                selected_ids.append(experiment_id)
                if membership["ready"]:
                    ready_ids.append(experiment_id)
            elif membership["membership"] == "related":
                related_ids.append(experiment_id)
            else:
                raise MetastudyContractError("meta-study membership must be selected or related")
    if len(experiments) != summary["experiment_count"] or membership_count != summary["membership_count"]:
        raise MetastudyContractError("readiness experiment or membership counts changed")
    if len(selected_ids) != selected or len(ready_ids) != ready:
        raise MetastudyContractError("selected readiness identities do not match summary counts")
    if len(related_ids) != summary["related_membership_count"]:
        raise MetastudyContractError("related readiness identities do not match summary count")
    if set(selected_ids) != set(DEFAULT_PROTOCOL.planned_kinetic_experiment_ids):
        raise MetastudyContractError("selected readiness identity set does not match the predeclared route cohort")
    if set(related_ids) != set(DEFAULT_PROTOCOL.excluded_snapshot_experiment_ids):
        raise MetastudyContractError("related readiness identity set must equal the excluded snapshot context")
    if set(blocked_ids) != set(selected_ids) - set(ready_ids):
        raise MetastudyContractError("selected blocker identities do not close the selected experiment set")
    complete = not blocked_ids and ready == selected and not contract_errors
    if not isinstance(payload["ok"], bool) or payload["ok"] is not complete:
        raise MetastudyContractError("readiness receipt ok does not match complete selected readiness")
    return EvidenceReadiness._from_validated_receipt(
        selected_experiment_count=selected,
        ready_experiment_count=ready,
        ready_experiment_ids=tuple(ready_ids),
        blocked_experiment_ids=tuple(blocked_ids),
        receipt_digest=canonical_digest({key: value for key, value in payload.items() if key != "reader_command"}),
    )


def readiness_from_live_bridge(*, phd_root: Path) -> EvidenceReadiness:
    """Run the exact bridge-owned route checker and authorize its typed receipt."""

    root = Path(phd_root).expanduser().resolve()
    skill_root = (root / ".agents/skills/retron-assay-study-bridge").resolve()
    registry = (skill_root / "references/reader-experiment-routes.json").resolve()
    checker = (skill_root / "scripts/check_reader_experiment_readiness.py").resolve()
    if not registry.is_file() or not checker.is_file():
        raise MetastudyContractError("canonical bridge registry or live-readiness checker is missing")
    command = [
        sys.executable,
        str(checker),
        "--registry",
        str(registry),
        "--phd-root",
        str(root),
        "--route-id",
        METASTUDY_ROUTE_ID,
    ]
    completed = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        diagnostics = "; ".join(
            value
            for value in (
                f"stdout={completed.stdout.strip()}" if completed.stdout.strip() else "",
                f"stderr={completed.stderr.strip()}" if completed.stderr.strip() else "",
            )
            if value
        )
        raise MetastudyContractError(
            f"live bridge checker exited with status {completed.returncode}: {diagnostics or '<no output>'}"
        )
    raw = completed.stdout.strip() or completed.stderr.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MetastudyContractError("live bridge checker returned invalid JSON") from exc
    structural = readiness_from_receipt(payload)
    return EvidenceReadiness._from_owner_bridge_receipt(
        selected_experiment_count=structural.selected_experiment_count,
        ready_experiment_count=structural.ready_experiment_count,
        ready_experiment_ids=structural.ready_experiment_ids,
        blocked_experiment_ids=structural.blocked_experiment_ids,
        receipt_digest=structural.receipt_digest,
    )


__all__ = ["decision_from_readiness", "readiness_from_live_bridge", "readiness_from_receipt"]

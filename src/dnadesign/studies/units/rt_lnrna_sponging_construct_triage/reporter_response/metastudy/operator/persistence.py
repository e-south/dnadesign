"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/operator/persistence.py

Atomic publication of one complete source-controlled meta-study state.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ..acquisition_projection import acquisition_projection_payload
from ..contracts._values import MetastudyContractError
from ..contracts.decision import decision_to_dict, validate_decision_payload
from ..contracts.protocol import DEFAULT_PROTOCOL
from ..sensitivity import sensitivity_evaluations_to_payload
from ..sensitivity_coverage import sensitivity_coverage_receipt_payload
from .state import (
    READINESS_SCHEMA_ID,
    RECEIPT_NORMALIZATION,
    ROUTE_ID,
    ROUTE_REGISTRY_PATH,
    STATE_FILE,
    STATE_SCHEMA_ID,
    UniqueKeySafeLoader,
    canonical_digest,
    digest_file,
    validate_state_payload,
)

if TYPE_CHECKING:
    from .regeneration import RegenerationResult


def write_source_controlled_state(result: RegenerationResult, *, destination: Path) -> tuple[Path]:
    """Atomically replace one combined readiness-and-decision generation."""

    from .regeneration import RegenerationResult

    if not isinstance(result, RegenerationResult):
        raise MetastudyContractError("state publication requires one complete regeneration result")
    target = Path(destination).resolve()
    if not target.is_dir():
        raise MetastudyContractError("state destination must be an existing directory")
    decision_payload = json.loads(json.dumps(decision_to_dict(result.decision), allow_nan=False))
    validate_decision_payload(decision_payload)
    route_registry = target.parents[5] / ROUTE_REGISTRY_PATH
    if not route_registry.is_file():
        raise MetastudyContractError("state destination does not resolve to the canonical PhD route registry")
    if digest_file(route_registry) != result.route_registry_digest:
        raise MetastudyContractError("route registry changed since regeneration")
    readiness = result.decision.readiness
    readiness_payload = {
        "schema_id": READINESS_SCHEMA_ID,
        "source_identity": {
            "route_id": ROUTE_ID,
            "route_registry_path": result.route_registry_path,
            "route_registry_digest": result.route_registry_digest,
            "normalized_full_receipt_digest": readiness.receipt_digest,
            "normalization": RECEIPT_NORMALIZATION,
        },
        "last_verified": date.today().isoformat(),
        "selected_experiment_count": readiness.selected_experiment_count,
        "related_experiment_count": len(DEFAULT_PROTOCOL.excluded_snapshot_experiment_ids),
        "related_experiment_ids": list(DEFAULT_PROTOCOL.excluded_snapshot_experiment_ids),
        "ready_experiment_count": readiness.ready_experiment_count,
        "ready_experiment_ids": list(readiness.ready_experiment_ids),
        "blocked_experiment_ids": list(readiness.blocked_experiment_ids),
    }
    body = {
        "readiness": readiness_payload,
        "decision": decision_payload,
        "objective_readiness": asdict(result.objective_readiness),
        "sensitivity_evaluations": sensitivity_evaluations_to_payload(result.sensitivity_evaluations),
        "sensitivity_coverage_receipts": [
            sensitivity_coverage_receipt_payload(row) for row in result.sensitivity_coverages
        ],
        "acquisition_projection": (
            acquisition_projection_payload(result.acquisition_projection)
            if result.acquisition_projection is not None
            else None
        ),
    }
    state_payload = {
        "schema_id": STATE_SCHEMA_ID,
        "generation_digest": canonical_digest(body),
        **body,
    }
    validate_state_payload(state_payload, route_registry=route_registry)
    state_path = target / STATE_FILE
    atomic_replace_yaml(state_path, state_payload)
    return (state_path,)


def atomic_replace_yaml(path: Path, payload: dict[str, object]) -> None:
    """Replace one YAML document only after strict staging round-trip validation."""

    canonical_payload = json.loads(json.dumps(payload, allow_nan=False))
    with tempfile.TemporaryDirectory(prefix=".metastudy-state-", dir=path.parent) as staging_name:
        staged = Path(staging_name) / path.name
        staged.write_text(yaml.safe_dump(canonical_payload, sort_keys=False), encoding="utf-8")
        if yaml.load(staged.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader) != canonical_payload:
            raise MetastudyContractError("staged combined meta-study state did not round-trip")
        os.replace(staged, path)

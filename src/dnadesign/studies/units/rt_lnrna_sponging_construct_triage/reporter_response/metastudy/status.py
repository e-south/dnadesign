"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/status.py

Report semantic readiness from the owner-bound live meta-study bridge route.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import (
    MetastudyContractError,
    canonical_digest,
)
from .operator import validate_live_source_controlled_state


def status_payload(*, phd_root: Path, state_path: Path) -> dict[str, object]:
    """Return a typed semantic status; path presence alone never means ready."""

    source = Path(state_path).expanduser().resolve()
    try:
        validation = validate_live_source_controlled_state(source, phd_root=phd_root)
    except (OSError, UnicodeError) as exc:
        raise MetastudyContractError(f"cannot read source state {source}: {exc}") from exc
    state = validation.state
    readiness = validation.regeneration.decision.readiness
    decision = state["decision"]
    objective_readiness = state["objective_readiness"]
    assert isinstance(decision, dict)
    assert isinstance(objective_readiness, dict)
    semantic_blockers: tuple[str, ...] = ()
    if decision["status"] != "selected":
        semantic_blockers += ("reduction_recommendation_is_blocked",)
    status = "ready" if not semantic_blockers else "blocked"
    measurement_readiness = (
        "ready"
        if readiness.ready_experiment_count == readiness.selected_experiment_count
        else "partial"
        if readiness.ready_experiment_count
        else "blocked"
    )
    return {
        "schema_id": "rt_lnrna_reporter_response_metastudy_status.v2",
        "status": status,
        "semantic_blockers": semantic_blockers,
        "measurement_readiness": measurement_readiness,
        "descriptive_visualization_readiness": "ready" if readiness.ready_experiment_count else "blocked",
        "reduction_recommendation_status": "ready" if decision["status"] == "selected" else "blocked",
        "objective_readiness_status": objective_readiness["status"],
        "objective_readiness_blockers": tuple(objective_readiness["blockers"]),
        "selected_reduction": tuple(decision["selected_reduction"]) if decision["selected_reduction"] else None,
        "evidence_grade": decision["evidence_grade"],
        "limitations": tuple(decision["limitations"]),
        "source_state_path": str(source),
        "source_state_generation_digest": state["generation_digest"],
        "source_decision_digest": canonical_digest(decision),
        "selected_experiment_count": readiness.selected_experiment_count,
        "ready_experiment_count": readiness.ready_experiment_count,
        "ready_experiment_ids": readiness.ready_experiment_ids,
        "blocked_experiment_ids": readiness.blocked_experiment_ids,
        "readiness_receipt_digest": readiness.receipt_digest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phd-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = status_payload(phd_root=args.phd_root, state_path=args.state)
    payload["status_digest"] = canonical_digest(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "status_payload"]

"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_state_semantics.py

Focused tests for shared OPS state aggregation semantics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dnadesign.ops.preflight import build_state_check, evaluate_preflight_checks
from dnadesign.ops.status.models import CampaignStatus, ProcedureStatus, combine_states

_STATES = ("ok", "attention", "missing")


def _procedure_status(*, registry_id: str, state: str) -> ProcedureStatus:
    return ProcedureStatus(
        registry_id=registry_id,
        title=registry_id,
        doc_path="docs/operations/README.md",
        owner_boundary="ops",
        status_kind="ops-audit-json",
        observes_plane="control",
        surface_type="artifact_state",
        cost_class="cheap",
        summary_scope="workspace",
        label=None,
        state=state,
        summary=state,
        evidence={},
    )


def test_combine_states_truth_table() -> None:
    expected = {
        ("ok", "ok"): "ok",
        ("ok", "attention"): "attention",
        ("ok", "missing"): "missing",
        ("attention", "ok"): "attention",
        ("attention", "attention"): "attention",
        ("attention", "missing"): "missing",
        ("missing", "ok"): "missing",
        ("missing", "attention"): "missing",
        ("missing", "missing"): "missing",
    }

    for states, result in expected.items():
        assert combine_states(states) == result


def test_combine_states_prefers_missing_for_longer_sequences() -> None:
    assert combine_states(("ok", "attention", "ok", "missing", "attention")) == "missing"
    assert combine_states(("ok", "ok", "attention", "ok")) == "attention"
    assert combine_states(()) == "ok"


@settings(derandomize=True, max_examples=9, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(_STATES), st.sampled_from(_STATES))
def test_combine_states_is_commutative(left: str, right: str) -> None:
    assert combine_states((left, right)) == combine_states((right, left))


@settings(derandomize=True, max_examples=27, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(_STATES), st.sampled_from(_STATES), st.sampled_from(_STATES))
def test_combine_states_is_associative(first: str, second: str, third: str) -> None:
    left = combine_states((combine_states((first, second)), third))
    right = combine_states((first, combine_states((second, third))))
    assert left == right


@settings(derandomize=True, max_examples=3, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(_STATES))
def test_combine_states_is_idempotent(state: str) -> None:
    assert combine_states((state, state, state)) == state


def test_campaign_overall_state_uses_shared_state_lattice() -> None:
    campaign = CampaignStatus(
        manifest_path=Path("campaign.yaml"),
        campaign_id="demo",
        steps=(
            _procedure_status(registry_id="ops.control-plane.orchestration", state="attention"),
            _procedure_status(registry_id="studies.stress-ethanol-cipro-growth.status", state="missing"),
        ),
    )

    assert campaign.counts() == {"ok": 0, "attention": 1, "missing": 1}
    assert campaign.overall_state() == "missing"


def test_preflight_blockers_sort_missing_before_attention() -> None:
    checks = (
        build_state_check(
            check_id="attention.check",
            check_group="demo",
            category="ops",
            check_set_id="demo_checks",
            state="attention",
            summary="attention",
        ),
        build_state_check(
            check_id="missing.check",
            check_group="demo",
            category="ops",
            check_set_id="demo_checks",
            state="missing",
            summary="missing",
        ),
    )

    evaluation = evaluate_preflight_checks(
        checks,
        scope_plan=type(
            "ScopePlan",
            (),
            {
                "scope": "full",
                "included_groups": ("demo",),
            },
        )(),
    )

    assert evaluation.blocked_by_ids == ("missing.check", "attention.check")

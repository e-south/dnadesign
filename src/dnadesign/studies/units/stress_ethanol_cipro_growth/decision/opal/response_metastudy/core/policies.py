"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/core/policies.py

response metric metastudy policy definitions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .contracts import PolicySpec

CANONICAL_SFXI_POLICY_ID = "sfxi_beta1_gamma1"
SCORE_SURFACE_POLICY_ID = "tradeoff_logic0p95"

PRIMARY_POLICIES: tuple[PolicySpec, ...] = (
    PolicySpec(
        id=CANONICAL_SFXI_POLICY_ID,
        label="Canonical SFXI beta=1 gamma=1",
        kind="multiplicative",
        beta=1.0,
        gamma=1.0,
        tier="canonical",
        plain_rule="Rank by logic fidelity multiplied by effect.",
    ),
    PolicySpec(
        id="logic_first_beta4_gamma05",
        label="Logic-weighted beta=4 gamma=0.5",
        kind="multiplicative",
        beta=4.0,
        gamma=0.5,
        tier="candidate",
        plain_rule="Rank by target-shape fidelity first while retaining a smaller effect term.",
    ),
    PolicySpec(
        id="logic_gate055_effect",
        label="Gate logic>=0.55, then effect",
        kind="logic_gate",
        beta=0.0,
        gamma=1.0,
        logic_gate=0.55,
        tier="candidate",
        plain_rule="Discard low-fidelity candidates, then rank the remaining candidates by effect.",
    ),
    PolicySpec(
        id="lexicographic_logic_effect",
        label="Lexicographic logic then effect",
        kind="lexicographic",
        beta=1.0,
        gamma=0.0,
        tier="candidate",
        plain_rule="Sort first by target-shape fidelity and use effect only as a tie-breaker.",
    ),
    PolicySpec(
        id="off_state_logic_eta2_beta2_gamma05",
        label="OFF-state logic penalty eta=2 beta=2 gamma=0.5",
        kind="off_state_logic_penalty",
        beta=2.0,
        gamma=0.5,
        off_state_logic_eta=2.0,
        tier="candidate",
        plain_rule="Rank by target-shape fidelity and effect while penalizing predicted logic level in OFF states.",
    ),
)


def logic_effect_tradeoff_policies() -> tuple[PolicySpec, ...]:
    """Return one representative for each multiplicative ranking tradeoff.

    For top-N selection, scaling both exponents by the same positive constant
    applies a monotone transform to the score and cannot change ranks. Fixing
    beta + gamma = 1 removes those redundant parameterizations.
    """

    weights = (0.0, 0.25, 0.4, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
    return tuple(
        PolicySpec(
            id=f"tradeoff_logic{_fmt_param(weight)}",
            label=f"Logic weight {weight:g}",
            kind="multiplicative",
            beta=weight,
            gamma=1.0 - weight,
            tier="sweep",
            plain_rule="Rank by a normalized geometric tradeoff between target-shape fidelity and effect.",
        )
        for weight in weights
    )


def logic_gate_policies() -> tuple[PolicySpec, ...]:
    return tuple(
        PolicySpec(
            id=f"gate{_fmt_param(gate)}_effect",
            label=f"Gate logic>={gate:g}, then effect",
            kind="logic_gate",
            beta=0.0,
            gamma=1.0,
            logic_gate=gate,
            tier="sweep",
            plain_rule="Discard low-fidelity candidates, then rank by effect.",
        )
        for gate in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65)
    )


def off_state_logic_penalty_policies() -> tuple[PolicySpec, ...]:
    return tuple(
        PolicySpec(
            id=f"off_state_logic_eta{_fmt_param(eta)}_beta2_gamma05",
            label=f"OFF-state logic penalty eta={eta:g}, beta=2, gamma=0.5",
            kind="off_state_logic_penalty",
            beta=2.0,
            gamma=0.5,
            off_state_logic_eta=eta,
            tier="sweep",
            plain_rule="Penalize predicted logic level in OFF states before retaining effect as a secondary term.",
        )
        for eta in (0.5, 1.0, 4.0)
    )


def audit_policy_specs() -> tuple[PolicySpec, ...]:
    by_id: dict[str, PolicySpec] = {}
    for policy in (
        *PRIMARY_POLICIES,
        *logic_effect_tradeoff_policies(),
        *logic_gate_policies(),
        *off_state_logic_penalty_policies(),
    ):
        if policy.id in by_id:
            raise ValueError(f"duplicate SFXI policy id: {policy.id}")
        by_id[policy.id] = policy
    return tuple(by_id.values())


def primary_policy_ids() -> list[str]:
    return [policy.id for policy in PRIMARY_POLICIES]


def _fmt_param(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p")

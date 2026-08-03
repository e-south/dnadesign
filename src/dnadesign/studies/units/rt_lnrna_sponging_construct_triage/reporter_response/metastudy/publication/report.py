"""Canonical rendering for reporter-response meta-study publications."""

from __future__ import annotations


def _render_report(payload: dict[str, object]) -> str:
    """Render the stable human-readable projection of one decision payload."""

    reduction = payload["selected_reduction"]
    reduction_text = "none" if reduction is None else f"{reduction[0]:g}-{reduction[1]:g} h"
    blockers = payload["blockers"]
    blocker_lines = "\n".join(f"- {value}" for value in blockers) if blockers else "- none"
    limitations = payload["limitations"]
    limitation_lines = "\n".join(f"- {value}" for value in limitations) if limitations else "- none"
    return (
        "# RT-lnRNA reporter-response reduction recommendation\n\n"
        f"- Protocol: `{payload['protocol_id']}`\n"
        f"- Status: `{payload['status']}`\n"
        f"- Evidence grade: `{payload['evidence_grade']}`\n"
        f"- Selected reduction: `{reduction_text}`\n"
        f"- Policy digest: `{payload['policy_digest']}`\n"
        f"- Evidence digest: `{payload['evidence_digest']}`\n\n"
        "## Blockers\n\n"
        f"{blocker_lines}\n\n"
        "## Limitations\n\n"
        f"{limitation_lines}\n"
    )

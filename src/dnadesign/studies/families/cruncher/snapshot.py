"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/families/cruncher/snapshot.py

Snapshot builder for read-only Cruncher study records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .record_normalizer import CruncherStudyResolvedContext


def build_cruncher_study_status(
    *,
    study_context: CruncherStudyResolvedContext,
    summary_scope: str,
) -> tuple[str, str, dict[str, object]]:
    evidence = dict(study_context.evidence)
    evidence.update(
        {
            "summary_scope": summary_scope,
            "study_id": study_context.study_id,
            "family": study_context.ops_contract.family,
            "title": study_context.ops_contract.title,
            "record_sources": dict(study_context.ops_contract.record_sources),
            "record_paths": {name: str(path) for name, path in study_context.record_paths.items()},
            "artifacts": {
                artifact_id: dict(payload) for artifact_id, payload in study_context.ops_contract.artifacts.items()
            },
            "current_phase": study_context.current_phase,
            "current_phase_status": _phase_status(study_context, study_context.current_phase),
            "phase_states": list(study_context.phase_states),
            "next_ready_phase": study_context.next_ready_phase,
            "next_planned_phase": study_context.next_planned_phase,
            "blocked_phases": list(study_context.blocked_phases),
            "status_excerpt": list(study_context.status_excerpt),
            "command_groups": [group.as_dict() for group in study_context.command_groups],
            "intent": dict(study_context.intent_payload),
            "native_agent_bootstrap": dict(study_context.native_agent_bootstrap),
            "execution_surfaces": {
                surface_id: dict(payload)
                for surface_id, payload in study_context.ops_contract.execution_surfaces.items()
            },
        }
    )

    state_label = str(study_context.intent_payload.get("state_label") or "phase").strip() or "phase"
    summary_parts = [f"{study_context.study_id}: {state_label} {study_context.current_phase or 'unknown'}"]
    primary_lane = str(study_context.intent_payload.get("primary_lane") or "").strip()
    if primary_lane:
        summary_parts.append(f"primary lane {primary_lane}")
    if state_label == "phase" and study_context.next_ready_phase is not None:
        summary_parts.append(f"next ready {study_context.next_ready_phase['id']}")
    elif state_label == "phase" and study_context.next_planned_phase is not None:
        summary_parts.append(f"next planned {study_context.next_planned_phase['id']}")
    if study_context.command_groups:
        summary_parts.append("command groups " + ", ".join(group.id for group in study_context.command_groups))

    attention_reasons: list[str] = []
    if not study_context.current_phase_is_known:
        attention_reasons.append("current_phase does not match any declared phase id")
    if not study_context.command_groups:
        attention_reasons.append("pipeline command_groups are missing")
    if not study_context.native_agent_bootstrap.get("open_first"):
        attention_reasons.append("native-agent bootstrap open_first list is missing")
    if study_context.blocked_phases:
        attention_reasons.append("one or more study phases are blocked")

    summary = "; ".join(summary_parts)
    if attention_reasons:
        evidence["attention_reasons"] = attention_reasons
        return ("attention", summary, evidence)
    return ("ok", summary, evidence)


def _phase_status(study_context: CruncherStudyResolvedContext, phase_id: str | None) -> str | None:
    normalized_phase_id = str(phase_id or "").strip()
    if not normalized_phase_id:
        return None
    for phase in study_context.phase_states:
        if str(phase.get("id") or "").strip() == normalized_phase_id:
            status = str(phase.get("status") or "").strip()
            return status or None
    return None


__all__ = ["build_cruncher_study_status"]

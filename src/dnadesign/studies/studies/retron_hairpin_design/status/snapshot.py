"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/status/snapshot.py

Snapshot builder for read-only Retron hairpin design study records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .record_normalizer import RetronHairpinDesignResolvedContext


def build_retron_hairpin_design_status(
    *,
    study_context: RetronHairpinDesignResolvedContext,
    summary_scope: str,
) -> tuple[str, str, dict[str, object]]:
    item_label = _lifecycle_item_label(study_context)
    item_key = _lifecycle_item_key(item_label)
    current_item_key = f"current_{item_key}"
    item_states_key = f"{item_key}_states"
    next_ready_item_key = f"next_ready_{item_key}"
    next_planned_item_key = f"next_planned_{item_key}"
    blocked_items_key = f"blocked_{item_key}s"

    evidence = dict(study_context.evidence)
    evidence.update(
        {
            "summary_scope": summary_scope,
            "study_id": study_context.study_id,
            "status_kind": study_context.ops_contract.status_kind,
            "preflight_kind": study_context.ops_contract.preflight_kind,
            "title": study_context.ops_contract.title,
            "lifecycle_mode": study_context.ops_contract.lifecycle_mode,
            "lifecycle_item_label": item_label,
            "record_sources": dict(study_context.ops_contract.record_sources),
            "record_paths": {name: str(path) for name, path in study_context.record_paths.items()},
            "artifacts": {
                artifact_id: dict(payload) for artifact_id, payload in study_context.ops_contract.artifacts.items()
            },
            current_item_key: study_context.current_phase,
            f"{current_item_key}_status": _phase_status(study_context, study_context.current_phase),
            item_states_key: list(study_context.phase_states),
            next_ready_item_key: study_context.next_ready_phase,
            next_planned_item_key: study_context.next_planned_phase,
            blocked_items_key: list(study_context.blocked_phases),
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

    if item_key == "phase":
        evidence.update(
            {
                "current_phase": study_context.current_phase,
                "current_phase_status": _phase_status(study_context, study_context.current_phase),
                "phase_states": list(study_context.phase_states),
                "next_ready_phase": study_context.next_ready_phase,
                "next_planned_phase": study_context.next_planned_phase,
                "blocked_phases": list(study_context.blocked_phases),
            }
        )

    state_label = str(study_context.intent_payload.get("state_label") or item_label).strip() or item_label
    summary_parts = [f"{study_context.study_id}: {state_label} {study_context.current_phase or 'unknown'}"]
    primary_lane = str(study_context.intent_payload.get("primary_lane") or "").strip()
    if primary_lane:
        summary_parts.append(f"primary lane {primary_lane}")
    if state_label == item_label and study_context.next_ready_phase is not None:
        summary_parts.append(f"next ready {item_label} {study_context.next_ready_phase['id']}")
    elif state_label == item_label and study_context.next_planned_phase is not None:
        summary_parts.append(f"next planned {item_label} {study_context.next_planned_phase['id']}")
    if study_context.command_groups:
        summary_parts.append("command groups " + ", ".join(group.id for group in study_context.command_groups))

    attention_reasons: list[str] = []
    if not study_context.current_phase_is_known:
        attention_reasons.append(f"{current_item_key} does not match any declared {item_label} id")
    if not study_context.command_groups:
        attention_reasons.append("pipeline command_groups are missing")
    if not study_context.native_agent_bootstrap.get("open_first"):
        attention_reasons.append("native-agent bootstrap open_first list is missing")
    if study_context.blocked_phases:
        attention_reasons.append(f"one or more study {item_label}s are blocked")

    summary = "; ".join(summary_parts)
    if attention_reasons:
        evidence["attention_reasons"] = attention_reasons
        return ("attention", summary, evidence)
    return ("ok", summary, evidence)


def _phase_status(study_context: RetronHairpinDesignResolvedContext, phase_id: str | None) -> str | None:
    normalized_phase_id = str(phase_id or "").strip()
    if not normalized_phase_id:
        return None
    for phase in study_context.phase_states:
        if str(phase.get("id") or "").strip() == normalized_phase_id:
            status = str(phase.get("status") or "").strip()
            return status or None
    return None


def _lifecycle_item_label(study_context: RetronHairpinDesignResolvedContext) -> str:
    return str(study_context.ops_contract.lifecycle_item_label or "phase").strip() or "phase"


def _lifecycle_item_key(item_label: str) -> str:
    normalized = item_label.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or "phase"


__all__ = ["build_retron_hairpin_design_status"]

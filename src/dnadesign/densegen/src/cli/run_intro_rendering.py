"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli/run_intro_rendering.py

Rendering helpers for DenseGen notebook run-details intro markdown.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .run_intro import (
    INTRO_TRACE_MAP,
    PLAN_INLINE_DETAIL_THRESHOLD,
    PlanGroup,
    PlanRow,
    RunContractSummary,
    RunDetailsModel,
    RunDetailsPathsContext,
    RunOutcomeSummary,
    Section,
    _bool_text,
    _format_percent,
    _format_seconds,
    _normalize_section_text,
    _plural,
    _render_markdown_table,
    _shorten,
    _summary_with_source,
    parse_plan_name,
)


def _build_plan_rows(contract: RunContractSummary, outcome: RunOutcomeSummary) -> tuple[PlanRow, ...]:
    outcome_by_plan = {item.name: item for item in outcome.per_plan}
    rows: list[PlanRow] = []

    for plan in contract.plans:
        parsed = parse_plan_name(plan.name)
        variants_text = (
            "; ".join(f"{key}={value}" for key, value in parsed.variants.items()) if parsed.variants else "-"
        )
        item = outcome_by_plan.get(plan.name)
        generated = item.generated if item is not None else None
        row_progress = None
        if item is not None and item.quota > 0:
            row_progress = float(item.generated) * 100.0 / float(item.quota)
        rows.append(
            PlanRow(
                raw_name=plan.name,
                base_name=parsed.base or plan.name,
                variants_text=variants_text,
                quota=int(plan.quota),
                acceptance_detail=plan.acceptance_detail,
                generated=generated,
                progress_pct=row_progress,
                stall_events=(item.stall_events if item is not None else 0),
                total_resamples=(item.total_resamples if item is not None else 0),
                failed_solutions=(item.failed_solutions if item is not None else 0),
            )
        )

    known = {plan.name for plan in contract.plans}
    for item in outcome.per_plan:
        if item.name in known:
            continue
        parsed = parse_plan_name(item.name)
        variants_text = (
            "; ".join(f"{key}={value}" for key, value in parsed.variants.items()) if parsed.variants else "-"
        )
        row_progress = float(item.generated) * 100.0 / float(item.quota) if item.quota > 0 else None
        rows.append(
            PlanRow(
                raw_name=item.name,
                base_name=parsed.base or item.name,
                variants_text=variants_text,
                quota=int(item.quota),
                acceptance_detail=None,
                generated=int(item.generated),
                progress_pct=row_progress,
                stall_events=int(item.stall_events),
                total_resamples=int(item.total_resamples),
                failed_solutions=int(item.failed_solutions),
            )
        )

    return tuple(rows)


def _group_plan_rows(rows: Sequence[PlanRow]) -> tuple[PlanGroup, ...]:
    grouped: dict[tuple[str, str, int | None, int | None], list[PlanRow]] = {}
    order: list[tuple[str, str, int | None, int | None]] = []
    for row in rows:
        key = (
            row.base_name,
            str(row.acceptance_detail or ""),
            int(row.quota),
            int(row.generated) if row.generated is not None else None,
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    groups: list[PlanGroup] = []
    for key in order:
        base_name, acceptance, _, _ = key
        label = base_name
        if acceptance:
            label = f"{base_name} ({_shorten(acceptance, max_len=72)})"
        groups.append(PlanGroup(label=label, rows=tuple(grouped[key])))
    return tuple(groups)


def _status_for_rows(rows: Sequence[PlanRow]) -> str:
    if not rows:
        return "none"
    known = [row for row in rows if row.generated is not None]
    if not known:
        return "-"
    complete = [row for row in known if row.quota > 0 and int(row.generated or 0) >= int(row.quota)]
    if len(complete) == len(known):
        return "all complete"
    none = [row for row in known if int(row.generated or 0) <= 0]
    if len(none) == len(known):
        return "none"
    return "partial"


def _build_summary_lines(contract: RunContractSummary, outcome: RunOutcomeSummary) -> tuple[str, ...]:
    plan_count = len(contract.plans)

    if contract.config_error:
        target_line = _summary_with_source(f"Target: unavailable ({contract.config_error}).", "config")
    else:
        if plan_count == 0:
            target_value = "Target: no plans declared in config."
        elif plan_count == 1:
            only = contract.plans[0]
            if contract.sequence_length_bp is not None:
                target_value = (
                    f"Target: {contract.total_quota} sequences × {contract.sequence_length_bp} bp for plan "
                    f"{only.name} ({only.quota})."
                )
            else:
                target_value = f"Target: {contract.total_quota} sequences for plan {only.name} ({only.quota})."
        else:
            if plan_count <= PLAN_INLINE_DETAIL_THRESHOLD:
                quota_items = "; ".join(f"{plan.name}: {plan.quota}" for plan in contract.plans)
            else:
                grouped_quotas: dict[int, int] = {}
                for plan in contract.plans:
                    grouped_quotas[int(plan.quota)] = int(grouped_quotas.get(int(plan.quota), 0)) + 1
                quota_items = "; ".join(
                    f"{count} {_plural(count, 'plan', 'plans')} at {quota}"
                    for quota, count in sorted(grouped_quotas.items(), key=lambda item: item[0])
                )
            if contract.sequence_length_bp is not None:
                target_value = (
                    f"Target: {contract.total_quota} sequences × "
                    f"{contract.sequence_length_bp} bp across {plan_count} plans "
                    f"({quota_items})."
                )
            else:
                target_value = f"Target: {contract.total_quota} sequences across {plan_count} plans ({quota_items})."
        target_line = _summary_with_source(target_value, "config")

    shared_constraints = contract.global_acceptance_detail or "none"
    plan_specific_details = sorted(
        {
            str(plan.acceptance_detail or "").strip()
            for plan in contract.plans
            if str(plan.acceptance_detail or "").strip()
        }
    )
    constrained_plans = sum(1 for plan in contract.plans if str(plan.acceptance_detail or "").strip())
    if plan_specific_details:
        acceptance_value = (
            "Acceptance: shared constraints "
            f"{_shorten(shared_constraints)}; plan-specific signatures: {len(plan_specific_details)} "
            f"across {constrained_plans}/{plan_count} plans."
        )
    else:
        acceptance_value = (
            f"Acceptance: shared constraints {_shorten(shared_constraints)}; plan-specific signatures: none."
        )
    acceptance_line = _summary_with_source(acceptance_value, "config")

    if contract.inputs_used:
        inputs_headline = ", ".join(contract.inputs_used)
    else:
        inputs_headline = "none declared"
    if contract.background_filtering_details:
        filtering_count = len(contract.background_filtering_details)
        input_value = (
            f"Inputs: {inputs_headline}; background filtering: "
            f"{filtering_count} {_plural(filtering_count, 'rule', 'rules')}."
        )
    else:
        input_value = f"Inputs: {inputs_headline}."
    inputs_line = _summary_with_source(input_value, "config")

    schedule = "round-robin" if contract.round_robin else "sequential" if contract.round_robin is False else "-"
    timeout_text = _format_seconds(contract.solver_attempt_timeout_seconds) or "-"
    resample_text = _format_seconds(contract.no_progress_seconds_before_resample) or "-"
    abort_text = (
        f"{contract.max_consecutive_no_progress_resamples} consecutive no-progress resamples"
        if contract.max_consecutive_no_progress_resamples is not None
        else "-"
    )
    execution_value = (
        "Execution: Stage-B samples one library, runs candidates, and accepts up to "
        f"{contract.max_accepted_per_library if contract.max_accepted_per_library is not None else '-'} "
        "sequences per library; "
        f"scheduling={schedule}; solver={contract.solver_backend or '-'} {contract.solver_strategy or '-'} "
        f"({timeout_text}); guards=resample at {resample_text}, abort after {abort_text}."
    )
    execution_line = _summary_with_source(execution_value, "config")

    if not outcome.available:
        outcome_line = _summary_with_source("Outcome: manifest not found (no outcome summary available).", "manifest")
        pressure_line = _summary_with_source("Pressure: manifest not found.", "manifest")
    else:
        quota_total = int(outcome.quota_total or 0)
        generated_total = int(outcome.generated_total or 0)
        pct_text = _format_percent(generated_total, quota_total)
        plans_at_quota = sum(
            1 for item in outcome.per_plan if item.quota > 0 and int(item.generated) >= int(item.quota)
        )
        plan_total = len(outcome.per_plan)
        outcome_progress = (
            f"Outcome: {generated_total}/{quota_total} generated ({pct_text}); "
            f"plans at quota: {plans_at_quota}/{plan_total}."
        )
        outcome_line = _summary_with_source(
            outcome_progress,
            "manifest",
        )

        pressure_parts: list[str] = []
        if outcome.stall_events > 0:
            pressure_parts.append(f"stall events={outcome.stall_events}")
        if outcome.total_resamples > 0:
            pressure_parts.append(f"resamples={outcome.total_resamples}")
        if outcome.failed_solutions > 0:
            pressure_parts.append(f"failed solves={outcome.failed_solutions}")
        if pressure_parts:
            pressure_line = _summary_with_source("Pressure: " + "; ".join(pressure_parts) + ".", "manifest")
        else:
            pressure_line = _summary_with_source("Pressure: none recorded.", "manifest")

    return (
        target_line,
        acceptance_line,
        inputs_line,
        execution_line,
        outcome_line,
        pressure_line,
    )


def _render_plans_section(contract: RunContractSummary, outcome: RunOutcomeSummary) -> str:
    rows = _build_plan_rows(contract, outcome)
    groups = _group_plan_rows(rows)

    group_rows: list[tuple[object, ...]] = []
    for group in groups:
        quotas = [int(row.quota) for row in group.rows]
        generated_rows = [row for row in group.rows if row.generated is not None]

        if len(set(quotas)) == 1:
            quota_text = str(quotas[0])
        else:
            quota_text = f"{min(quotas)}–{max(quotas)}"

        if not generated_rows:
            progress_text = "-"
        else:
            generated_total = sum(int(row.generated or 0) for row in generated_rows)
            quota_total = sum(int(row.quota) for row in generated_rows)
            progress_text = f"{generated_total}/{quota_total}"

        group_rows.append(
            (
                group.label,
                len(group.rows),
                quota_text,
                progress_text,
                _status_for_rows(group.rows),
            )
        )

    lines: list[str] = []
    schema_text = contract.schema_version or "-"
    output_targets_text = ", ".join(contract.output_targets) if contract.output_targets else "-"
    lines.append(f"Schema version: {schema_text}. [config]")
    lines.append(f"Output targets: {output_targets_text}. [config]")
    lines.append("")
    lines.append(
        _render_markdown_table(
            ("Group", "Plans", "Quota", "Progress", "Status"),
            group_rows,
        )
    )
    return "\n".join(lines)


def _render_all_plans_section(contract: RunContractSummary, outcome: RunOutcomeSummary) -> str:
    rows = _build_plan_rows(contract, outcome)

    raw_rows: list[tuple[object, ...]] = []
    for row in rows:
        generated_text = "-" if row.generated is None else str(row.generated)
        progress_text = "-"
        if row.generated is not None and row.quota > 0:
            progress_text = f"{row.generated}/{row.quota} ({_format_percent(row.generated, row.quota)})"
        raw_rows.append(
            (
                row.base_name,
                row.variants_text,
                row.quota,
                generated_text,
                progress_text,
                f"`{row.raw_name}`",
            )
        )

    return _render_markdown_table(
        ("Plan", "Variants", "Quota", "Generated", "Progress", "Raw plan id"),
        raw_rows,
    )


def _render_constraints_section(contract: RunContractSummary) -> str:
    lines: list[str] = []

    shared_constraints = contract.global_acceptance_detail or "none"
    lines.append(f"Shared constraints: {shared_constraints}. [config]")

    plan_groups: dict[str, int] = {}
    shared_only = 0
    for plan in contract.plans:
        detail = str(plan.acceptance_detail or "").strip()
        if not detail:
            shared_only += 1
            continue
        plan_groups[detail] = int(plan_groups.get(detail, 0)) + 1

    lines.append("")
    lines.append("Plan constraints (grouped): [config]")
    if shared_only > 0:
        lines.append(f"- {shared_only} {_plural(shared_only, 'plan', 'plans')} use shared constraints only.")
    for detail, count in sorted(plan_groups.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {count} {_plural(count, 'plan', 'plans')} require {_shorten(detail, max_len=140)}.")

    if contract.background_filtering_details:
        lines.append("")
        lines.append("Background filtering: [config]")
        for detail in contract.background_filtering_details:
            lines.append(f"- {detail}.")

    if contract.expansion_details:
        lines.append("")
        lines.append("Plan expansion summary: [config]")
        for detail in contract.expansion_details:
            lines.append(f"- {detail}.")

    return "\n".join(lines).strip()


def _render_constraint_literals_section(contract: RunContractSummary) -> str:
    if not contract.expansion_detail_blocks:
        return "No literal sets recorded for this run."

    lines: list[str] = []
    for index, block in enumerate(contract.expansion_detail_blocks, start=1):
        if index > 1:
            lines.append("")
        lines.append(f"Literal set {index}:")
        lines.extend(block.splitlines())
    return "\n".join(lines)


def _render_definitions_section() -> str:
    return "\n".join(
        [
            "- Library = sampled set of motifs offered to the solver for one plan pass.",
            "- Candidate = one solver attempt input built from a library.",
            "- Accepted sequence = candidate that passes constraints and is written to outputs.",
            "- Resample = rebuild or refresh the library and continue attempts.",
            "- No progress = active Stage-B time with zero newly accepted sequences.",
        ]
    )


def _render_execution_section(contract: RunContractSummary) -> str:
    max_accepted_configured = (
        str(contract.max_accepted_per_library) if contract.max_accepted_per_library is not None else "-"
    )
    solver_timeout = _format_seconds(contract.solver_attempt_timeout_seconds) or "-"
    no_progress_seconds = _format_seconds(contract.no_progress_seconds_before_resample) or "-"
    max_no_progress = (
        str(contract.max_consecutive_no_progress_resamples)
        if contract.max_consecutive_no_progress_resamples is not None
        else "-"
    )
    max_failed = str(contract.max_failed_solutions) if contract.max_failed_solutions is not None else "-"
    max_failed_per_target = (
        str(contract.max_failed_solutions_per_target) if contract.max_failed_solutions_per_target is not None else "-"
    )
    lines = [
        "Stage-B sampling [config]",
        f"- pool_strategy: {contract.pool_strategy or '-'}",
        f"- library_sampling_strategy: {contract.library_sampling_strategy or '-'}",
        f"- library_size: {contract.library_size if contract.library_size is not None else '-'}",
        f"- unique_binding_sites: {_bool_text(contract.unique_binding_sites)}",
        f"- unique_binding_cores: {_bool_text(contract.unique_binding_cores)}",
        "",
        "Scheduling and caps [config]",
        f"- round_robin: {_bool_text(contract.round_robin)}",
        f"- max_accepted_per_library (configured): {max_accepted_configured}",
        "- max_accepted_per_library (effective): not computed (config value shown)",
        "",
        "Solver [config]",
        f"- backend: {contract.solver_backend or '-'}",
        f"- strategy: {contract.solver_strategy or '-'}",
        f"- solver_attempt_timeout_seconds: {solver_timeout}",
        "",
        "Guards [config]",
        f"- no_progress_seconds_before_resample: {no_progress_seconds}",
        f"- max_consecutive_no_progress_resamples: {max_no_progress}",
        f"- max_failed_solutions: {max_failed}",
        f"- max_failed_solutions_per_target: {max_failed_per_target}",
    ]
    return "\n".join(lines)


def _render_outcome_section(contract: RunContractSummary, outcome: RunOutcomeSummary) -> str:
    lines: list[str] = []
    rows = _build_plan_rows(contract, outcome)

    if not outcome.available:
        lines.append("Outcome: manifest not found. [manifest]")
        lines.append("Pressure: manifest not found. [manifest]")
        lines.append("Plan-local pressure is not recorded in this manifest.")
        return "\n".join(lines)

    quota_total = int(outcome.quota_total or 0)
    generated_total = int(outcome.generated_total or 0)
    outcome_pct = _format_percent(generated_total, quota_total)
    lines.append(f"Outcome: {generated_total}/{quota_total} generated ({outcome_pct}). [manifest]")

    plans_at_quota = sum(1 for item in outcome.per_plan if item.quota > 0 and int(item.generated) >= int(item.quota))
    lines.append(f"Plans at quota: {plans_at_quota}/{len(outcome.per_plan)}. [manifest]")

    grouped: dict[str, list[PlanRow]] = {}
    order: list[str] = []
    for row in rows:
        key = row.base_name
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    completion_rows: list[tuple[object, ...]] = []
    for key in order:
        group_rows = grouped[key]
        completed = 0
        partial = 0
        none = 0
        for row in group_rows:
            if row.generated is None:
                continue
            if row.quota > 0 and int(row.generated) >= int(row.quota):
                completed += 1
            elif int(row.generated) <= 0:
                none += 1
            else:
                partial += 1
        completion_rows.append((key, len(group_rows), completed, partial, none))

    lines.append("")
    lines.append(_render_markdown_table(("Group", "Plans", "Completed", "Partial", "None"), completion_rows))

    pressure_parts: list[str] = []
    if outcome.stall_events > 0:
        pressure_parts.append(f"stall events={outcome.stall_events}")
    if outcome.total_resamples > 0:
        pressure_parts.append(f"resamples={outcome.total_resamples}")
    if outcome.failed_solutions > 0:
        pressure_parts.append(f"failed solves={outcome.failed_solutions}")

    lines.append("")
    if pressure_parts:
        lines.append("Pressure: " + "; ".join(pressure_parts) + ". [manifest]")
    else:
        lines.append("Pressure: none recorded. [manifest]")
    return "\n".join(lines)


def _render_pressure_by_plan_section(contract: RunContractSummary, outcome: RunOutcomeSummary) -> str:
    rows = _build_plan_rows(contract, outcome)
    pressure_rows = [
        (
            row.base_name,
            row.variants_text,
            row.stall_events,
            row.total_resamples,
            row.failed_solutions,
        )
        for row in rows
        if row.stall_events > 0 or row.total_resamples > 0 or row.failed_solutions > 0
    ]
    if not pressure_rows:
        return "No non-zero plan-local pressure counters."
    return _render_markdown_table(
        ("Plan", "Variants", "Stall events", "Resamples", "Failed solves"),
        pressure_rows,
    )


def _file_mtime(path: Path | None) -> float | None:
    if path is None:
        return None
    if not path.exists():
        return None
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return None


def _format_timestamp(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_sources_section(
    contract: RunContractSummary,
    paths_context: RunDetailsPathsContext | None,
    outcome: RunOutcomeSummary,
) -> str:
    context = paths_context or RunDetailsPathsContext()
    run_root = context.run_root
    config_path = context.config_path
    records_path = context.records_path
    manifest_path = context.manifest_path
    notebook_path = context.notebook_path

    def _code(value: str) -> str:
        return f"`{value}`" if value else "`-`"

    def _short_hash(value: str | None) -> str | None:
        if value is None:
            return None
        token = str(value).strip()
        if len(token) <= 28:
            return token
        return token[:12] + "..." + token[-12:]

    run_root_text = str(run_root) if run_root is not None else "-"
    config_hash = _file_sha256(config_path)
    config_hash_text = _short_hash(config_hash)
    config_text = str(config_path) if config_path is not None else "-"
    if run_root is not None and config_path is not None:
        try:
            config_text = str(config_path.resolve().relative_to(run_root.resolve()))
        except Exception:
            pass

    records_text = str(records_path) if records_path is not None else "-"
    if run_root is not None and records_path is not None:
        try:
            records_text = str(records_path.resolve().relative_to(run_root.resolve()))
        except Exception:
            pass

    manifest_text = str(manifest_path) if manifest_path is not None else "-"
    if outcome.created_at:
        manifest_text += f" (created_at={outcome.created_at})"
    else:
        manifest_mtime = _file_mtime(manifest_path)
        if manifest_mtime is not None:
            manifest_text += f" (mtime={_format_timestamp(manifest_mtime)})"

    notebook_text = str(notebook_path) if notebook_path is not None else "-"
    notebook_mtime = _file_mtime(notebook_path)
    if notebook_mtime is not None:
        notebook_text += f" (mtime={_format_timestamp(notebook_mtime)})"

    lines = [
        f"- Run root: {_code(run_root_text)} `[config]`",
        (
            f"- Config: {_code(config_text)}"
            + (f" (sha256={_code(config_hash_text or '-')})" if config_hash_text else "")
            + " `[config]`"
        ),
        f"- Records path: {_code(records_text)} `[config]`",
        f"- Manifest: {_code(manifest_text)} `[manifest]`",
        f"- Notebook: {_code(notebook_text)} `[manifest]`",
    ]

    if contract.config_error is not None and str(contract.config_error).strip():
        lines.append(f"- Config validation note: {contract.config_error}. `[config]`")

    manifest_mtime = _file_mtime(manifest_path)
    if manifest_mtime is not None and notebook_mtime is not None:
        if manifest_mtime > notebook_mtime + 2.0:
            lines.append(
                "- Freshness: Manifest is newer than this notebook file. "
                "Regenerate the notebook to update the narrative."
            )
        else:
            lines.append("- Freshness: Notebook narrative matches the current manifest timestamp.")
    elif manifest_mtime is None:
        lines.append("- Freshness: Manifest file is not available for freshness checks.")
    elif notebook_mtime is None:
        lines.append("- Freshness: Notebook file timestamp is not available for freshness checks.")

    return "\n".join(lines)


def build_run_details_model(
    contract: RunContractSummary,
    outcome: RunOutcomeSummary,
    *,
    paths_context: RunDetailsPathsContext | None = None,
) -> RunDetailsModel:
    sections: list[Section] = [
        Section(
            title="Definitions",
            body_md=_normalize_section_text(_render_definitions_section()),
        ),
        Section(
            title=f"Scope and quotas ({len(contract.plans)} plans)",
            body_md=_normalize_section_text(_render_plans_section(contract, outcome)),
        ),
        Section(
            title="Acceptance and inputs",
            body_md=_normalize_section_text(_render_constraints_section(contract)),
        ),
        Section(
            title="Constraint literals",
            body_md=_normalize_section_text(_render_constraint_literals_section(contract)),
        ),
        Section(
            title="Execution policy",
            body_md=_normalize_section_text(_render_execution_section(contract)),
        ),
        Section(
            title="Outcome and pressure",
            body_md=_normalize_section_text(_render_outcome_section(contract, outcome)),
        ),
    ]

    if outcome.available:
        sections.append(
            Section(
                title="Pressure by plan",
                body_md=_normalize_section_text(_render_pressure_by_plan_section(contract, outcome)),
            )
        )

    if len(contract.plans) > PLAN_INLINE_DETAIL_THRESHOLD:
        sections.append(
            Section(
                title=f"All plans (raw list, {len(contract.plans)} plans)",
                body_md=_normalize_section_text(_render_all_plans_section(contract, outcome)),
            )
        )

    sections.append(
        Section(
            title="Sources and freshness",
            body_md=_normalize_section_text(_render_sources_section(contract, paths_context, outcome)),
        )
    )

    return RunDetailsModel(summary_lines=tuple(), sections=tuple(sections))


def build_run_details_payload(
    contract: RunContractSummary,
    outcome: RunOutcomeSummary,
    *,
    paths_context: RunDetailsPathsContext | None = None,
) -> dict[str, object]:
    model = build_run_details_model(contract, outcome, paths_context=paths_context)
    return {
        "summary_lines": list(model.summary_lines),
        "sections": [
            {
                "title": section.title,
                "body_md": section.body_md,
            }
            for section in model.sections
        ],
    }


def render_run_details_md(model: RunDetailsModel, *, profile: str = "notebook") -> str:
    lines: list[str] = ["## Run details"]
    lines.extend(model.summary_lines)
    for section in model.sections:
        lines.append("")
        lines.append(f"### {section.title}")
        lines.append(section.body_md)

    if profile == "compact":
        compact = ["Run details:"]
        compact.extend(line[2:] for line in model.summary_lines if line.startswith("- "))
        return "\n".join(compact)

    return "\n".join(lines).strip()


def _render_trace_details() -> str:
    header = "| Intro clause | Config path(s) | Runtime owner | Outcome source |"
    divider = "|---|---|---|---|"
    rows: list[str] = [header, divider]
    for clause, entry in INTRO_TRACE_MAP.items():
        config_paths = ", ".join(entry.config_paths) if entry.config_paths else "-"
        rows.append(f"| {clause} | {config_paths} | {entry.runtime_owner} | {entry.outcome_source} |")
    return "\n".join(rows)


def render_intro_md(
    contract: RunContractSummary,
    outcome: RunOutcomeSummary,
    *,
    style: str = "didactic",
    include_trace: bool = False,
    profile: str = "notebook",
    paths_context: RunDetailsPathsContext | None = None,
) -> str:
    resolved_profile = profile
    if style == "compact" and profile == "notebook":
        resolved_profile = "compact"

    model = build_run_details_model(contract, outcome, paths_context=paths_context)
    rendered = render_run_details_md(model, profile=resolved_profile)

    if include_trace:
        rendered = "\n\n".join([rendered, "### Run intro trace map", _render_trace_details()])
    return rendered.strip()

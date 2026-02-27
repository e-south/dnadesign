"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli/run_intro.py

Render DenseGen notebook run details from validated config payload and run manifest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

PLAN_INLINE_DETAIL_THRESHOLD = 8


@dataclass(frozen=True)
class PlanContract:
    name: str
    quota: int
    acceptance_detail: str | None


@dataclass(frozen=True)
class RunContractSummary:
    sequence_length_bp: int | None
    plans: tuple[PlanContract, ...]
    total_quota: int
    global_acceptance_detail: str | None
    inputs_used: tuple[str, ...]
    max_accepted_per_library: int | None
    round_robin: bool | None
    solver_backend: str | None
    solver_strategy: str | None
    solver_attempt_timeout_seconds: float | None
    no_progress_seconds_before_resample: int | None
    max_consecutive_no_progress_resamples: int | None
    background_filtering_details: tuple[str, ...] = tuple()
    expansion_details: tuple[str, ...] = tuple()
    expansion_detail_blocks: tuple[str, ...] = tuple()
    pool_strategy: str | None = None
    library_sampling_strategy: str | None = None
    library_size: int | None = None
    unique_binding_sites: bool | None = None
    unique_binding_cores: bool | None = None
    max_failed_solutions: int | None = None
    max_failed_solutions_per_target: float | None = None
    schema_version: str | None = None
    output_targets: tuple[str, ...] = tuple()
    config_error: str | None = None


@dataclass(frozen=True)
class PlanOutcome:
    name: str
    generated: int
    quota: int
    stall_events: int
    total_resamples: int
    failed_solutions: int


@dataclass(frozen=True)
class RunOutcomeSummary:
    available: bool
    message: str
    generated_total: int | None
    quota_total: int | None
    progress_pct: float | None
    per_plan: tuple[PlanOutcome, ...]
    stall_events: int
    total_resamples: int
    failed_solutions: int
    created_at: str | None = None


@dataclass(frozen=True)
class IntroTraceEntry:
    config_paths: tuple[str, ...]
    runtime_owner: str
    outcome_source: str


@dataclass(frozen=True)
class RunDetailsPathsContext:
    run_root: Path | None = None
    config_path: Path | None = None
    records_path: Path | None = None
    manifest_path: Path | None = None
    notebook_path: Path | None = None


@dataclass(frozen=True)
class ParsedPlanName:
    base: str
    variants: dict[str, str]


@dataclass(frozen=True)
class PlanRow:
    raw_name: str
    base_name: str
    variants_text: str
    quota: int
    acceptance_detail: str | None
    generated: int | None
    progress_pct: float | None
    stall_events: int
    total_resamples: int
    failed_solutions: int


@dataclass(frozen=True)
class PlanGroup:
    label: str
    rows: tuple[PlanRow, ...]


@dataclass(frozen=True)
class Section:
    title: str
    body_md: str


@dataclass(frozen=True)
class RunDetailsModel:
    summary_lines: tuple[str, ...]
    sections: tuple[Section, ...]


INTRO_TRACE_MAP: dict[str, IntroTraceEntry] = {
    "scope": IntroTraceEntry(
        config_paths=(
            "generation.sequence_length",
            "generation.plan[*].sequences",
            "generation.plan[*].name",
        ),
        runtime_owner="densegen/src/core/pipeline/stage_b_plan_setup.py",
        outcome_source="config",
    ),
    "acceptance_global": IntroTraceEntry(
        config_paths=(
            "generation.sequence_constraints.forbid_kmers",
            "generation.sequence_constraints.allowlist",
        ),
        runtime_owner="densegen/src/core/sequence_constraints/engine.py",
        outcome_source="config",
    ),
    "acceptance_plan": IntroTraceEntry(
        config_paths=(
            "generation.plan[*].fixed_elements.promoter_constraints",
            "generation.plan[*].fixed_elements.fixed_element_matrix",
            "generation.plan[*].regulator_constraints.groups",
        ),
        runtime_owner="densegen/src/core/pipeline/stage_b_plan_setup.py",
        outcome_source="config",
    ),
    "inputs": IntroTraceEntry(
        config_paths=(
            "densegen.inputs[*]",
            "generation.plan[*].sampling.include_inputs",
        ),
        runtime_owner="densegen/src/core/pipeline/plan_execution.py",
        outcome_source="config",
    ),
    "search_policy": IntroTraceEntry(
        config_paths=(
            "generation.sampling.pool_strategy",
            "generation.sampling.library_sampling_strategy",
            "generation.sampling.subsample_size",
            "runtime.max_accepted_per_library",
            "runtime.round_robin",
        ),
        runtime_owner="densegen/src/core/pipeline/stage_b_sampler.py",
        outcome_source="config",
    ),
    "solver_policy": IntroTraceEntry(
        config_paths=(
            "solver.backend",
            "solver.strategy",
            "solver.solver_attempt_timeout_seconds",
        ),
        runtime_owner="densegen/src/core/pipeline/stage_b_runtime_runner.py",
        outcome_source="config",
    ),
    "runtime_guards": IntroTraceEntry(
        config_paths=(
            "runtime.no_progress_seconds_before_resample",
            "runtime.max_consecutive_no_progress_resamples",
        ),
        runtime_owner="densegen/src/core/runtime_policy.py",
        outcome_source="config",
    ),
    "outcome_totals": IntroTraceEntry(
        config_paths=(),
        runtime_owner="densegen/src/core/pipeline/run_finalization.py",
        outcome_source="manifest",
    ),
    "pressure_counters": IntroTraceEntry(
        config_paths=(),
        runtime_owner="densegen/src/core/pipeline/stage_b_runtime_callbacks.py",
        outcome_source="manifest",
    ),
}


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _plural(count: int, singular: str, plural: str) -> str:
    if int(count) == 1:
        return singular
    return plural


def _format_human_name(raw_name: str) -> str:
    cleaned = str(raw_name or "").strip()
    if not cleaned:
        return ""
    base = cleaned
    for suffix in ("_pwm", "_sites", "_input"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    tokens = [token for token in base.replace("-", "_").split("_") if token]
    if not tokens:
        return cleaned
    return " ".join(token[:1].upper() + token[1:] for token in tokens)


def _promoter_display_name(name: str) -> str:
    raw = str(name).strip()
    if raw == "":
        return "promoter"
    normalized = raw.replace("_", "").replace("-", "").lower()
    if normalized.startswith("sigma"):
        suffix = normalized[len("sigma") :]
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if digits:
            return f"σ{digits}"
        return "σ"
    return raw.replace("_", " ")


def _format_int_range(range_value: object, *, units: str) -> str | None:
    if not isinstance(range_value, (list, tuple)):
        return None
    if len(range_value) != 2:
        return None
    lo = _safe_int(range_value[0])
    hi = _safe_int(range_value[1])
    if lo is None or hi is None:
        return None
    if lo == hi:
        return f"{lo} {units}"
    return f"{lo}–{hi} {units}"


def _format_seconds(seconds: float | int | None) -> str | None:
    if seconds is None:
        return None
    value = float(seconds)
    if value.is_integer():
        return f"{int(value)} s"
    return f"{value:.1f} s"


def _format_percent(numerator: int | None, denominator: int | None) -> str:
    if numerator is None or denominator is None or denominator <= 0:
        return "n/a"
    return f"{(float(numerator) / float(denominator)) * 100.0:.1f}%"


def _bool_text(value: bool | None) -> str:
    if value is None:
        return "-"
    return "true" if bool(value) else "false"


def _shorten(text: str, *, max_len: int = 96) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def _summary_with_source(text: str, source: str) -> str:
    return f"- {text} [{source}]"


def _normalize_section_text(text: str) -> str:
    lines = [line.rstrip() for line in str(text).splitlines()]
    normalized: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                normalized.append("")
            continue
        blank_run = 0
        normalized.append(line)
    return "\n".join(normalized).strip()


def _escape_markdown_cell(value: object) -> str:
    text = str(value)
    text = text.replace("|", "\\|")
    text = text.replace("\n", "<br>")
    return text


def _render_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    header_line = "| " + " | ".join(_escape_markdown_cell(item) for item in headers) + " |"
    divider_line = "|" + "|".join("---" for _ in headers) + "|"
    body_lines: list[str] = []
    for row in rows:
        body_lines.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |")
    if not body_lines:
        body_lines.append("| " + " | ".join("-" for _ in headers) + " |")
    return "\n".join([header_line, divider_line, *body_lines])


def parse_plan_name(name: str) -> ParsedPlanName:
    raw = str(name or "").strip()
    if not raw:
        return ParsedPlanName(base="", variants={})
    tokens = raw.split("__")
    base = tokens[0]
    variants: dict[str, str] = {}
    tag_index = 0
    for token in tokens[1:]:
        item = str(token).strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                variants[key] = value
                continue
        tag_index += 1
        variants[f"tag{tag_index}"] = item
    return ParsedPlanName(base=base, variants=variants)


def _summarize_global_constraints(generation_cfg: Mapping[str, object]) -> str | None:
    constraints = generation_cfg.get("sequence_constraints")
    if not isinstance(constraints, Mapping):
        return None

    parts: list[str] = []
    forbid_kmers = constraints.get("forbid_kmers")
    if isinstance(forbid_kmers, list) and forbid_kmers:
        strands: list[str] = []
        include_rc = False
        for rule in forbid_kmers:
            if not isinstance(rule, Mapping):
                continue
            strand = str(rule.get("strands") or "").strip()
            if strand:
                strands.append(strand)
            if bool(rule.get("include_reverse_complements")):
                include_rc = True
        strand_summary = "/".join(sorted(set(strands))) if strands else "both"
        rc_summary = "on" if include_rc else "off"
        parts.append(
            f"{len(forbid_kmers)} forbid_kmers {_plural(len(forbid_kmers), 'rule', 'rules')} "
            f"(strands={strand_summary}, reverse-complements={rc_summary})"
        )

    allowlist = constraints.get("allowlist")
    if isinstance(allowlist, list) and allowlist:
        parts.append(f"{len(allowlist)} allowlist {_plural(len(allowlist), 'selector', 'selectors')}")

    if not parts:
        return None
    return "; ".join(parts)


def _promoter_constraint_detail(constraint: Mapping[str, object]) -> str | None:
    spacer = _format_int_range(constraint.get("spacer_length"), units="bp")
    promoter_name = _promoter_display_name(str(constraint.get("name") or ""))
    upstream = str(constraint.get("upstream") or "").strip()
    downstream = str(constraint.get("downstream") or "").strip()

    pos_parts: list[str] = []
    upstream_pos = _format_int_range(constraint.get("upstream_pos"), units="bp")
    if upstream_pos is not None:
        pos_parts.append(f"upstream window {upstream_pos}")
    downstream_pos = _format_int_range(constraint.get("downstream_pos"), units="bp")
    if downstream_pos is not None:
        pos_parts.append(f"downstream window {downstream_pos}")

    if upstream and downstream and spacer is not None:
        if promoter_name != "promoter":
            base = f"{promoter_name} promoter upstream/downstream motifs with a {spacer} spacer"
        else:
            base = f"promoter upstream/downstream motifs with a {spacer} spacer"
    elif spacer is not None:
        base = f"promoter spacer {spacer}"
    else:
        base = "promoter constraints"

    if pos_parts:
        base += f" ({'; '.join(pos_parts)})"
    return base


def _fixed_element_matrix_detail(matrix: Mapping[str, object]) -> str:
    name = _promoter_display_name(str(matrix.get("name") or "matrix"))
    pieces: list[str] = [f"fixed-element matrix {name}"]
    spacer = _format_int_range(matrix.get("spacer_length"), units="bp")
    if spacer is not None:
        pieces.append(f"spacer {spacer}")
    upstream_pos = _format_int_range(matrix.get("upstream_pos"), units="bp")
    if upstream_pos is not None:
        pieces.append(f"upstream window {upstream_pos}")
    downstream_pos = _format_int_range(matrix.get("downstream_pos"), units="bp")
    if downstream_pos is not None:
        pieces.append(f"downstream window {downstream_pos}")
    if len(pieces) == 1:
        return pieces[0]
    return f"{pieces[0]} ({', '.join(pieces[1:])})"


def _regulator_groups_detail(groups: Sequence[object]) -> str | None:
    entries: list[str] = []
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        name = str(group.get("name") or "").strip()
        members = group.get("members")
        min_required = _safe_int(group.get("min_required"))
        if not name or not isinstance(members, list) or not members:
            continue
        member_count = len([item for item in members if str(item or "").strip()])
        if member_count <= 0:
            continue
        if min_required is not None:
            entries.append(f"{name} min_required={min_required}/{member_count}")
        else:
            entries.append(f"{name} ({member_count} members)")
    if not entries:
        return None
    return "regulator groups: " + ", ".join(entries)


def _plan_acceptance_detail(plan: Mapping[str, object]) -> str | None:
    details: list[str] = []
    fixed_elements = plan.get("fixed_elements")
    if isinstance(fixed_elements, Mapping):
        promoter_constraints = fixed_elements.get("promoter_constraints")
        if isinstance(promoter_constraints, list):
            for constraint in promoter_constraints:
                if not isinstance(constraint, Mapping):
                    continue
                detail = _promoter_constraint_detail(constraint)
                if detail:
                    details.append(detail)
                    break
        matrix = fixed_elements.get("fixed_element_matrix")
        if isinstance(matrix, Mapping):
            details.append(_fixed_element_matrix_detail(matrix))

    regulator_constraints = plan.get("regulator_constraints")
    if isinstance(regulator_constraints, Mapping):
        groups = regulator_constraints.get("groups")
        if isinstance(groups, list) and groups:
            group_detail = _regulator_groups_detail(groups)
            if group_detail is not None:
                details.append(group_detail)

    if not details:
        return None
    return "; ".join(details)


def _extract_plans(generation_cfg: Mapping[str, object]) -> tuple[tuple[PlanContract, ...], int]:
    plans_raw = generation_cfg.get("plan")
    if not isinstance(plans_raw, list):
        return (tuple(), 0)

    plans: list[PlanContract] = []
    total_quota = 0
    for plan in plans_raw:
        if not isinstance(plan, Mapping):
            continue
        name = str(plan.get("name") or "").strip()
        quota = _safe_int(plan.get("sequences"))
        if not name or quota is None:
            continue
        total_quota += int(quota)
        plans.append(
            PlanContract(
                name=name,
                quota=int(quota),
                acceptance_detail=_plan_acceptance_detail(plan),
            )
        )
    return tuple(plans), int(total_quota)


def _extract_inputs_used(*, plans_raw: object, inputs_cfg: Sequence[object]) -> tuple[str, ...]:
    included: set[str] = set()
    if isinstance(plans_raw, list):
        for plan in plans_raw:
            if not isinstance(plan, Mapping):
                continue
            sampling = plan.get("sampling")
            if not isinstance(sampling, Mapping):
                continue
            include_inputs = sampling.get("include_inputs")
            if not isinstance(include_inputs, list):
                continue
            for entry in include_inputs:
                name = str(entry or "").strip()
                if name:
                    included.add(name)

    rendered: list[str] = []
    background_labels: list[str] = []
    for input_cfg in inputs_cfg:
        if not isinstance(input_cfg, Mapping):
            continue
        name = str(input_cfg.get("name") or "").strip()
        if not name or name not in included:
            continue
        display = _format_human_name(name)
        if not display:
            continue
        input_type = str(input_cfg.get("type") or "").strip()
        if input_type == "background_pool":
            background_labels.append(display)
            continue
        rendered.append(display)

    if rendered:
        return tuple(rendered)
    if background_labels:
        return tuple(background_labels)
    return tuple()


def _extract_background_filtering_details(*, plans_raw: object, inputs_cfg: Sequence[object]) -> tuple[str, ...]:
    included: set[str] = set()
    if isinstance(plans_raw, list):
        for plan in plans_raw:
            if not isinstance(plan, Mapping):
                continue
            sampling = plan.get("sampling")
            if not isinstance(sampling, Mapping):
                continue
            include_inputs = sampling.get("include_inputs")
            if not isinstance(include_inputs, list):
                continue
            for entry in include_inputs:
                name = str(entry or "").strip()
                if name:
                    included.add(name)

    details: list[str] = []
    for input_cfg in inputs_cfg:
        if not isinstance(input_cfg, Mapping):
            continue
        input_name = str(input_cfg.get("name") or "").strip()
        if input_name == "" or input_name not in included:
            continue
        sampling = input_cfg.get("sampling")
        if not isinstance(sampling, Mapping):
            continue
        filters = sampling.get("filters")
        if not isinstance(filters, Mapping):
            continue
        fimo_exclude = filters.get("fimo_exclude")
        if not isinstance(fimo_exclude, Mapping):
            continue
        pwms_input = fimo_exclude.get("pwms_input")
        if not isinstance(pwms_input, list) or not pwms_input:
            continue
        excluded: list[str] = []
        seen: set[str] = set()
        for item in pwms_input:
            label = _format_human_name(str(item or ""))
            if label and label not in seen:
                excluded.append(label)
                seen.add(label)
        if not excluded:
            continue
        detail = f"{_format_human_name(input_name)} excludes FIMO hits for " + ", ".join(excluded)
        if bool(fimo_exclude.get("allow_zero_hit_only")):
            detail += " (zero-hit only)"
        details.append(detail)
    return tuple(details)


def _resolve_motif_set_name(*, motif_sets: Mapping[str, object], variant_id: str, sequence: str) -> str | None:
    if variant_id == "" or sequence == "":
        return None
    sequence_u = str(sequence).strip().upper()
    for set_name, set_payload in motif_sets.items():
        if not isinstance(set_payload, Mapping):
            continue
        candidate = set_payload.get(variant_id)
        if candidate is None:
            continue
        if str(candidate).strip().upper() == sequence_u:
            return str(set_name)
    return None


def _motif_set_size(motif_sets: Mapping[str, object], set_name: str | None) -> int | None:
    if set_name is None:
        return None
    payload = motif_sets.get(set_name)
    if isinstance(payload, Mapping):
        return len([key for key in payload.keys() if str(key).strip()])
    if isinstance(payload, list):
        return len([entry for entry in payload if str(entry).strip()])
    return None


def _format_motif_set_literals(
    *, motif_sets: Mapping[str, object], set_name: str | None, selected_ids: set[str]
) -> tuple[str, ...]:
    if set_name is None or not selected_ids:
        return tuple()
    payload = motif_sets.get(set_name)
    if not isinstance(payload, Mapping):
        return tuple()
    ordered_ids = [str(key).strip() for key in payload.keys() if str(key).strip()]
    literals: list[str] = []
    for variant_id in ordered_ids:
        if variant_id not in selected_ids:
            continue
        sequence = str(payload.get(variant_id) or "").strip().upper()
        if sequence:
            literals.append(f"`{variant_id}={sequence}`")
    return tuple(literals)


def _collect_promoter_expansion_groups(
    *, plans_raw: Sequence[object], motif_sets: Mapping[str, object]
) -> dict[
    tuple[str, str, str | None, str | None, str | None, str | None, str | None],
    dict[str, set[object]],
]:
    grouped: dict[
        tuple[str, str, str | None, str | None, str | None, str | None, str | None],
        dict[str, set[object]],
    ] = {}
    for plan in plans_raw:
        if not isinstance(plan, Mapping):
            continue
        plan_name = str(plan.get("name") or "").strip()
        if plan_name == "":
            continue
        fixed_elements = plan.get("fixed_elements")
        if not isinstance(fixed_elements, Mapping):
            continue
        promoter_constraints = fixed_elements.get("promoter_constraints")
        if not isinstance(promoter_constraints, list) or not promoter_constraints:
            continue
        for constraint in promoter_constraints:
            if not isinstance(constraint, Mapping):
                continue
            upstream_variant_id = str(constraint.get("upstream_variant_id") or "").strip()
            downstream_variant_id = str(constraint.get("downstream_variant_id") or "").strip()
            if upstream_variant_id == "" or downstream_variant_id == "":
                continue
            upstream = str(constraint.get("upstream") or "").strip()
            downstream = str(constraint.get("downstream") or "").strip()
            upstream_set = _resolve_motif_set_name(
                motif_sets=motif_sets,
                variant_id=upstream_variant_id,
                sequence=upstream,
            )
            downstream_set = _resolve_motif_set_name(
                motif_sets=motif_sets,
                variant_id=downstream_variant_id,
                sequence=downstream,
            )
            spacer = _format_int_range(constraint.get("spacer_length"), units="bp")
            upstream_window = _format_int_range(constraint.get("upstream_pos"), units="bp")
            downstream_window = _format_int_range(constraint.get("downstream_pos"), units="bp")
            base_name = plan_name.split("__", 1)[0].replace("_", " ")
            key = (
                base_name,
                _promoter_display_name(str(constraint.get("name") or "promoter")),
                upstream_set,
                downstream_set,
                spacer,
                upstream_window,
                downstream_window,
            )
            state = grouped.setdefault(
                key,
                {
                    "upstream_ids": set(),
                    "downstream_ids": set(),
                    "pairs": set(),
                },
            )
            state["upstream_ids"].add(upstream_variant_id)
            state["downstream_ids"].add(downstream_variant_id)
            state["pairs"].add((upstream_variant_id, downstream_variant_id))
    return grouped


def _resolve_expansion_mode(
    *,
    pair_count: int,
    expected_pairs: int,
    upstream_ids: set[str],
    downstream_ids: set[str],
    upstream_available: int | None,
    downstream_available: int | None,
) -> str:
    mode = "partial cross-product"
    if expected_pairs > 0 and pair_count == expected_pairs:
        if (
            upstream_available is not None
            and downstream_available is not None
            and len(upstream_ids) == int(upstream_available)
            and len(downstream_ids) == int(downstream_available)
        ):
            return "exhaustive cross-product"
        return "cross-product"
    return mode


def _build_expansion_detail_text(
    *,
    base_name: str,
    promoter_name: str,
    mode: str,
    upstream_set: str | None,
    downstream_set: str | None,
    upstream_ids: set[str],
    downstream_ids: set[str],
    upstream_available: int | None,
    downstream_available: int | None,
    pair_count: int,
    upstream_window: str | None,
    spacer: str | None,
    downstream_window: str | None,
) -> str:
    upstream_part = (
        f"{upstream_set} ({len(upstream_ids)}/{upstream_available} variants)"
        if upstream_set is not None and upstream_available is not None
        else f"{len(upstream_ids)} upstream variants"
    )
    downstream_part = (
        f"{downstream_set} ({len(downstream_ids)}/{downstream_available} variants)"
        if downstream_set is not None and downstream_available is not None
        else f"{len(downstream_ids)} downstream variants"
    )

    detail = (
        f"{base_name} expands {promoter_name} via {mode} from "
        f"{upstream_part} × {downstream_part} ({pair_count} variants)"
    )
    if upstream_window is not None:
        upstream_source = upstream_set if upstream_set is not None else "the upstream set"
        detail += (
            ", where the upstream motif from "
            f"{upstream_source} must start within positions {upstream_window} "
            "of each output sequence"
        )
    if spacer is not None:
        detail += f", with spacer {spacer} between upstream and downstream motifs"
    if downstream_window is not None:
        detail += f", and downstream motif constrained to positions {downstream_window}"
    return detail


def _build_expansion_detail_block(
    *,
    motif_sets: Mapping[str, object],
    upstream_set: str | None,
    downstream_set: str | None,
    upstream_ids: set[str],
    downstream_ids: set[str],
    mode: str,
    pair_count: int,
    upstream_window: str | None,
    spacer: str | None,
    downstream_window: str | None,
) -> str:
    upstream_literals = _format_motif_set_literals(
        motif_sets=motif_sets,
        set_name=upstream_set,
        selected_ids=upstream_ids,
    )
    downstream_literals = _format_motif_set_literals(
        motif_sets=motif_sets,
        set_name=downstream_set,
        selected_ids=downstream_ids,
    )
    block_lines: list[str] = []
    if upstream_set is not None and upstream_literals:
        block_lines.append(f"Upstream set `{upstream_set}` literals:")
        block_lines.extend(f"- {literal}" for literal in upstream_literals)
        block_lines.append("")
    if downstream_set is not None and downstream_literals:
        block_lines.append(f"Downstream set `{downstream_set}` literals:")
        block_lines.extend(f"- {literal}" for literal in downstream_literals)
        block_lines.append("")
    block_lines.append(f"Combinatorics: {mode} over {pair_count} expanded plan variants.")
    if upstream_window is not None:
        upstream_source = upstream_set if upstream_set is not None else "upstream set"
        block_lines.append(
            f"Placement constraint: upstream motif from `{upstream_source}` starts within positions "
            f"{upstream_window} of each output sequence."
        )
    if spacer is not None:
        block_lines.append(f"Spacer constraint: {spacer} between upstream and downstream motifs.")
    if downstream_window is not None:
        block_lines.append(f"Downstream placement window: {downstream_window}.")
    return "\n".join(block_lines)


def _extract_expansion_details(
    *, generation_cfg: Mapping[str, object], motif_sets: Mapping[str, object]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    plans_raw = generation_cfg.get("plan")
    if not isinstance(plans_raw, list):
        return (tuple(), tuple())

    grouped = _collect_promoter_expansion_groups(plans_raw=plans_raw, motif_sets=motif_sets)

    details: list[str] = []
    detail_blocks: list[str] = []
    for key, state in sorted(grouped.items(), key=lambda item: item[0]):
        (
            base_name,
            promoter_name,
            upstream_set,
            downstream_set,
            spacer,
            upstream_window,
            downstream_window,
        ) = key
        upstream_ids = set(str(value) for value in state["upstream_ids"])
        downstream_ids = set(str(value) for value in state["downstream_ids"])
        pair_count = int(len(state["pairs"]))
        expected_pairs = int(len(upstream_ids) * len(downstream_ids))
        upstream_available = _motif_set_size(motif_sets, upstream_set)
        downstream_available = _motif_set_size(motif_sets, downstream_set)

        mode = _resolve_expansion_mode(
            pair_count=pair_count,
            expected_pairs=expected_pairs,
            upstream_ids=upstream_ids,
            downstream_ids=downstream_ids,
            upstream_available=upstream_available,
            downstream_available=downstream_available,
        )
        detail = _build_expansion_detail_text(
            base_name=base_name,
            promoter_name=promoter_name,
            mode=mode,
            upstream_set=upstream_set,
            downstream_set=downstream_set,
            upstream_ids=upstream_ids,
            downstream_ids=downstream_ids,
            upstream_available=upstream_available,
            downstream_available=downstream_available,
            pair_count=pair_count,
            upstream_window=upstream_window,
            spacer=spacer,
            downstream_window=downstream_window,
        )
        details.append(detail)

        detail_block = _build_expansion_detail_block(
            motif_sets=motif_sets,
            upstream_set=upstream_set,
            downstream_set=downstream_set,
            upstream_ids=upstream_ids,
            downstream_ids=downstream_ids,
            mode=mode,
            pair_count=pair_count,
            upstream_window=upstream_window,
            spacer=spacer,
            downstream_window=downstream_window,
        )
        detail_blocks.append(detail_block)

    return (tuple(details), tuple(detail_blocks))


def extract_contract(
    config_payload: Mapping[str, object] | None, *, config_error: str | None = None
) -> RunContractSummary:
    payload = config_payload if isinstance(config_payload, Mapping) else {}
    densegen_cfg = payload.get("densegen")
    if not isinstance(densegen_cfg, Mapping):
        densegen_cfg = {}

    generation_cfg = densegen_cfg.get("generation")
    if not isinstance(generation_cfg, Mapping):
        generation_cfg = {}

    runtime_cfg = densegen_cfg.get("runtime")
    if not isinstance(runtime_cfg, Mapping):
        runtime_cfg = {}

    solver_cfg = densegen_cfg.get("solver")
    if not isinstance(solver_cfg, Mapping):
        solver_cfg = {}

    output_cfg = densegen_cfg.get("output")
    if not isinstance(output_cfg, Mapping):
        output_cfg = {}

    sampling_cfg = generation_cfg.get("sampling")
    if not isinstance(sampling_cfg, Mapping):
        sampling_cfg = {}

    inputs_cfg = densegen_cfg.get("inputs")
    inputs_seq: Sequence[object] = inputs_cfg if isinstance(inputs_cfg, list) else ()
    motif_sets_cfg = densegen_cfg.get("motif_sets")
    motif_sets_map: Mapping[str, object] = motif_sets_cfg if isinstance(motif_sets_cfg, Mapping) else {}

    plans, total_quota = _extract_plans(generation_cfg)

    solver_backend = str(solver_cfg.get("backend") or "").strip() or None
    solver_strategy = str(solver_cfg.get("strategy") or "").strip() or None

    raw_round_robin = runtime_cfg.get("round_robin")
    round_robin = raw_round_robin if isinstance(raw_round_robin, bool) else None

    expansion_details, expansion_detail_blocks = _extract_expansion_details(
        generation_cfg=generation_cfg,
        motif_sets=motif_sets_map,
    )

    raw_unique_sites = sampling_cfg.get("unique_binding_sites")
    unique_binding_sites = raw_unique_sites if isinstance(raw_unique_sites, bool) else None
    raw_unique_cores = sampling_cfg.get("unique_binding_cores")
    unique_binding_cores = raw_unique_cores if isinstance(raw_unique_cores, bool) else None
    output_targets_raw = output_cfg.get("targets")
    output_targets: tuple[str, ...] = tuple()
    if isinstance(output_targets_raw, list):
        output_targets = tuple(text for text in (str(item).strip() for item in output_targets_raw) if text)
    schema_version = str(densegen_cfg.get("schema_version") or "").strip() or None

    return RunContractSummary(
        sequence_length_bp=_safe_int(generation_cfg.get("sequence_length")),
        plans=plans,
        total_quota=int(total_quota),
        global_acceptance_detail=_summarize_global_constraints(generation_cfg),
        inputs_used=_extract_inputs_used(plans_raw=generation_cfg.get("plan"), inputs_cfg=inputs_seq),
        max_accepted_per_library=_safe_int(runtime_cfg.get("max_accepted_per_library")),
        round_robin=round_robin,
        solver_backend=solver_backend,
        solver_strategy=solver_strategy,
        solver_attempt_timeout_seconds=_safe_float(solver_cfg.get("solver_attempt_timeout_seconds")),
        no_progress_seconds_before_resample=_safe_int(runtime_cfg.get("no_progress_seconds_before_resample")),
        max_consecutive_no_progress_resamples=_safe_int(runtime_cfg.get("max_consecutive_no_progress_resamples")),
        background_filtering_details=_extract_background_filtering_details(
            plans_raw=generation_cfg.get("plan"),
            inputs_cfg=inputs_seq,
        ),
        expansion_details=expansion_details,
        expansion_detail_blocks=expansion_detail_blocks,
        pool_strategy=str(sampling_cfg.get("pool_strategy") or "").strip() or None,
        library_sampling_strategy=str(sampling_cfg.get("library_sampling_strategy") or "").strip() or None,
        library_size=_safe_int(sampling_cfg.get("subsample_size")) or _safe_int(sampling_cfg.get("library_size")),
        unique_binding_sites=unique_binding_sites,
        unique_binding_cores=unique_binding_cores,
        max_failed_solutions=_safe_int(runtime_cfg.get("max_failed_solutions")),
        max_failed_solutions_per_target=_safe_float(runtime_cfg.get("max_failed_solutions_per_target")),
        schema_version=schema_version,
        output_targets=output_targets,
        config_error=config_error,
    )


def _aggregate_per_plan(items: object, *, plan_order: Sequence[str]) -> tuple[tuple[PlanOutcome, ...], int, int, int]:
    if not isinstance(items, list):
        return (tuple(), 0, 0, 0)

    by_plan: dict[str, dict[str, int]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("plan_name") or "").strip()
        if not name:
            continue
        state = by_plan.setdefault(
            name,
            {
                "generated": 0,
                "quota": 0,
                "stall_events": 0,
                "total_resamples": 0,
                "failed_solutions": 0,
            },
        )
        state["generated"] += _safe_int(item.get("generated")) or 0
        state["quota"] += _safe_int(item.get("quota")) or 0
        state["stall_events"] += _safe_int(item.get("stall_events")) or 0
        state["total_resamples"] += _safe_int(item.get("total_resamples")) or 0
        state["failed_solutions"] += _safe_int(item.get("failed_solutions")) or 0

    ordered_names: list[str] = []
    seen: set[str] = set()
    for name in plan_order:
        if name in by_plan and name not in seen:
            ordered_names.append(name)
            seen.add(name)
    for name in by_plan:
        if name not in seen:
            ordered_names.append(name)
            seen.add(name)

    per_plan: list[PlanOutcome] = []
    stall_events = 0
    total_resamples = 0
    failed_solutions = 0
    for name in ordered_names:
        state = by_plan[name]
        per_plan.append(
            PlanOutcome(
                name=name,
                generated=int(state["generated"]),
                quota=int(state["quota"]),
                stall_events=int(state["stall_events"]),
                total_resamples=int(state["total_resamples"]),
                failed_solutions=int(state["failed_solutions"]),
            )
        )
        stall_events += int(state["stall_events"])
        total_resamples += int(state["total_resamples"])
        failed_solutions += int(state["failed_solutions"])

    return (tuple(per_plan), int(stall_events), int(total_resamples), int(failed_solutions))


def extract_outcome(
    manifest_payload: Mapping[str, object] | None,
    *,
    plan_order: Sequence[str],
    error_message: str | None = None,
) -> RunOutcomeSummary:
    if error_message is not None:
        return RunOutcomeSummary(
            available=False,
            message=error_message,
            generated_total=None,
            quota_total=None,
            progress_pct=None,
            per_plan=tuple(),
            stall_events=0,
            total_resamples=0,
            failed_solutions=0,
            created_at=None,
        )

    if not isinstance(manifest_payload, Mapping):
        return RunOutcomeSummary(
            available=False,
            message="manifest not found",
            generated_total=None,
            quota_total=None,
            progress_pct=None,
            per_plan=tuple(),
            stall_events=0,
            total_resamples=0,
            failed_solutions=0,
            created_at=None,
        )

    generated_total = _safe_int(manifest_payload.get("total_generated"))
    if generated_total is None:
        return RunOutcomeSummary(
            available=False,
            message="manifest not found",
            generated_total=None,
            quota_total=None,
            progress_pct=None,
            per_plan=tuple(),
            stall_events=0,
            total_resamples=0,
            failed_solutions=0,
            created_at=None,
        )

    per_plan, stall_events, total_resamples, failed_solutions = _aggregate_per_plan(
        manifest_payload.get("items"),
        plan_order=plan_order,
    )
    quota_total = _safe_int(manifest_payload.get("total_quota"))
    if quota_total is None and per_plan:
        quota_total = int(sum(item.quota for item in per_plan))

    progress_pct = _safe_float(manifest_payload.get("quota_progress_pct"))
    if progress_pct is None and quota_total is not None and quota_total > 0:
        progress_pct = float(generated_total) * 100.0 / float(quota_total)

    created_at = str(manifest_payload.get("created_at") or "").strip() or None

    return RunOutcomeSummary(
        available=True,
        message="",
        generated_total=int(generated_total),
        quota_total=int(quota_total) if quota_total is not None else None,
        progress_pct=progress_pct,
        per_plan=per_plan,
        stall_events=int(stall_events),
        total_resamples=int(total_resamples),
        failed_solutions=int(failed_solutions),
        created_at=created_at,
    )


def build_run_details_model(
    contract: RunContractSummary,
    outcome: RunOutcomeSummary,
    *,
    paths_context: RunDetailsPathsContext | None = None,
) -> RunDetailsModel:
    from .run_intro_rendering import build_run_details_model as _impl

    return _impl(contract, outcome, paths_context=paths_context)


def build_run_details_payload(
    contract: RunContractSummary,
    outcome: RunOutcomeSummary,
    *,
    paths_context: RunDetailsPathsContext | None = None,
) -> dict[str, object]:
    from .run_intro_rendering import build_run_details_payload as _impl

    return _impl(contract, outcome, paths_context=paths_context)


def render_run_details_md(model: RunDetailsModel, *, profile: str = "notebook") -> str:
    from .run_intro_rendering import render_run_details_md as _impl

    return _impl(model, profile=profile)


def render_intro_md(
    contract: RunContractSummary,
    outcome: RunOutcomeSummary,
    *,
    style: str = "didactic",
    include_trace: bool = False,
    profile: str = "notebook",
    paths_context: RunDetailsPathsContext | None = None,
) -> str:
    from .run_intro_rendering import render_intro_md as _impl

    return _impl(
        contract,
        outcome,
        style=style,
        include_trace=include_trace,
        profile=profile,
        paths_context=paths_context,
    )

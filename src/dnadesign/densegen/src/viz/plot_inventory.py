"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_inventory.py

Shared plot inventory helpers for manifest metadata, notebook gallery discovery,
and availability status resolution.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from typing_extensions import Literal

from .plot_registry import PLOT_SPECS

PlotAvailabilityState = Literal["generated", "recoverable_read_only", "requires_local_artifacts"]

HIDDEN_VISUAL_PLOT_TYPES = frozenset({"run_health/summary_table"})


def ordered_unique(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if not token or token in seen:
            continue
        ordered.append(token)
        seen.add(token)
    return ordered


def base_plot_id(plot_type: str) -> str:
    token = str(plot_type or "").strip()
    if "/" in token:
        return str(token.split("/", 1)[0]).strip()
    return token


def plot_spec(plot_id: str) -> Mapping[str, object]:
    return PLOT_SPECS.get(base_plot_id(plot_id), {})


def plot_missing_state(plot_id: str) -> PlotAvailabilityState:
    raw = str(plot_spec(plot_id).get("missing_state") or "").strip()
    if raw in {"generated", "recoverable_read_only", "requires_local_artifacts"}:
        return raw
    return "requires_local_artifacts"


def plot_required_artifacts(plot_id: str) -> tuple[str, ...]:
    raw = plot_spec(plot_id).get("required_artifacts")
    if not isinstance(raw, (list, tuple)):
        return tuple()
    return tuple(str(item).strip() for item in raw if str(item).strip())


def plot_missing_hint(plot_id: str) -> str:
    return str(plot_spec(plot_id).get("missing_hint") or "").strip()


def stage_b_scope_seed_plot_ids() -> list[str]:
    return [str(plot_id) for plot_id, spec in PLOT_SPECS.items() if bool(spec.get("seed_stage_b_scope_when_missing"))]


def infer_plot_id_from_path(relative_parts: Sequence[str], stem: str) -> str:
    if not relative_parts:
        return ""
    head = str(relative_parts[0]).strip().lower()
    normalized_stem = str(stem or "").strip().lower()
    if head == "stage_a":
        return "stage_a_summary"
    if head == "stage_b":
        if "showcase" in normalized_stem:
            return "dense_array_video_showcase"
        if "usage" in normalized_stem:
            return "tfbs_usage"
        return "placement_map"
    if head == "run_health":
        return "run_health"
    return ""


def build_visual_plot_type(plot_id: str, *, plot_name: str, variant: str, stem: str) -> str:
    base = str(plot_id or "").strip()
    variant_token = str(variant or "").strip()
    plot_name_token = str(plot_name or "").strip()
    stem_token = str(stem or "").strip()
    if base == "dense_array_video_showcase":
        return base
    if base and variant_token and variant_token != base:
        return f"{base}/{variant_token}"
    if base:
        return base
    if variant_token:
        return variant_token
    if plot_name_token:
        return plot_name_token
    return stem_token


def manifest_path_fields(name: str, rel_path: Path) -> dict[str, str]:
    fields: dict[str, str] = {"plot_id": str(name)}
    parts = rel_path.parts
    stem = rel_path.stem
    if name == "dense_array_video_showcase":
        fields["group"] = "stage_b"
        fields["family"] = "showcase"
        if len(parts) >= 2:
            fields["plan_name"] = parts[1]
        fields["variant"] = stem
        return fields
    if name == "stage_a_summary":
        fields["group"] = "stage_a"
        fields["family"] = "stage_a"
        fields["plan_name"] = "stage_a"
        if stem == "background_logo":
            fields["variant"] = "background_logo"
            fields["input_name"] = "background"
        elif stem.endswith("__background_logo"):
            fields["variant"] = "background_logo"
            fields["input_name"] = stem[: -len("__background_logo")]
        else:
            fields["variant"] = stem
        return fields
    if name == "placement_map":
        fields["group"] = "stage_b"
        fields["family"] = "plan"
        if len(parts) >= 3:
            fields["plan_name"] = parts[1]
        if len(parts) >= 4:
            fields["input_name"] = parts[2]
        fields["variant"] = stem
        return fields
    if name == "tfbs_usage":
        fields["group"] = "stage_b"
        fields["family"] = "plan"
        if len(parts) >= 3:
            fields["plan_name"] = parts[1]
        if len(parts) >= 4:
            fields["input_name"] = parts[2]
        fields["variant"] = stem
        return fields
    if name == "run_health":
        fields["group"] = "run"
        fields["family"] = "run_health"
        fields["variant"] = stem
        return fields
    fields["variant"] = stem
    return fields


def resolve_plot_record(
    *,
    plot_root: Path,
    plot_path: Path,
    manifest_entry: Mapping[str, object] | None = None,
    source_rank: int = 1,
) -> dict[str, object]:
    entry = dict(manifest_entry or {})
    root = Path(plot_root).expanduser().resolve()
    resolved_path = Path(plot_path).expanduser().resolve()
    try:
        relative_path = resolved_path.relative_to(root)
    except Exception:
        relative_path = Path(resolved_path.name)
    relative_parts = tuple(str(part) for part in relative_path.parts)
    plot_id = str(entry.get("plot_id") or entry.get("name") or "").strip()
    if not plot_id:
        plot_id = infer_plot_id_from_path(relative_parts, resolved_path.stem)
    path_fields = manifest_path_fields(plot_id, relative_path) if plot_id else {}
    plan_name = str(entry.get("plan_name") or path_fields.get("plan_name") or "").strip()
    if not plan_name:
        if len(relative_parts) >= 2 and relative_parts[0] == "stage_b":
            plan_name = str(relative_parts[1]).strip() or "unscoped"
        elif len(relative_parts) >= 1 and relative_parts[0] == "stage_a":
            plan_name = "stage_a"
        else:
            plan_name = "unscoped"
    input_name = str(entry.get("input_name") or path_fields.get("input_name") or "").strip()
    plot_name = str(entry.get("name") or resolved_path.stem)
    variant = str(entry.get("variant") or path_fields.get("variant") or resolved_path.stem or "")
    group = str(entry.get("group") or path_fields.get("group") or "").strip()
    family = str(entry.get("family") or path_fields.get("family") or "").strip()
    return {
        "path": resolved_path,
        "plot_id": plot_id,
        "visual_plot_type": build_visual_plot_type(
            plot_id,
            plot_name=plot_name,
            variant=variant,
            stem=resolved_path.stem,
        ),
        "plan_name": plan_name,
        "input_name": input_name,
        "plot_name": plot_name,
        "variant": variant,
        "description": str(entry.get("description") or ""),
        "group": group,
        "family": family,
        "_source_rank": int(source_rank),
    }


def resolve_plot_availability(
    plot_id: str,
    *,
    generated_plot_ids: Iterable[str],
) -> PlotAvailabilityState:
    token = str(plot_id or "").strip()
    generated_tokens = ordered_unique(generated_plot_ids)
    generated_base_ids = {base_plot_id(item) for item in generated_tokens if base_plot_id(item)}
    if token in generated_tokens or base_plot_id(token) in generated_base_ids:
        return "generated"
    return plot_missing_state(token)


def build_plot_ids_by_scope(
    plot_entries: Iterable[Mapping[str, object]],
    *,
    stage_b_scope_names: Iterable[str] = (),
    known_plot_ids: Iterable[str] = (),
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    entries = list(plot_entries)
    plot_ids_by_scope: dict[str, list[str]] = {}
    generated_plot_ids_by_scope: dict[str, list[str]] = {}

    generated_all = sorted(
        {
            str(entry.get("visual_plot_type") or "").strip()
            for entry in entries
            if str(entry.get("visual_plot_type") or "").strip()
            and str(entry.get("visual_plot_type") or "").strip() not in HIDDEN_VISUAL_PLOT_TYPES
        }
    )
    generated_plot_ids_by_scope["all"] = generated_all
    plot_ids_by_scope["all"] = ordered_unique(
        [
            *(str(name).strip() for name in known_plot_ids),
            *generated_all,
        ]
    )

    scope_names = ordered_unique(
        [
            *(str(entry.get("plan_name") or "").strip() for entry in entries),
            *(str(name).strip() for name in stage_b_scope_names),
        ]
    )
    stage_b_scope_set = {name for name in scope_names if name not in {"stage_a", "unscoped"}}
    for scope_name in scope_names:
        generated_scope = sorted(
            {
                str(entry.get("visual_plot_type") or "").strip()
                for entry in entries
                if str(entry.get("plan_name") or "").strip() == scope_name
                and str(entry.get("visual_plot_type") or "").strip()
                and str(entry.get("visual_plot_type") or "").strip() not in HIDDEN_VISUAL_PLOT_TYPES
            }
        )
        generated_plot_ids_by_scope[scope_name] = generated_scope
        available_scope = list(generated_scope)
        generated_scope_base_ids = {base_plot_id(item) for item in generated_scope if base_plot_id(item)}
        if scope_name in stage_b_scope_set:
            available_scope.extend(
                plot_id for plot_id in stage_b_scope_seed_plot_ids() if plot_id not in generated_scope_base_ids
            )
        plot_ids_by_scope[scope_name] = ordered_unique(available_scope)
    return plot_ids_by_scope, generated_plot_ids_by_scope

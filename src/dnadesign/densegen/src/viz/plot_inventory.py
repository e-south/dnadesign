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

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from typing_extensions import Literal

from .plot_registry import PLOT_SPECS

PlotAvailabilityState = Literal["generated", "recoverable_read_only", "requires_local_artifacts"]
InventoryPayloadSource = Literal["current_inventory", "plot_manifest", "missing", "invalid"]

HIDDEN_VISUAL_PLOT_TYPES = frozenset()
LEGACY_PUBLIC_PLOT_IDS = frozenset(
    {
        "accepted_arrays_by_plan",
        "dataset_metadata_heatmap",
        "dataset_source_inventory",
        "dense_array_video_showcase",
        "plan_by_regulator_heatmap",
        "placement_map",
        "retained_vs_deployed_length_shift",
        "retained_vs_deployed_tier_mix",
        "run_health",
        "stage_a_summary",
        "tfbs_usage",
        "upstream_evidence_quality_summary",
        "used_unique_vs_retained",
    }
)
PLOT_MANIFEST_FILENAME = "plot_manifest.json"
CURRENT_INVENTORY_FILENAME = "current_inventory.json"
ARTIFACT_LEDGER_FILENAME = "artifact_ledger.json"
ANALYSIS_SURFACE_CONTRACT_VERSION = "densegen.analysis_surface.v2"
CURRENT_INVENTORY_SCHEMA_VERSION = "densegen.current_inventory.v2"
ARTIFACT_LEDGER_SCHEMA_VERSION = "densegen.artifact_ledger.v1"
_PLAN_BASE_LABELS = {
    "background_only": "Background",
    "ciprofloxacin": "Cipro",
    "ethanol": "EtOH",
    "ethanol_ciprofloxacin": "EtOH + Cipro",
}
_PLAN_VARIANT_LABELS = {
    "sig35": "σ70",
    "sigma70": "σ70",
}
_STAGE_A_PLOT_IDS = frozenset(
    {
        "background_sequence_logo",
        "stage_a_pool_diversity",
        "stage_a_pool_score_strata",
        "stage_a_sampling_yield",
    }
)
_STAGE_B_SCOPED_PLOT_IDS = frozenset({"placement_occupancy_map", "tfbs_concentration_profile"})
_STAGE_B_SUMMARY_PLOT_IDS = frozenset(
    {
        "plan_regulator_deployment_heatmap",
        "score_strata_and_deployed_length_bridge",
        "retained_pool_coverage_by_regulator",
        "retained_vs_deployed_length_mix_by_regulator",
        "retained_vs_deployed_tier_mix_by_regulator",
        "upstream_motif_supply_and_pwm_strength",
    }
)
_RUN_PLOT_IDS = frozenset(
    {
        "attempt_outcome_timeline",
        "compression_ratio_by_plan",
        "solve_pressure_and_progress",
    }
)
_DATASET_PLOT_IDS = frozenset({"source_cohort_concentration", "source_plan_input_heatmap"})


def _analysis_surface_lists() -> tuple[list[str], list[str]]:
    from dnadesign.densegen.analysis_surface import (
        operator_visible_surface_plot_ids,
        optional_surface_plot_ids,
    )

    return list(operator_visible_surface_plot_ids()), list(optional_surface_plot_ids())


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


def notebook_visible_plot_ids() -> list[str]:
    core_ids, optional_ids = _analysis_surface_lists()
    return ordered_unique([*core_ids, *optional_ids])


def required_notebook_plot_ids() -> list[str]:
    return notebook_visible_plot_ids()


def missing_notebook_visible_plot_ids(plot_entries: Iterable[Mapping[str, object]]) -> list[str]:
    generated_base_ids = {
        base_plot_id(_entry_plot_id(entry)) for entry in plot_entries if base_plot_id(_entry_plot_id(entry))
    }
    return [plot_id for plot_id in required_notebook_plot_ids() if plot_id not in generated_base_ids]


def plot_manifest_path(plot_root: Path) -> Path:
    return Path(plot_root) / PLOT_MANIFEST_FILENAME


def current_inventory_path(plot_root: Path) -> Path:
    return Path(plot_root) / CURRENT_INVENTORY_FILENAME


def artifact_ledger_path(plot_root: Path) -> Path:
    return Path(plot_root) / ARTIFACT_LEDGER_FILENAME


def load_inventory_payload(plot_root: Path) -> tuple[dict[str, object], InventoryPayloadSource]:
    plot_root = Path(plot_root)
    for source_name, candidate in (
        ("current_inventory", current_inventory_path(plot_root)),
        ("plot_manifest", plot_manifest_path(plot_root)),
    ):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return {}, "invalid"
        if isinstance(payload, dict):
            return payload, source_name
        return {}, "invalid"
    return {}, "missing"


def _entry_plot_id(entry: Mapping[str, object]) -> str:
    return str(entry.get("plot_id") or entry.get("visual_plot_type") or entry.get("name") or "").strip()


def _contains_legacy_plot_id(token: str) -> bool:
    base = base_plot_id(token)
    return bool(token in LEGACY_PUBLIC_PLOT_IDS or base in LEGACY_PUBLIC_PLOT_IDS)


def load_current_inventory_strict(
    plot_root: Path,
    *,
    required_plot_ids: Iterable[str] | None = None,
    config_path: Path | str | None = None,
    root_cfg: object | None = None,
) -> dict[str, object]:
    inventory_path = current_inventory_path(plot_root)
    if not inventory_path.exists():
        raise ValueError("missing required notebook plot ids: current_inventory.json is missing")
    try:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("current_inventory.json is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("current_inventory.json is invalid")
    if str(payload.get("schema_version") or "").strip() != CURRENT_INVENTORY_SCHEMA_VERSION:
        raise ValueError("legacy/unsupported inventory taxonomy detected")
    plot_entries = payload.get("plots", [])
    if not isinstance(plot_entries, list):
        raise ValueError("current_inventory.json is invalid")
    if any(_contains_legacy_plot_id(_entry_plot_id(entry)) for entry in plot_entries if isinstance(entry, Mapping)):
        raise ValueError("legacy/unsupported inventory taxonomy detected")
    required_ids = ordered_unique(required_plot_ids) if required_plot_ids is not None else required_notebook_plot_ids()
    scope_keys = [_entry_scope_key(entry, plot_root=plot_root) for entry in plot_entries if isinstance(entry, Mapping)]
    generated_ids = {base_plot_id(plot_id) for plot_id, _scope_name in scope_keys if plot_id}
    missing_ids = _missing_required_inventory_expectations(
        scope_keys,
        required_plot_ids=required_ids,
        generated_ids=generated_ids,
        config_path=config_path,
        root_cfg=root_cfg,
    )
    if missing_ids:
        raise ValueError("missing required notebook plot ids: " + ", ".join(missing_ids))
    return payload


def _entry_scope_key(entry: Mapping[str, object], *, plot_root: Path) -> tuple[str, str | None]:
    rel_path = str(entry.get("path") or "").strip()
    plot_path = (plot_root / rel_path) if rel_path else plot_root
    record = resolve_plot_record(
        plot_root=plot_root,
        plot_path=plot_path,
        manifest_entry=entry,
        source_rank=0,
    )
    plot_id = base_plot_id(str(record.get("plot_id") or record.get("visual_plot_type") or _entry_plot_id(entry)))
    plan_name = str(record.get("plan_name") or "").strip()
    if plot_id not in _STAGE_B_SCOPED_PLOT_IDS:
        return plot_id, None
    return plot_id, (plan_name or "unscoped")


def _load_root_cfg_for_inventory_validation(*, config_path: Path | str | None, root_cfg: object | None):
    if root_cfg is not None:
        return root_cfg
    if config_path is None:
        return None
    from ..config import load_config

    return load_config(Path(config_path)).root


def _expected_stage_b_scope_names(*, plot_id: str, root_cfg) -> list[str]:
    plots_cfg = getattr(root_cfg, "plots", None)
    raw_options = dict((getattr(plots_cfg, "options", {}) or {}).get(plot_id) or {})
    scope = str(raw_options.get("scope", "auto")).strip() or "auto"
    try:
        max_plans = int(raw_options.get("max_plans", 12))
    except Exception:
        max_plans = 12
    plan_names = ordered_unique(
        str(getattr(plan, "name", "")).strip()
        for plan in list(getattr(getattr(root_cfg.densegen, "generation", None), "plan", []) or [])
    )
    plan_names = [name for name in plan_names if name]
    if not plan_names:
        return []
    if scope == "per_plan":
        return plan_names
    from .plot_common import plan_group_from_name

    grouped_names = ordered_unique(plan_group_from_name(name) for name in plan_names if str(name).strip())
    if scope == "per_group":
        return grouped_names
    if len(plan_names) <= max_plans:
        return plan_names
    if len(grouped_names) < len(plan_names):
        return grouped_names
    return plan_names


def _missing_required_inventory_expectations(
    scope_keys: Iterable[tuple[str, str | None]],
    *,
    required_plot_ids: list[str],
    generated_ids: set[str],
    config_path: Path | str | None,
    root_cfg: object | None,
) -> list[str]:
    generated_scope_keys = {key for key in scope_keys if key[0]}
    loaded_root_cfg = _load_root_cfg_for_inventory_validation(config_path=config_path, root_cfg=root_cfg)
    missing: list[str] = []
    for plot_id in required_plot_ids:
        if plot_id in _STAGE_B_SCOPED_PLOT_IDS and loaded_root_cfg is not None:
            expected_scopes = _expected_stage_b_scope_names(plot_id=plot_id, root_cfg=loaded_root_cfg)
            if expected_scopes:
                for scope_name in expected_scopes:
                    if (plot_id, scope_name) not in generated_scope_keys:
                        missing.append(f"{plot_id}[{scope_name}]")
                continue
        if plot_id not in generated_ids:
            missing.append(plot_id)
    return missing


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


def _humanize_token(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    parts = [part for part in token.replace("/", "_").replace("-", "_").split("_") if part]
    words: list[str] = []
    for part in parts:
        lowered = part.lower()
        if lowered == "tfbs":
            words.append("TFBS")
        elif lowered == "dna":
            words.append("DNA")
        elif lowered == "usr":
            words.append("USR")
        elif lowered and len(part) <= 4 and part.isupper():
            words.append(part)
        else:
            words.append(part[:1].upper() + part[1:])
    return " ".join(words)


def compact_plan_label(plan_name: str) -> str:
    token = str(plan_name or "").strip()
    if not token or token == "unscoped":
        return "Run-level"
    if token == "stage_a":
        return "Stage A"
    if token == "all_plans":
        return "All plans"
    parts = [part for part in token.split("__") if part]
    base_token = str(parts[0] if parts else token).strip().replace("__", "-")
    base_label = _PLAN_BASE_LABELS.get(base_token.lower(), _humanize_token(base_token) or base_token)
    variant_tokens: list[str] = []
    for raw_token in parts[1:]:
        raw_token = str(raw_token).strip()
        if not raw_token:
            continue
        if "=" in raw_token:
            key, value = raw_token.split("=", 1)
        elif "_" in raw_token:
            key, value = raw_token.split("_", 1)
        else:
            key, value = raw_token, ""
        key = str(key).strip()
        value = str(value).strip()
        if key and value:
            key_label = _PLAN_VARIANT_LABELS.get(key.lower(), _humanize_token(key) or key)
            variant_tokens.append(f"{key_label} {value}")
    if not variant_tokens:
        return base_label
    return f"{base_label} [{' | '.join(variant_tokens)}]"


def _plot_label(plot_id: str) -> str:
    label = str(plot_spec(plot_id).get("label") or "").strip()
    if label:
        return label
    return _humanize_token(base_plot_id(plot_id))


def plot_title(plot_id: str, *, variant: str = "") -> str:
    del variant
    return _plot_label(plot_id)


def plot_description(plot_id: str, *, variant: str = "") -> str:
    del variant
    return str(plot_spec(plot_id).get("description") or "").strip()


def _plot_scope_note(
    plot_id: str,
    *,
    plan_name: str = "",
    input_name: str = "",
) -> str:
    notes: list[str] = []
    plan_token = str(plan_name or "").strip()
    input_token = str(input_name or "").strip()
    if plot_id in {"placement_occupancy_map", "tfbs_concentration_profile"} and plan_token not in {"", "unscoped"}:
        notes.append(f"This view is scoped to the {compact_plan_label(plan_token)} plan")
    elif plot_id == "dense_array_showcase_video" and plan_token == "all_plans":
        notes.append("The sequence examples are drawn across all plans")
    if input_token:
        if notes:
            notes[-1] += f" and references the {_humanize_token(input_token)} input"
        else:
            notes.append(f"This view references the {_humanize_token(input_token)} input")
    if not notes:
        return ""
    return ". ".join(notes).rstrip(".") + "."


def _plot_supporting_caption_base(plot_id: str) -> str:
    custom_captions = {
        "retained_pool_coverage_by_regulator": (
            "Compare how many unique Stage-A TFBS were retained for each regulator "
            "with how many of those retained motifs were actually deployed into "
            "accepted DenseGen arrays"
        ),
        "retained_vs_deployed_length_mix_by_regulator": (
            "Compare each regulator's retained Stage-A TFBS length mix with the "
            "TFBS lengths that were actually deployed into accepted arrays"
        ),
        "retained_vs_deployed_tier_mix_by_regulator": (
            "Compare each regulator's retained Stage-A score-tier mix with the "
            "mapped tier mix that was actually deployed into accepted arrays"
        ),
        "score_strata_and_deployed_length_bridge": (
            "Bridge Stage A score support to Stage B deployment by showing where "
            "the minimum retained score sits inside each regulator's eligible "
            "score distribution and how TFBS counts in DenseGen arrays shift "
            "across lengths"
        ),
        "solve_pressure_and_progress": (
            "The left panel counts failed-solve pressure by reason family, and "
            "the right panel shows accepted progress by plan"
        ),
        "source_cohort_concentration": (
            "Break DenseGen arrays down by source-derived part composition so "
            "the dominant composition cohorts are visible"
        ),
        "source_plan_input_heatmap": (
            "The left heatmap counts final DenseGen records by source cohort and "
            "DenseGen plan, and the right heatmap counts the same source cohorts "
            "against DenseGen input names so both panels can be compared against "
            "the same source ordering"
        ),
        "upstream_motif_supply_and_pwm_strength": (
            "The left panel compares source hits, eligible unique motifs, and "
            "retained Stage-A motifs per regulator, and the right panel shows "
            "each regulator's PWM consensus score as a fraction of that PWM's "
            "theoretical maximum"
        ),
    }
    if plot_id in custom_captions:
        return custom_captions[plot_id]
    return plot_description(plot_id) or plot_title(plot_id)


def describe_visual_plot_type(plot_type: str) -> str:
    token = str(plot_type or "").strip()
    if not token:
        return ""
    return plot_title(token)


def plot_supporting_caption(
    plot_id: str,
    *,
    variant: str = "",
    plan_name: str = "",
    input_name: str = "",
) -> str:
    del variant
    caption = _plot_supporting_caption_base(plot_id).rstrip(".")
    scope_note = _plot_scope_note(plot_id, plan_name=plan_name, input_name=input_name)
    if scope_note:
        caption += " " + scope_note.rstrip(".")
    return caption.rstrip(".") + "."


def plot_media_alt_text(
    plot_id: str,
    *,
    variant: str = "",
    plan_name: str = "",
    input_name: str = "",
) -> str:
    title = plot_title(plot_id, variant=variant)
    scope_note = _plot_scope_note(plot_id, plan_name=plan_name, input_name=input_name)
    detailed_alt = {
        "retained_pool_coverage_by_regulator": (
            "Horizontal bars compare two regulator-level counts. Retained means "
            "the unique Stage-A TFBS sequences that survived sampling and "
            "selection before any Stage-B layout. Unique deployed means the "
            "unique TFBS sequences from that retained pool that actually appear "
            "at least once in accepted DenseGen outputs."
        ),
        "retained_vs_deployed_length_mix_by_regulator": (
            "Paired stacked bars compare regulator-level length shares. "
            "Retained means the Stage-A TFBS pool kept after sampling and "
            "selection. Deployed means TFBS annotations that actually appear in "
            "accepted DenseGen outputs. "
            "Length mix means the within-regulator share at each TFBS length."
        ),
        "retained_vs_deployed_tier_mix_by_regulator": (
            "Paired stacked bars compare regulator-level score-tier shares. "
            "Retained means the Stage-A TFBS pool kept after sampling and "
            "selection before Stage-B placement. Deployed means TFBS "
            "annotations that actually appear in accepted DenseGen outputs. "
            "Tier mix means the within-regulator share at each mapped Stage-A "
            "score tier, with Tier 0 as the highest-scoring band and larger "
            "tier numbers representing lower-score bands."
        ),
        "score_strata_and_deployed_length_bridge": (
            "Two-panel bridge view. Left panel: three stacked score "
            "distributions show the eligible-unique Stage-A PWM matches for "
            "each regulator. The vertical lollipop marks the minimum retained "
            "score for that regulator. The adjacent annotation reports how "
            "many unique TFBS sequences from that regulator were actually "
            "deployed into DenseGen arrays, plus the core average pairwise "
            "Hamming distance across the mapped deployed PWM-core sequences. "
            "Right panel: grouped horizontal bars count unique deployed TFBS "
            "sequences in DenseGen arrays by length for each regulator, plus "
            "background when a control pool is present, with lengths ordered "
            "from longest to shortest."
        ),
        "solve_pressure_and_progress": (
            "Two-panel run diagnostic. The left panel counts failed or rejected "
            "attempts by reason family so the dominant sources of solve "
            "pressure are visible. The right panel shows cumulative accepted "
            "progress for each plan, "
            "normalized to that plan's final accepted total."
        ),
        "source_plan_input_heatmap": (
            "Two aligned provenance heatmaps that share the same source-cohort "
            "rows. The left panel counts final DenseGen records by source cohort "
            "and DenseGen plan. The right panel counts final DenseGen records by "
            "source cohort and DenseGen input name so the two downstream "
            "groupings can be compared against the same cohort ordering."
        ),
        "source_cohort_concentration": (
            "A horizontal bar chart of DenseGen arrays traced back to their "
            "source-derived part-composition cohorts. The bar colors separate "
            "the main plan families, and each annotation gives the DenseGen "
            "array count for one cohort so the dominant composition groups are "
            "visible at a glance."
        ),
        "upstream_motif_supply_and_pwm_strength": (
            "Two-panel Stage-A summary. Left panel: source hits are all mined "
            "candidate windows with at least one motif hit for a regulator; "
            "eligible unique are candidates that still satisfy the Stage-A "
            "eligibility and uniqueness rules after collapse; retained are the "
            "top Stage-A candidates kept in the retained pool before Stage-B "
            "placement. If retained bars look absent, they are usually much "
            "smaller than the source-hit and eligible-unique bars rather than "
            "missing from the data. Right panel: the PWM confidence proxy is "
            "the PWM consensus score divided by that PWM's theoretical maximum "
            "score for the regulator. It is computed from the source PWM "
            "during Stage-A sampling, not averaged over deployed DenseGen "
            "arrays."
        ),
    }.get(plot_id)
    if detailed_alt:
        if scope_note:
            return f"{title}. {detailed_alt} {scope_note}".strip()
        return f"{title}. {detailed_alt}".strip()
    caption = plot_supporting_caption(
        plot_id,
        variant=variant,
        plan_name=plan_name,
        input_name=input_name,
    )
    return f"{title}. {caption}".strip()


def build_plot_text_contract(
    plot_id: str,
    *,
    variant: str = "",
    plan_name: str = "",
    input_name: str = "",
) -> dict[str, str]:
    title = plot_title(plot_id, variant=variant)
    description = plot_description(plot_id, variant=variant)
    caption = plot_supporting_caption(
        plot_id,
        variant=variant,
        plan_name=plan_name,
        input_name=input_name,
    )
    alt_text = plot_media_alt_text(
        plot_id,
        variant=variant,
        plan_name=plan_name,
        input_name=input_name,
    )
    return {
        "title": title,
        "description": description,
        "caption": caption,
        "alt_text": alt_text,
    }


def stage_b_scope_seed_plot_ids() -> list[str]:
    return [str(plot_id) for plot_id, spec in PLOT_SPECS.items() if bool(spec.get("seed_stage_b_scope_when_missing"))]


def infer_plot_id_from_path(relative_parts: Sequence[str], stem: str) -> str:
    if not relative_parts:
        return ""
    head = str(relative_parts[0]).strip().lower()
    normalized_stem = str(stem or "").strip().lower()
    if head == "stage_a":
        if normalized_stem.endswith("__background_sequence_logo"):
            return "background_sequence_logo"
        if normalized_stem in _STAGE_A_PLOT_IDS:
            return normalized_stem
        return ""
    if head == "stage_b":
        if normalized_stem in _STAGE_B_SCOPED_PLOT_IDS:
            return normalized_stem
        if normalized_stem == "showcase" or normalized_stem.endswith(".mp4"):
            return "dense_array_showcase_video"
        return "dense_array_showcase_video" if Path(stem).suffix.lower() == ".mp4" else ""
    if head == "stage_b_summary" and normalized_stem in _STAGE_B_SUMMARY_PLOT_IDS:
        return normalized_stem
    if head == "run_health" and normalized_stem in _RUN_PLOT_IDS:
        return normalized_stem
    if head == "dataset" and normalized_stem in _DATASET_PLOT_IDS:
        return normalized_stem
    return ""


def build_visual_plot_type(plot_id: str, *, plot_name: str, variant: str, stem: str) -> str:
    del plot_name, variant, stem
    base = str(plot_id or "").strip()
    return base


def manifest_path_fields(name: str, rel_path: Path) -> dict[str, str]:
    fields: dict[str, str] = {"plot_id": str(name)}
    parts = rel_path.parts
    stem = rel_path.stem
    if name == "dense_array_showcase_video":
        fields["group"] = "stage_b"
        fields["family"] = "showcase"
        fields["plan_name"] = parts[1] if len(parts) >= 2 else "all_plans"
        fields["variant"] = "showcase"
        return fields
    if name in _STAGE_A_PLOT_IDS:
        fields["group"] = "stage_a"
        fields["plan_name"] = "stage_a"
        if name in {"stage_a_pool_diversity", "stage_a_sampling_yield"}:
            fields["family"] = "stage_a_health"
        else:
            fields["family"] = "stage_a_context"
        if name == "background_sequence_logo" and stem.endswith("__background_sequence_logo"):
            fields["input_name"] = stem[: -len("__background_sequence_logo")]
        elif name == "background_sequence_logo":
            fields["input_name"] = "background"
        fields["variant"] = name
        return fields
    if name in _STAGE_B_SCOPED_PLOT_IDS:
        fields["group"] = "stage_b"
        fields["family"] = "stage_b_deployment"
        if len(parts) >= 3:
            fields["plan_name"] = parts[1]
        if len(parts) >= 4:
            fields["input_name"] = parts[2]
        fields["variant"] = name
        return fields
    if name in _STAGE_B_SUMMARY_PLOT_IDS:
        fields["group"] = "stage_b_summary"
        fields["plan_name"] = "unscoped"
        if name == "plan_regulator_deployment_heatmap":
            fields["family"] = "stage_b_deployment"
        elif name == "upstream_motif_supply_and_pwm_strength":
            fields["family"] = "stage_a_context"
        else:
            fields["family"] = "stage_b_bridge"
        fields["variant"] = name
        return fields
    if name in _RUN_PLOT_IDS:
        fields["group"] = "run"
        fields["family"] = "run_diagnostics"
        fields["plan_name"] = "unscoped"
        fields["variant"] = name
        return fields
    if name in _DATASET_PLOT_IDS:
        fields["group"] = "dataset"
        fields["family"] = "provenance"
        fields["plan_name"] = "unscoped"
        fields["variant"] = name
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
    plot_name = str(entry.get("name") or plot_id or resolved_path.stem)
    variant = str(entry.get("variant") or path_fields.get("variant") or plot_id or resolved_path.stem or "")
    group = str(entry.get("group") or path_fields.get("group") or "").strip()
    family = str(entry.get("family") or path_fields.get("family") or "").strip()
    text_contract = build_plot_text_contract(
        plot_id,
        variant=variant,
        plan_name=plan_name,
        input_name=input_name,
    )
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
        "title": str(entry.get("title") or text_contract["title"]),
        "description": str(entry.get("description") or text_contract["description"]),
        "caption": str(entry.get("caption") or text_contract["caption"]),
        "alt_text": str(entry.get("alt_text") or text_contract["alt_text"]),
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
            *(str(name).strip() for name in known_plot_ids if str(name).strip() not in HIDDEN_VISUAL_PLOT_TYPES),
            *generated_all,
        ]
    )

    scope_names = ordered_unique(
        [
            *(str(entry.get("plan_name") or "").strip() for entry in entries),
            *(str(name).strip() for name in stage_b_scope_names),
        ]
    )
    stage_b_scope_set = {name for name in scope_names if name not in {"stage_a", "unscoped", "all_plans"}}
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

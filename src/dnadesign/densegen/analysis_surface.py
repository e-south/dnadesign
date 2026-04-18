"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/analysis_surface.py

Public DenseGen analysis-surface contract.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typing_extensions import Literal

from .src.config import load_config, resolve_outputs_scoped_path, resolve_run_root, resolve_usr_root_scoped_path
from .src.viz.plot_inventory import (
    ANALYSIS_SURFACE_CONTRACT_VERSION,
    HIDDEN_VISUAL_PLOT_TYPES,
    artifact_ledger_path,
    current_inventory_path,
    load_current_inventory_strict,
    plot_manifest_path,
    resolve_plot_record,
)
from .src.viz.plot_registry import PLOT_SPECS

ArtifactState = Literal["current", "stale", "missing", "partial", "degraded", "historical_only"]
SourcePolicy = Literal["workspace", "usr", "auto"]
ArtifactType = Literal["pdf", "png", "svg", "html", "mp4", "json", "tsv"]
ProvenanceSource = Literal["manifest", "filesystem", "generated"]
DiagnosticSeverity = Literal["info", "warning", "error"]

_DEFAULT_NOTEBOOK_FILENAME = "densegen_run_overview.py"
_INTERNAL_HIDDEN_PLOT_IDS = frozenset()
_OPERATOR_VISIBLE_PLOT_IDS = (
    "source_cohort_concentration",
    "stage_a_sampling_yield",
    "stage_a_pool_diversity",
    "plan_regulator_deployment_heatmap",
    "placement_occupancy_map",
    "retained_pool_coverage_by_regulator",
    "attempt_outcome_timeline",
    "solve_pressure_and_progress",
)
_OPTIONAL_PLOT_IDS = frozenset(
    {
        "source_plan_input_heatmap",
        "background_sequence_logo",
        "stage_a_pool_score_strata",
        "score_strata_and_deployed_length_bridge",
        "tfbs_concentration_profile",
        "retained_vs_deployed_tier_mix_by_regulator",
        "retained_vs_deployed_length_mix_by_regulator",
        "upstream_motif_supply_and_pwm_strength",
        "compression_ratio_by_plan",
        "dense_array_showcase_video",
    }
)


def operator_visible_surface_plot_ids() -> list[str]:
    return list(_OPERATOR_VISIBLE_PLOT_IDS)


def optional_surface_plot_ids() -> list[str]:
    return sorted(_OPTIONAL_PLOT_IDS)


@dataclass(frozen=True)
class RuntimeSummary:
    run_id: str
    workspace_name: str
    sink_topology: list[str]
    record_source: str
    overlay_aware: bool
    dataset_row_count: int | None
    expanded_plan_count: int | None


@dataclass(frozen=True)
class PlotTaxonomyEntry:
    plot_id: str
    family: str
    generated_by_default: bool
    operator_visible_by_default: bool
    notebook_visible_by_default: bool
    optional: bool
    internal_hidden: bool
    semantic_contract_version: str
    required_inputs: list[str]
    degraded_modes: list[str]
    research_question: str | None


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    plot_id: str
    variant: str | None
    relative_path: str
    artifact_type: ArtifactType
    materialized: bool
    current: bool
    visible: bool
    stale: bool
    state: ArtifactState
    provenance_source: ProvenanceSource
    generated_at: datetime | None
    render_inputs_mtime: datetime | None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NotebookSurface:
    notebook_path: str | None
    fresh: bool | None
    gallery_visible_artifact_ids: list[str]
    hidden_artifact_ids: list[str]
    preview_cache_status: str


@dataclass(frozen=True)
class FreshnessSummary:
    current_inventory_exact: bool
    inventory_source: str
    notebook_fresh: bool | None
    manifest_freshness: ArtifactState
    stale_but_present_artifacts: list[str]


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: DiagnosticSeverity
    message: str
    blocking: bool


@dataclass(frozen=True)
class DenseGenAnalysisSurface:
    workspace_id: str
    workspace_root: str
    source_dataset_id: str | None
    source_policy: SourcePolicy
    contract_version: str
    generated_at: datetime
    runtime_summary: RuntimeSummary
    taxonomy: list[PlotTaxonomyEntry]
    current_inventory: list[ArtifactRecord]
    notebook: NotebookSurface
    freshness: FreshnessSummary
    diagnostics: list[Diagnostic]
    generated_surface: list[str]
    operator_visible_surface: list[str]
    optional_surface: list[str]
    internal_or_hidden_surface: list[str]
    historical_ledger_surface: list[str]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


def inspect_analysis_surface(config_path: Path | str) -> DenseGenAnalysisSurface:
    loaded = load_config(Path(config_path))
    root_cfg = loaded.root
    run_root = resolve_run_root(loaded.path, root_cfg.densegen.run.root)
    plots_cfg = root_cfg.plots
    plots_out_dir = plots_cfg.out_dir if plots_cfg is not None else "outputs/plots"
    plot_root = resolve_outputs_scoped_path(loaded.path, run_root, plots_out_dir, label="plots.out_dir")
    inventory_payload: dict[str, object] = {}
    inventory_source: str = "missing"
    diagnostics: list[Diagnostic] = []
    try:
        inventory_payload = load_current_inventory_strict(plot_root, required_plot_ids=())
        inventory_source = "current_inventory"
    except ValueError as exc:
        message = str(exc).strip()
        if "current_inventory.json is missing" in message:
            inventory_source = "missing"
        else:
            inventory_source = "invalid"
    if inventory_source == "missing":
        diagnostics.append(
            Diagnostic(
                code="current_inventory_missing",
                severity="warning",
                message="DenseGen current inventory is missing; analysis surface has no materialized plot entries.",
                blocking=False,
            )
        )
    elif inventory_source == "invalid":
        diagnostics.append(
            Diagnostic(
                code="inventory_invalid",
                severity="error",
                message="DenseGen plot inventory exists but could not be parsed.",
                blocking=True,
            )
        )

    actual_record_source = _actual_record_source(root_cfg)
    source_policy = _source_policy(root_cfg)
    taxonomy = _build_taxonomy(root_cfg)
    taxonomy_by_plot = {entry.plot_id: entry for entry in taxonomy}
    current_inventory = _build_current_inventory(
        config_path=loaded.path,
        plot_root=plot_root,
        payload=inventory_payload,
        inventory_source=inventory_source,
        run_root=run_root,
        root_cfg=root_cfg,
        taxonomy_by_plot=taxonomy_by_plot,
    )
    notebook_surface = _build_notebook_surface(
        plot_root=plot_root,
        run_root=run_root,
        current_inventory=current_inventory,
    )
    freshness = _build_freshness_summary(
        plot_root=plot_root,
        inventory_source=inventory_source,
        current_inventory=current_inventory,
        notebook_surface=notebook_surface,
    )
    runtime_summary = RuntimeSummary(
        run_id=str(root_cfg.densegen.run.id),
        workspace_name=str(loaded.path.parent.name),
        sink_topology=[str(item) for item in root_cfg.densegen.output.targets],
        record_source=("usr(include_overlays)" if actual_record_source == "usr" else "workspace(parquet)"),
        overlay_aware=bool(actual_record_source == "usr"),
        dataset_row_count=_dataset_row_count(loaded.path, root_cfg, run_root, actual_record_source),
        expanded_plan_count=len(list(root_cfg.densegen.generation.plan or [])),
    )
    generated_surface = [entry.plot_id for entry in taxonomy if entry.generated_by_default]
    operator_visible_surface = [entry.plot_id for entry in taxonomy if entry.operator_visible_by_default]
    optional_surface = [entry.plot_id for entry in taxonomy if entry.optional]
    internal_or_hidden_surface = [entry.plot_id for entry in taxonomy if entry.internal_hidden]
    generated_at = _parse_datetime(inventory_payload.get("generated_at") or inventory_payload.get("updated_at"))
    if generated_at is None:
        generated_at = datetime.now(timezone.utc)

    return DenseGenAnalysisSurface(
        workspace_id=str(root_cfg.densegen.run.id),
        workspace_root=str(run_root),
        source_dataset_id=(
            str(root_cfg.densegen.output.usr.dataset) if root_cfg.densegen.output.usr is not None else None
        ),
        source_policy=source_policy,
        contract_version=ANALYSIS_SURFACE_CONTRACT_VERSION,
        generated_at=generated_at,
        runtime_summary=runtime_summary,
        taxonomy=taxonomy,
        current_inventory=current_inventory,
        notebook=notebook_surface,
        freshness=freshness,
        diagnostics=diagnostics,
        generated_surface=generated_surface,
        operator_visible_surface=operator_visible_surface,
        optional_surface=optional_surface,
        internal_or_hidden_surface=internal_or_hidden_surface,
        historical_ledger_surface=_build_historical_ledger_surface(
            plot_root=plot_root,
            current_inventory=current_inventory,
        ),
    )


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def _actual_record_source(root_cfg) -> str:
    targets = [str(item) for item in root_cfg.densegen.output.targets]
    if len(targets) > 1:
        plots_cfg = root_cfg.plots
        if plots_cfg is None or plots_cfg.source is None:
            raise ValueError("plots.source must be set when output.targets has multiple sinks")
        return str(plots_cfg.source)
    return str(targets[0])


def _source_policy(root_cfg) -> SourcePolicy:
    plots_cfg = root_cfg.plots
    if len([str(item) for item in root_cfg.densegen.output.targets]) > 1 and (
        plots_cfg is None or plots_cfg.source is None
    ):
        raise ValueError("plots.source must be set when output.targets has multiple sinks")
    if plots_cfg is None or plots_cfg.source is None:
        return "auto"
    if str(plots_cfg.source) == "usr":
        return "usr"
    return "workspace"


def _resolved_default_plot_ids(root_cfg) -> list[str]:
    plots_cfg = root_cfg.plots
    return [str(item) for item in (plots_cfg.default if plots_cfg is not None else [])]


def _family_for_plot(plot_id: str) -> str:
    if plot_id in {"source_cohort_concentration", "source_plan_input_heatmap"}:
        return "provenance"
    if plot_id in {"stage_a_sampling_yield", "stage_a_pool_diversity"}:
        return "stage_a_health"
    if plot_id in {"background_sequence_logo", "stage_a_pool_score_strata"}:
        return "stage_a_context"
    if plot_id in {"plan_regulator_deployment_heatmap", "placement_occupancy_map", "tfbs_concentration_profile"}:
        return "stage_b_deployment"
    if plot_id in {
        "score_strata_and_deployed_length_bridge",
        "retained_pool_coverage_by_regulator",
        "retained_vs_deployed_length_mix_by_regulator",
        "retained_vs_deployed_tier_mix_by_regulator",
    }:
        return "stage_b_bridge"
    if plot_id in {"attempt_outcome_timeline", "solve_pressure_and_progress", "compression_ratio_by_plan"}:
        return "run_diagnostics"
    return "showcase"


def _research_question_for_plot(plot_id: str) -> str | None:
    return {
        "attempt_outcome_timeline": (
            "How did accepted, rejected, duplicate, and failed solve outcomes accumulate over time?"
        ),
        "background_sequence_logo": "What fixed-sequence background context was available during Stage A sampling?",
        "compression_ratio_by_plan": "How does compression ratio vary across plans?",
        "dense_array_showcase_video": "What does a read-only showcase of accepted DenseGen arrays look like?",
        "plan_regulator_deployment_heatmap": "How is deployed TFBS usage distributed across plans and regulators?",
        "placement_occupancy_map": "Where do accepted Stage B placements land across the promoter array?",
        "retained_pool_coverage_by_regulator": (
            "How much of each regulator's retained Stage A pool is actually covered by uniquely deployed TFBS?"
        ),
        "score_strata_and_deployed_length_bridge": (
            "Where does the deployed subset sit inside each regulator's eligible and retained Stage A score "
            "distribution, and how does that deployed subset collapse by TFBS length?"
        ),
        "retained_vs_deployed_length_mix_by_regulator": (
            "How do retained Stage A TFBS length distributions shift when motifs "
            "are actually deployed in accepted arrays?"
        ),
        "retained_vs_deployed_tier_mix_by_regulator": (
            "Does Stage B preferentially deploy specific Stage A selection tiers?"
        ),
        "solve_pressure_and_progress": (
            "How did solver pressure and accepted-progress trajectories evolve across plans?"
        ),
        "source_cohort_concentration": "How is the shared DenseGen source dataset distributed across source cohorts?",
        "source_plan_input_heatmap": "What source-to-plan and source-to-input provenance is present in the dataset?",
        "stage_a_pool_diversity": "How diverse are the retained Stage A pools across regulators?",
        "stage_a_pool_score_strata": "How are Stage A retained pools distributed across score strata and tiers?",
        "stage_a_sampling_yield": "How much Stage A yield was retained, and where did sampling bias appear?",
        "tfbs_concentration_profile": "How concentrated is deployed TFBS usage across ranked motif placements?",
        "upstream_motif_supply_and_pwm_strength": (
            "How asymmetric are the upstream source-hit, eligible-unique, retained-pool, and PWM-strength signals?"
        ),
    }.get(plot_id)


def _build_taxonomy(root_cfg) -> list[PlotTaxonomyEntry]:
    default_plot_ids = set(_resolved_default_plot_ids(root_cfg))
    entries: list[PlotTaxonomyEntry] = []
    for plot_id, spec in PLOT_SPECS.items():
        internal_hidden = plot_id in _INTERNAL_HIDDEN_PLOT_IDS
        optional = plot_id in _OPTIONAL_PLOT_IDS
        generated_by_default = plot_id in default_plot_ids
        operator_visible = plot_id in _OPERATOR_VISIBLE_PLOT_IDS
        notebook_visible = plot_id in set(_OPERATOR_VISIBLE_PLOT_IDS) | _OPTIONAL_PLOT_IDS
        missing_state = str(spec.get("missing_state") or "").strip()
        degraded_modes = [missing_state] if missing_state else []
        entries.append(
            PlotTaxonomyEntry(
                plot_id=str(plot_id),
                family=_family_for_plot(str(plot_id)),
                generated_by_default=generated_by_default,
                operator_visible_by_default=operator_visible,
                notebook_visible_by_default=notebook_visible,
                optional=optional,
                internal_hidden=internal_hidden,
                semantic_contract_version=ANALYSIS_SURFACE_CONTRACT_VERSION,
                required_inputs=[str(item) for item in list(spec.get("requires") or [])],
                degraded_modes=degraded_modes,
                research_question=_research_question_for_plot(str(plot_id)),
            )
        )
    return entries


def _build_current_inventory(
    *,
    config_path: Path,
    plot_root: Path,
    payload: dict[str, object],
    inventory_source: str,
    run_root: Path,
    root_cfg,
    taxonomy_by_plot: dict[str, PlotTaxonomyEntry],
) -> list[ArtifactRecord]:
    entries: list[ArtifactRecord] = []
    source_rank = 0 if inventory_source == "current_inventory" else 1
    for entry in list(payload.get("plots") or []):
        rel_path = str(entry.get("path") or "").strip()
        if not rel_path:
            continue
        candidate = plot_root / rel_path
        record = resolve_plot_record(
            plot_root=plot_root,
            plot_path=candidate,
            manifest_entry=entry,
            source_rank=source_rank,
        )
        plot_id = str(record.get("plot_id") or "").strip()
        visual_plot_type = str(record.get("visual_plot_type") or "").strip()
        taxonomy = taxonomy_by_plot.get(plot_id)
        visible = bool(
            taxonomy is not None
            and taxonomy.notebook_visible_by_default
            and visual_plot_type not in HIDDEN_VISUAL_PLOT_TYPES
        )
        materialized = candidate.exists()
        render_inputs_mtime = _render_inputs_mtime(
            config_path=config_path,
            run_root=run_root,
            root_cfg=root_cfg,
            plot_id=plot_id,
        )
        generated_at = _parse_datetime(entry.get("generated_at")) or _datetime_from_path(candidate)
        stale = bool(
            materialized
            and render_inputs_mtime is not None
            and _datetime_from_path(candidate) is not None
            and _datetime_from_path(candidate) < render_inputs_mtime
        )
        if not materialized:
            state: ArtifactState = "missing"
        elif stale:
            state = "stale"
        else:
            state = "current"
        variant = str(record.get("variant") or "").strip() or None
        artifact_id = plot_id if not variant or variant == plot_id else f"{plot_id}:{variant}"
        entries.append(
            ArtifactRecord(
                artifact_id=artifact_id,
                plot_id=plot_id,
                variant=variant,
                relative_path=rel_path,
                artifact_type=_artifact_type_from_path(candidate),
                materialized=materialized,
                current=bool(state == "current"),
                visible=visible,
                stale=stale,
                state=state,
                provenance_source="manifest",
                generated_at=generated_at,
                render_inputs_mtime=render_inputs_mtime,
                notes=[],
            )
        )
    return sorted(entries, key=lambda item: (item.plot_id, item.relative_path))


def _build_notebook_surface(
    *,
    plot_root: Path,
    run_root: Path,
    current_inventory: list[ArtifactRecord],
) -> NotebookSurface:
    notebook_path = run_root / "outputs" / "notebooks" / _DEFAULT_NOTEBOOK_FILENAME
    notebook_dt = _datetime_from_path(notebook_path)
    inventory_dt = _active_inventory_datetime(plot_root)
    gallery_visible = _unique_in_order([entry.plot_id for entry in current_inventory if entry.visible])
    hidden_ids = _unique_in_order([entry.plot_id for entry in current_inventory if not entry.visible])
    fresh = None
    if notebook_path.exists():
        fresh = bool(inventory_dt is None or (notebook_dt is not None and notebook_dt >= inventory_dt))
    preview_cache_status = (
        "available" if (run_root / "outputs" / "notebooks" / ".baserender_preview_cache").exists() else "missing"
    )
    return NotebookSurface(
        notebook_path=(str(notebook_path) if notebook_path.exists() else None),
        fresh=fresh,
        gallery_visible_artifact_ids=gallery_visible,
        hidden_artifact_ids=hidden_ids,
        preview_cache_status=preview_cache_status,
    )


def _active_inventory_datetime(plot_root: Path) -> datetime | None:
    for candidate in (
        current_inventory_path(plot_root),
        plot_manifest_path(plot_root),
        artifact_ledger_path(plot_root),
    ):
        inventory_dt = _datetime_from_path(candidate)
        if inventory_dt is not None:
            return inventory_dt
    return None


def _build_freshness_summary(
    *,
    plot_root: Path,
    inventory_source: str,
    current_inventory: list[ArtifactRecord],
    notebook_surface: NotebookSurface,
) -> FreshnessSummary:
    inventory_path = current_inventory_path(plot_root)
    has_historical_surface = any(
        candidate.exists()
        for candidate in (
            plot_manifest_path(plot_root),
            artifact_ledger_path(plot_root),
        )
    )
    manifest_freshness: ArtifactState
    if inventory_source == "missing":
        manifest_freshness = "historical_only" if has_historical_surface else "missing"
    elif inventory_source == "invalid":
        manifest_freshness = "degraded"
    elif inventory_path.exists():
        manifest_freshness = "current"
    else:
        manifest_freshness = "historical_only"
    stale_ids = [entry.artifact_id for entry in current_inventory if entry.stale and entry.materialized]
    return FreshnessSummary(
        current_inventory_exact=bool(inventory_source == "current_inventory"),
        inventory_source=str(inventory_source),
        notebook_fresh=notebook_surface.fresh,
        manifest_freshness=manifest_freshness,
        stale_but_present_artifacts=stale_ids,
    )


def _build_historical_ledger_surface(
    *,
    plot_root: Path,
    current_inventory: list[ArtifactRecord],
) -> list[str]:
    ledger_path = artifact_ledger_path(plot_root)
    if not ledger_path.exists():
        return []
    try:
        payload = _load_dict_payload(ledger_path)
    except ValueError:
        return []
    current_plot_ids = {record.plot_id for record in current_inventory}
    historical_plot_ids: list[str] = []
    seen: set[str] = set()
    for entry in list(payload.get("plots") or []):
        plot_id = str(entry.get("plot_id") or entry.get("name") or "").strip()
        if not plot_id:
            rel_path = str(entry.get("path") or "").strip()
            if not rel_path:
                continue
            record = resolve_plot_record(
                plot_root=plot_root,
                plot_path=plot_root / rel_path,
                manifest_entry=entry,
                source_rank=1,
            )
            plot_id = str(record.get("plot_id") or "").strip()
        if not plot_id or plot_id in current_plot_ids or plot_id in seen:
            continue
        seen.add(plot_id)
        historical_plot_ids.append(plot_id)
    return historical_plot_ids


def _load_dict_payload(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON payload: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object payload: {path}")
    return payload


def _dataset_row_count(config_path: Path, root_cfg, run_root: Path, actual_record_source: str) -> int | None:
    if actual_record_source == "usr":
        usr_cfg = root_cfg.densegen.output.usr
        if usr_cfg is None:
            return None
        usr_root = resolve_usr_root_scoped_path(
            config_path,
            usr_cfg.root,
            label="output.usr.root",
            scope=usr_cfg.root_scope,
        )
        try:
            from dnadesign.usr import Dataset
        except Exception:
            return None
        dataset = Dataset(usr_root, usr_cfg.dataset)
        if not dataset.records_path.exists():
            return None
        try:
            stats = dataset.stats()
        except Exception:
            return None
        return int(getattr(stats, "rows", None)) if getattr(stats, "rows", None) is not None else None
    parquet_cfg = root_cfg.densegen.output.parquet
    if parquet_cfg is None:
        return None
    path = resolve_outputs_scoped_path(config_path, run_root, parquet_cfg.path, label="output.parquet.path")
    if path.exists():
        try:
            import pyarrow.parquet as pq
        except Exception:
            return None
        return int(pq.ParquetFile(str(path)).metadata.num_rows)
    return None


def _render_inputs_mtime(*, config_path: Path, run_root: Path, root_cfg, plot_id: str) -> datetime | None:
    candidates: list[Path] = []
    actual_record_source = _actual_record_source(root_cfg)
    if actual_record_source == "usr" and root_cfg.densegen.output.usr is not None:
        usr_cfg = root_cfg.densegen.output.usr
        usr_root = resolve_usr_root_scoped_path(
            config_path,
            usr_cfg.root,
            label="output.usr.root",
            scope=usr_cfg.root_scope,
        )
        candidates.append(Path(usr_root) / str(usr_cfg.dataset) / "records.parquet")
    elif root_cfg.densegen.output.parquet is not None:
        candidates.append(
            resolve_outputs_scoped_path(
                config_path,
                run_root,
                root_cfg.densegen.output.parquet.path,
                label="output.parquet.path",
            )
        )
    if plot_id in {"placement_occupancy_map", "tfbs_concentration_profile"}:
        candidates.append(run_root / "outputs" / "tables" / "composition.parquet")
    if plot_id in {"attempt_outcome_timeline", "compression_ratio_by_plan", "solve_pressure_and_progress"}:
        candidates.append(run_root / "outputs" / "tables" / "attempts.parquet")
        candidates.append(run_root / "outputs" / "meta" / "effective_config.json")
    if plot_id in {
        "background_sequence_logo",
        "stage_a_pool_diversity",
        "stage_a_pool_score_strata",
        "stage_a_sampling_yield",
        "retained_pool_coverage_by_regulator",
        "retained_vs_deployed_length_mix_by_regulator",
        "retained_vs_deployed_tier_mix_by_regulator",
        "upstream_motif_supply_and_pwm_strength",
    }:
        candidates.append(run_root / "outputs" / "pools" / "pool_manifest.json")
    mtimes = [_datetime_from_path(path) for path in candidates if path.exists()]
    mtimes = [value for value in mtimes if value is not None]
    if not mtimes:
        return None
    return max(mtimes)


def _datetime_from_path(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _parse_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _artifact_type_from_path(path: Path) -> ArtifactType:
    suffix = str(path.suffix).lower().lstrip(".")
    if suffix in {"pdf", "png", "svg", "html", "mp4", "json", "tsv"}:
        return suffix  # type: ignore[return-value]
    return "json"


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        token = str(value).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


__all__ = [
    "ArtifactRecord",
    "DenseGenAnalysisSurface",
    "Diagnostic",
    "FreshnessSummary",
    "NotebookSurface",
    "PlotTaxonomyEntry",
    "RuntimeSummary",
    "inspect_analysis_surface",
]

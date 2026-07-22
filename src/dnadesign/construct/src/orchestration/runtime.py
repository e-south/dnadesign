"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/orchestration/runtime.py

Construct runtime orchestration for planning, realization, and run results.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dnadesign.usr import Dataset

from ..contracts.config import (
    JobConfig,
    load_job_config,
)
from ..contracts.errors import ValidationError
from ..persistence.records import (
    BuiltRecord as _BuiltRecord,
)
from ..persistence.records import (
    records_to_write as _records_to_write,
)
from ..persistence.records import (
    require_output_conflict_policy as _require_output_conflict_policy,
)
from ..persistence.records import (
    validate_duplicate_output_aliases as _validate_duplicate_output_aliases,
)
from ..persistence.write_session import (
    ensure_output_dataset as _ensure_output_dataset,
)
from ..persistence.write_session import (
    write_output_records as _write_output_records,
)
from ..persistence.write_session import (
    write_planned_sequence_views as _write_planned_sequence_views,
)
from ..products.classic import (
    build_classic_record as _build_classic_record,
)
from ..products.classic import (
    build_variant_record as _build_variant_record,
)
from ..products.normalize_anchor import (
    build_normalize_anchor_record as _build_normalize_anchor_record,
)
from ..products.specs import (
    build_classic_spec_id as _build_classic_spec_id,
)
from ..products.specs import (
    build_normalize_spec_id as _build_normalize_spec_id,
)
from ..realization.normalize_anchor import (
    require_normalize_target_length_match as _require_normalize_target_length_match,
)
from ..realization.placement import (
    PlannedPlacement,
)
from ..realization.placement import (
    planned_placements as _planned_placements,
)
from ..realization.placement import (
    resolved_placement_sites as _resolved_placement_sites,
)
from ..realization.placement import (
    validate_placements as _validate_placements,
)
from ..sources.input_rows import (
    classic_input_scan_fields as _input_scan_fields,
)
from ..sources.input_rows import (
    normalize_input_scan_fields as _normalize_input_scan_fields,
)
from ..sources.input_rows import (
    require_distinct_input_output_or_opt_in as _require_distinct_input_output_or_opt_in,
)
from ..sources.input_rows import (
    scan_usr_rows as _scan_usr_rows,
)
from ..sources.paths import (
    default_usr_root as _source_default_usr_root,
)
from ..sources.paths import (
    resolve_usr_root as _resolve_usr_root,
)
from ..sources.templates import (
    load_normalize_template as _load_normalize_template,
)
from ..sources.templates import (
    load_template_sequence as _load_template_sequence,
)


@dataclass(frozen=True)
class RunResult:
    job_id: str
    input_dataset: str
    output_dataset: str
    output_root: Path
    records_total: int
    records_written: int
    records_skipped_existing: int
    spec_id: str
    dry_run: bool


@dataclass(frozen=True)
class PlannedRow:
    input_id: str
    output_id: str
    input_length: int
    focal_part_length: int | None
    output_length: int
    full_construct_length: int


@dataclass(frozen=True)
class PreflightResult:
    job_id: str
    input_dataset: str
    output_dataset: str
    input_root: Path
    output_root: Path
    template_id: str
    template_kind: str
    template_source: str
    template_dataset: str | None
    template_field: str | None
    template_record_id: str | None
    template_sha256: str
    template_length: int
    template_circular: bool
    realize_mode: str
    focal_part: str | None
    window_semantics: str | None
    window_reference: str | None
    window_direction: str | None
    window_size_bp: int | None
    window_upstream_bp: int | None
    window_downstream_bp: int | None
    window_offset_bp: int | None
    spec_id: str
    records_total: int
    existing_output_collisions: int
    output_on_conflict: str
    placements: List[PlannedPlacement]
    planned_rows: List[PlannedRow]


@dataclass(frozen=True)
class _PlannedRun:
    cfg: JobConfig
    preflight: PreflightResult
    built: List[_BuiltRecord]


def _default_usr_root() -> Path:
    return _source_default_usr_root()


def _plan_classic_loaded_config(
    cfg: JobConfig,
    *,
    config_path: Path,
    input_root: Path,
    output_root: Path,
) -> tuple[PreflightResult, List[_BuiltRecord]]:
    if cfg.job.template is None or cfg.job.realize is None:
        raise ValidationError("job.template and job.realize are required when job.mode='realize_template'.")
    realize = cfg.job.realize
    base_dir = config_path.parent
    input_ds = Dataset(input_root, cfg.job.input.source.dataset)
    if not input_ds.records_path.exists():
        raise ValidationError(f"Input dataset not initialized: {input_ds.records_path}")
    _require_distinct_input_output_or_opt_in(cfg=cfg, input_root=input_root, output_root=output_root)

    template = _load_template_sequence(base_dir, cfg)
    resolved_sites = _resolved_placement_sites(template, cfg.job.parts)
    ordered_placements = _validate_placements(len(template.sequence), cfg.job.parts, resolved_sites=resolved_sites)
    template_sha256 = hashlib.sha256(template.sequence.encode("utf-8")).hexdigest()
    spec_id = _build_classic_spec_id(
        cfg,
        template=template,
        template_sha256=template_sha256,
        input_root=input_root,
        output_root=output_root,
    )

    rows = _scan_usr_rows(input_ds, columns=_input_scan_fields(input_ds, cfg), ids=cfg.job.input.ids)
    if not rows:
        raise ValidationError("Input selection resolved to zero rows.")

    forward_records = [
        _build_classic_record(
            row=row,
            cfg=cfg,
            template=template,
            template_sha256=template_sha256,
            spec_id=spec_id,
            ordered_placements=ordered_placements,
        )
        for row in rows
    ]
    built: list[_BuiltRecord] = []
    if cfg.job.output_variants:
        for record in forward_records:
            for variant in cfg.job.output_variants:
                built.append(
                    _build_variant_record(
                        forward_record=record,
                        variant=variant,
                        output_dataset_id=cfg.job.output.target.dataset,
                    )
                )
    else:
        built = forward_records

    _validate_duplicate_output_aliases(built)
    collision_count = _require_output_conflict_policy(
        built,
        output_root=output_root,
        output_dataset=cfg.job.output.target.dataset,
        on_conflict=cfg.job.output.on_conflict,
    )
    planned_rows = [
        PlannedRow(
            input_id=str(record.metadata["construct__input_id"]),
            output_id=record.output_id,
            input_length=int(record.metadata["construct__input_length"]),
            focal_part_length=(
                int(record.metadata["construct__focal_part_length"])
                if record.metadata["construct__focal_part_length"] is not None
                else None
            ),
            output_length=len(record.sequence),
            full_construct_length=int(record.metadata["construct__full_construct_length"]),
        )
        for record in built
    ]
    window = realize.window
    preflight = PreflightResult(
        job_id=cfg.job.id,
        input_dataset=cfg.job.input.source.dataset,
        output_dataset=cfg.job.output.target.dataset,
        input_root=input_root,
        output_root=output_root,
        template_id=template.id,
        template_kind=template.kind,
        template_source=template.source,
        template_dataset=template.dataset,
        template_field=template.field,
        template_record_id=template.record_id,
        template_sha256=template_sha256,
        template_length=len(template.sequence),
        template_circular=bool(template.circular),
        realize_mode=realize.mode,
        focal_part=realize.focal_part,
        window_semantics=window.semantics if window is not None else None,
        window_reference=window.reference if window is not None else None,
        window_direction=window.direction if window is not None else None,
        window_size_bp=int(window.size_bp) if window is not None and window.size_bp is not None else None,
        window_upstream_bp=(int(window.upstream_bp) if window is not None and window.upstream_bp is not None else None),
        window_downstream_bp=(
            int(window.downstream_bp) if window is not None and window.downstream_bp is not None else None
        ),
        window_offset_bp=int(window.offset_bp) if window is not None else None,
        spec_id=spec_id,
        records_total=len(built),
        existing_output_collisions=collision_count,
        output_on_conflict=cfg.job.output.on_conflict,
        placements=_planned_placements(
            [resolved.part for resolved in ordered_placements],
            template=template,
            resolved_sites=resolved_sites,
        ),
        planned_rows=planned_rows,
    )
    return preflight, built


def _plan_normalize_loaded_config(
    cfg: JobConfig,
    *,
    config_path: Path,
    input_root: Path,
    output_root: Path,
) -> tuple[PreflightResult, List[_BuiltRecord]]:
    if cfg.job.normalize_anchor is None:
        raise ValidationError("job.normalize_anchor is required when job.mode='normalize_anchor'.")
    normalize_cfg = cfg.job.normalize_anchor
    _require_normalize_target_length_match(cfg=cfg)
    base_dir = config_path.parent
    input_ds = Dataset(input_root, cfg.job.input.source.dataset)
    if not input_ds.records_path.exists():
        raise ValidationError(f"Input dataset not initialized: {input_ds.records_path}")
    _require_distinct_input_output_or_opt_in(cfg=cfg, input_root=input_root, output_root=output_root)
    spec_id = _build_normalize_spec_id(cfg=cfg, input_root=input_root, output_root=output_root)
    rows = _scan_usr_rows(input_ds, columns=_normalize_input_scan_fields(input_ds, cfg), ids=cfg.job.input.ids)
    if not rows:
        raise ValidationError("Input selection resolved to zero rows.")
    built = [
        _build_normalize_anchor_record(
            row=row,
            cfg=cfg,
            spec_id=spec_id,
            output_dataset_id=cfg.job.output.target.dataset,
            load_template=lambda template_cfg: _load_normalize_template(base_dir=base_dir, cfg=template_cfg),
        )
        for row in rows
    ]
    _validate_duplicate_output_aliases(built)
    collision_count = _require_output_conflict_policy(
        built,
        output_root=output_root,
        output_dataset=cfg.job.output.target.dataset,
        on_conflict=cfg.job.output.on_conflict,
    )
    policy = normalize_cfg.under_length_policy
    template = _load_normalize_template(base_dir=base_dir, cfg=policy.template) if policy is not None else None
    template_sha256 = hashlib.sha256(template.sequence.encode("utf-8")).hexdigest() if template is not None else ""
    preflight = PreflightResult(
        job_id=cfg.job.id,
        input_dataset=cfg.job.input.source.dataset,
        output_dataset=cfg.job.output.target.dataset,
        input_root=input_root,
        output_root=output_root,
        template_id=template.id if template is not None else "",
        template_kind=template.kind if template is not None else "",
        template_source=template.source if template is not None else "",
        template_dataset=template.dataset if template is not None else None,
        template_field=template.field if template is not None else None,
        template_record_id=template.record_id if template is not None else None,
        template_sha256=template_sha256,
        template_length=len(template.sequence) if template is not None else 0,
        template_circular=bool(template.circular) if template is not None else False,
        realize_mode="normalize_anchor",
        focal_part="analysis_window",
        window_semantics="normalize_anchor",
        window_reference="annotation_or_sequence_focal",
        window_direction="symmetric",
        window_size_bp=normalize_cfg.target_length,
        window_upstream_bp=None,
        window_downstream_bp=None,
        window_offset_bp=None,
        spec_id=spec_id,
        records_total=len(built),
        existing_output_collisions=collision_count,
        output_on_conflict=cfg.job.output.on_conflict,
        placements=[],
        planned_rows=[
            PlannedRow(
                input_id=str(record.metadata["construct__input_id"]),
                output_id=record.output_id,
                input_length=int(record.metadata["construct__input_length"]),
                focal_part_length=int(record.metadata["construct__focal_part_length"]),
                output_length=len(record.sequence),
                full_construct_length=int(record.metadata["construct__full_construct_length"]),
            )
            for record in built
        ],
    )
    return preflight, built


def _plan_loaded_config(
    cfg: JobConfig,
    *,
    config_path: Path,
) -> tuple[PreflightResult, List[_BuiltRecord]]:
    base_dir = config_path.parent
    input_root = _resolve_usr_root(base_dir, cfg.job.input.source.root, label="job.input.source.root")
    output_root = _resolve_usr_root(
        base_dir,
        cfg.job.output.target.root or cfg.job.input.source.root,
        label="job.output.target.root or job.input.source.root",
    )
    if cfg.job.mode == "normalize_anchor":
        return _plan_normalize_loaded_config(
            cfg,
            config_path=config_path,
            input_root=input_root,
            output_root=output_root,
        )
    return _plan_classic_loaded_config(
        cfg,
        config_path=config_path,
        input_root=input_root,
        output_root=output_root,
    )


def _planned_run_from_config(path: str | Path) -> _PlannedRun:
    cfg, config_path = load_job_config(path)
    preflight, built = _plan_loaded_config(cfg, config_path=config_path)
    return _PlannedRun(cfg=cfg, preflight=preflight, built=built)


def _plan_from_config(path: str | Path) -> tuple[PreflightResult, List[_BuiltRecord]]:
    planned = _planned_run_from_config(path)
    return planned.preflight, planned.built


def _dry_run_result(planned: _PlannedRun) -> RunResult:
    cfg = planned.cfg
    preflight = planned.preflight
    return RunResult(
        job_id=cfg.job.id,
        input_dataset=cfg.job.input.source.dataset,
        output_dataset=cfg.job.output.target.dataset,
        output_root=preflight.output_root,
        records_total=preflight.records_total,
        records_written=0,
        records_skipped_existing=preflight.existing_output_collisions,
        spec_id=preflight.spec_id,
        dry_run=True,
    )


def _persist_construct_run(planned: _PlannedRun) -> RunResult:
    cfg = planned.cfg
    preflight = planned.preflight
    output_ds = _ensure_output_dataset(
        output_root=preflight.output_root,
        output_dataset=cfg.job.output.target.dataset,
    )
    built_to_write = _records_to_write(
        planned.built,
        output_root=preflight.output_root,
        output_dataset=cfg.job.output.target.dataset,
        on_conflict=cfg.job.output.on_conflict,
    )
    _write_output_records(
        output_ds,
        job_id=cfg.job.id,
        record_source=cfg.job.output.record_source,
        records=built_to_write,
    )
    _write_planned_sequence_views(output_ds, job_id=cfg.job.id, records=planned.built)
    return RunResult(
        job_id=cfg.job.id,
        input_dataset=cfg.job.input.source.dataset,
        output_dataset=cfg.job.output.target.dataset,
        output_root=preflight.output_root,
        records_total=preflight.records_total,
        records_written=len(built_to_write),
        records_skipped_existing=preflight.records_total - len(built_to_write),
        spec_id=preflight.spec_id,
        dry_run=False,
    )


def preflight_from_config(path: str | Path) -> PreflightResult:
    return _planned_run_from_config(path).preflight


def run_from_config(path: str | Path, *, dry_run: bool = False) -> RunResult:
    planned = _planned_run_from_config(path)
    if dry_run:
        return _dry_run_result(planned)
    return _persist_construct_run(planned)

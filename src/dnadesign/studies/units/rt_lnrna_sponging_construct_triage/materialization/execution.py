"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/materialization/execution.py

Execution workflow for RT-lnRNA Construct materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.construct import RunResult, run_from_config

from .common import _list, _mapping, _resolve_repo_root
from .contracts import MaterializationContractError, _ConstructViewRunPlan, _MaterializationContext
from .manifest import (
    _load_projection_manifest,
    _require_genbank_authority,
    _require_valid_projection_manifest,
    _target_context_bounds,
    _template_sequence,
)
from .usr_io import (
    _require_construct_infer_ready,
    _write_construct_output_subject_bridge,
    _write_construct_subject_dataset,
)
from .views import (
    _construct_config,
    _context_output_variants,
    _slot_anchor_output_variants,
    _write_config,
)


@dataclass(frozen=True)
class _ConstructMaterializationRun:
    usr_root: Path
    input_ids_by_subject_id: dict[str, str]
    config_paths: tuple[Path, ...]
    run_results: tuple[RunResult, ...]


def _load_materialization_context(repo_root: Path | None) -> _MaterializationContext:
    root = _resolve_repo_root(repo_root)
    manifest = _load_projection_manifest(root)
    _require_valid_projection_manifest(manifest)
    authority = _require_genbank_authority(root)
    template_sequence = _template_sequence(manifest=manifest, authority=authority)
    target_start, target_end = _target_context_bounds(manifest)
    slots = tuple(_mapping(slot, label="slots[]") for slot in _list(manifest["slots"], label="slots"))
    return _MaterializationContext(
        root=root,
        manifest=manifest,
        authority=authority,
        template_sequence=template_sequence,
        target_start=target_start,
        target_end=target_end,
        template_context_sequence=template_sequence[target_start:target_end],
        slots=slots,
    )


def _materialize_construct_view_plans(
    *,
    work_root: Path,
    context: _MaterializationContext,
    rows: list[dict[str, object]],
    plans: tuple[_ConstructViewRunPlan, ...],
) -> _ConstructMaterializationRun:
    if not rows:
        raise MaterializationContractError("Construct materialization selected no input rows.")
    if not plans:
        raise MaterializationContractError("Construct materialization selected no Construct run plans.")

    work = Path(work_root).resolve()
    usr_root = work / "usr"
    config_dir = work / "construct_configs"
    usr_root.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    input_ids_by_subject_id = _write_construct_subject_dataset(usr_root=usr_root, rows=rows)
    config_paths: list[Path] = []
    run_results: list[RunResult] = []
    for plan in plans:
        _require_plan_subjects(plan, input_ids_by_subject_id=input_ids_by_subject_id)
        context_config = _construct_config(
            manifest=context.manifest,
            template_sequence=context.template_sequence,
            usr_root=usr_root,
            input_ids_by_subject_id=input_ids_by_subject_id,
            job_id=plan.context_job_id,
            output_on_conflict="error",
            output_variants=_context_output_variants(),
            construct_subject_ids=plan.subject_ids,
            window_offset_bp=plan.window_offset_bp,
        )
        slot_anchor_config = _construct_config(
            manifest=context.manifest,
            template_sequence=context.template_sequence,
            usr_root=usr_root,
            input_ids_by_subject_id=input_ids_by_subject_id,
            job_id=plan.slot_anchor_job_id,
            output_on_conflict="ignore",
            output_variants=_slot_anchor_output_variants(),
            construct_subject_ids=plan.subject_ids,
            window_offset_bp=plan.window_offset_bp,
        )
        context_path = _write_config(config_dir / plan.context_config_name, context_config)
        slot_anchor_path = _write_config(config_dir / plan.slot_anchor_config_name, slot_anchor_config)
        config_paths.extend([context_path, slot_anchor_path])
        run_results.extend([run_from_config(context_path), run_from_config(slot_anchor_path)])

    _write_construct_output_subject_bridge(
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
    )
    _require_construct_infer_ready(
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
    )
    return _ConstructMaterializationRun(
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
        config_paths=tuple(config_paths),
        run_results=tuple(run_results),
    )


def _require_plan_subjects(
    plan: _ConstructViewRunPlan,
    *,
    input_ids_by_subject_id: dict[str, str],
) -> None:
    if not plan.subject_ids:
        raise MaterializationContractError(f"Construct run plan {plan.context_job_id} selected no subjects.")
    missing = sorted(set(plan.subject_ids) - set(input_ids_by_subject_id))
    if missing:
        raise MaterializationContractError(
            f"Construct run plan {plan.context_job_id} references unmaterialized subject id(s): "
            + ", ".join(missing[:8])
        )

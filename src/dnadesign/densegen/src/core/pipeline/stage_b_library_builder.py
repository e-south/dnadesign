"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/stage_b_library_builder.py

Stage-B library construction and feasibility validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ...config.generation import ResolvedPlanItem, SamplingConfig
from ..artifacts.library import LibraryRecord
from ..artifacts.pool import PoolData
from .outputs import _emit_event
from .sequence_validation import _validate_library_constraints
from .stage_b import (
    assess_library_feasibility,
    build_library_for_plan,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LibraryContext:
    library_for_opt: list[str]
    tfbs_parts: list[str]
    regulator_labels: list[str]
    sampling_info: dict
    library_tfbs: list[str]
    library_tfs: list[str]
    library_site_ids: list[str | None]
    library_sources: list[str | None]
    library_tfbs_ids: list[str | None]
    library_motif_ids: list[str | None]
    sampling_library_index: int
    sampling_library_hash: str
    required_regulators: list[str]
    fixed_bp: int
    min_required_bp: int
    slack_bp: int
    infeasible: bool
    min_required_len: int
    min_breakdown: dict


@dataclass
class LibraryBuilder:
    source_label: str
    plan_item: ResolvedPlanItem
    pool: PoolData
    sampling_cfg: SamplingConfig
    seq_len: int
    min_count_per_tf: int
    usage_counts: dict[tuple[str, str], int]
    failure_counts: dict[tuple[str, str, str, str, str | None], dict[str, int]] | None
    rng: random.Random
    np_rng: np.random.Generator
    library_source_label: str
    library_records: dict[tuple[str, str], list[LibraryRecord]] | None
    library_cursor: dict[tuple[str, str], int] | None
    events_path: Path | None
    library_build_rows: list[dict] | None
    library_member_rows: list[dict] | None

    def build_next(self, *, library_index_start: int) -> LibraryContext:
        pool_strategy = str(self.sampling_cfg.pool_strategy)
        library_sampling_strategy = str(self.sampling_cfg.library_sampling_strategy)
        constraints = self.plan_item.regulator_constraints
        groups = list(constraints.groups or [])
        plan_min_count_by_regulator = dict(constraints.min_count_by_regulator or {})

        if self.library_source_label == "artifact":
            library_for_opt, tfbs_parts, regulator_labels, sampling_info = self._select_library_from_artifact(
                pool_strategy=pool_strategy,
                library_sampling_strategy=library_sampling_strategy,
                groups=groups,
                plan_min_count_by_regulator=plan_min_count_by_regulator,
            )
        else:
            library_for_opt, tfbs_parts, regulator_labels, sampling_info = build_library_for_plan(
                source_label=self.source_label,
                plan_item=self.plan_item,
                pool=self.pool,
                sampling_cfg=self.sampling_cfg,
                seq_len=self.seq_len,
                min_count_per_tf=self.min_count_per_tf,
                usage_counts=self.usage_counts,
                failure_counts=self.failure_counts if self.failure_counts else None,
                rng=self.rng,
                np_rng=self.np_rng,
                library_index_start=library_index_start,
            )

        site_id_by_index = sampling_info.get("site_id_by_index")
        source_by_index = sampling_info.get("source_by_index")
        tfbs_id_by_index = sampling_info.get("tfbs_id_by_index")
        motif_id_by_index = sampling_info.get("motif_id_by_index")
        sampling_library_index = int(sampling_info.get("library_index") or 0)
        sampling_library_hash = str(sampling_info.get("library_hash") or "")
        library_tfbs = list(library_for_opt)
        library_tfs = list(regulator_labels) if regulator_labels else []
        library_site_ids = list(site_id_by_index) if site_id_by_index else []
        library_sources = list(source_by_index) if source_by_index else []
        library_tfbs_ids = list(tfbs_id_by_index) if tfbs_id_by_index else []
        library_motif_ids = list(motif_id_by_index) if motif_id_by_index else []
        required_regulators = list(dict.fromkeys(sampling_info.get("required_regulators_selected") or []))
        if groups and not required_regulators:
            raise RuntimeError(
                f"Stage-B sampling did not record required_regulators_selected for {self.source_label}/"
                f"{self.plan_item.name}. Rebuild libraries with the current version."
            )

        min_required_len, min_breakdown, feasibility = assess_library_feasibility(
            library_tfbs=library_tfbs,
            library_tfs=library_tfs,
            fixed_elements=self.plan_item.fixed_elements,
            groups=groups,
            min_count_by_regulator=plan_min_count_by_regulator,
            min_count_per_tf=self.min_count_per_tf,
            sequence_length=self.seq_len,
        )
        fixed_bp = int(feasibility["fixed_bp"])
        min_required_bp = int(feasibility["min_required_bp"])
        slack_bp = int(feasibility["slack_bp"])
        infeasible = bool(feasibility["infeasible"])
        self._record_library_build(
            sampling_info=sampling_info,
            library_tfbs=library_tfbs,
            library_tfs=library_tfs,
            library_tfbs_ids=library_tfbs_ids,
            library_motif_ids=library_motif_ids,
            library_site_ids=library_site_ids,
            library_sources=library_sources,
            fixed_bp=fixed_bp,
            min_required_bp=min_required_bp,
            slack_bp=slack_bp,
            infeasible=infeasible,
            sequence_length=self.seq_len,
        )

        return LibraryContext(
            library_for_opt=list(library_for_opt),
            tfbs_parts=list(tfbs_parts),
            regulator_labels=list(regulator_labels) if regulator_labels else [],
            sampling_info=dict(sampling_info),
            library_tfbs=library_tfbs,
            library_tfs=library_tfs,
            library_site_ids=library_site_ids,
            library_sources=library_sources,
            library_tfbs_ids=library_tfbs_ids,
            library_motif_ids=library_motif_ids,
            sampling_library_index=sampling_library_index,
            sampling_library_hash=sampling_library_hash,
            required_regulators=required_regulators,
            fixed_bp=fixed_bp,
            min_required_bp=min_required_bp,
            slack_bp=slack_bp,
            infeasible=infeasible,
            min_required_len=min_required_len,
            min_breakdown=min_breakdown,
        )

    def _select_library_from_artifact(
        self,
        *,
        pool_strategy: str,
        library_sampling_strategy: str,
        groups: list,
        plan_min_count_by_regulator: dict[str, int],
    ) -> tuple[list[str], list[str], list[str], dict]:
        if self.library_records is None or self.library_cursor is None:
            raise RuntimeError("Library artifacts requested but no library records were provided.")
        key = (self.source_label, self.plan_item.name)
        records = self.library_records.get(key) or []
        if not records:
            raise RuntimeError(
                f"No libraries available in artifact for {self.source_label}/{self.plan_item.name}. "
                "Build libraries with `dense stage-b build-libraries` and re-run."
            )
        cursor = int(self.library_cursor.get(key, 0))
        if cursor >= len(records):
            raise RuntimeError(
                f"Library artifact exhausted for {self.source_label}/{self.plan_item.name} "
                f"(requested index={cursor + 1}, available={len(records)}). "
                "Build more libraries or reduce Stage-B resampling."
            )
        record = records[cursor]
        self.library_cursor[key] = cursor + 1
        if record.pool_strategy is None or record.library_sampling_strategy is None:
            raise RuntimeError(
                f"Library artifact missing Stage-B sampling metadata for {self.source_label}/{self.plan_item.name} "
                f"(library_index={record.library_index}). Rebuild libraries with the current version."
            )
        if str(record.pool_strategy) != str(pool_strategy):
            raise RuntimeError(
                f"Library artifact pool_strategy mismatch for {self.source_label}/{self.plan_item.name}: "
                f"artifact={record.pool_strategy} config={pool_strategy}."
            )
        if str(record.library_sampling_strategy) != str(library_sampling_strategy):
            raise RuntimeError(
                f"Library artifact Stage-B sampling strategy mismatch for {self.source_label}/{self.plan_item.name}: "
                f"artifact={record.library_sampling_strategy} config={library_sampling_strategy}."
            )
        if pool_strategy != "full" and record.library_size != int(self.sampling_cfg.library_size):
            raise RuntimeError(
                f"Library artifact size mismatch for {self.source_label}/{self.plan_item.name}: "
                f"artifact={record.library_size} config={self.sampling_cfg.library_size}."
            )
        _validate_library_constraints(
            record,
            groups=groups,
            min_count_by_regulator=plan_min_count_by_regulator,
            input_name=self.source_label,
            plan_name=self.plan_item.name,
        )
        tfbs_parts_local = []
        for idx, tfbs in enumerate(record.library_tfbs):
            tf = record.library_tfs[idx] if idx < len(record.library_tfs) else ""
            tfbs_parts_local.append(f"{tf}:{tfbs}" if tf else str(tfbs))
        if self.events_path is not None:
            try:
                _emit_event(
                    self.events_path,
                    event="LIBRARY_SELECTED",
                    payload={
                        "input_name": self.source_label,
                        "plan_name": self.plan_item.name,
                        "library_index": int(record.library_index),
                        "library_hash": str(record.library_hash),
                        "library_size": int(record.library_size),
                    },
                )
            except Exception:
                log.debug("Failed to emit LIBRARY_SELECTED event.", exc_info=True)
        return record.library_tfbs, tfbs_parts_local, record.library_tfs, record.sampling_info()

    def _record_library_build(
        self,
        *,
        sampling_info: dict,
        library_tfbs: list[str],
        library_tfs: list[str],
        library_tfbs_ids: list[str | None],
        library_motif_ids: list[str | None],
        library_site_ids: list[str | None],
        library_sources: list[str | None],
        fixed_bp: int,
        min_required_bp: int,
        slack_bp: int,
        infeasible: bool,
        sequence_length: int,
    ) -> None:
        if str(self.sampling_cfg.library_source).lower() == "artifact":
            return
        library_index = int(sampling_info.get("library_index") or 0)
        library_hash = str(sampling_info.get("library_hash") or "")
        library_id = library_hash or f"{self.source_label}:{self.plan_item.name}:{library_index}"
        library_size = int(sampling_info.get("library_size") or len(library_tfbs))
        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_name": self.source_label,
            "plan_name": self.plan_item.name,
            "library_index": library_index,
            "library_id": library_id,
            "library_hash": library_hash,
            "pool_strategy": sampling_info.get("pool_strategy"),
            "library_sampling_strategy": sampling_info.get("library_sampling_strategy"),
            "library_size": library_size,
            "achieved_length": sampling_info.get("achieved_length"),
            "relaxed_cap": sampling_info.get("relaxed_cap"),
            "final_cap": sampling_info.get("final_cap"),
            "iterative_max_libraries": sampling_info.get("iterative_max_libraries"),
            "iterative_min_new_solutions": sampling_info.get("iterative_min_new_solutions"),
            "required_regulators_selected": sampling_info.get("required_regulators_selected"),
            "fixed_bp": int(fixed_bp),
            "min_required_bp": int(min_required_bp),
            "slack_bp": int(slack_bp),
            "infeasible": bool(infeasible),
            "sequence_length": int(sequence_length),
        }
        if self.library_build_rows is not None:
            self.library_build_rows.append(row)
        if self.events_path is not None:
            try:
                _emit_event(
                    self.events_path,
                    event="LIBRARY_BUILT",
                    payload={
                        "input_name": self.source_label,
                        "plan_name": self.plan_item.name,
                        "library_index": library_index,
                        "library_hash": library_hash,
                        "library_size": library_size,
                    },
                )
            except Exception:
                log.debug("Failed to emit LIBRARY_BUILT event.", exc_info=True)
            if sampling_info.get("sampling_weight_by_tf"):
                try:
                    _emit_event(
                        self.events_path,
                        event="LIBRARY_SAMPLING_PRESSURE",
                        payload={
                            "input_name": self.source_label,
                            "plan_name": self.plan_item.name,
                            "library_index": library_index,
                            "library_hash": library_hash,
                            "sampling_strategy": sampling_info.get("library_sampling_strategy"),
                            "weight_by_tf": sampling_info.get("sampling_weight_by_tf"),
                            "weight_fraction_by_tf": sampling_info.get("sampling_weight_fraction_by_tf"),
                            "usage_count_by_tf": sampling_info.get("sampling_usage_count_by_tf"),
                            "failure_count_by_tf": sampling_info.get("sampling_failure_count_by_tf"),
                        },
                    )
                except Exception:
                    log.debug("Failed to emit LIBRARY_SAMPLING_PRESSURE event.", exc_info=True)
        if self.library_member_rows is not None:
            for idx, tfbs in enumerate(library_tfbs):
                self.library_member_rows.append(
                    {
                        "library_id": library_id,
                        "library_hash": library_hash,
                        "library_index": library_index,
                        "input_name": self.source_label,
                        "plan_name": self.plan_item.name,
                        "position": int(idx),
                        "tf": library_tfs[idx] if idx < len(library_tfs) else "",
                        "tfbs": tfbs,
                        "tfbs_id": library_tfbs_ids[idx] if idx < len(library_tfbs_ids) else None,
                        "motif_id": library_motif_ids[idx] if idx < len(library_motif_ids) else None,
                        "site_id": library_site_ids[idx] if idx < len(library_site_ids) else None,
                        "source": library_sources[idx] if idx < len(library_sources) else None,
                    }
                )

"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/status_adapters/cruncher_status/adapter.py

Cruncher study status adapter for tracked study snapshots and preflights.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dnadesign.studies.core.models import StudyStatusAdapter, StudyStatusContext

from .preflight import (
    CruncherPreflightDependencies,
    build_cruncher_preflight_progress,
    resolve_cruncher_preflight_context,
)
from .record_normalizer import CruncherStudyResolvedContext, resolve_cruncher_study_context
from .snapshot import build_cruncher_study_status


@dataclass(frozen=True)
class CruncherStatusAdapterContext:
    study_context: CruncherStudyResolvedContext


class CruncherStudyStatusAdapter(StudyStatusAdapter):
    status_kind = "cruncher-study-status"
    preflight_kind = "cruncher-study-preflight"

    def load_context(self, *, repo_root: Path | None, study_root: Path | None) -> StudyStatusContext:
        study_context = resolve_cruncher_study_context(
            study_root,
            repo_root=repo_root,
            status_kind="cruncher-study-status",
        )
        contract = study_context.ops_contract
        if contract.status_kind != self.status_kind:
            raise ValueError(
                f"ops.study.yaml ops_surfaces.status_kind mismatch for {study_context.resolved_study_dir}: "
                f"expected {self.status_kind}, found {contract.status_kind}"
            )
        if contract.preflight_kind != self.preflight_kind:
            raise ValueError(
                f"ops.study.yaml ops_surfaces.preflight_kind mismatch for {study_context.resolved_study_dir}: "
                f"expected {self.preflight_kind}, found {contract.preflight_kind}"
            )
        if contract.study_id != study_context.study_id:
            raise ValueError(
                f"ops.study.yaml study_id mismatch for {study_context.resolved_study_dir}: "
                f"expected {study_context.study_id}, found {contract.study_id}"
            )
        return StudyStatusContext(
            repo_root=study_context.study_repo_root,
            study_root=study_context.resolved_study_dir,
            contract=contract,
            adapter_context=CruncherStatusAdapterContext(study_context=study_context),
        )

    def build_snapshot(self, context: StudyStatusContext) -> tuple[str, str, dict[str, object]]:
        study_context = _study_adapter_context(context).study_context
        missing_result = _missing_cruncher_study_result(study_context=study_context)
        if missing_result is not None:
            return missing_result
        return build_cruncher_study_status(
            study_context=study_context,
            summary_scope=context.contract.snapshot_summary_scope,
        )

    def build_preflight(
        self,
        context: StudyStatusContext,
        *,
        scope: str | None,
    ) -> tuple[str, str, dict[str, object]]:
        study_context = _study_adapter_context(context).study_context
        missing_result = _missing_cruncher_study_result(study_context=study_context)
        if missing_result is not None:
            return missing_result
        resolved_context = resolve_cruncher_preflight_context(
            study_context=study_context,
            scope=scope,
            contract=context.contract,
        )
        return build_cruncher_preflight_progress(
            context=resolved_context,
            dependencies=CruncherPreflightDependencies(
                environ=os.environ,
            ),
        )


STUDY_STATUS_ADAPTER = CruncherStudyStatusAdapter()


def _missing_cruncher_study_result(
    *,
    study_context: CruncherStudyResolvedContext,
) -> tuple[str, str, dict[str, object]] | None:
    if not study_context.missing_required_files:
        return None
    evidence = dict(study_context.evidence)
    evidence.update(
        {
            "missing_required_files": list(study_context.missing_required_files),
            "record_paths": {name: str(path) for name, path in study_context.record_paths.items()},
        }
    )
    summary = f"{study_context.study_id}: missing study files " + ", ".join(study_context.missing_required_files)
    return ("missing", summary, evidence)


def _study_adapter_context(context: StudyStatusContext) -> CruncherStatusAdapterContext:
    if not isinstance(context.adapter_context, CruncherStatusAdapterContext):
        raise ValueError("cruncher status context has invalid adapter_context payload")
    return context.adapter_context


__all__ = ["STUDY_STATUS_ADAPTER", "CruncherStatusAdapterContext", "CruncherStudyStatusAdapter"]

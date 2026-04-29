"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/families/cruncher/adapter.py

Cruncher study-family adapter for tracked study snapshots and preflights.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dnadesign.studies.core.models import StudyFamilyAdapter, StudyStatusContext

from .preflight import (
    CruncherPreflightDependencies,
    build_cruncher_preflight_progress,
    resolve_cruncher_preflight_context,
)
from .record_normalizer import CruncherStudyResolvedContext, resolve_cruncher_study_context
from .snapshot import build_cruncher_study_status


@dataclass(frozen=True)
class CruncherFamilyContext:
    study_context: CruncherStudyResolvedContext


class CruncherStudyFamilyAdapter(StudyFamilyAdapter):
    family_id = "cruncher"

    def load_context(self, *, repo_root: Path | None, study_root: Path | None) -> StudyStatusContext:
        study_context = resolve_cruncher_study_context(
            study_root,
            repo_root=repo_root,
            status_kind="cruncher-study-status",
        )
        contract = study_context.ops_contract
        if contract.family != self.family_id:
            raise ValueError(
                f"ops.study.yaml family mismatch for {study_context.resolved_study_dir}: "
                f"expected {self.family_id}, found {contract.family}"
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
            family_context=CruncherFamilyContext(study_context=study_context),
        )

    def build_snapshot(self, context: StudyStatusContext) -> tuple[str, str, dict[str, object]]:
        study_context = _study_family_context(context).study_context
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
        study_context = _study_family_context(context).study_context
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


STUDY_FAMILY_ADAPTER = CruncherStudyFamilyAdapter()


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


def _study_family_context(context: StudyStatusContext) -> CruncherFamilyContext:
    if not isinstance(context.family_context, CruncherFamilyContext):
        raise ValueError("cruncher status context has invalid family_context payload")
    return context.family_context


__all__ = ["STUDY_FAMILY_ADAPTER", "CruncherFamilyContext", "CruncherStudyFamilyAdapter"]

"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/status/service.py

Retron hairpin design study status service for tracked study snapshots and preflights.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dnadesign.studies.core.models import StudyStatusContext, StudyStatusService

from .preflight import (
    RetronHairpinDesignPreflightDependencies,
    build_retron_hairpin_design_preflight_progress,
    resolve_retron_hairpin_design_preflight_context,
)
from .record_normalizer import RetronHairpinDesignResolvedContext, resolve_retron_hairpin_design_context
from .snapshot import build_retron_hairpin_design_status


@dataclass(frozen=True)
class RetronHairpinDesignStatusServiceContext:
    study_context: RetronHairpinDesignResolvedContext


class RetronHairpinDesignStatusService(StudyStatusService):
    study_id = "retron_hairpin_design"
    status_kind = "retron-hairpin-design-status"
    preflight_kind = "retron-hairpin-design-preflight"

    def load_context(self, *, repo_root: Path | None, study_root: Path | None) -> StudyStatusContext:
        study_context = resolve_retron_hairpin_design_context(
            study_root,
            repo_root=repo_root,
            status_kind="retron-hairpin-design-status",
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
        if contract.study_id != self.study_id:
            raise ValueError(
                f"{self.status_kind} only serves study_id {self.study_id!r}; "
                f"found {contract.study_id!r} in {study_context.resolved_study_dir / 'operations' / 'ops.study.yaml'}"
            )
        return StudyStatusContext(
            repo_root=study_context.study_repo_root,
            study_root=study_context.resolved_study_dir,
            contract=contract,
            service_context=RetronHairpinDesignStatusServiceContext(study_context=study_context),
        )

    def build_snapshot(self, context: StudyStatusContext) -> tuple[str, str, dict[str, object]]:
        study_context = _study_service_context(context).study_context
        missing_result = _missing_retron_hairpin_design_result(study_context=study_context)
        if missing_result is not None:
            return missing_result
        return build_retron_hairpin_design_status(
            study_context=study_context,
            summary_scope=context.contract.snapshot_summary_scope,
        )

    def build_preflight(
        self,
        context: StudyStatusContext,
        *,
        scope: str | None,
    ) -> tuple[str, str, dict[str, object]]:
        study_context = _study_service_context(context).study_context
        missing_result = _missing_retron_hairpin_design_result(study_context=study_context)
        if missing_result is not None:
            return missing_result
        resolved_context = resolve_retron_hairpin_design_preflight_context(
            study_context=study_context,
            scope=scope,
            contract=context.contract,
        )
        return build_retron_hairpin_design_preflight_progress(
            context=resolved_context,
            dependencies=RetronHairpinDesignPreflightDependencies(
                environ=os.environ,
            ),
        )


STUDY_STATUS_SERVICE = RetronHairpinDesignStatusService()


def _missing_retron_hairpin_design_result(
    *,
    study_context: RetronHairpinDesignResolvedContext,
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


def _study_service_context(context: StudyStatusContext) -> RetronHairpinDesignStatusServiceContext:
    if not isinstance(context.service_context, RetronHairpinDesignStatusServiceContext):
        raise ValueError("retron hairpin design status context has invalid service_context payload")
    return context.service_context


__all__ = ["STUDY_STATUS_SERVICE", "RetronHairpinDesignStatusServiceContext", "RetronHairpinDesignStatusService"]

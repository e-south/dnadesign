"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/downstream_surfaces.py

Study-owned downstream surface inspection for stress_ethanol_cipro_growth.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

from dnadesign.ops.status import resolve_repo_relative_path

from .opal_surface import inspect_opal_surface
from .record_normalizer import StressEthanolCiproGrowthResolvedContext

_STATUS_KIND = "stress-ethanol-cipro-growth-status"
_DEFAULT_CLUSTER_DOC = "src/dnadesign/cluster/docs/workflows/exploratory-clustering.md"
_DEFAULT_OPAL_DOC = "src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md"


def inspect_stress_ethanol_cipro_growth_downstream_surfaces(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
) -> dict[str, dict[str, object]]:
    downstream_candidate_dataset = _downstream_candidate_dataset_id(study_context=study_context)
    return {
        "cluster": _inspect_declared_surface(
            study_context=study_context,
            config_key="cluster",
            default_doc=_DEFAULT_CLUSTER_DOC,
            default_state="planned",
            surface_key="results_root",
            entry_artifact_default=downstream_candidate_dataset,
        ),
        "opal": _inspect_opal_surface(
            study_context=study_context,
            downstream_candidate_dataset=downstream_candidate_dataset,
        ),
    }


def _inspect_opal_surface(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
    downstream_candidate_dataset: str | None,
) -> dict[str, object]:
    payload = inspect_opal_surface(study_context=study_context, default_doc=_DEFAULT_OPAL_DOC)
    if "entry_artifact" not in payload and downstream_candidate_dataset is not None:
        payload["entry_artifact"] = downstream_candidate_dataset
    return payload


def _inspect_declared_surface(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
    config_key: str,
    default_doc: str,
    default_state: str,
    surface_key: str,
    entry_artifact_default: str | None,
) -> dict[str, object]:
    surface_config = study_context.study_pipeline.get(config_key)
    if not isinstance(surface_config, Mapping):
        surface_config = {}

    doc_path = _string_or_none(surface_config.get("doc")) or default_doc
    raw_state = _string_or_none(surface_config.get("state")) or default_state
    raw_surface = _string_or_none(surface_config.get(surface_key))
    entry_artifact = _string_or_none(surface_config.get("entry_artifact")) or entry_artifact_default

    configured = False
    resolved_surface_ref: str | None = None
    if raw_surface is not None and raw_surface.lower() != "n/a":
        configured = True
        if study_context.study_repo_root is not None:
            resolved_surface_ref = str(
                resolve_repo_relative_path(
                    repo_root=study_context.study_repo_root,
                    raw_path=raw_surface,
                    status_kind=_STATUS_KIND,
                )
            )
        else:
            resolved_surface_ref = raw_surface

    payload: dict[str, object] = {
        "configured": configured,
        "state": raw_state if configured or raw_state in {"planned", "not_configured"} else default_state,
        "doc": doc_path,
        "surface_ref": resolved_surface_ref,
    }
    if entry_artifact is not None:
        payload["entry_artifact"] = entry_artifact
    payload[surface_key] = raw_surface
    return payload


def _downstream_candidate_dataset_id(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
) -> str | None:
    preferred_roles = {
        "opal_candidate_feature_table",
    }
    for dataset_state in study_context.dataset_states:
        if _string_or_none(dataset_state.get("role")) in preferred_roles:
            return _string_or_none(dataset_state.get("dataset"))
    for dataset_state in study_context.dataset_states:
        dataset_id = _string_or_none(dataset_state.get("dataset"))
        if dataset_id and "opal_candidates" in dataset_id:
            return dataset_id
    return None


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None

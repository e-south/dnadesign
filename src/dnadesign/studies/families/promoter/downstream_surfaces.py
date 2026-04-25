"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/families/promoter/downstream_surfaces.py

Uniform downstream surface inspection for promoter-study snapshot evidence.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

from dnadesign.ops.status.paths import resolve_repo_relative_path

from .record_normalizer import PromoterStudyResolvedContext

_STATUS_KIND = "promoter-study-status"
_DEFAULT_CLUSTER_DOC = "src/dnadesign/cluster/docs/workflows/exploratory-clustering.md"
_DEFAULT_OPAL_DOC = "src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md"


def inspect_promoter_downstream_surfaces(
    *,
    study_context: PromoterStudyResolvedContext,
) -> dict[str, dict[str, object]]:
    feature_matrix_dataset = _feature_matrix_dataset_id(study_context=study_context)
    return {
        "cluster": _inspect_declared_surface(
            study_context=study_context,
            config_key="cluster",
            default_doc=_DEFAULT_CLUSTER_DOC,
            default_state="planned",
            surface_key="results_root",
            entry_artifact_default=feature_matrix_dataset,
        ),
        "opal": _inspect_declared_surface(
            study_context=study_context,
            config_key="opal",
            default_doc=_DEFAULT_OPAL_DOC,
            default_state="not_configured",
            surface_key="config",
            entry_artifact_default=feature_matrix_dataset,
        ),
    }


def _inspect_declared_surface(
    *,
    study_context: PromoterStudyResolvedContext,
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


def _feature_matrix_dataset_id(
    *,
    study_context: PromoterStudyResolvedContext,
) -> str | None:
    for dataset_state in study_context.dataset_states:
        if _string_or_none(dataset_state.get("role")) == "feature_matrix":
            return _string_or_none(dataset_state.get("dataset"))
    for dataset_state in study_context.dataset_states:
        dataset_id = _string_or_none(dataset_state.get("dataset"))
        if dataset_id and "feature_matrix" in dataset_id:
            return dataset_id
    return None


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None

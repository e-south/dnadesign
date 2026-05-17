"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/opal_surface.py

Study-owned status surface helpers for the stress / ethanol / ciprofloxacin
growth study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

from dnadesign.ops.status import resolve_repo_relative_path

_STATUS_KIND = "stress-ethanol-cipro-growth-status"
_OPAL_CANDIDATE_TABLE_ARTIFACT = "opal_candidate_feature_table"
_OPAL_CANDIDATE_TABLE_ROLE = "opal_candidate_feature_table"


def inspect_opal_surface(*, study_context: object, default_doc: str) -> dict[str, object]:
    opal_config = _mapping(_mapping(getattr(study_context, "study_pipeline", {})).get("opal"))
    doc_path = _string_or_none(opal_config.get("doc")) or default_doc
    raw_config = _string_or_none(opal_config.get("config"))
    raw_state = _string_or_none(opal_config.get("state")) or "not_configured"
    candidate_table = _resolve_candidate_feature_table(study_context=study_context, opal_config=opal_config)
    configured = raw_config is not None or candidate_table is not None

    if configured and candidate_table is None:
        raise ValueError(
            "stress_ethanol_cipro_growth OPAL surface must define "
            "opal.candidate_feature_table in pipeline.yaml or "
            "artifacts.opal_candidate_feature_table in ops.study.yaml"
        )

    resolved_config_ref = _resolve_optional_repo_path(study_context=study_context, raw_path=raw_config)
    payload: dict[str, object] = {
        "configured": configured,
        "state": raw_state if configured or raw_state in {"planned", "not_configured"} else "not_configured",
        "doc": doc_path,
        "surface_ref": resolved_config_ref,
        "config": raw_config,
    }
    if isinstance(opal_config.get("configs"), Mapping):
        payload["configs"] = dict(opal_config["configs"])
    if candidate_table is not None:
        payload["entry_artifact"] = candidate_table["dataset"]
        payload["candidate_feature_table"] = candidate_table
    return payload


def _resolve_candidate_feature_table(
    *,
    study_context: object,
    opal_config: Mapping[str, object],
) -> dict[str, object] | None:
    pipeline_payload = _mapping(opal_config.get("candidate_feature_table"))
    contract_payload = _mapping(
        getattr(getattr(study_context, "ops_contract", None), "artifacts", {}).get(_OPAL_CANDIDATE_TABLE_ARTIFACT)
    )
    if not pipeline_payload and not contract_payload:
        return None

    dataset = _string_or_none(pipeline_payload.get("dataset")) or _string_or_none(contract_payload.get("dataset_id"))
    role = _string_or_none(pipeline_payload.get("role")) or _string_or_none(contract_payload.get("role"))
    x_column = _string_or_none(pipeline_payload.get("x_column")) or _string_or_none(contract_payload.get("x_column"))
    x_source = _string_or_none(pipeline_payload.get("x_source")) or _string_or_none(contract_payload.get("x_source"))
    ref = _string_or_none(contract_payload.get("ref"))

    missing = [
        name
        for name, value in (
            ("dataset", dataset),
            ("role", role),
            ("x_column", x_column),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "stress_ethanol_cipro_growth OPAL candidate_feature_table is missing required field(s): "
            + ", ".join(missing)
        )
    if role != _OPAL_CANDIDATE_TABLE_ROLE:
        raise ValueError(
            "stress_ethanol_cipro_growth OPAL candidate_feature_table role must be "
            f"{_OPAL_CANDIDATE_TABLE_ROLE!r}, found {role!r}"
        )

    table: dict[str, object] = {
        "dataset": dataset,
        "role": role,
        "x_column": x_column,
    }
    if x_source is not None:
        table["x_source"] = x_source
    if ref is not None:
        table["ref"] = ref
        table["resolved_ref"] = _resolve_optional_repo_path(study_context=study_context, raw_path=ref)
    return table


def _resolve_optional_repo_path(*, study_context: object, raw_path: str | None) -> str | None:
    if raw_path is None:
        return None
    repo_root = getattr(study_context, "study_repo_root", None)
    if repo_root is None:
        return raw_path
    return str(resolve_repo_relative_path(repo_root=repo_root, raw_path=raw_path, status_kind=_STATUS_KIND))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


__all__ = ["inspect_opal_surface"]

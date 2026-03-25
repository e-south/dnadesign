"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/status/campaign.py

Metadata-driven campaign and procedure status assembly.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml

from ..catalog import (
    RunbookCatalog,
    load_catalog_procedure_owner_boundary,
    load_catalog_related_registry_ids,
    resolve_catalog_procedure_entry,
)
from .models import CampaignProgress, CampaignScaffold, CampaignScaffoldStep, ProcedureProgress
from .path_ref import PathBase
from .service import load_status_kind_spec, run_status_kind


def build_procedure_progress(
    catalog: RunbookCatalog,
    registry_id: str,
    *,
    raw_inputs: Mapping[str, object] | None,
    manifest_dir: Path | None = None,
    default_path_base: PathBase | None = None,
) -> ProcedureProgress:
    entry = resolve_catalog_procedure_entry(catalog, registry_id)
    state, summary, evidence = run_status_kind(
        entry.progress_kind,
        repo_root=catalog.repo_root,
        raw_inputs=raw_inputs,
        manifest_dir=manifest_dir,
        default_path_base=default_path_base,
    )
    return ProcedureProgress(
        registry_id=entry.registry_id,
        title=entry.title,
        doc_path=entry.doc_path,
        owner_boundary=load_catalog_procedure_owner_boundary(catalog, entry),
        progress_kind=entry.progress_kind,
        label=None,
        state=state,
        summary=summary,
        evidence=evidence,
    )


def load_campaign_progress(catalog: RunbookCatalog, *, manifest_path: Path) -> CampaignProgress:
    resolved_manifest = manifest_path.expanduser().resolve()
    manifest_dir = resolved_manifest.parent
    if not resolved_manifest.exists():
        raise ValueError(f"campaign manifest not found: {resolved_manifest}")
    payload = yaml.safe_load(resolved_manifest.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("campaign manifest must be a mapping with 'campaign_id' and 'steps'")

    manifest_version = int(payload.get("version") or 0)
    if manifest_version != 2:
        raise ValueError("campaign manifest must declare version: 2")

    campaign_id = str(payload.get("campaign_id") or resolved_manifest.stem).strip()
    steps_payload = payload.get("steps")
    if not isinstance(steps_payload, list) or not steps_payload:
        raise ValueError("campaign manifest must define a non-empty 'steps' list")

    manifest_path_base = _resolve_campaign_path_base(payload=payload)

    steps: list[ProcedureProgress] = []
    for index, step_payload in enumerate(steps_payload, start=1):
        if not isinstance(step_payload, dict):
            raise ValueError(f"campaign manifest step {index} must be a mapping")
        registry_id = str(step_payload.get("registry_id") or "").strip()
        if not registry_id:
            raise ValueError(f"campaign manifest step {index} missing 'registry_id'")
        if catalog.find_procedure(registry_id) is None:
            raise ValueError(f"unknown registry id: {registry_id}")
        raw_inputs = _load_campaign_step_inputs(
            step_payload=step_payload,
            step_index=index,
        )
        try:
            step = build_procedure_progress(
                catalog,
                registry_id,
                raw_inputs=raw_inputs,
                manifest_dir=manifest_dir,
                default_path_base=manifest_path_base,
            )
        except FileNotFoundError as exc:
            missing_path = exc.filename or str(exc)
            raise ValueError(
                f"campaign manifest step {index} ({registry_id}) references a missing file: {missing_path}"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"campaign manifest step {index} ({registry_id}): {exc}") from exc

        label = str(step_payload.get("label") or "").strip()
        if label:
            step = ProcedureProgress(
                registry_id=step.registry_id,
                title=step.title,
                doc_path=step.doc_path,
                owner_boundary=step.owner_boundary,
                progress_kind=step.progress_kind,
                label=label,
                state=step.state,
                summary=step.summary,
                evidence=dict(step.evidence),
            )
        steps.append(step)
    return CampaignProgress(
        manifest_path=resolved_manifest,
        campaign_id=campaign_id or resolved_manifest.stem,
        steps=tuple(steps),
        manifest_version=manifest_version,
        path_base=manifest_path_base,
    )


def build_campaign_scaffold(
    catalog: RunbookCatalog,
    *,
    registry_ids: Sequence[str],
    campaign_id: str | None = None,
    related_to: str | None = None,
) -> CampaignScaffold:
    normalized_registry_ids = _resolve_campaign_scaffold_registry_ids(
        catalog,
        registry_ids=registry_ids,
        related_to=related_to,
    )

    resolved_campaign_id = str(campaign_id or "progress_campaign").strip() or "progress_campaign"
    used_labels: Counter[str] = Counter()
    steps: list[CampaignScaffoldStep] = []
    for registry_id in normalized_registry_ids:
        entry = resolve_catalog_procedure_entry(catalog, registry_id)
        spec = load_status_kind_spec(entry.progress_kind)
        label = _suggest_scaffold_label(entry.registry_id, used_labels)
        steps.append(
            CampaignScaffoldStep(
                registry_id=entry.registry_id,
                title=entry.title,
                doc_path=entry.doc_path,
                owner_boundary=load_catalog_procedure_owner_boundary(catalog, entry),
                progress_kind=spec.progress_kind,
                label=label,
                input_schema=spec.input_schema,
            )
        )
    return CampaignScaffold(campaign_id=resolved_campaign_id, steps=tuple(steps))


def _resolve_campaign_scaffold_registry_ids(
    catalog: RunbookCatalog,
    *,
    registry_ids: Sequence[str],
    related_to: str | None,
) -> tuple[str, ...]:
    ordered_registry_ids: list[str] = []
    seen_registry_ids: set[str] = set()

    normalized_related_to = str(related_to or "").strip()
    if normalized_related_to:
        for registry_id in load_catalog_related_registry_ids(catalog, normalized_related_to, include_self=True):
            if registry_id in seen_registry_ids:
                continue
            ordered_registry_ids.append(registry_id)
            seen_registry_ids.add(registry_id)

    for registry_id in registry_ids:
        normalized_registry_id = registry_id.strip()
        if not normalized_registry_id or normalized_registry_id in seen_registry_ids:
            continue
        ordered_registry_ids.append(normalized_registry_id)
        seen_registry_ids.add(normalized_registry_id)

    if not ordered_registry_ids:
        raise ValueError("progress scaffold requires at least one registry id or --related-to")
    return tuple(ordered_registry_ids)


def _resolve_campaign_path_base(
    *,
    payload: Mapping[str, object],
) -> PathBase:
    path_base = str(payload.get("path_base") or "").strip().lower()
    if path_base not in {"repo", "manifest", "cwd"}:
        raise ValueError("campaign manifest must define path_base as one of: repo, manifest, cwd")
    return cast(PathBase, path_base)


def _load_campaign_step_inputs(
    *,
    step_payload: Mapping[str, object],
    step_index: int,
) -> dict[str, object]:
    allowed_keys = {"inputs", "label", "registry_id"}
    unexpected = sorted(
        {str(key).strip() for key in step_payload if str(key).strip() and str(key).strip() not in allowed_keys}
    )
    if unexpected:
        raise ValueError(
            f"campaign manifest step {step_index} must place provider inputs under 'inputs': {', '.join(unexpected)}"
        )
    if "inputs" in step_payload:
        inputs_payload = step_payload.get("inputs") or {}
        if not isinstance(inputs_payload, dict):
            raise ValueError(f"campaign manifest step {step_index} inputs must be a mapping")
        return {str(name).strip(): value for name, value in inputs_payload.items() if str(name).strip()}
    return {}


def _suggest_scaffold_label(registry_id: str, used_labels: Counter[str]) -> str:
    base = registry_id.split(".")[-1].strip() or "step"
    used_labels[base] += 1
    if used_labels[base] == 1:
        return base
    return f"{base}-{used_labels[base]}"


__all__ = [
    "build_campaign_scaffold",
    "build_procedure_progress",
    "load_campaign_progress",
]

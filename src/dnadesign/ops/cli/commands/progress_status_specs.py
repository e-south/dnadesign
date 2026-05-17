"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli/commands/progress_status_specs.py

Lazy status-spec accessors for OPS progress commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dnadesign.ops.catalog import CatalogProcedureEntry

if TYPE_CHECKING:
    from dnadesign.ops.status import CampaignScaffold, CampaignStatus, InputFieldSpec, ProcedureStatus
    from dnadesign.ops.status.models import StatusKindSpec


def build_campaign_scaffold(*args, **kwargs) -> CampaignScaffold:
    from dnadesign.ops.status.campaign import build_campaign_scaffold

    return build_campaign_scaffold(*args, **kwargs)


def build_procedure_status(*args, **kwargs) -> ProcedureStatus:
    from dnadesign.ops.status.campaign import build_procedure_status

    return build_procedure_status(*args, **kwargs)


def list_status_kind_specs() -> tuple[StatusKindSpec, ...]:
    from dnadesign.ops.status.registry_loader import list_status_kind_specs

    return list_status_kind_specs()


def load_campaign_status(*args, **kwargs) -> CampaignStatus:
    from dnadesign.ops.status.campaign import load_campaign_status

    return load_campaign_status(*args, **kwargs)


def load_status_kind_spec(status_kind: str) -> StatusKindSpec:
    from dnadesign.ops.status.registry_loader import load_status_kind_spec

    return load_status_kind_spec(status_kind)


def status_required_inputs(status_kind: str) -> tuple[InputFieldSpec, ...]:
    return load_status_kind_spec(status_kind).required_inputs


def status_optional_inputs(status_kind: str) -> tuple[tuple[str, str], ...]:
    spec = load_status_kind_spec(status_kind)
    return tuple((field.cli_flag, field.summary) for field in spec.optional_inputs)


def status_notes(entry: CatalogProcedureEntry) -> tuple[str, ...]:
    return load_status_kind_spec(entry.status_kind).notes


__all__ = [
    "build_campaign_scaffold",
    "build_procedure_status",
    "list_status_kind_specs",
    "load_campaign_status",
    "load_status_kind_spec",
    "status_notes",
    "status_optional_inputs",
    "status_required_inputs",
]

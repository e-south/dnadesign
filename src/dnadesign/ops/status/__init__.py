"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/status/__init__.py

Neutral status/observation package surface with lazy execution-module exports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .models import (
    CampaignScaffold,
    CampaignScaffoldStep,
    CampaignStatus,
    InputFieldSpec,
    ProcedureStatus,
    StatusKindSpec,
)


def build_campaign_scaffold(*args, **kwargs):
    from .campaign import build_campaign_scaffold as _build_campaign_scaffold

    return _build_campaign_scaffold(*args, **kwargs)


def build_procedure_status(*args, **kwargs):
    from .campaign import build_procedure_status as _build_procedure_status

    return _build_procedure_status(*args, **kwargs)


def build_status_inputs(*args, **kwargs):
    from .service import build_status_inputs as _build_status_inputs

    return _build_status_inputs(*args, **kwargs)


def list_status_kind_specs(*args, **kwargs):
    from .registry_loader import list_status_kind_specs as _list_status_kind_specs

    return _list_status_kind_specs(*args, **kwargs)


def load_campaign_status(*args, **kwargs):
    from .campaign import load_campaign_status as _load_campaign_status

    return _load_campaign_status(*args, **kwargs)


def load_status_kind_spec(*args, **kwargs):
    from .registry_loader import load_status_kind_spec as _load_status_kind_spec

    return _load_status_kind_spec(*args, **kwargs)


def run_status_kind(*args, **kwargs):
    from .service import run_status_kind as _run_status_kind

    return _run_status_kind(*args, **kwargs)


__all__ = [
    "CampaignScaffold",
    "CampaignScaffoldStep",
    "CampaignStatus",
    "InputFieldSpec",
    "ProcedureStatus",
    "StatusKindSpec",
    "build_campaign_scaffold",
    "build_procedure_status",
    "build_status_inputs",
    "list_status_kind_specs",
    "load_campaign_status",
    "load_status_kind_spec",
    "run_status_kind",
]

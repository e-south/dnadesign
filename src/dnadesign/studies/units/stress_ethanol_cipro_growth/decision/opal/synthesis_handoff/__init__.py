"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/__init__.py

Study-owned OPAL synthesis handoff public facade.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_EXPORTS = {
    "CloningStrategy": ".contracts",
    "DEFAULT_SYNTHESIS_HANDOFF_RECORD": ".records",
    "DEFAULT_STRESS_OPAL_CAMPAIGN_CONFIGS": ".campaigns",
    "SelectedCandidate": ".contracts",
    "apply_handoff_record_lifecycle": ".records",
    "artifact_status_for_handoff_record": ".records",
    "build_batch0_selected_candidates": ".batch0_source",
    "build_genbank_feature_table": ".genbank",
    "build_synthesis_manifest": ".manifest",
    "campaign_synthesis_artifact_paths": ".exports",
    "campaign_synthesis_output_dir": ".exports",
    "genbank_record_filename": ".genbank",
    "get_synthesis_handoff_record": ".records",
    "handoff_record_payload": ".records",
    "load_cloning_strategy": ".strategy",
    "load_synthesis_handoff_records": ".records",
    "read_azenta_workbook": ".azenta",
    "read_genbank_records": ".genbank",
    "render_azenta_workbook": ".azenta",
    "render_campaign_scoped_exports": ".exports",
    "render_genbank_record_set": ".genbank",
    "run_id_by_campaign_from_handoff_record": ".records",
    "selected_candidates_from_batch0_review": ".batch0_source",
    "selected_candidates_from_opal_round_campaigns": ".opal_round_source",
    "source_mode_from_handoff_record": ".records",
    "validate_azenta_workbook": ".azenta",
    "validate_genbank_record_set": ".genbank",
    "validate_manifest_against_handoff_record": ".records",
}

__all__ = sorted(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _PUBLIC_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_PUBLIC_EXPORTS])

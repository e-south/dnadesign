"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/adapter_loader.py

Explicit study-family adapter loading for normalized checked-in study records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib

from .models import StudyFamilyAdapter


def load_study_family_adapter(family: str) -> StudyFamilyAdapter:
    family_id = str(family or "").strip()
    if not family_id:
        raise ValueError("study family must be non-empty")
    module = importlib.import_module(f"dnadesign.studies.families.{family_id}.adapter")
    adapter = getattr(module, "STUDY_FAMILY_ADAPTER", None)
    if adapter is None:
        raise ValueError(f"study family adapter module did not expose STUDY_FAMILY_ADAPTER: {family_id}")
    if not callable(getattr(adapter, "load_context", None)):
        raise ValueError(f"study family adapter is missing load_context: {family_id}")
    return adapter


__all__ = ["load_study_family_adapter"]

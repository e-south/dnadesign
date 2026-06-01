"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_api.py

Adapter for the Reader-owned SPOP scoring API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any


class ReaderSpopApiError(ValueError):
    """Raised when the Reader SPOP public API cannot be loaded."""


@dataclass(frozen=True, slots=True)
class ReaderSpopApi:
    """Loaded Reader SPOP API surface used by the RT-lnRNA bridge."""

    metric_id: str
    metric_family: str
    numeric_scope: str
    normalization_basis: str
    reporter_readout: str
    viability_readout: str
    default_lambda: float
    dose_value_factory: Callable[..., Any]
    score_endpoint: Callable[..., Any]
    scoring_error_type: type[Exception]
    source_path: str


def load_reader_spop_api(reader_root: Path) -> ReaderSpopApi:
    """Load Reader's public SPOP scorer from a sibling checkout or package."""

    return _load_reader_spop_api(str(Path(reader_root).expanduser().resolve()))


@cache
def _load_reader_spop_api(reader_root: str) -> ReaderSpopApi:
    root = Path(reader_root)
    module_path = root / "src" / "reader" / "domains" / "plate_reader" / "analysis" / "spop.py"
    module = _load_module_from_path(module_path) if module_path.exists() else _load_module_from_package()
    return _api_from_module(module)


def _load_module_from_path(module_path: Path) -> ModuleType:
    module_name = "_dnadesign_reader_spop_api"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ReaderSpopApiError(f"Reader SPOP API cannot be loaded from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_module_from_package() -> ModuleType:
    try:
        return importlib.import_module("reader.domains.plate_reader.analysis.spop")
    except ImportError as exc:
        raise ReaderSpopApiError(
            "Reader SPOP API is required. Provide --reader-root pointing at a Reader checkout "
            "with src/reader/domains/plate_reader/analysis/spop.py, or install the reader package."
        ) from exc


def _api_from_module(module: ModuleType) -> ReaderSpopApi:
    required = {
        "SPOP_METRIC_ID": str,
        "SPOP_ACRONYM": str,
        "SPOP_NUMERIC_SCOPE": str,
        "SPOP_NORMALIZATION_BASIS": str,
        "SPOP_REPORTER_READOUT": str,
        "SPOP_VIABILITY_READOUT": str,
        "SPOP_DEFAULT_LAMBDA": (int, float),
        "SpopDoseValue": type,
        "SpopScoringError": type,
    }
    for name, expected_type in required.items():
        if not hasattr(module, name):
            raise ReaderSpopApiError(f"Reader SPOP API missing required symbol: {name}")
        value = getattr(module, name)
        if not isinstance(value, expected_type):
            raise ReaderSpopApiError(f"Reader SPOP API symbol {name} has unexpected type: {type(value).__name__}")
    score_endpoint = getattr(module, "score_spop_endpoint", None)
    if not callable(score_endpoint):
        raise ReaderSpopApiError("Reader SPOP API missing callable score_spop_endpoint")
    source_path = str(getattr(module, "__file__", "reader.domains.plate_reader.analysis.spop"))
    return ReaderSpopApi(
        metric_id=str(module.SPOP_METRIC_ID),
        metric_family=str(module.SPOP_ACRONYM),
        numeric_scope=str(module.SPOP_NUMERIC_SCOPE),
        normalization_basis=str(module.SPOP_NORMALIZATION_BASIS),
        reporter_readout=str(module.SPOP_REPORTER_READOUT),
        viability_readout=str(module.SPOP_VIABILITY_READOUT),
        default_lambda=float(module.SPOP_DEFAULT_LAMBDA),
        dose_value_factory=module.SpopDoseValue,
        score_endpoint=score_endpoint,
        scoring_error_type=module.SpopScoringError,
        source_path=source_path,
    )

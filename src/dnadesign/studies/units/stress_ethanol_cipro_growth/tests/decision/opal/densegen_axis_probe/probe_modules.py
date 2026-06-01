from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

PROBE_PACKAGE = "dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.densegen_axis_probe"


def probe_module(relative_module: str) -> ModuleType:
    return import_module(f"{PROBE_PACKAGE}.{relative_module}")


def probe_object(relative_module: str, name: str) -> Any:
    return getattr(probe_module(relative_module), name)

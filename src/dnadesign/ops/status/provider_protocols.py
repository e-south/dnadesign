"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/status/provider_protocols.py

Provider protocols for lazy-loaded ops status providers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

StatusProviderResult = tuple[str, str, dict[str, object]]


class StatusProvider(Protocol):
    def __call__(
        self,
        *,
        repo_root: Path | None,
        inputs: Mapping[str, object],
    ) -> StatusProviderResult: ...


__all__ = ["StatusProvider", "StatusProviderResult"]

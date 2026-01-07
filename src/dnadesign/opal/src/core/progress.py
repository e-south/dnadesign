"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/core/progress.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/core/progress.py

Minimal progress hooks for runtime workflows (CLI-agnostic).
Provides a NullProgress implementation and a small protocol surface that
CLI/TUI layers can implement (e.g., Rich progress bars).
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


class ProgressTracker(Protocol):
    """Minimal progress surface used by runtime code."""

    def advance(self, n: int = 1) -> None:  # pragma: no cover - interface
        ...

    def close(self) -> None:  # pragma: no cover - interface
        ...

    def __enter__(self) -> "ProgressTracker":  # pragma: no cover - interface
        ...

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - interface
        ...


ProgressFactory = Callable[[str, int], ProgressTracker]


@dataclass
class NullProgress:
    """No-op progress tracker for non-TUI or test contexts."""

    def __enter__(self) -> "NullProgress":
        return self

    def advance(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

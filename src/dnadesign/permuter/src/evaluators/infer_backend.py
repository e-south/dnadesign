"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/evaluators/infer_backend.py

Lazy public Infer facade loading for Evo2-backed evaluators.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_RUN_EXTRACT: Callable[..., Any] | None = None


def infer_run_extract() -> Callable[..., Any]:
    global _RUN_EXTRACT
    if _RUN_EXTRACT is not None:
        return _RUN_EXTRACT
    try:
        import dnadesign.infer as infer_api
    except Exception as exc:
        raise RuntimeError(
            "Evo2 backend unavailable: public dnadesign.infer facade is not importable. "
            "Install the Infer/Evo2 runtime dependencies for closed-loop scoring."
        ) from exc
    run_extract = getattr(infer_api, "run_extract", None)
    if not callable(run_extract):
        raise RuntimeError("Evo2 backend unavailable: dnadesign.infer.run_extract is not callable.")
    _RUN_EXTRACT = run_extract
    return run_extract

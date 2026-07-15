"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/model_evidence/evaluator_protocol.py

Scientific evaluator source fingerprint for protocol comparability.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .contracts import ModelEvidenceError
from .fields import required_mapping, sha256_digest

_PACKAGE = "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/"
_PREFIXES = (_PACKAGE + "core/", _PACKAGE + "evaluation/")
_EXACT = {
    _PACKAGE + "runtime/response_screen.py",
    _PACKAGE + "config/response_model_screen_selection.yaml",
}
_REQUIRED = {
    _PACKAGE + "core/response_contracts.py",
    _PACKAGE + "evaluation/greedy_support.py",
    _PACKAGE + "evaluation/grouped_models.py",
    _PACKAGE + "evaluation/model_screen.py",
    _PACKAGE + "evaluation/response_uncertainty.py",
    _PACKAGE + "runtime/response_screen.py",
    _PACKAGE + "config/response_model_screen_selection.yaml",
}


def evaluator_sources(source: dict[str, object]) -> list[dict[str, str]]:
    """Return only sources that can change the scientific evaluation result."""

    files = source.get("files")
    if not isinstance(files, list):
        raise ModelEvidenceError("source.files must be a provenance list.")
    selection_config = required_mapping(required_mapping(source, "response_measurement_selection"), "config")
    rows: list[dict[str, str]] = []
    for raw in [*files, selection_config]:
        if not isinstance(raw, dict):
            continue
        path = raw.get("path")
        if not isinstance(path, str) or not _is_evaluator_source(path):
            continue
        rows.append({"path": path, "sha256": sha256_digest(raw.get("sha256"), f"evaluator source {path}")})
    observed = {row["path"] for row in rows}
    missing = sorted(_REQUIRED - observed)
    if missing:
        raise ModelEvidenceError(f"source.files is missing required scientific evaluator {missing[0]!r}.")
    if len(observed) != len(rows):
        raise ModelEvidenceError("source.files contains duplicate scientific evaluator paths.")
    return sorted(rows, key=lambda row: row["path"])


def _is_evaluator_source(path: str) -> bool:
    if path in _EXACT:
        return True
    return path.endswith(".py") and not path.endswith("/__init__.py") and path.startswith(_PREFIXES)


__all__ = ["evaluator_sources"]

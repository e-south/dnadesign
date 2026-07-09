"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/renderer.py

Notebook-set template builders for renderer OPAL analysis notebook set template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ...core.utils import ExitCodes, OpalError
from .cells import OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION, render_campaign_set_template


def render_campaign_set_notebook(
    config_paths: list[Path],
    *,
    round_selector: str,
    run_id: str | None = None,
    collection_manifest_path: str | Path | None = None,
    collection_visual_index_path: str | Path | None = None,
) -> str:
    """Render a marimo notebook template for an OPAL campaign set."""

    config_path_literals = _campaign_review_config_path_literals(config_paths)
    try:
        import marimo as _marimo
    except Exception:
        _marimo = None
    marimo_version = "unknown" if _marimo is None else getattr(_marimo, "__version__", "unknown")
    return (
        render_campaign_set_template()
        .replace("__CONFIG_PATHS__", config_path_literals)
        .replace(
            "__COLLECTION_MANIFEST_PATH__",
            _optional_path_literal(collection_manifest_path),
        )
        .replace(
            "__COLLECTION_VISUAL_INDEX_PATH__",
            _optional_path_literal(collection_visual_index_path),
        )
        .replace("__DEFAULT_ROUND__", repr(str(round_selector)))
        .replace("__DEFAULT_RUN_ID__", repr(str(run_id)) if run_id else "None")
        .replace("__OPAL_NOTEBOOK_TEMPLATE_SCHEMA__", OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION)
        .replace("__GENERATED_WITH__", str(marimo_version))
        + "\n"
    )


def _campaign_review_config_path_literals(config_paths: list[Path]) -> str:
    paths = [Path(path) for path in config_paths]
    if not paths:
        raise OpalError("Campaign review notebooks require at least one campaign config.", ExitCodes.BAD_ARGS)
    resolved = [str(path.resolve()) for path in paths]
    duplicates = sorted({path for path in resolved if resolved.count(path) > 1})
    if duplicates:
        raise OpalError(
            "Campaign review notebooks require distinct campaign configs; duplicates: " + ", ".join(duplicates),
            ExitCodes.BAD_ARGS,
        )
    items = [_wrapped_string_item(str(path)) for path in paths]
    return "[\n" + ",\n".join(items) + "\n]"


def _optional_path_literal(path: str | Path | None) -> str:
    if path is None:
        return "None"
    return _wrapped_string_expression(str(Path(path)))


def _wrapped_string_item(value: str) -> str:
    expression = _wrapped_string_expression(value, base_indent="    ")
    return "    " + expression.replace("\n", "\n    ")


def _wrapped_string_expression(value: str, *, base_indent: str = "") -> str:
    literal = repr(value)
    if len(base_indent) + len(literal) <= 100:
        return literal
    chunk_size = 88
    chunks = [value[index : index + chunk_size] for index in range(0, len(value), chunk_size)]
    lines = ["("]
    lines.extend(f"{base_indent}    {chunk!r}" for chunk in chunks)
    lines.append(f"{base_indent})")
    return "\n".join(lines)


__all__ = ["render_campaign_set_notebook"]

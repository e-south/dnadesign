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
            repr(str(Path(collection_manifest_path))) if collection_manifest_path is not None else "None",
        )
        .replace(
            "__COLLECTION_VISUAL_INDEX_PATH__",
            repr(str(Path(collection_visual_index_path))) if collection_visual_index_path is not None else "None",
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
    return repr([str(path) for path in paths])


__all__ = ["render_campaign_set_notebook"]

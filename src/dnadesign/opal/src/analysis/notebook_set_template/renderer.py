from __future__ import annotations

from pathlib import Path

from .cells import OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION, render_campaign_set_template


def render_campaign_set_notebook(
    config_paths: list[Path],
    *,
    round_selector: str,
    run_id: str | None = None,
    collection_manifest_path: str | Path | None = None,
) -> str:
    """Render a marimo notebook template for an OPAL campaign set."""

    try:
        import marimo as _marimo
    except Exception:
        _marimo = None
    marimo_version = "unknown" if _marimo is None else getattr(_marimo, "__version__", "unknown")
    path_literals = repr([str(Path(path)) for path in config_paths])
    return (
        render_campaign_set_template()
        .replace("__CONFIG_PATHS__", path_literals)
        .replace(
            "__COLLECTION_MANIFEST_PATH__",
            repr(str(Path(collection_manifest_path))) if collection_manifest_path is not None else "None",
        )
        .replace("__DEFAULT_ROUND__", repr(str(round_selector)))
        .replace("__DEFAULT_RUN_ID__", repr(str(run_id)) if run_id else "None")
        .replace("__OPAL_NOTEBOOK_TEMPLATE_SCHEMA__", OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION)
        .replace("__GENERATED_WITH__", str(marimo_version))
        + "\n"
    )


__all__ = ["render_campaign_set_notebook"]

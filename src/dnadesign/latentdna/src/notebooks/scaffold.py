"""
Notebook scaffold rendering helpers for latentdna.
"""

from __future__ import annotations

import json
from textwrap import dedent


def _marimo_version() -> str:
    try:
        import marimo as _marimo
    except Exception:
        return "unknown"
    return getattr(_marimo, "__version__", "unknown")


def render_artifact_review_notebook(
    *,
    workspace_id: str,
    notebook_id: str,
    title: str,
    description: str | None,
    artifacts: list[dict[str, str]],
) -> str:
    artifact_payload = json.dumps(artifacts, indent=2, sort_keys=True)
    description_text = description or "Load persisted latentdna artifacts without recomputing them."
    template = dedent(
        """\
        import marimo

        __generated_with = "__GENERATED_WITH__"

        app = marimo.App(width="full")


        @app.cell
        def _():
            import json
            from pathlib import Path

            import marimo as mo
            import numpy as np
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq

            TITLE = __TITLE__
            DESCRIPTION = __DESCRIPTION__
            WORKSPACE_ID = __WORKSPACE_ID__
            NOTEBOOK_ID = __NOTEBOOK_ID__
            ARTIFACTS = json.loads(__ARTIFACTS_JSON__)
            WORKSPACE_DIR = Path(__file__).resolve().parents[4]
            PLOT_ARTIFACT_ROOT = WORKSPACE_DIR / "outputs" / "latentdna" / "plots"
            PREVIEW_ROWS = 200
            RENDERABLE_SUFFIXES = {".svg", ".png", ".jpg", ".jpeg", ".webp", ".html"}

            def format_bytes(num_bytes: int) -> str:
                size = float(num_bytes)
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if size < 1024.0 or unit == "TB":
                        return f"{size:.1f} {unit}"
                    size /= 1024.0
                return f"{size:.1f} TB"

            def pretty_json(payload: object) -> str:
                return json.dumps(payload, indent=2, sort_keys=True)

            def artifact_record(alias: str) -> dict[str, str]:
                for artifact in ARTIFACTS:
                    if artifact["alias"] == alias:
                        return artifact
                raise KeyError(f"unknown artifact alias: {alias}")

            def artifact_path(alias: str) -> Path:
                artifact = artifact_record(alias)
                return WORKSPACE_DIR / artifact["path"]

            def discover_plot_artifacts() -> list[dict[str, str]]:
                if not PLOT_ARTIFACT_ROOT.is_dir():
                    return []
                discovered: list[dict[str, str]] = []
                for candidate in sorted(PLOT_ARTIFACT_ROOT.iterdir()):
                    if not candidate.is_dir():
                        continue
                    if not (candidate / "manifest.json").is_file():
                        continue
                    discovered.append(
                        {
                            "alias": candidate.name,
                            "kind": "plot",
                            "id": candidate.name,
                            "path": candidate.relative_to(WORKSPACE_DIR).as_posix(),
                        }
                    )
                return discovered

            def artifact_files(base: Path, manifest: dict[str, object] | None) -> list[Path]:
                files: list[Path] = []
                if manifest is not None:
                    for output in manifest.get("outputs", []):
                        output_path = output.get("path") if isinstance(output, dict) else None
                        if not output_path:
                            continue
                        candidate = base / str(output_path)
                        if candidate.is_file():
                            files.append(candidate)
                if not files and base.exists():
                    files = sorted(candidate for candidate in base.iterdir() if candidate.is_file())
                return files

            def preview_parquet(path: Path, *, limit: int = PREVIEW_ROWS) -> tuple[pd.DataFrame, dict[str, object]]:
                parquet = pq.ParquetFile(path)
                metadata = parquet.metadata
                batch = next(parquet.iter_batches(batch_size=limit), None)
                if batch is None:
                    preview_df = pd.DataFrame()
                else:
                    preview_df = pa.Table.from_batches([batch]).to_pandas()
                return preview_df, {
                    "rows": int(metadata.num_rows),
                    "columns": int(metadata.num_columns),
                    "row_groups": int(metadata.num_row_groups),
                }

            def summarize_array(path: Path) -> list[dict[str, object]]:
                if path.suffix == ".npy":
                    array = np.load(path, mmap_mode="r")
                    return [
                        {
                            "file": path.name,
                            "array": path.stem,
                            "shape": list(array.shape),
                            "dtype": str(array.dtype),
                        }
                    ]
                bundle = np.load(path, mmap_mode="r")
                try:
                    return [
                        {
                            "file": path.name,
                            "array": key,
                            "shape": list(bundle[key].shape),
                            "dtype": str(bundle[key].dtype),
                        }
                        for key in bundle.files
                    ]
                finally:
                    bundle.close()

            def render_file(path: Path):
                suffix = path.suffix.lower()
                if suffix == ".svg":
                    svg = path.read_text(encoding="utf-8")
                    return mo.Html(
                        "<div style='width: 100%; overflow-x: auto; padding: 0.5rem 0;'>"
                        f"{svg}"
                        "</div>"
                    )
                if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                    return mo.image(path.read_bytes(), alt=path.name, width="100%")
                if suffix == ".html":
                    return mo.Html(path.read_text(encoding="utf-8"))
                return mo.callout(f"No inline renderer for `{path.name}`.", kind="info")

            def load_artifact_record(artifact: dict[str, str]) -> dict[str, object]:
                base = WORKSPACE_DIR / artifact["path"]
                manifest_path = base / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
                files = artifact_files(base, manifest)

                file_rows: list[dict[str, object]] = []
                table_previews: dict[str, dict[str, object]] = {}
                array_rows: list[dict[str, object]] = []
                renderable_files: list[Path] = []

                for file_path in files:
                    file_rows.append(
                        {
                            "file": file_path.name,
                            "suffix": file_path.suffix or "<none>",
                            "size": format_bytes(file_path.stat().st_size),
                        }
                    )
                    if file_path.suffix.lower() == ".parquet":
                        preview_df, summary = preview_parquet(file_path)
                        table_previews[file_path.name] = {
                            "preview": preview_df,
                            "summary": summary,
                        }
                    elif file_path.suffix.lower() in {".npy", ".npz"}:
                        array_rows.extend(summarize_array(file_path))
                    if file_path.suffix.lower() in RENDERABLE_SUFFIXES:
                        renderable_files.append(file_path)

                return {
                    "artifact": artifact,
                    "path": base,
                    "manifest": manifest,
                    "files": files,
                    "file_rows": file_rows,
                    "table_previews": table_previews,
                    "array_rows": array_rows,
                    "renderable_files": renderable_files,
                }

            def load_artifact(alias: str) -> dict[str, object]:
                return load_artifact_record(artifact_record(alias))

            def build_artifact_details_view(selected_name: str, payload: dict[str, object]):
                artifact = payload["artifact"]
                manifest = payload["manifest"]
                file_rows = payload["file_rows"]
                overview_lines = [
                    "## Selected artifact",
                    "",
                    f"- Alias: `{selected_name}`",
                    f"- Kind: `{artifact['kind']}`",
                    f"- Id: `{artifact['id']}`",
                    f"- Path: `{payload['path']}`",
                ]
                overview = mo.md("\\n".join(overview_lines))
                manifest_view = (
                    mo.md(f"```json\\n{pretty_json(manifest)}\\n```")
                    if manifest is not None
                    else mo.callout("No manifest.json found for this artifact.", kind="warn")
                )
                files_view = (
                    mo.ui.table(pd.DataFrame(file_rows), page_size=min(max(len(file_rows), 1), 10), show_download=False)
                    if file_rows
                    else mo.callout("This artifact directory does not currently expose files to preview.", kind="warn")
                )
                return mo.ui.tabs(
                    {
                        "Overview": overview,
                        "Manifest": manifest_view,
                        "Files": files_view,
                    }
                )

            def build_artifact_preview_view(payload: dict[str, object]):
                artifact = payload["artifact"]
                renderable_files = payload["renderable_files"]
                table_previews = payload["table_previews"]
                array_rows = payload["array_rows"]

                preview_tabs: dict[str, object] = {}
                if renderable_files:
                    renderable_views = {path.name: render_file(path) for path in renderable_files}
                    preview_tabs["Plot outputs" if artifact["kind"] == "plot" else "Rendered files"] = (
                        mo.ui.tabs(renderable_views)
                        if len(renderable_views) > 1
                        else next(iter(renderable_views.values()))
                    )
                if table_previews:
                    table_views = {}
                    for file_name, table_payload in table_previews.items():
                        summary = table_payload["summary"]
                        table_views[file_name] = mo.vstack(
                            [
                                mo.md(
                                    f"**Rows:** {summary['rows']:,}  "
                                    f"**Columns:** {summary['columns']}  "
                                    f"**Row groups:** {summary['row_groups']}"
                                ),
                                mo.ui.dataframe(table_payload["preview"], page_size=10, show_download=False),
                            ],
                            gap=0.75,
                        )
                    preview_tabs["Table previews"] = (
                        mo.ui.tabs(table_views) if len(table_views) > 1 else next(iter(table_views.values()))
                    )
                if array_rows:
                    preview_tabs["Array summaries"] = mo.ui.table(
                        pd.DataFrame(array_rows),
                        page_size=min(max(len(array_rows), 1), 10),
                        show_download=False,
                    )
                if not preview_tabs:
                    preview_tabs["Preview"] = mo.callout(
                        "No inline preview is available for this artifact yet.", kind="info"
                    )
                return mo.ui.tabs(preview_tabs)

            return (
                ARTIFACTS,
                DESCRIPTION,
                NOTEBOOK_ID,
                PLOT_ARTIFACT_ROOT,
                TITLE,
                WORKSPACE_DIR,
                WORKSPACE_ID,
                artifact_path,
                artifact_record,
                build_artifact_details_view,
                build_artifact_preview_view,
                discover_plot_artifacts,
                format_bytes,
                load_artifact,
                load_artifact_record,
                mo,
                pd,
                pretty_json,
                render_file,
            )


        @app.cell
        def _(DESCRIPTION, NOTEBOOK_ID, TITLE, WORKSPACE_ID, mo):
            _header_lines = [
                f"# {TITLE}",
                "",
                DESCRIPTION,
                "",
                f"- Workspace: `{WORKSPACE_ID}`",
                f"- Notebook: `{NOTEBOOK_ID}`",
                f"- Regenerate with: `uv run latentdna notebook generate {NOTEBOOK_ID} --workspace <workspace>`",
                f"- Run as app: `uv run marimo run outputs/latentdna/notebooks/{NOTEBOOK_ID}/notebook.py`",
            ]
            mo.md("\\n".join(_header_lines))
            return


        @app.cell
        def _(ARTIFACTS, mo, pd):
            artifact_catalog = pd.DataFrame(ARTIFACTS)
            _artifact_options = [artifact["alias"] for artifact in ARTIFACTS]
            artifact_picker = mo.ui.dropdown(
                _artifact_options,
                value=ARTIFACTS[0]["alias"],
                label="Artifact",
                searchable=True,
                full_width=True,
            )
            return artifact_catalog, artifact_picker


        @app.cell
        def _(artifact_picker, load_artifact):
            selected_alias = str(artifact_picker.value)
            selected_payload = load_artifact(selected_alias)
            return selected_alias, selected_payload


        @app.cell
        def _(
            artifact_catalog,
            artifact_picker,
            build_artifact_details_view,
            build_artifact_preview_view,
            mo,
            selected_alias,
            selected_payload,
        ):
            _inventory = mo.ui.table(
                artifact_catalog,
                page_size=min(max(len(artifact_catalog), 1), 10),
                show_download=False,
                label="Artifact inventory",
            )
            declared_browser_panel = mo.vstack(
                [
                    mo.md("Notebook-declared artifacts remain the read-only primary review surface."),
                    artifact_picker,
                    _inventory,
                    build_artifact_details_view(selected_alias, selected_payload),
                    build_artifact_preview_view(selected_payload),
                ],
                gap=1.0,
            )
            return (declared_browser_panel,)


        @app.cell
        def _(discover_plot_artifacts, mo, pd):
            workspace_plot_artifacts = discover_plot_artifacts()
            workspace_plot_catalog = pd.DataFrame(
                [
                    {
                        "plot_id": artifact["id"],
                        "path": artifact["path"],
                    }
                    for artifact in workspace_plot_artifacts
                ]
            )
            if workspace_plot_artifacts:
                workspace_plot_picker = mo.ui.dropdown(
                    [artifact["id"] for artifact in workspace_plot_artifacts],
                    value=workspace_plot_artifacts[0]["id"],
                    label="Workspace plot",
                    searchable=True,
                    full_width=True,
                )
            else:
                workspace_plot_picker = None
            return workspace_plot_artifacts, workspace_plot_catalog, workspace_plot_picker


        @app.cell
        def _(load_artifact_record, workspace_plot_artifacts, workspace_plot_picker):
            if not workspace_plot_artifacts or workspace_plot_picker is None:
                selected_workspace_plot_id = None
                selected_workspace_plot_payload = None
            else:
                selected_workspace_plot_id = str(workspace_plot_picker.value)
                selected_workspace_plot = next(
                    artifact for artifact in workspace_plot_artifacts if artifact["id"] == selected_workspace_plot_id
                )
                selected_workspace_plot_payload = load_artifact_record(selected_workspace_plot)
            return selected_workspace_plot_id, selected_workspace_plot_payload


        @app.cell
        def _(
            build_artifact_details_view,
            build_artifact_preview_view,
            mo,
            selected_workspace_plot_id,
            selected_workspace_plot_payload,
            workspace_plot_artifacts,
            workspace_plot_catalog,
            workspace_plot_picker,
        ):
            if not workspace_plot_artifacts or workspace_plot_picker is None or selected_workspace_plot_payload is None:
                workspace_plot_browser_panel = mo.callout(
                    "No persisted plot artifacts were found under `outputs/latentdna/plots` yet.",
                    kind="info",
                )
            else:
                _plot_inventory = mo.ui.table(
                    workspace_plot_catalog,
                    page_size=min(max(len(workspace_plot_catalog), 1), 10),
                    show_download=False,
                    label="Workspace plot inventory",
                )
                workspace_plot_browser_panel = mo.vstack(
                    [
                        mo.md(
                            "This browser scans `outputs/latentdna/plots` at runtime, "
                            "so newly rendered plot artifacts appear here without regenerating the notebook."
                        ),
                        workspace_plot_picker,
                        _plot_inventory,
                        build_artifact_details_view(selected_workspace_plot_id, selected_workspace_plot_payload),
                        build_artifact_preview_view(selected_workspace_plot_payload),
                    ],
                    gap=1.0,
                )
            return (workspace_plot_browser_panel,)


        @app.cell
        def _(declared_browser_panel, mo, workspace_plot_browser_panel):
            mo.ui.tabs(
                {
                    "Declared artifacts": declared_browser_panel,
                    "Workspace plots": workspace_plot_browser_panel,
                }
            )
            return


        if __name__ == "__main__":
            app.run()
        """
    )
    return (
        template.replace("__GENERATED_WITH__", _marimo_version())
        .replace("__TITLE__", repr(title))
        .replace("__DESCRIPTION__", repr(description_text))
        .replace("__WORKSPACE_ID__", repr(workspace_id))
        .replace("__NOTEBOOK_ID__", repr(notebook_id))
        .replace("__ARTIFACTS_JSON__", repr(artifact_payload))
    )
